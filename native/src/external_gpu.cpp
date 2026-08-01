#include "external_gpu.h"
#include "moge2_native.h"

#include "gpu_model.h"
#include "gpu_preprocess.h"
#include "graph_gpu.h"
#include "operators.h"
#include "safetensors.h"
#include "vulkan.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace moge2_native {
namespace {
#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kMaximumJobs = 3u;

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) throw std::runtime_error(
        std::string(operation) + " failed with HRESULT " +
        std::to_string(static_cast<long>(result)));
}

ComPtr<ID3D12Device> matching_device(std::uint64_t luid) {
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT status = factory->EnumAdapters1(index, &adapter);
        if (status == DXGI_ERROR_NOT_FOUND) break;
        check(status, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 descriptor{};
        check(adapter->GetDesc1(&descriptor), "GetDesc1");
        std::uint64_t candidate = 0u;
        std::memcpy(&candidate, &descriptor.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        return device;
    }
    return {};
}

void validate_texture(ID3D12Device* device, std::uintptr_t handle,
    std::uint32_t width, std::uint32_t height, DXGI_FORMAT format) {
    ComPtr<ID3D12Resource> resource;
    check(device->OpenSharedHandle(reinterpret_cast<HANDLE>(handle),
        IID_PPV_ARGS(&resource)), "OpenSharedHandle(MoGe-2)");
    const D3D12_RESOURCE_DESC descriptor = resource->GetDesc();
    if (descriptor.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        descriptor.Width != width || descriptor.Height != height ||
        descriptor.DepthOrArraySize != 1u || descriptor.MipLevels != 1u ||
        descriptor.SampleDesc.Count != 1u || descriptor.Format != format)
        throw std::invalid_argument("MoGe-2 shared texture descriptor mismatch");
}

class Job final : public ExternalJob {
public:
    Job(std::shared_ptr<ExternalGpu> owner, da3_native::VulkanImage input,
        da3_native::VulkanImage output, da3_native::VulkanSubmission submission,
        std::mutex& context_mutex)
        : owner_(std::move(owner)), input_(std::move(input)),
          output_(std::move(output)), submission_(std::move(submission)),
          context_mutex_(&context_mutex) {}
    ~Job() override {
        try { submission_.wait(); } catch (...) {}
        std::lock_guard<std::mutex> lock(*context_mutex_);
        submission_ = {};
        output_ = {};
        input_ = {};
    }
    ExternalJobState state() const override {
        if (cancelled_.load(std::memory_order_relaxed))
            return ExternalJobState::cancelled;
        return submission_.ready() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { cancelled_.store(true, std::memory_order_relaxed); }
private:
    std::shared_ptr<ExternalGpu> owner_;
    da3_native::VulkanImage input_;
    da3_native::VulkanImage output_;
    da3_native::VulkanSubmission submission_;
    std::mutex* context_mutex_ = nullptr;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(const std::string& path, std::uint32_t index)
        : weights_(path), config_(read_model_config(weights_)),
          context_(index), model_(weights_, context_),
          operators_(context_), moge_operators_(context_),
          preprocessor_(context_)
#if defined(_WIN32)
          , d3d12_(matching_device(context_.adapter_luid()))
#endif
    {}

    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const auto& caps = context_.external_capabilities();
        const bool available = d3d12_ && caps.timeline_semaphore &&
            caps.d3d12_resource_import && caps.d3d12_fence_import &&
            caps.d3d12_bgra8_sampled_image_import &&
            caps.d3d12_r32_storage_image_import;
        return {available, available ? context_.adapter_luid() : 0u,
            available ? kMaximumJobs : 0u};
#else
        return {};
#endif
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error("MoGe-2 D3D12 interop requires Windows");
#else
        const auto caps = capabilities();
        if (!caps.available) throw std::runtime_error(
            "MoGe-2 D3D12/Vulkan interop is unavailable");
        if (!request.input_texture || !request.output_texture ||
            !request.wait_fence || !request.signal_fence || !request.width ||
            !request.height || request.num_tokens < 16u ||
            (request.rgba && !context_.external_capabilities().
                d3d12_rgba8_sampled_image_import))
            throw std::invalid_argument("invalid MoGe-2 external request");
        const DXGI_FORMAT input_dxgi = request.rgba ?
            DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
        const VkFormat input_vk = request.rgba ?
            VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
        validate_texture(d3d12_.Get(), request.input_texture,
            request.width, request.height, input_dxgi);
        validate_texture(d3d12_.Get(), request.output_texture,
            request.width, request.height, DXGI_FORMAT_R32_FLOAT);
        const float aspect = float(request.width) / float(request.height);
        const std::uint32_t token_height = std::max(1u, static_cast<std::uint32_t>(
            std::nearbyint(std::sqrt(float(request.num_tokens) / aspect))));
        const std::uint32_t token_width = std::max(1u, static_cast<std::uint32_t>(
            std::nearbyint(std::sqrt(float(request.num_tokens) * aspect))));
        std::lock_guard<std::mutex> lock(record_mutex_);
        auto input = context_.import_d3d12_image(
            reinterpret_cast<void*>(request.input_texture), request.width,
            request.height, input_vk,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        auto output = context_.import_d3d12_image(
            reinterpret_cast<void*>(request.output_texture), request.width,
            request.height, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        auto wait = context_.import_d3d12_fence(
            reinterpret_cast<void*>(request.wait_fence), request.wait_value);
        auto signal = context_.import_d3d12_fence(
            reinterpret_cast<void*>(request.signal_fence), request.signal_value);
        auto submission = context_.segmented_batch_async(
            std::move(wait), std::move(signal), [&] {
                context_.acquire_external_image(input,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT);
                context_.acquire_external_image(output,
                    VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
                const std::uint32_t encoder_width = token_width * 14u;
                const std::uint32_t encoder_height = token_height * 14u;
                auto image = context_.create_device_buffer(
                    std::uint64_t(encoder_width) * encoder_height * 3u * sizeof(float));
                preprocessor_.run_texture(image, input, encoder_width, encoder_height);
                (void)infer_vits_normal(context_, model_, operators_,
                    moge_operators_, config_, std::move(image),
                    encoder_width, encoder_height,
                    request.width, request.height, &output);
                context_.release_external_image(input,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT);
                context_.release_external_image(output,
                    VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
            });
        return std::make_shared<Job>(shared_from_this(), std::move(input),
            std::move(output), std::move(submission), record_mutex_);
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes, std::uint64_t& download_bytes) const override {
        context_.transfer_counters(upload_bytes, download_bytes);
    }
private:
    da3_native::SafeTensors weights_;
    ModelConfig config_;
    da3_native::VulkanContext context_;
    da3_native::GpuModel model_;
    da3_native::VulkanOperators operators_;
    MoGeOperators moge_operators_;
    GpuPreprocessor preprocessor_;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    std::mutex record_mutex_;
#endif
};
}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& path, std::uint32_t index) {
    return std::make_shared<ExternalGpuImpl>(path, index);
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t index) {
#if defined(_WIN32)
    da3_native::VulkanContext context(index);
    const auto& caps = context.external_capabilities();
    const bool available = matching_device(context.adapter_luid()) &&
        caps.timeline_semaphore && caps.d3d12_resource_import &&
        caps.d3d12_fence_import && caps.d3d12_bgra8_sampled_image_import &&
        caps.d3d12_r32_storage_image_import;
    return {available, available ? context.adapter_luid() : 0u,
        available ? kMaximumJobs : 0u};
#else
    (void)index;
    return {};
#endif
}

}  // namespace moge2_native

extern "C" MOGE2_API int moge2_get_transfer_counters(
    std::uint64_t* upload_bytes, std::uint64_t* download_bytes) {
    if (!upload_bytes || !download_bytes) return 1;
    da3_native::global_transfer_counters(*upload_bytes, *download_bytes);
    return 0;
}
