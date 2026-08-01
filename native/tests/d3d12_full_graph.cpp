#include "moge2_native.h"
#include "inferbridge_harness.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Microsoft::WRL::ComPtr;

void check(HRESULT value, const char* operation) {
    if (FAILED(value)) throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::to_string(static_cast<long>(value)));
}
void check(ibrh_result value, const char* operation) {
    if (value != IBRH_OK) throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::to_string(static_cast<unsigned>(value)));
}
void check_counter(int value, const char* operation) {
    if (value != 0) throw std::runtime_error(
        std::string(operation) + " failed");
}

struct Capture {
    ComPtr<ID3D12Resource> texture;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE texture_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t value = 1u;
};

std::vector<std::uint8_t> pixels(
    std::uint32_t width, std::uint32_t height, std::uint32_t frame) {
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(width) * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width + x) * 4u;
            result[offset] = static_cast<std::uint8_t>((x * 11u + y + frame) & 255u);
            result[offset + 1u] = static_cast<std::uint8_t>((x + y * 7u + frame * 3u) & 255u);
            result[offset + 2u] = static_cast<std::uint8_t>((x * 3u + y * 5u + frame * 13u) & 255u);
            result[offset + 3u] = 255u;
        }
    }
    return result;
}

Capture upload_texture(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const std::vector<std::uint8_t>& source,
    std::uint32_t width, std::uint32_t height, bool signal = true) {
    Capture result;
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC texture_desc{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_B8G8R8A8_UNORM, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    check(device->CreateCommittedResource(
        &default_heap, D3D12_HEAP_FLAG_SHARED, &texture_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&result.texture)), "CreateCommittedResource(input)");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0u;
    UINT64 row_bytes = 0u;
    UINT64 upload_bytes = 0u;
    device->GetCopyableFootprints(
        &texture_desc, 0, 1, 0, &footprint, &rows,
        &row_bytes, &upload_bytes);
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC buffer_desc{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, upload_bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE};
    check(device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&result.upload)), "CreateCommittedResource(upload)");
    std::uint8_t* mapped = nullptr;
    const D3D12_RANGE no_read{0, 0};
    check(result.upload->Map(
        0, &no_read, reinterpret_cast<void**>(&mapped)), "Map(upload)");
    for (std::uint32_t y = 0; y < height; ++y)
        std::memcpy(
            mapped + footprint.Offset +
                static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
            source.data() + static_cast<std::size_t>(y) * width * 4u,
            static_cast<std::size_t>(width) * 4u);
    result.upload->Unmap(0, nullptr);

    check(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&result.allocator)),
        "CreateCommandAllocator");
    check(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, result.allocator.Get(), nullptr,
        IID_PPV_ARGS(&result.commands)), "CreateCommandList");
    const D3D12_TEXTURE_COPY_LOCATION destination{
        result.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION upload_location{};
    upload_location.pResource = result.upload.Get();
    upload_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    upload_location.PlacedFootprint = footprint;
    result.commands->CopyTextureRegion(
        &destination, 0, 0, 0, &upload_location, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = result.texture.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    result.commands->ResourceBarrier(1, &barrier);
    check(result.commands->Close(), "Close(upload list)");
    ID3D12CommandList* lists[] = {result.commands.Get()};
    queue->ExecuteCommandLists(1, lists);
    check(device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&result.fence)),
        "CreateFence(input)");
    if (signal) check(queue->Signal(result.fence.Get(), result.value), "Signal(input)");
    check(device->CreateSharedHandle(
        result.texture.Get(), nullptr, GENERIC_ALL, nullptr,
        &result.texture_handle), "CreateSharedHandle(input)");
    check(device->CreateSharedHandle(
        result.fence.Get(), nullptr, GENERIC_ALL, nullptr,
        &result.fence_handle), "CreateSharedHandle(input fence)");
    return result;
}

void close_capture(Capture& value) {
    if (value.texture_handle) CloseHandle(value.texture_handle);
    if (value.fence_handle) CloseHandle(value.fence_handle);
    value.texture_handle = nullptr;
    value.fence_handle = nullptr;
}

void wait_fence(ID3D12Device* device, const ibrh_synchronization& ready) {
    ComPtr<ID3D12Fence> fence;
    check(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(ready.native_handle),
        IID_PPV_ARGS(&fence)), "OpenSharedHandle(output fence)");
    if (fence->GetCompletedValue() >= ready.value) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed");
    check(fence->SetEventOnCompletion(ready.value, event), "SetEventOnCompletion");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
}

std::vector<float> read_output(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const ibrh_output_descriptor& output) {
    wait_fence(device, output.ready);
    ComPtr<ID3D12Resource> texture;
    check(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(output.resource.native_handle),
        IID_PPV_ARGS(&texture)), "OpenSharedHandle(output texture)");
    const D3D12_RESOURCE_DESC description = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
    UINT64 bytes = 0;
    device->GetCopyableFootprints(
        &description, 0, 1, 0, &footprint, &rows, &row_bytes, &bytes);
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC buffer{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> readback;
    check(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback)), "CreateCommittedResource(readback)");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator(readback)");
    check(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
        IID_PPV_ARGS(&list)), "CreateCommandList(readback)");
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = texture.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    check(list->Close(), "Close(readback list)");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> done;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)),
          "CreateFence(readback)");
    check(queue->Signal(done.Get(), 1), "Signal(readback)");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(done->SetEventOnCompletion(1, event), "SetEventOnCompletion(readback)");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    const std::uint8_t* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    check(readback->Map(0, &range, reinterpret_cast<void**>(
        const_cast<std::uint8_t**>(&mapped))), "Map(readback)");
    std::vector<float> result(
        static_cast<std::size_t>(output.resource.width) * output.resource.height);
    for (std::uint32_t y = 0; y < output.resource.height; ++y)
        std::memcpy(
            result.data() + static_cast<std::size_t>(y) * output.resource.width,
            mapped + footprint.Offset +
                static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
            static_cast<std::size_t>(output.resource.width) * sizeof(float));
    readback->Unmap(0, nullptr);
    return result;
}

}  // namespace

namespace {
struct Abi2Device {
    ComPtr<ID3D12Device> device;
    std::string luid_json;
    std::string name;
};

Abi2Device select_abi2_device(const ibrh_api& api) {
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        const auto* bytes = reinterpret_cast<const unsigned char*>(&description.AdapterLuid);
        char json[48]{};
        std::snprintf(json, sizeof(json),
            "{\"luid\":\"%02x%02x%02x%02x%02x%02x%02x%02x\"}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7]);
        ibrh_runtime_create_request request{};
        request.struct_size = sizeof(request);
        request.api_version = IBRH_CURRENT_API_VERSION;
        request.backend = {"native", 6u};
        request.requested_device_json = {json, std::strlen(json)};
        ibrh_runtime* runtime = nullptr;
        if (api.runtime_create(sizeof(request), &request, &runtime) != IBRH_OK)
            continue;
        api.runtime_destroy(runtime);
        Abi2Device result;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&result.device)), "D3D12CreateDevice");
        result.luid_json = json;
        char name[128]{};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1,
            name, sizeof(name), nullptr, nullptr);
        result.name = name;
        return result;
    }
    throw std::runtime_error("no D3D12 adapter accepted by ZipDepth Vulkan");
}

struct CoreOutput {
    ComPtr<ID3D12Resource> texture;
    ComPtr<ID3D12Fence> fence;
    HANDLE texture_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t value = 1u;
};

CoreOutput create_core_output(
    ID3D12Device* device, std::uint32_t width, std::uint32_t height) {
    CoreOutput result;
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC texture{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_R32_FLOAT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    check(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_SHARED, &texture,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&result.texture)), "CreateCommittedResource(output)");
    check(device->CreateSharedHandle(result.texture.Get(), nullptr, GENERIC_ALL,
        nullptr, &result.texture_handle), "CreateSharedHandle(output)");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
        IID_PPV_ARGS(&result.fence)), "CreateFence(output)");
    check(device->CreateSharedHandle(result.fence.Get(), nullptr, GENERIC_ALL,
        nullptr, &result.fence_handle), "CreateSharedHandle(output fence)");
    return result;
}

void close_output(CoreOutput& output) {
    if (output.texture_handle) CloseHandle(output.texture_handle);
    if (output.fence_handle) CloseHandle(output.fence_handle);
    output.texture_handle = nullptr;
    output.fence_handle = nullptr;
}
}  // namespace

int main() try {
    const char* model_environment = std::getenv("MOGE2_MODEL");
    if (!model_environment || !std::filesystem::exists(model_environment)) return 77;
    ibrh_api api{};
    check(ibrh_get_api(IBRH_CURRENT_API_VERSION, sizeof(api), &api), "ibrh_get_api");
    ibrh_capabilities capabilities{};
    check(api.query_capabilities(sizeof(capabilities), &capabilities), "capabilities");
    const std::uint64_t required = IBRH_CAP_ASYNC_SUBMIT |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    if ((capabilities.flags & required) != required)
        throw std::runtime_error("MoGe-2 ABI2 GPU capability is incomplete");
    Abi2Device selected = select_abi2_device(api);
    std::cout << "device=" << selected.name << '\n';
    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    check(selected.device->CreateCommandQueue(&queue_description,
        IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    ibrh_runtime_create_request runtime_request{};
    runtime_request.struct_size = sizeof(runtime_request);
    runtime_request.api_version = IBRH_CURRENT_API_VERSION;
    runtime_request.backend = {"native", 6u};
    runtime_request.requested_device_json = {
        selected.luid_json.data(), selected.luid_json.size()};
    ibrh_runtime* runtime = nullptr;
    check(api.runtime_create(sizeof(runtime_request), &runtime_request, &runtime),
        "runtime_create");
    const std::string model_path = model_environment;
    const std::string parameters = "{\"NumTokens\":\"64\"}";
    ibrh_model_load_request load{};
    load.struct_size = sizeof(load);
    load.api_version = IBRH_CURRENT_API_VERSION;
    load.model_path = {model_path.data(), model_path.size()};
    load.parameters_json = {parameters.data(), parameters.size()};
    ibrh_model* model = nullptr;
    check(api.model_load(runtime, sizeof(load), &load, &model), "model_load");

    constexpr std::uint32_t width = 64u, height = 64u;
    const auto source_pixels = pixels(width, height, 7u);
    Capture source = upload_texture(selected.device.Get(), queue.Get(),
        source_pixels, width, height);
    CoreOutput output = create_core_output(selected.device.Get(), width, height);
    ibrh_transfer_binding bindings[2]{};
    auto& input = bindings[0];
    input.struct_size = sizeof(input); input.api_version = IBRH_CURRENT_API_VERSION;
    input.resource.struct_size = sizeof(input.resource);
    input.resource.api_version = IBRH_CURRENT_API_VERSION;
    input.resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
    input.resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    input.resource.access = IBRH_RESOURCE_ACCESS_READ;
    input.resource.pixel_format = IBRH_PIXEL_BGRA8;
    input.resource.width = width; input.resource.height = height; input.resource.depth = 1u;
    input.resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    input.resource.native_handle = reinterpret_cast<std::uintptr_t>(source.texture_handle);
    input.synchronization.struct_size = sizeof(input.synchronization);
    input.synchronization.api_version = IBRH_CURRENT_API_VERSION;
    input.synchronization.kind = IBRH_SYNC_D3D12_FENCE;
    input.synchronization.operation = IBRH_SYNC_WAIT;
    input.synchronization.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    input.synchronization.native_handle = reinterpret_cast<std::uintptr_t>(source.fence_handle);
    input.synchronization.value = source.value;
    auto& target = bindings[1];
    target.struct_size = sizeof(target); target.api_version = IBRH_CURRENT_API_VERSION;
    target.resource.struct_size = sizeof(target.resource);
    target.resource.api_version = IBRH_CURRENT_API_VERSION;
    target.resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
    target.resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    target.resource.access = IBRH_RESOURCE_ACCESS_WRITE;
    target.resource.pixel_format = IBRH_PIXEL_DEPTH_METRIC_FLOAT32;
    target.resource.width = width; target.resource.height = height; target.resource.depth = 1u;
    target.resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    target.resource.native_handle = reinterpret_cast<std::uintptr_t>(output.texture_handle);
    target.synchronization.struct_size = sizeof(target.synchronization);
    target.synchronization.api_version = IBRH_CURRENT_API_VERSION;
    target.synchronization.kind = IBRH_SYNC_D3D12_FENCE;
    target.synchronization.operation = IBRH_SYNC_SIGNAL;
    target.synchronization.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    target.synchronization.native_handle = reinterpret_cast<std::uintptr_t>(output.fence_handle);
    target.synchronization.value = output.value;
    ibrh_submit_request submit{};
    submit.struct_size = sizeof(submit); submit.api_version = IBRH_CURRENT_API_VERSION;
    submit.inputs = &input; submit.input_count = 1u;
    submit.outputs = &target; submit.output_count = 1u;
    submit.source_frame_id = 7007u; submit.timestamp_ns = 123456789u;
    submit.parameters_json = {parameters.data(), parameters.size()};
    std::uint64_t upload_before = 0u, download_before = 0u;
    check_counter(moge2_get_transfer_counters(&upload_before, &download_before),
        "transfer counters before");
    const auto start = std::chrono::steady_clock::now();
    ibrh_job* job = nullptr;
    check(api.submit(model, sizeof(submit), &submit, &job), "submit");
    const double submit_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (submit_ms > 5.0) throw std::runtime_error("MoGe-2 submit exceeded 5 ms");
    ibrh_synchronization ready = target.synchronization;
    wait_fence(selected.device.Get(), ready);
    ibrh_job_status status{};
    for (unsigned attempt = 0; attempt < 10000; ++attempt) {
        check(api.job_poll(job, sizeof(status), &status), "job_poll");
        if (status.state != IBRH_JOB_QUEUED && status.state != IBRH_JOB_RUNNING) break;
        Sleep(1);
    }
    if (status.state != IBRH_JOB_COMPLETE || status.source_frame_id != 7007u)
        throw std::runtime_error("MoGe-2 ABI2 job did not complete with correlation");
    std::uint64_t upload_after = 0u, download_after = 0u;
    check_counter(moge2_get_transfer_counters(&upload_after, &download_after),
        "transfer counters after");
    if (upload_after != upload_before || download_after != download_before)
        throw std::runtime_error("MoGe-2 external path staged tensor bytes through host");
    ibrh_output_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.api_version = IBRH_CURRENT_API_VERSION;
    descriptor.resource = target.resource;
    descriptor.ready = target.synchronization;
    const auto gpu = read_output(selected.device.Get(), queue.Get(), descriptor);
    const auto finite = [](float value) { return std::isfinite(value); };
    const auto finite_count = static_cast<std::size_t>(
        std::count_if(gpu.begin(), gpu.end(), finite));
    if (finite_count < gpu.size() / 2u)
        throw std::runtime_error("MoGe-2 external output has too few valid mask pixels");
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (float value : gpu) if (finite(value)) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    if (!(maximum > minimum))
        throw std::runtime_error("MoGe-2 external output is not varying");
    double maximum_submit_ms = submit_ms;
    for (std::uint64_t frame = 7008u; frame < 7032u; ++frame) {
        Capture next_source = upload_texture(selected.device.Get(), queue.Get(),
            pixels(width, height, static_cast<std::uint32_t>(frame)), width, height);
        CoreOutput next_output = create_core_output(selected.device.Get(), width, height);
        input.resource.native_handle =
            reinterpret_cast<std::uintptr_t>(next_source.texture_handle);
        input.synchronization.native_handle =
            reinterpret_cast<std::uintptr_t>(next_source.fence_handle);
        input.synchronization.value = next_source.value;
        target.resource.native_handle =
            reinterpret_cast<std::uintptr_t>(next_output.texture_handle);
        target.synchronization.native_handle =
            reinterpret_cast<std::uintptr_t>(next_output.fence_handle);
        target.synchronization.value = next_output.value;
        submit.source_frame_id = frame;
        const auto next_start = std::chrono::steady_clock::now();
        ibrh_job* next_job = nullptr;
        check(api.submit(model, sizeof(submit), &submit, &next_job),
            "stress submit");
        maximum_submit_ms = std::max(maximum_submit_ms,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - next_start).count());
        api.job_release(job);
        close_capture(source);
        close_output(output);
        source = std::move(next_source);
        output = std::move(next_output);
        job = next_job;
        ready = target.synchronization;
        wait_fence(selected.device.Get(), ready);
        status = {};
        for (unsigned attempt = 0; attempt < 10000; ++attempt) {
            check(api.job_poll(job, sizeof(status), &status), "stress poll");
            if (status.state != IBRH_JOB_QUEUED &&
                status.state != IBRH_JOB_RUNNING) break;
            Sleep(1);
        }
        if (status.state != IBRH_JOB_COMPLETE || status.source_frame_id != frame)
            throw std::runtime_error("MoGe-2 stress correlation failed");
    }
    api.job_release(job);
    close_capture(source); close_output(output);
    api.model_unload(model); api.runtime_destroy(runtime);
    std::cout << "MoGe-2 ABI2 D3D12/Vulkan passed; frames=25; max_submit_ms="
              << maximum_submit_ms
              << "; upload_delta=0; download_delta=0; sourceFrameId=7007..7031\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
