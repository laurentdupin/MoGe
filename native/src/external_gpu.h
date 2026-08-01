#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace moge2_native {

struct ExternalGpuCapabilities {
    bool available = false;
    std::uint64_t adapter_luid = 0u;
    std::uint32_t maximum_in_flight_jobs = 0u;
};

struct ExternalTextureRequest {
    std::uintptr_t input_texture = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t num_tokens = 1200u;
    bool rgba = false;
    std::uintptr_t wait_fence = 0u;
    std::uint64_t wait_value = 0u;
    std::uintptr_t output_texture = 0u;
    std::uintptr_t signal_fence = 0u;
    std::uint64_t signal_value = 0u;
};

enum class ExternalJobState { running, complete, cancelled };

class ExternalJob {
public:
    virtual ~ExternalJob() = default;
    virtual ExternalJobState state() const = 0;
    virtual void cancel() = 0;
};

class ExternalGpu : public std::enable_shared_from_this<ExternalGpu> {
public:
    virtual ~ExternalGpu() = default;
    virtual ExternalGpuCapabilities capabilities() const = 0;
    virtual std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) = 0;
    virtual void transfer_counters(
        std::uint64_t& upload_bytes, std::uint64_t& download_bytes) const = 0;
};

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& model_path, std::uint32_t device_index);
ExternalGpuCapabilities probe_external_gpu(std::uint32_t device_index);

}  // namespace moge2_native
