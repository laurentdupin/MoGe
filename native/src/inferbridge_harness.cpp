#include "inferbridge_harness.h"

#include "external_gpu.h"
#include "inferbridge/native_harness_precision.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

struct ibrh_runtime {
    std::string error;
    std::uint32_t device_index = 0u;
    std::uint64_t adapter_luid = 0u;
};

struct ibrh_job {
    std::atomic<std::uint32_t> references{1u};
    std::atomic<std::uint32_t> state{IBRH_JOB_QUEUED};
    std::atomic<bool> cancel_requested{false};
    std::shared_ptr<std::atomic<std::uint32_t>> occupied_slots;
    std::shared_ptr<moge2_native::ExternalJob> gpu;
    std::mutex gpu_mutex;
    moge2_native::ExternalTextureRequest request{};
    const std::uint8_t* host_input = nullptr;
    float* host_output = nullptr;
    std::size_t host_row_stride = 0u;
    std::uint32_t host_width = 0u;
    std::uint32_t host_height = 0u;
    std::uint32_t host_num_tokens = 1200u;
    float host_background_distance_metres = 50.0f;
    bool host_rgba = false;
    std::uint64_t source_frame_id = 0u;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    std::shared_ptr<moge2_native::ExternalGpu> gpu;
    std::uint32_t num_tokens = 1200u;
    std::uint32_t background_distance_metres = 50u;
    std::shared_ptr<std::atomic<std::uint32_t>> occupied_slots =
        std::make_shared<std::atomic<std::uint32_t>>(0u);
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    std::deque<ibrh_job*> queue;
    bool stopping = false;
    std::thread worker;
};

namespace {
thread_local std::string g_error;
#ifndef MOGE2_HARNESS_ID
#define MOGE2_HARNESS_ID "inferbridge.moge-2.native"
#endif
constexpr char kHarnessId[] = MOGE2_HARNESS_ID;
constexpr char kHarnessVersion[] = "0.2.0";

std::string text(ibrh_string_view value) {
    return value.data && value.size ? std::string(value.data, value.size) :
        std::string();
}

ibrh_result fail(ibrh_runtime* runtime, ibrh_result code,
    const std::string& message) {
    g_error = message;
    if (runtime) runtime->error = message;
    return code;
}

bool json_uint(const std::string& json, const std::string& key,
    std::uint32_t& result) {
    std::size_t position = json.find("\"" + key + "\"");
    if (position == std::string::npos) return false;
    position = json.find(':', position);
    if (position == std::string::npos) return false;
    position = json.find_first_of("0123456789", position + 1u);
    if (position == std::string::npos) return false;
    std::uint64_t value = 0u;
    while (position < json.size() && json[position] >= '0' && json[position] <= '9') {
        value = value * 10u + static_cast<unsigned>(json[position++] - '0');
        if (value > std::numeric_limits<std::uint32_t>::max()) return false;
    }
    result = static_cast<std::uint32_t>(value);
    return true;
}

bool json_string(const std::string& json, const std::string& key,
    std::string& result) {
    std::size_t position = json.find("\"" + key + "\"");
    if (position == std::string::npos) return false;
    position = json.find(':', position);
    position = json.find('"', position + 1u);
    if (position == std::string::npos) return false;
    const std::size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    result = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool parse_luid(const std::string& value, std::uint64_t& result) {
    if (value.size() != 16u) return false;
    std::uint8_t bytes[8]{};
    for (std::size_t index = 0; index < 8u; ++index) {
        auto digit = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int high = digit(value[index * 2u]);
        const int low = digit(value[index * 2u + 1u]);
        if (high < 0 || low < 0) return false;
        bytes[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

void retain(ibrh_job* job) { job->references.fetch_add(1u); }
void release(ibrh_job* job) {
    if (job && job->references.fetch_sub(1u) == 1u) {
        if (job->occupied_slots) job->occupied_slots->fetch_sub(1u);
        delete job;
    }
}

void worker_loop(ibrh_model* model) {
    for (;;) {
        ibrh_job* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(model->queue_mutex);
            model->queue_condition.wait(lock, [&] {
                return model->stopping || !model->queue.empty();
            });
            if (model->stopping && model->queue.empty()) return;
            job = model->queue.front();
            model->queue.pop_front();
        }
        if (job->cancel_requested.load()) {
            job->state.store(IBRH_JOB_CANCELLED);
            release(job);
            continue;
        }
        try {
            if (job->host_input != nullptr) {
                job->state.store(IBRH_JOB_RUNNING);
                model->gpu->infer_host(job->host_input, job->host_width,
                    job->host_height, job->host_row_stride, job->host_rgba,
                    job->host_num_tokens, job->host_background_distance_metres,
                    job->host_output);
                job->state.store(IBRH_JOB_COMPLETE);
                release(job);
                continue;
            }
            auto gpu = model->gpu->submit_texture(job->request);
            {
                std::lock_guard<std::mutex> lock(job->gpu_mutex);
                job->gpu = std::move(gpu);
            }
            job->state.store(IBRH_JOB_RUNNING);
        } catch (const std::exception& exception) {
            fail(model->runtime, IBRH_ERROR_INTERNAL, exception.what());
            job->state.store(IBRH_JOB_FAILED);
        }
        release(job);
    }
}

ibrh_result IBRH_CALL query_capabilities(
    std::size_t size, ibrh_capabilities* output) {
    if (!output) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *output = {};
    output->struct_size = sizeof(*output);
    output->api_version = IBRH_CURRENT_API_VERSION;
    output->flags = IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_HOST_MEMORY;
    output->input_domain_mask = 1ull << IBRH_RESOURCE_DOMAIN_HOST;
    output->output_domain_mask = 1ull << IBRH_RESOURCE_DOMAIN_HOST;
#if defined(_WIN32)
    output->flags |= IBRH_CAP_GPU_RESOURCES |
        IBRH_CAP_EXTERNAL_SYNCHRONIZATION | IBRH_CAP_GPU_RESIDENT_OUTPUT;
    output->input_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_D3D12;
    output->output_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_D3D12;
    output->synchronization_mask = 1ull << IBRH_SYNC_D3D12_FENCE;
#elif defined(MOGE2_WITH_METAL) && defined(__APPLE__)
    output->flags |= IBRH_CAP_GPU_RESOURCES |
        IBRH_CAP_EXTERNAL_SYNCHRONIZATION | IBRH_CAP_GPU_RESIDENT_OUTPUT;
    output->input_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_METAL;
    output->output_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_METAL;
    output->synchronization_mask = 1ull << IBRH_SYNC_METAL_SHARED_EVENT;
#endif
    output->maximum_inputs = 1u;
    output->maximum_outputs = 1u;
    output->maximum_in_flight_jobs = 3u;
    output->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    output->harness_version = {kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(std::size_t size,
    const ibrh_runtime_create_request* request, ibrh_runtime** output) {
    if (!request || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (size < sizeof(*request) || request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto runtime = std::unique_ptr<ibrh_runtime>(new (std::nothrow) ibrh_runtime());
    if (!runtime) return IBRH_ERROR_INTERNAL;
    const std::string device = text(request->requested_device_json);
    (void)json_uint(device, "index", runtime->device_index);
    std::string luid_text;
    if (json_string(device, "luid", luid_text)) {
#if defined(_WIN32)
        if (!parse_luid(luid_text, runtime->adapter_luid))
            return IBRH_ERROR_INVALID_ARGUMENT;
#else
        runtime->adapter_luid = 0u;
#endif
    }
    if (runtime->adapter_luid) {
#if defined(MOGE2_WITH_VULKAN) && defined(_WIN32)
        bool found = false;
        for (std::uint32_t index = 0; index < 32u; ++index) {
            try {
                const auto caps = moge2_native::probe_external_gpu(index);
                if (caps.available && caps.adapter_luid == runtime->adapter_luid) {
                    runtime->device_index = index;
                    found = true;
                    break;
                }
            } catch (...) { if (index == 0u) break; }
        }
        if (!found) return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
#elif !defined(MOGE2_WITH_METAL)
        return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
#endif
    }
    *output = runtime.release();
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) { delete runtime; }

ibrh_result IBRH_CALL model_load(ibrh_runtime* runtime, std::size_t size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (!runtime || !request || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (size < sizeof(*request) || request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    const std::string path = text(request->model_path);
    if (path.empty()) return IBRH_ERROR_INVALID_ARGUMENT;
    auto model = std::unique_ptr<ibrh_model>(new (std::nothrow) ibrh_model());
    if (!model) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    const std::string parameters = text(request->parameters_json);
    inferbridge::native::Precision precision;
    try {
        precision = inferbridge::native::precision_from_parameters_json(parameters);
    } catch (const std::exception& error) {
        return fail(runtime, IBRH_ERROR_INVALID_ARGUMENT, error.what());
    }
    const inferbridge::native::ScopedPrecisionRequest precision_scope(precision);
    (void)json_uint(parameters, "NumTokens", model->num_tokens);
    (void)json_uint(parameters, "BackgroundDistanceMetres",
        model->background_distance_metres);
    if (model->num_tokens < 16u || model->num_tokens > 4096u)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (model->background_distance_metres < 1u ||
        model->background_distance_metres > 1000u)
        return IBRH_ERROR_INVALID_ARGUMENT;
    try {
        model->gpu =
#if defined(MOGE2_WITH_METAL)
            moge2_native::create_metal_gpu(path);
#else
            moge2_native::create_external_gpu(path, runtime->device_index);
#endif
        const auto caps = model->gpu->capabilities();
#if defined(_WIN32)
        if (runtime->adapter_luid && (!caps.available ||
            caps.adapter_luid != runtime->adapter_luid))
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
#else
        (void)caps;
#endif
        model->worker = std::thread(worker_loop, model.get());
    } catch (const std::exception& exception) {
        return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY, exception.what());
    }
    *output = model.release();
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (!model) return;
    {
        std::lock_guard<std::mutex> lock(model->queue_mutex);
        model->stopping = true;
        for (ibrh_job* job : model->queue) {
            job->cancel_requested.store(true);
            job->state.store(IBRH_JOB_CANCELLED);
            release(job);
        }
        model->queue.clear();
    }
    model->queue_condition.notify_all();
    if (model->worker.joinable()) model->worker.join();
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(const ibrh_model* model,
    std::size_t size, ibrh_model_io_descriptor* output) {
    if (!model || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *output = {sizeof(*output), IBRH_CURRENT_API_VERSION, 1u, 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL model_get_port(const ibrh_model* model,
    std::uint32_t direction, std::uint32_t index, std::size_t size,
    ibrh_port_descriptor* output) {
    if (!model || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (index || (direction != IBRH_PORT_INPUT && direction != IBRH_PORT_OUTPUT))
        return IBRH_ERROR_NOT_FOUND;
    *output = {};
    output->struct_size = sizeof(*output);
    output->api_version = IBRH_CURRENT_API_VERSION;
    output->index = 0u;
    output->direction = direction;
    output->semantic = direction == IBRH_PORT_INPUT ?
        IBRH_SEMANTIC_IMAGE : IBRH_SEMANTIC_DEPTH;
    output->payload_type = direction == IBRH_PORT_INPUT ?
        IBRH_PIXEL_BGRA8 : IBRH_PIXEL_DEPTH_METRIC_FLOAT32;
    output->pixel_format = output->payload_type;
    output->accepted_pixel_format_mask = direction == IBRH_PORT_INPUT ?
        (1ull << IBRH_PIXEL_BGRA8) | (1ull << IBRH_PIXEL_RGBA8) :
        (1ull << IBRH_PIXEL_DEPTH_METRIC_FLOAT32);
    output->resource_kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    output->depth = 1u;
    output->flags = IBRH_DESCRIPTOR_DYNAMIC_WIDTH | IBRH_DESCRIPTOR_DYNAMIC_HEIGHT;
    return IBRH_OK;
}

ibrh_result IBRH_CALL model_plan_outputs(const ibrh_model* model,
    std::size_t size, const ibrh_output_plan_request* request,
    std::uint32_t capacity, ibrh_port_descriptor* outputs) {
    if (!model || !request || !outputs) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*request) || request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (capacity < 1u) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || !request->inputs ||
        !request->inputs[0].width || !request->inputs[0].height)
        return IBRH_ERROR_INVALID_ARGUMENT;
    const auto result = model_get_port(model, IBRH_PORT_OUTPUT, 0u,
        sizeof(outputs[0]), &outputs[0]);
    if (result != IBRH_OK) return result;
    outputs[0].width = request->inputs[0].width;
    outputs[0].height = request->inputs[0].height;
    outputs[0].flags = 0u;
    return IBRH_OK;
}

bool valid_binding(const ibrh_transfer_binding& input,
    const ibrh_transfer_binding& output) {
#if defined(_WIN32)
    constexpr uint32_t domain = IBRH_RESOURCE_DOMAIN_D3D12;
    constexpr uint32_t texture_handle = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    constexpr uint32_t sync_kind = IBRH_SYNC_D3D12_FENCE;
    constexpr uint32_t sync_handle = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    const bool wait_valid = input.synchronization.kind == sync_kind &&
        input.synchronization.operation == IBRH_SYNC_WAIT;
#elif defined(MOGE2_WITH_METAL) && defined(__APPLE__)
    constexpr uint32_t domain = IBRH_RESOURCE_DOMAIN_METAL;
    constexpr uint32_t texture_handle = IBRH_NATIVE_HANDLE_METAL_TEXTURE;
    constexpr uint32_t sync_kind = IBRH_SYNC_METAL_SHARED_EVENT;
    constexpr uint32_t sync_handle = IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT;
    const bool wait_valid =
        (input.synchronization.kind == IBRH_SYNC_NONE &&
         input.synchronization.native_handle == 0u) ||
        (input.synchronization.kind == sync_kind &&
         input.synchronization.operation == IBRH_SYNC_WAIT &&
         input.synchronization.native_handle_type == sync_handle &&
         input.synchronization.native_handle != 0u);
#endif
#if defined(_WIN32) || (defined(MOGE2_WITH_METAL) && defined(__APPLE__))
    return input.resource.domain == domain &&
        output.resource.domain == domain &&
        input.resource.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
        output.resource.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
        (input.resource.pixel_format == IBRH_PIXEL_BGRA8 ||
         input.resource.pixel_format == IBRH_PIXEL_RGBA8) &&
        output.resource.pixel_format == IBRH_PIXEL_DEPTH_METRIC_FLOAT32 &&
        input.resource.native_handle_type == texture_handle &&
        output.resource.native_handle_type == texture_handle &&
        wait_valid &&
        output.synchronization.kind == sync_kind &&
        output.synchronization.operation == IBRH_SYNC_SIGNAL &&
        output.synchronization.native_handle_type == sync_handle &&
        output.synchronization.native_handle != 0u &&
        input.resource.width == output.resource.width &&
        input.resource.height == output.resource.height;
#else
    return false;
#endif
}

bool valid_host_binding(const ibrh_transfer_binding& input,
    const ibrh_transfer_binding& output) {
    const std::uint64_t input_bytes = static_cast<std::uint64_t>(
        input.resource.row_stride_bytes) * input.resource.height;
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(
        output.resource.width) * output.resource.height * sizeof(float);
    return input.resource.domain == IBRH_RESOURCE_DOMAIN_HOST &&
        output.resource.domain == IBRH_RESOURCE_DOMAIN_HOST &&
        input.resource.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
        output.resource.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
        (input.resource.pixel_format == IBRH_PIXEL_BGRA8 ||
         input.resource.pixel_format == IBRH_PIXEL_RGBA8) &&
        output.resource.pixel_format == IBRH_PIXEL_DEPTH_METRIC_FLOAT32 &&
        input.resource.native_handle_type == IBRH_NATIVE_HANDLE_HOST_POINTER &&
        output.resource.native_handle_type == IBRH_NATIVE_HANDLE_HOST_POINTER &&
        input.resource.native_handle != 0u && output.resource.native_handle != 0u &&
        input.resource.width == output.resource.width &&
        input.resource.height == output.resource.height &&
        input.resource.row_stride_bytes >= input.resource.width * 4u &&
        input.resource.byte_size >= input_bytes &&
        output.resource.byte_size >= output_bytes;
}

ibrh_result IBRH_CALL submit(ibrh_model* model, std::size_t size,
    const ibrh_submit_request* request, ibrh_job** output) {
    if (!model || !request || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (size < sizeof(*request) || request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || request->output_count != 1u ||
        !request->inputs || !request->outputs)
        return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
    const bool host = valid_host_binding(request->inputs[0], request->outputs[0]);
    const bool external = valid_binding(request->inputs[0], request->outputs[0]);
    if (!host && !external) return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
    std::uint32_t previous = model->occupied_slots->load();
    do {
        if (previous >= 3u) return IBRH_ERROR_INVALID_STATE;
    } while (!model->occupied_slots->compare_exchange_weak(previous, previous + 1u));
    auto job = std::unique_ptr<ibrh_job>(new (std::nothrow) ibrh_job());
    if (!job) {
        model->occupied_slots->fetch_sub(1u);
        return IBRH_ERROR_INTERNAL;
    }
    job->occupied_slots = model->occupied_slots;
    job->source_frame_id = request->source_frame_id;
    const auto& input = request->inputs[0];
    const auto& target = request->outputs[0];
    if (host) {
        job->host_input = reinterpret_cast<const std::uint8_t*>(
            input.resource.native_handle);
        job->host_output = reinterpret_cast<float*>(target.resource.native_handle);
        job->host_row_stride = input.resource.row_stride_bytes;
        job->host_width = input.resource.width;
        job->host_height = input.resource.height;
        job->host_num_tokens = model->num_tokens;
        job->host_background_distance_metres =
            static_cast<float>(model->background_distance_metres);
        job->host_rgba = input.resource.pixel_format == IBRH_PIXEL_RGBA8;
    } else {
        job->request = {
            static_cast<std::uintptr_t>(input.resource.native_handle),
            input.resource.auxiliary_handle,
            input.resource.width, input.resource.height, model->num_tokens,
            static_cast<float>(model->background_distance_metres),
            input.resource.pixel_format == IBRH_PIXEL_RGBA8,
            static_cast<std::uintptr_t>(input.synchronization.native_handle),
            input.synchronization.value,
            static_cast<std::uintptr_t>(target.resource.native_handle),
            target.resource.auxiliary_handle,
            static_cast<std::uintptr_t>(target.synchronization.native_handle),
            target.synchronization.value};
    }
    {
        std::lock_guard<std::mutex> lock(model->queue_mutex);
        if (model->stopping) {
            job->occupied_slots.reset();
            model->occupied_slots->fetch_sub(1u);
            return IBRH_ERROR_INVALID_STATE;
        }
        retain(job.get());
        model->queue.push_back(job.get());
    }
    model->queue_condition.notify_one();
    *output = job.release();
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_poll(const ibrh_job* job, std::size_t size,
    ibrh_job_status* output) {
    if (!job || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::uint32_t state = job->state.load();
    if (state == IBRH_JOB_RUNNING) {
        std::lock_guard<std::mutex> lock(const_cast<ibrh_job*>(job)->gpu_mutex);
        if (job->gpu) {
            const auto gpu_state = job->gpu->state();
            if (gpu_state == moge2_native::ExternalJobState::complete)
                state = IBRH_JOB_COMPLETE;
            else if (gpu_state == moge2_native::ExternalJobState::cancelled)
                state = IBRH_JOB_CANCELLED;
            const_cast<ibrh_job*>(job)->state.store(state);
        }
    }
    *output = {};
    output->struct_size = sizeof(*output);
    output->state = state;
    output->output_count = 1u;
    output->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (!job) return IBRH_ERROR_INVALID_ARGUMENT;
    job->cancel_requested.store(true);
    std::lock_guard<std::mutex> lock(job->gpu_mutex);
    if (job->gpu) job->gpu->cancel();
    else job->state.store(IBRH_JOB_CANCELLED);
    return IBRH_OK;
}

void IBRH_CALL job_release(ibrh_job* job) { release(job); }

ibrh_result IBRH_CALL get_last_error(const void* object, char* destination,
    std::size_t destination_size, std::size_t* required_size) {
    const auto runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message = runtime && !runtime->error.empty() ?
        runtime->error : g_error;
    const std::size_t required = message.size() + 1u;
    if (required_size) *required_size = required;
    if (!destination || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}
}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    std::uint32_t requested_version, std::size_t size, ibrh_api* api) {
    if (!api) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload = model_unload;
    api->model_describe_io = model_describe_io;
    api->model_get_port = model_get_port;
    api->model_plan_outputs = model_plan_outputs;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
