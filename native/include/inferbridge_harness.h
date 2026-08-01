#ifndef INFERBRIDGE_INFERBRIDGE_HARNESS_H
#define INFERBRIDGE_INFERBRIDGE_HARNESS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
#    define IBRH_API __declspec(dllexport)
#  else
#    define IBRH_API __declspec(dllimport)
#  endif
#  define IBRH_CALL __cdecl
#else
#  define IBRH_API __attribute__((visibility("default")))
#  define IBRH_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IBRH_API_VERSION_MAJOR 2u
#define IBRH_API_VERSION_MINOR 0u
#define IBRH_MAKE_API_VERSION(major, minor) (((major) << 16u) | (minor))
#define IBRH_CURRENT_API_VERSION \
    IBRH_MAKE_API_VERSION(IBRH_API_VERSION_MAJOR, IBRH_API_VERSION_MINOR)

typedef int32_t ibrh_result;

enum {
    IBRH_OK = 0,
    IBRH_ERROR_INVALID_ARGUMENT = 1,
    IBRH_ERROR_STRUCT_TOO_SMALL = 2,
    IBRH_ERROR_UNSUPPORTED_API = 3,
    IBRH_ERROR_INTERNAL = 4,
    IBRH_ERROR_INVALID_STATE = 5,
    IBRH_ERROR_NOT_FOUND = 6,
    IBRH_ERROR_TIMEOUT = 7,
    IBRH_ERROR_UNSUPPORTED_CAPABILITY = 8,
    IBRH_ERROR_CANCELLED = 9,
    IBRH_ERROR_DEVICE_LOST = 10
};

typedef struct ibrh_runtime ibrh_runtime;
typedef struct ibrh_model ibrh_model;
typedef struct ibrh_job ibrh_job;

typedef struct ibrh_string_view {
    const char* data;
    size_t size;
} ibrh_string_view;

enum {
    IBRH_RESOURCE_DOMAIN_HOST = 1u,
    IBRH_RESOURCE_DOMAIN_VULKAN = 2u,
    IBRH_RESOURCE_DOMAIN_D3D12 = 3u,
    IBRH_RESOURCE_DOMAIN_DMA_BUF = 4u,
    IBRH_RESOURCE_DOMAIN_ANDROID_HARDWARE_BUFFER = 5u,
    IBRH_RESOURCE_DOMAIN_METAL = 6u
};

enum {
    IBRH_PORT_INPUT = 1u,
    IBRH_PORT_OUTPUT = 2u
};

enum {
    IBRH_SEMANTIC_IMAGE = 1u,
    IBRH_SEMANTIC_DEPTH = 2u,
    IBRH_SEMANTIC_TEXT = 3u,
    IBRH_SEMANTIC_GAUSSIAN_SPLAT = 4u
};

enum {
    IBRH_DESCRIPTOR_DYNAMIC_WIDTH = 1u << 0u,
    IBRH_DESCRIPTOR_DYNAMIC_HEIGHT = 1u << 1u,
    IBRH_DESCRIPTOR_DYNAMIC_DEPTH = 1u << 2u
};

enum {
    IBRH_RESOURCE_KIND_BUFFER = 1u,
    IBRH_RESOURCE_KIND_IMAGE_2D = 2u
};

enum {
    IBRH_RESOURCE_ACCESS_READ = 1u,
    IBRH_RESOURCE_ACCESS_WRITE = 2u,
    IBRH_RESOURCE_ACCESS_READ_WRITE = 3u
};

enum {
    IBRH_PIXEL_BGRA8 = 1u,
    IBRH_PIXEL_DEPTH_FLOAT32 = 2u,
    IBRH_PIXEL_DEPTH_FLOAT16 = 3u,
    IBRH_PIXEL_DEPTH_METRIC_FLOAT16 = 4u,
    IBRH_PIXEL_DEPTH_METRIC_FLOAT32 = 5u,
    IBRH_PIXEL_DEPTH_UNORM8 = 6u,
    IBRH_PIXEL_RGBA8 = 7u,
    IBRH_PAYLOAD_UTF8_JSON = 32u,
    IBRH_PAYLOAD_GAUSSIAN_SPLAT_FLOAT32 = 64u
};

enum {
    IBRH_NATIVE_HANDLE_NONE = 0u,
    IBRH_NATIVE_HANDLE_HOST_POINTER = 1u,
    IBRH_NATIVE_HANDLE_VULKAN_BUFFER = 2u,
    IBRH_NATIVE_HANDLE_VULKAN_IMAGE = 3u,
    IBRH_NATIVE_HANDLE_D3D12_RESOURCE = 4u,
    IBRH_NATIVE_HANDLE_WIN32_SHARED = 5u,
    IBRH_NATIVE_HANDLE_POSIX_FD = 6u,
    IBRH_NATIVE_HANDLE_ANDROID_HARDWARE_BUFFER = 7u,
    IBRH_NATIVE_HANDLE_METAL_BUFFER = 8u,
    IBRH_NATIVE_HANDLE_METAL_TEXTURE = 9u,
    IBRH_NATIVE_HANDLE_VULKAN_SEMAPHORE = 10u,
    IBRH_NATIVE_HANDLE_D3D12_FENCE = 11u,
    IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT = 12u,
    /* Win32 NT handle for an ID3D12Heap. The receiver creates an
     * identically-described placed resource at offset zero. */
    IBRH_NATIVE_HANDLE_D3D12_HEAP = 13u
};

enum {
    IBRH_SYNC_NONE = 0u,
    IBRH_SYNC_VULKAN_SEMAPHORE = 1u,
    IBRH_SYNC_D3D12_FENCE = 2u,
    IBRH_SYNC_SYNC_FD = 3u,
    IBRH_SYNC_METAL_SHARED_EVENT = 4u
};

enum {
    IBRH_SYNC_WAIT = 1u,
    IBRH_SYNC_SIGNAL = 2u
};

enum {
    IBRH_JOB_QUEUED = 0u,
    IBRH_JOB_RUNNING = 1u,
    IBRH_JOB_COMPLETE = 2u,
    IBRH_JOB_FAILED = 3u,
    IBRH_JOB_CANCELLED = 4u
};

enum {
    IBRH_CAP_ASYNC_SUBMIT = 1ull << 0u,
    IBRH_CAP_CANCELLATION = 1ull << 1u,
    IBRH_CAP_HOST_MEMORY = 1ull << 2u,
    IBRH_CAP_GPU_RESOURCES = 1ull << 3u,
    IBRH_CAP_EXTERNAL_SYNCHRONIZATION = 1ull << 4u,
    IBRH_CAP_GPU_RESIDENT_OUTPUT = 1ull << 5u
};

typedef void(IBRH_CALL* ibrh_log_fn)(
    void* user_data, uint32_t level, ibrh_string_view message);

typedef struct ibrh_runtime_create_request {
    uint32_t struct_size;
    uint32_t api_version;
    ibrh_string_view backend;
    ibrh_string_view requested_device_json;
    ibrh_string_view cache_path;
    ibrh_log_fn log;
    void* log_user_data;
} ibrh_runtime_create_request;

typedef struct ibrh_capabilities {
    uint32_t struct_size;
    uint32_t api_version;
    uint64_t flags;
    uint64_t input_domain_mask;
    uint64_t output_domain_mask;
    uint64_t synchronization_mask;
    uint32_t maximum_inputs;
    uint32_t maximum_outputs;
    uint32_t maximum_in_flight_jobs;
    uint32_t reserved;
    ibrh_string_view harness_id;
    ibrh_string_view harness_version;
} ibrh_capabilities;

typedef struct ibrh_model_load_request {
    uint32_t struct_size;
    uint32_t api_version;
    ibrh_string_view model_path;
    ibrh_string_view parameters_json;
} ibrh_model_load_request;

/*
 * A model port describes meaning and shape, never ownership. InferBridge uses
 * these descriptors to select and allocate the platform transfer resources.
 * A zero dimension is valid only when its matching DYNAMIC flag is set.
 */
typedef struct ibrh_port_descriptor {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t index;
    uint32_t direction;
    uint32_t semantic;
    uint32_t payload_type;
    uint32_t pixel_format;
    uint32_t resource_kind;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t flags;
    uint64_t accepted_pixel_format_mask;
} ibrh_port_descriptor;

typedef struct ibrh_model_io_descriptor {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t input_count;
    uint32_t output_count;
} ibrh_model_io_descriptor;

typedef struct ibrh_resource {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t domain;
    uint32_t kind;
    uint32_t access;
    uint32_t pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t row_stride_bytes;
    uint32_t plane_index;
    uint32_t native_handle_type;
    uint64_t byte_size;
    uint64_t byte_offset;
    uint64_t native_handle;
    uint64_t auxiliary_handle;
    uint32_t queue_family_index;
    uint32_t reserved;
} ibrh_resource;

typedef struct ibrh_synchronization {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t kind;
    uint32_t operation;
    uint32_t native_handle_type;
    uint32_t reserved;
    uint64_t native_handle;
    uint64_t value;
} ibrh_synchronization;

/* A resource and its synchronization operation form one unambiguous binding. */
typedef struct ibrh_transfer_binding {
    uint32_t struct_size;
    uint32_t api_version;
    ibrh_resource resource;
    ibrh_synchronization synchronization;
} ibrh_transfer_binding;

/*
 * Planning is side-effect free. It resolves dynamic output dimensions from
 * the concrete input descriptions and parameters before InferBridge reserves
 * an output slot.
 */
typedef struct ibrh_output_plan_request {
    uint32_t struct_size;
    uint32_t api_version;
    const ibrh_resource* inputs;
    uint32_t input_count;
    uint32_t reserved;
    ibrh_string_view parameters_json;
} ibrh_output_plan_request;

typedef struct ibrh_submit_request {
    uint32_t struct_size;
    uint32_t api_version;
    const ibrh_transfer_binding* inputs;
    uint32_t input_count;
    const ibrh_transfer_binding* outputs;
    uint32_t output_count;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
    ibrh_string_view parameters_json;
} ibrh_submit_request;

typedef struct ibrh_job_status {
    uint32_t struct_size;
    uint32_t state;
    uint32_t output_count;
    uint32_t reserved;
    uint64_t source_frame_id;
} ibrh_job_status;

typedef struct ibrh_output_descriptor {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t output_index;
    uint32_t payload_type;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
    ibrh_resource resource;
    ibrh_synchronization ready;
} ibrh_output_descriptor;

typedef struct ibrh_api {
    uint32_t struct_size;
    uint32_t api_version;

    ibrh_result(IBRH_CALL* query_capabilities)(
        size_t capabilities_size, ibrh_capabilities* out_capabilities);

    ibrh_result(IBRH_CALL* runtime_create)(
        size_t request_size, const ibrh_runtime_create_request* request,
        ibrh_runtime** out_runtime);
    void(IBRH_CALL* runtime_destroy)(ibrh_runtime* runtime);

    ibrh_result(IBRH_CALL* model_load)(
        ibrh_runtime* runtime, size_t request_size,
        const ibrh_model_load_request* request, ibrh_model** out_model);
    void(IBRH_CALL* model_unload)(ibrh_model* model);

    ibrh_result(IBRH_CALL* model_describe_io)(
        const ibrh_model* model, size_t descriptor_size,
        ibrh_model_io_descriptor* out_descriptor);
    ibrh_result(IBRH_CALL* model_get_port)(
        const ibrh_model* model, uint32_t direction, uint32_t index,
        size_t descriptor_size, ibrh_port_descriptor* out_descriptor);
    ibrh_result(IBRH_CALL* model_plan_outputs)(
        const ibrh_model* model, size_t request_size,
        const ibrh_output_plan_request* request, uint32_t output_capacity,
        ibrh_port_descriptor* out_outputs);

    ibrh_result(IBRH_CALL* submit)(
        ibrh_model* model, size_t request_size,
        const ibrh_submit_request* request, ibrh_job** out_job);
    ibrh_result(IBRH_CALL* job_poll)(
        const ibrh_job* job, size_t status_size, ibrh_job_status* out_status);
    ibrh_result(IBRH_CALL* job_cancel)(ibrh_job* job);
    void(IBRH_CALL* job_release)(ibrh_job* job);

    ibrh_result(IBRH_CALL* get_last_error)(
        const void* object, char* destination, size_t destination_size,
        size_t* out_required_size);
} ibrh_api;

typedef ibrh_result(IBRH_CALL* ibrh_get_api_fn)(
    uint32_t requested_api_version, size_t api_size, ibrh_api* out_api);

IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* out_api);

#ifdef __cplusplus
}
#endif

#endif
