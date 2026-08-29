#pragma once

#if defined(_WIN32)
#  if !defined(WIN32_LEAN_AND_MEAN)
#    define WIN32_LEAN_AND_MEAN
#  endif
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  if !defined(VK_USE_PLATFORM_WIN32_KHR)
#    define VK_USE_PLATFORM_WIN32_KHR
#  endif
#endif
#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace da3_native {

void global_transfer_counters(
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes);

class VulkanContext;
class VulkanPipeline;
class VulkanBuffer;
class VulkanImage;

struct VulkanDeferredBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    bool cacheable = false;
};

struct VulkanBatchedDescriptor {
    VulkanPipeline* pipeline = nullptr;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct VulkanDispatchResource {
    const VulkanBuffer* buffer = nullptr;
    const VulkanImage* image = nullptr;
};

struct VulkanExternalCapabilities {
    bool timeline_semaphore = false;
    bool d3d12_resource_import = false;
    bool d3d12_fence_import = false;
    bool d3d12_bgra8_sampled_image_import = false;
    bool d3d12_rgba8_sampled_image_import = false;
    bool d3d12_r32_storage_image_import = false;
};

class VulkanSemaphore {
public:
    VulkanSemaphore() = default;
    VulkanSemaphore(VulkanSemaphore&& other) noexcept;
    VulkanSemaphore& operator=(VulkanSemaphore&& other) noexcept;
    VulkanSemaphore(const VulkanSemaphore&) = delete;
    VulkanSemaphore& operator=(const VulkanSemaphore&) = delete;
    ~VulkanSemaphore();

private:
    friend class VulkanContext;
    VulkanContext* owner_ = nullptr;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    std::uint64_t value_ = 0;
};

class VulkanSubmission {
public:
    VulkanSubmission() = default;
    VulkanSubmission(VulkanSubmission&& other) noexcept;
    VulkanSubmission& operator=(VulkanSubmission&& other) noexcept;
    VulkanSubmission(const VulkanSubmission&) = delete;
    VulkanSubmission& operator=(const VulkanSubmission&) = delete;
    ~VulkanSubmission();

    bool ready() const;
    void wait() const;

private:
    friend class VulkanContext;
    struct Resources;
    VulkanContext* owner_ = nullptr;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    Resources* resources_ = nullptr;
};

class VulkanBuffer {
public:
    VulkanBuffer() = default;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    ~VulkanBuffer();

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }

private:
    friend class VulkanContext;
    VulkanContext* owner_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
    bool cacheable_ = false;
};

class VulkanImage {
public:
    VulkanImage() = default;
    VulkanImage(VulkanImage&& other) noexcept;
    VulkanImage& operator=(VulkanImage&& other) noexcept;
    VulkanImage(const VulkanImage&) = delete;
    VulkanImage& operator=(const VulkanImage&) = delete;
    ~VulkanImage();

    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    VkFormat format() const { return format_; }

private:
    friend class VulkanContext;
    VulkanContext* owner_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
};

class VulkanPipeline {
public:
    VulkanPipeline() = default;
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&& other) noexcept;
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    ~VulkanPipeline();
    void set_debug_name(const char* name);

private:
    friend class VulkanContext;
    VulkanContext* owner_ = nullptr;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorType> descriptor_types_;
    std::vector<VkAccessFlags> descriptor_access_;
    mutable std::vector<VkDescriptorSet> cached_descriptor_sets_;
    std::uint32_t push_constant_bytes_ = 0;
    std::string debug_name_;
};

class VulkanContext {
public:
    explicit VulkanContext(
        std::uint32_t device_index,
        bool track_resource_hazards = true);
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    ~VulkanContext();

    const std::string& device_name() const { return device_name_; }
    bool supports_float16() const { return float16_supported_; }
    bool supports_packed_int8_dot() const {
        return packed_int8_dot_supported_;
    }
    const VulkanExternalCapabilities& external_capabilities() const {
        return external_capabilities_;
    }
#if defined(_WIN32)
    std::uint64_t adapter_luid() const { return adapter_luid_; }
    // Input handles are borrowed. VDA duplicates them before importing, so
    // the caller may close its handle after this function returns.
    VulkanBuffer import_d3d12_buffer(
        void* shared_handle,
        VkDeviceSize bytes);
    VulkanSemaphore import_d3d12_fence(
        void* shared_handle,
        std::uint64_t value);
    VulkanImage import_d3d12_image(
        void* shared_handle,
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format,
        VkImageUsageFlags usage);
#endif

    VulkanBuffer create_device_buffer(VkDeviceSize bytes);
    VulkanBuffer create_host_buffer(VkDeviceSize bytes);
    void discard(VulkanBuffer& buffer) noexcept;
    void write_host(
        VulkanBuffer& destination,
        const void* data,
        std::size_t bytes);
    void upload(VulkanBuffer& destination, const void* data, std::size_t bytes);
    void download(
        const VulkanBuffer& source,
        void* data,
        std::size_t bytes);
    void copy(
        VulkanBuffer& destination,
        VkDeviceSize destination_offset,
        const VulkanBuffer& source,
        VkDeviceSize source_offset,
        VkDeviceSize bytes);
    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const;
    void acquire_external_buffer(
        const VulkanBuffer& buffer,
        VkAccessFlags destination_access);
    void release_external_buffer(
        const VulkanBuffer& buffer,
        VkAccessFlags source_access);
    void acquire_external_image(
        const VulkanImage& image,
        VkImageLayout layout,
        VkAccessFlags destination_access);
    void release_external_image(
        const VulkanImage& image,
        VkImageLayout layout,
        VkAccessFlags source_access);

    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        std::uint32_t binding_count,
        std::uint32_t push_constant_bytes);
    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        const std::vector<VkDescriptorType>& descriptor_types,
        std::uint32_t push_constant_bytes);
    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        const std::vector<VkDescriptorType>& descriptor_types,
        const std::vector<VkAccessFlags>& descriptor_access,
        std::uint32_t push_constant_bytes);

    void dispatch(
        const VulkanPipeline& pipeline,
        const std::vector<const VulkanBuffer*>& buffers,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1,
        const VulkanSemaphore* wait = nullptr);
    void dispatch_image_to_buffer(
        const VulkanPipeline& pipeline,
        const VulkanImage& image,
        VulkanBuffer& buffer,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1);
    void dispatch_buffer_to_image(
        const VulkanPipeline& pipeline,
        const VulkanBuffer& buffer,
        VulkanImage& image,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1);
    void dispatch_buffers_to_image(
        const VulkanPipeline& pipeline,
        const std::vector<const VulkanBuffer*>& buffers,
        VulkanImage& image,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1);

    template <typename Function>
    void batch(Function&& function) {
        if (profile_dispatches_) {
            std::forward<Function>(function)();
            return;
        }
        if (batch_command_ != VK_NULL_HANDLE || segmented_batch_) {
            std::forward<Function>(function)();
            return;
        }
        begin_batch();
        try {
            std::forward<Function>(function)();
            end_batch();
        } catch (...) {
            cancel_batch();
            throw;
        }
    }

    template <typename Function>
    VulkanSubmission batch_async(
        VulkanSemaphore wait,
        VulkanSemaphore signal,
        Function&& function) {
        if (batch_command_ != VK_NULL_HANDLE || segmented_batch_) {
            throw std::logic_error(
                "asynchronous Vulkan batch cannot be nested");
        }
        begin_batch();
        try {
            std::forward<Function>(function)();
            return end_batch_async(
                std::move(wait), std::move(signal));
        } catch (...) {
            cancel_batch();
            throw;
        }
    }

    template <typename Function>
    VulkanSubmission segmented_batch_async(
        VulkanSemaphore wait,
        VulkanSemaphore signal,
        Function&& function) {
        if (batch_command_ != VK_NULL_HANDLE || segmented_batch_) {
            throw std::logic_error(
                "asynchronous Vulkan batch cannot be nested");
        }
        begin_segmented_batch();
        try {
            std::forward<Function>(function)();
            return end_segmented_batch_async(
                std::move(wait), std::move(signal));
        } catch (...) {
            cancel_batch();
            throw;
        }
    }

private:
    friend class VulkanSemaphore;
    friend class VulkanSubmission;
    friend class VulkanBuffer;
    friend class VulkanImage;
    friend class VulkanPipeline;

    std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags properties) const;
    VulkanBuffer create_buffer(
        VkDeviceSize bytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties);
    void copy_buffer_raw(
        VkBuffer source,
        VkBuffer destination,
        VkDeviceSize source_offset,
        VkDeviceSize destination_offset,
        VkDeviceSize bytes);
    VkCommandBuffer begin_commands();
    VulkanSubmission submit_commands(
        VkCommandBuffer command_buffer,
        const VulkanSemaphore* wait = nullptr,
        const VulkanSemaphore* signal = nullptr);
    void dispatch_resources(
        const VulkanPipeline& pipeline,
        const std::vector<VulkanDispatchResource>& resources,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y,
        std::uint32_t group_z,
        const VulkanSemaphore* wait);
    void end_commands(
        VkCommandBuffer command_buffer,
        const VulkanSemaphore* wait = nullptr);
    void begin_batch();
    void end_batch();
    VulkanSubmission end_batch_async(
        VulkanSemaphore wait,
        VulkanSemaphore signal);
    void begin_segmented_batch();
    VulkanSubmission end_segmented_batch_async(
        VulkanSemaphore wait,
        VulkanSemaphore signal);
    void finish_segment();
    VkCommandBuffer active_batch_command();
    VulkanSemaphore create_timeline_semaphore(std::uint64_t initial_value);
    void cancel_batch() noexcept;
    void release_batch_resources() noexcept;
    void recycle_or_destroy(VulkanDeferredBuffer buffer) noexcept;
    void destroy(VulkanSemaphore& semaphore) noexcept;
    void destroy(VulkanSubmission& submission) noexcept;
    void destroy(VulkanBuffer& buffer) noexcept;
    void destroy(VulkanImage& image) noexcept;
    void destroy(VulkanPipeline& pipeline) noexcept;
    void release_submission_resources(
        VulkanSubmission& submission) noexcept;
    void release() noexcept;
    void record_profile(
        const VulkanPipeline& pipeline,
        std::uint64_t ticks);
    void print_profile() const noexcept;

    struct ProfileStat {
        std::uint64_t total_ticks = 0;
        std::uint64_t maximum_ticks = 0;
        std::uint64_t dispatches = 0;
    };
    static void check(VkResult result, const char* operation);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkQueryPool profile_query_pool_ = VK_NULL_HANDLE;
    bool profile_dispatches_ = false;
    float timestamp_period_ns_ = 0.0f;
    std::unordered_map<std::string, ProfileStat> profile_stats_;
    VkCommandBuffer batch_command_ = VK_NULL_HANDLE;
    bool batch_has_dispatch_ = false;
    bool segmented_batch_ = false;
    bool segment_has_commands_ = false;
    std::vector<VkCommandBuffer> batch_commands_;
    bool track_resource_hazards_ = true;
    std::vector<VulkanBatchedDescriptor> batch_descriptor_sets_;
    std::vector<VulkanDeferredBuffer> batch_deferred_buffers_;
    std::unordered_map<VkBuffer, VkAccessFlags>
        batch_buffer_access_;
    std::unordered_map<VkImage, VkAccessFlags>
        batch_image_access_;
    std::unordered_map<VkImage, VkImageLayout>
        batch_image_layout_;
    std::vector<VulkanDeferredBuffer> device_buffer_pool_;
    std::vector<VulkanDeferredBuffer> host_buffer_pool_;
    VkDeviceSize pooled_device_bytes_ = 0;
    VkDeviceSize pooled_host_bytes_ = 0;
    VulkanExternalCapabilities external_capabilities_{};
    bool float16_supported_ = false;
    bool packed_int8_dot_supported_ = false;
#if defined(_WIN32)
    std::uint64_t adapter_luid_ = 0;
    PFN_vkGetMemoryWin32HandlePropertiesKHR
        get_memory_win32_handle_properties_ = nullptr;
    PFN_vkImportSemaphoreWin32HandleKHR
        import_semaphore_win32_handle_ = nullptr;
#endif
    std::atomic<std::uint64_t> tensor_upload_bytes_{0};
    std::atomic<std::uint64_t> tensor_download_bytes_{0};
    std::string device_name_;
};

}  // namespace da3_native
