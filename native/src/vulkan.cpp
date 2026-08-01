#include "vulkan.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace da3_native {
namespace {

std::atomic<std::uint64_t> g_tensor_upload_bytes{0u};
std::atomic<std::uint64_t> g_tensor_download_bytes{0u};

template <typename Handle>
void exchange_handle(Handle& left, Handle& right) {
    std::swap(left, right);
}

#if defined(_WIN32)
bool has_extension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* name) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}
#endif

}  // namespace

struct VulkanSubmission::Resources {
    std::vector<VulkanBatchedDescriptor> descriptor_sets;
    std::vector<VulkanDeferredBuffer> deferred_buffers;
    std::vector<VkCommandBuffer> commands;
    VulkanSemaphore wait;
    VulkanSemaphore signal;
    VulkanSemaphore timeline;
};

void VulkanContext::check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with Vulkan error " +
            std::to_string(static_cast<int>(result)));
    }
}

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
    *this = std::move(other);
}

VulkanSemaphore& VulkanSemaphore::operator=(
    VulkanSemaphore&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        semaphore_ = std::exchange(
            other.semaphore_, VK_NULL_HANDLE);
        value_ = std::exchange(other.value_, 0);
    }
    return *this;
}

VulkanSemaphore::~VulkanSemaphore() {
    if (owner_) owner_->destroy(*this);
}

VulkanSubmission::VulkanSubmission(VulkanSubmission&& other) noexcept {
    *this = std::move(other);
}

VulkanSubmission& VulkanSubmission::operator=(
    VulkanSubmission&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        command_ = std::exchange(other.command_, VK_NULL_HANDLE);
        fence_ = std::exchange(other.fence_, VK_NULL_HANDLE);
        resources_ = std::exchange(other.resources_, nullptr);
    }
    return *this;
}

VulkanSubmission::~VulkanSubmission() {
    if (owner_) owner_->destroy(*this);
}

bool VulkanSubmission::ready() const {
    if (owner_ == nullptr || fence_ == VK_NULL_HANDLE) {
        throw std::logic_error("invalid Vulkan submission");
    }
    const VkResult result =
        vkGetFenceStatus(owner_->device_, fence_);
    if (result == VK_SUCCESS) return true;
    if (result == VK_NOT_READY) return false;
    VulkanContext::check(result, "vkGetFenceStatus");
    return false;
}

void VulkanSubmission::wait() const {
    if (owner_ == nullptr || fence_ == VK_NULL_HANDLE) {
        throw std::logic_error("invalid Vulkan submission");
    }
    VulkanContext::check(
        vkWaitForFences(
            owner_->device_, 1, &fence_, VK_TRUE, UINT64_MAX),
        "vkWaitForFences");
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
        size_ = std::exchange(other.size_, 0);
        mapped_ = std::exchange(other.mapped_, nullptr);
        cacheable_ = std::exchange(other.cacheable_, false);
    }
    return *this;
}

VulkanBuffer::~VulkanBuffer() {
    if (owner_) owner_->destroy(*this);
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        image_ = std::exchange(other.image_, VK_NULL_HANDLE);
        memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
        view_ = std::exchange(other.view_, VK_NULL_HANDLE);
        sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
        format_ = std::exchange(
            other.format_, VK_FORMAT_UNDEFINED);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
    }
    return *this;
}

VulkanImage::~VulkanImage() {
    if (owner_) owner_->destroy(*this);
}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept {
    *this = std::move(other);
}

VulkanPipeline& VulkanPipeline::operator=(VulkanPipeline&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        descriptor_layout_ =
            std::exchange(other.descriptor_layout_, VK_NULL_HANDLE);
        layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
        pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
        descriptor_types_ = std::move(other.descriptor_types_);
        descriptor_access_ = std::move(other.descriptor_access_);
        debug_name_ = std::move(other.debug_name_);
        cached_descriptor_sets_ =
            std::move(other.cached_descriptor_sets_);
        push_constant_bytes_ = std::exchange(other.push_constant_bytes_, 0);
    }
    return *this;
}

VulkanPipeline::~VulkanPipeline() {
    if (owner_) owner_->destroy(*this);
}

void VulkanPipeline::set_debug_name(const char* name) {
    debug_name_ = name != nullptr ? name : "";
}

VulkanContext::VulkanContext(
    std::uint32_t device_index,
    bool track_resource_hazards)
    : track_resource_hazards_(track_resource_hazards) {
    try {
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Depth Anything V2 C",
        1,
        "VDA",
        1,
        VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application,
        0,
        nullptr,
        0,
        nullptr,
    };
    check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
        "vkEnumeratePhysicalDevices");
    if (device_count == 0 || device_index >= device_count) {
        throw std::runtime_error("requested Vulkan device does not exist");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
        "vkEnumeratePhysicalDevices");
    physical_device_ = devices[device_index];

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    device_name_ = properties.deviceName;
#if defined(_WIN32)
    VkPhysicalDeviceIDProperties identity{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &identity,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    if (identity.deviceLUIDValid) {
        static_assert(VK_LUID_SIZE == sizeof(adapter_luid_));
        std::memcpy(
            &adapter_luid_, identity.deviceLUID, VK_LUID_SIZE);
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(
        physical_device_, &memory_properties_);

    std::uint32_t extension_count = 0;
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, nullptr),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<const char*> enabled_extensions;
#if defined(_WIN32)
    const bool has_external_memory_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    const bool has_external_semaphore_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    if (has_external_memory_win32) {
        const VkPhysicalDeviceExternalBufferInfo external_buffer{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            nullptr,
            0,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
        };
        VkExternalBufferProperties external_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalBufferProperties(
            physical_device_,
            &external_buffer,
            &external_properties);
        external_capabilities_.d3d12_resource_import =
            (external_properties.externalMemoryProperties
                 .externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        const auto image_importable =
            [&](VkFormat format, VkImageUsageFlags usage) {
                const VkPhysicalDeviceExternalImageFormatInfo external{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                    nullptr,
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                };
                const VkPhysicalDeviceImageFormatInfo2 image_info{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                    &external,
                    format,
                    VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    0,
                };
                VkExternalImageFormatProperties external_properties{
                    VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                };
                VkImageFormatProperties2 image_properties{
                    VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                    &external_properties,
                };
                const VkResult queried =
                    vkGetPhysicalDeviceImageFormatProperties2(
                        physical_device_,
                        &image_info,
                        &image_properties);
                return queried == VK_SUCCESS &&
                    (external_properties.externalMemoryProperties
                         .externalMemoryFeatures &
                     VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
            };
        external_capabilities_
            .d3d12_bgra8_sampled_image_import =
            image_importable(
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_rgba8_sampled_image_import =
            image_importable(
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_r32_storage_image_import =
            image_importable(
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
    if (has_external_semaphore_win32) {
        const VkPhysicalDeviceExternalSemaphoreInfo external_semaphore{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            nullptr,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
        };
        VkExternalSemaphoreProperties semaphore_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalSemaphoreProperties(
            physical_device_,
            &external_semaphore,
            &semaphore_properties);
        external_capabilities_.d3d12_fence_import =
            (semaphore_properties.externalSemaphoreFeatures &
             VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#else
    (void)extensions;
#endif

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, families.data());
    auto family = std::find_if(
        families.begin(), families.end(), [](const auto& candidate) {
            return candidate.queueCount > 0 &&
                (candidate.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        });
    if (family == families.end()) {
        throw std::runtime_error("Vulkan device has no compute queue");
    }
    queue_family_ = static_cast<std::uint32_t>(
        std::distance(families.begin(), family));
    const char* profile_environment =
        std::getenv("DA3_VULKAN_PROFILE");
    profile_dispatches_ =
        profile_environment != nullptr &&
        profile_environment[0] != '\0' &&
        profile_environment[0] != '0' &&
        family->timestampValidBits != 0;
    timestamp_period_ns_ = properties.limits.timestampPeriod;

    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        queue_family_,
        1,
        &priority,
    };
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    };
    VkPhysicalDeviceFeatures2 features2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &timeline_features,
    };
    vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
    external_capabilities_.timeline_semaphore =
        timeline_features.timelineSemaphore == VK_TRUE;
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &timeline_features,
        0,
        1,
        &queue_info,
        0,
        nullptr,
        static_cast<std::uint32_t>(enabled_extensions.size()),
        enabled_extensions.data(),
        nullptr,
    };
    check(
        vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
#if defined(_WIN32)
    get_memory_win32_handle_properties_ =
        reinterpret_cast<
            PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkGetMemoryWin32HandlePropertiesKHR"));
    import_semaphore_win32_handle_ =
        reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkImportSemaphoreWin32HandleKHR"));
    external_capabilities_.d3d12_resource_import =
        external_capabilities_.d3d12_resource_import &&
        get_memory_win32_handle_properties_ != nullptr;
    external_capabilities_.d3d12_fence_import =
        external_capabilities_.d3d12_fence_import &&
        import_semaphore_win32_handle_ != nullptr;
#endif
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_,
    };
    check(
        vkCreateCommandPool(
            device_, &command_pool_info, nullptr, &command_pool_),
        "vkCreateCommandPool");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
    };
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        4096,
        static_cast<std::uint32_t>(std::size(pool_sizes)),
        pool_sizes,
    };
    check(
        vkCreateDescriptorPool(
            device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
    if (profile_dispatches_) {
        const VkQueryPoolCreateInfo query_pool_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            nullptr,
            0,
            VK_QUERY_TYPE_TIMESTAMP,
            2,
            0,
        };
        check(
            vkCreateQueryPool(
                device_,
                &query_pool_info,
                nullptr,
                &profile_query_pool_),
            "vkCreateQueryPool");
    }
    } catch (...) {
        release();
        throw;
    }
}

VulkanContext::~VulkanContext() {
    release();
}

void VulkanContext::release() noexcept {
    if (device_) {
        vkDeviceWaitIdle(device_);
        print_profile();
        cancel_batch();
        for (VulkanDeferredBuffer& buffer : device_buffer_pool_) {
            buffer.cacheable = false;
            recycle_or_destroy(buffer);
        }
        device_buffer_pool_.clear();
        pooled_device_bytes_ = 0;
        for (VulkanDeferredBuffer& buffer : host_buffer_pool_) {
            buffer.cacheable = false;
            recycle_or_destroy(buffer);
        }
        host_buffer_pool_.clear();
        pooled_host_bytes_ = 0;
        if (descriptor_pool_) {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        }
        if (profile_query_pool_) {
            vkDestroyQueryPool(
                device_, profile_query_pool_, nullptr);
        }
        if (command_pool_) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        descriptor_pool_ = VK_NULL_HANDLE;
        profile_query_pool_ = VK_NULL_HANDLE;
        command_pool_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::record_profile(
    const VulkanPipeline& pipeline,
    std::uint64_t ticks) {
    const std::string& name = pipeline.debug_name_.empty()
        ? std::string("unnamed")
        : pipeline.debug_name_;
    ProfileStat& stat = profile_stats_[name];
    stat.total_ticks += ticks;
    stat.maximum_ticks = std::max(stat.maximum_ticks, ticks);
    ++stat.dispatches;
}

void VulkanContext::print_profile() const noexcept {
    if (!profile_dispatches_ || profile_stats_.empty()) return;
    std::vector<std::pair<std::string, ProfileStat>> sorted(
        profile_stats_.begin(), profile_stats_.end());
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const auto& left, const auto& right) {
            return left.second.total_ticks >
                right.second.total_ticks;
        });
    std::fprintf(
        stderr,
        "VDA Vulkan profile: %s (GPU timestamps)\n",
        device_name_.c_str());
    for (const auto& [name, stat] : sorted) {
        const double total_ms =
            static_cast<double>(stat.total_ticks) *
            timestamp_period_ns_ / 1.0e6;
        const double average_ms =
            total_ms / static_cast<double>(stat.dispatches);
        const double maximum_ms =
            static_cast<double>(stat.maximum_ticks) *
            timestamp_period_ns_ / 1.0e6;
        std::fprintf(
            stderr,
            "  %-34s total=%10.3f ms count=%6llu "
            "avg=%8.4f ms max=%8.4f ms\n",
            name.c_str(),
            total_ms,
            static_cast<unsigned long long>(stat.dispatches),
            average_ms,
            maximum_ms);
    }
}

std::uint32_t VulkanContext::find_memory_type(
    std::uint32_t type_bits,
    VkMemoryPropertyFlags properties) const {
    for (std::uint32_t index = 0;
         index < memory_properties_.memoryTypeCount;
         ++index) {
        if ((type_bits & (1u << index)) != 0 &&
            (memory_properties_.memoryTypes[index].propertyFlags & properties) ==
                properties) {
            return index;
        }
    }
    throw std::runtime_error("no compatible Vulkan memory type");
}

VulkanBuffer VulkanContext::create_buffer(
    VkDeviceSize bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties) {
    if (bytes == 0) {
        throw std::invalid_argument("cannot create an empty Vulkan buffer");
    }
    VulkanBuffer result;
    result.owner_ = this;
    result.size_ = bytes;
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        bytes,
        usage,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    check(
        vkCreateBuffer(device_, &buffer_info, nullptr, &result.buffer_),
        "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, result.buffer_, &requirements);
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        requirements.size,
        find_memory_type(requirements.memoryTypeBits, properties),
    };
    check(
        vkAllocateMemory(device_, &allocate_info, nullptr, &result.memory_),
        "vkAllocateMemory");
    check(
        vkBindBufferMemory(device_, result.buffer_, result.memory_, 0),
        "vkBindBufferMemory");
    if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        check(
            vkMapMemory(
                device_, result.memory_, 0, bytes, 0, &result.mapped_),
            "vkMapMemory");
    }
    return result;
}

VulkanBuffer VulkanContext::create_device_buffer(VkDeviceSize bytes) {
    auto best = device_buffer_pool_.end();
    for (auto candidate = device_buffer_pool_.begin();
         candidate != device_buffer_pool_.end();
         ++candidate) {
        if (candidate->size >= bytes &&
            (best == device_buffer_pool_.end() ||
             candidate->size < best->size)) {
            best = candidate;
        }
    }
    if (best != device_buffer_pool_.end()) {
        VulkanBuffer result;
        result.owner_ = this;
        result.buffer_ = best->buffer;
        result.memory_ = best->memory;
        result.size_ = best->size;
        result.mapped_ = nullptr;
        result.cacheable_ = true;
        pooled_device_bytes_ -= best->size;
        device_buffer_pool_.erase(best);
        return result;
    }
    VulkanBuffer result = create_buffer(
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    result.cacheable_ = true;
    return result;
}

#if defined(_WIN32)
VulkanBuffer VulkanContext::import_d3d12_buffer(
    void* shared_handle,
    VkDeviceSize bytes) {
    if (!external_capabilities_.d3d12_resource_import) {
        throw std::runtime_error(
            "Vulkan device cannot import D3D12 resources");
    }
    if (shared_handle == nullptr || bytes == 0) {
        throw std::invalid_argument(
            "invalid D3D12 shared buffer");
    }
    HANDLE duplicated = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(),
            static_cast<HANDLE>(shared_handle),
            GetCurrentProcess(),
            &duplicated,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error(
            "failed to duplicate D3D12 shared handle");
    }

    VulkanBuffer result;
    result.owner_ = this;
    result.size_ = bytes;
    const VkExternalMemoryBufferCreateInfo external_buffer{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        nullptr,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
    };
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        &external_buffer,
        0,
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    try {
        check(
            vkCreateBuffer(
                device_, &buffer_info, nullptr, &result.buffer_),
            "vkCreateBuffer(D3D12 import)");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            device_, result.buffer_, &requirements);
        VkMemoryWin32HandlePropertiesKHR handle_properties{
            VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
        };
        check(
            get_memory_win32_handle_properties_(
                device_,
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                duplicated,
                &handle_properties),
            "vkGetMemoryWin32HandlePropertiesKHR");
        const VkMemoryDedicatedAllocateInfo dedicated{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            nullptr,
            VK_NULL_HANDLE,
            result.buffer_,
        };
        const VkImportMemoryWin32HandleInfoKHR import{
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
            &dedicated,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            duplicated,
            nullptr,
        };
        const VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            &import,
            requirements.size,
            find_memory_type(
                requirements.memoryTypeBits &
                    handle_properties.memoryTypeBits,
                0),
        };
        check(
            vkAllocateMemory(
                device_, &allocation, nullptr, &result.memory_),
            "vkAllocateMemory(D3D12 import)");
        CloseHandle(duplicated);
        duplicated = nullptr;
        check(
            vkBindBufferMemory(
                device_, result.buffer_, result.memory_, 0),
            "vkBindBufferMemory(D3D12 import)");
        return result;
    } catch (...) {
        if (duplicated != nullptr) CloseHandle(duplicated);
        throw;
    }
}

VulkanImage VulkanContext::import_d3d12_image(
    void* shared_handle,
    std::uint32_t width,
    std::uint32_t height,
    VkFormat format,
    VkImageUsageFlags usage) {
    if (!external_capabilities_.d3d12_resource_import) {
        throw std::runtime_error(
            "Vulkan device cannot import D3D12 resources");
    }
    if (shared_handle == nullptr || width == 0 || height == 0 ||
        format == VK_FORMAT_UNDEFINED ||
        (usage & (VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_STORAGE_BIT)) == 0) {
        throw std::invalid_argument(
            "invalid D3D12 shared image");
    }
    HANDLE duplicated = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(),
            static_cast<HANDLE>(shared_handle),
            GetCurrentProcess(),
            &duplicated,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error(
            "failed to duplicate D3D12 image handle");
    }

    VulkanImage result;
    result.owner_ = this;
    result.format_ = format;
    result.width_ = width;
    result.height_ = height;
    const VkExternalMemoryImageCreateInfo external_image{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        nullptr,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
    };
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        &external_image,
        0,
        VK_IMAGE_TYPE_2D,
        format,
        {width, height, 1},
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        usage,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED,
    };
    try {
        check(
            vkCreateImage(
                device_, &image_info, nullptr, &result.image_),
            "vkCreateImage(D3D12 import)");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(
            device_, result.image_, &requirements);
        VkMemoryWin32HandlePropertiesKHR handle_properties{
            VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
        };
        check(
            get_memory_win32_handle_properties_(
                device_,
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                duplicated,
                &handle_properties),
            "vkGetMemoryWin32HandlePropertiesKHR(image)");
        const VkMemoryDedicatedAllocateInfo dedicated{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            nullptr,
            result.image_,
            VK_NULL_HANDLE,
        };
        const VkImportMemoryWin32HandleInfoKHR import{
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
            &dedicated,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            duplicated,
            nullptr,
        };
        const VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            &import,
            requirements.size,
            find_memory_type(
                requirements.memoryTypeBits &
                    handle_properties.memoryTypeBits,
                0),
        };
        check(
            vkAllocateMemory(
                device_, &allocation, nullptr, &result.memory_),
            "vkAllocateMemory(D3D12 image import)");
        CloseHandle(duplicated);
        duplicated = nullptr;
        check(
            vkBindImageMemory(
                device_, result.image_, result.memory_, 0),
            "vkBindImageMemory(D3D12 import)");
        const VkImageViewCreateInfo view_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            result.image_,
            VK_IMAGE_VIEW_TYPE_2D,
            format,
            {
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        };
        check(
            vkCreateImageView(
                device_, &view_info, nullptr, &result.view_),
            "vkCreateImageView(D3D12 import)");
        if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) {
            const VkSamplerCreateInfo sampler_info{
                VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                nullptr,
                0,
                VK_FILTER_NEAREST,
                VK_FILTER_NEAREST,
                VK_SAMPLER_MIPMAP_MODE_NEAREST,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                0.0f,
                VK_FALSE,
                1.0f,
                VK_FALSE,
                VK_COMPARE_OP_ALWAYS,
                0.0f,
                0.0f,
                VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                VK_FALSE,
            };
            check(
                vkCreateSampler(
                    device_,
                    &sampler_info,
                    nullptr,
                    &result.sampler_),
                "vkCreateSampler(D3D12 import)");
        }
        return result;
    } catch (...) {
        if (duplicated != nullptr) CloseHandle(duplicated);
        throw;
    }
}

VulkanSemaphore VulkanContext::import_d3d12_fence(
    void* shared_handle,
    std::uint64_t value) {
    if (!external_capabilities_.d3d12_fence_import) {
        throw std::runtime_error(
            "Vulkan device cannot import D3D12 fences");
    }
    if (shared_handle == nullptr) {
        throw std::invalid_argument(
            "invalid D3D12 shared fence");
    }
    HANDLE duplicated = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(),
            static_cast<HANDLE>(shared_handle),
            GetCurrentProcess(),
            &duplicated,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error(
            "failed to duplicate D3D12 fence handle");
    }
    VulkanSemaphore result;
    result.owner_ = this;
    result.value_ = value;
    const VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        nullptr,
        0,
    };
    try {
        check(
            vkCreateSemaphore(
                device_,
                &semaphore_info,
                nullptr,
                &result.semaphore_),
            "vkCreateSemaphore(D3D12 import)");
        const VkImportSemaphoreWin32HandleInfoKHR import{
            VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
            nullptr,
            result.semaphore_,
            0,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
            duplicated,
            nullptr,
        };
        check(
            import_semaphore_win32_handle_(device_, &import),
            "vkImportSemaphoreWin32HandleKHR");
        CloseHandle(duplicated);
        return result;
    } catch (...) {
        CloseHandle(duplicated);
        throw;
    }
}

#endif

VulkanBuffer VulkanContext::create_host_buffer(VkDeviceSize bytes) {
    auto best = host_buffer_pool_.end();
    for (auto candidate = host_buffer_pool_.begin();
         candidate != host_buffer_pool_.end();
         ++candidate) {
        if (candidate->size >= bytes &&
            (best == host_buffer_pool_.end() ||
             candidate->size < best->size)) {
            best = candidate;
        }
    }
    if (best != host_buffer_pool_.end()) {
        VulkanBuffer result;
        result.owner_ = this;
        result.buffer_ = best->buffer;
        result.memory_ = best->memory;
        result.size_ = best->size;
        result.mapped_ = best->mapped;
        result.cacheable_ = true;
        pooled_host_bytes_ -= best->size;
        host_buffer_pool_.erase(best);
        return result;
    }
    VulkanBuffer result = create_buffer(
        bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    result.cacheable_ = true;
    return result;
}

void VulkanContext::write_host(
    VulkanBuffer& destination,
    const void* data,
    std::size_t bytes) {
    if (destination.owner_ != this || destination.mapped_ == nullptr ||
        data == nullptr || bytes > destination.size_) {
        throw std::invalid_argument("invalid Vulkan host write");
    }
    std::memcpy(destination.mapped_, data, bytes);
}

VkCommandBuffer VulkanContext::begin_commands() {
    const VkCommandBufferAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        command_pool_,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(
        vkAllocateCommandBuffers(device_, &allocate_info, &command),
        "vkAllocateCommandBuffers");
    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr,
    };
    check(vkBeginCommandBuffer(command, &begin_info), "vkBeginCommandBuffer");
    return command;
}

void VulkanContext::end_commands(
    VkCommandBuffer command,
    const VulkanSemaphore* wait) {
    VulkanSubmission submission = submit_commands(command, wait);
    submission.wait();
}

VulkanSubmission VulkanContext::submit_commands(
    VkCommandBuffer command,
    const VulkanSemaphore* wait,
    const VulkanSemaphore* signal) {
    if (wait != nullptr && wait->owner_ != this) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw std::invalid_argument(
            "foreign Vulkan wait semaphore");
    }
    if (signal != nullptr && signal->owner_ != this) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw std::invalid_argument(
            "foreign Vulkan signal semaphore");
    }
    try {
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    } catch (...) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw;
    }
    VkFence fence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        0,
    };
    try {
        check(
            vkCreateFence(device_, &fence_info, nullptr, &fence),
            "vkCreateFence");
    } catch (...) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw;
    }
    const VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
#if defined(_WIN32)
    const std::uint64_t wait_value =
        wait != nullptr ? wait->value_ : 0;
    const std::uint64_t signal_value =
        signal != nullptr ? signal->value_ : 0;
    const VkD3D12FenceSubmitInfoKHR d3d12_values{
        VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR,
        nullptr,
        wait != nullptr ? 1u : 0u,
        wait != nullptr ? &wait_value : nullptr,
        signal != nullptr ? 1u : 0u,
        signal != nullptr ? &signal_value : nullptr,
    };
#endif
    const VkSemaphore wait_handle =
        wait != nullptr ? wait->semaphore_ : VK_NULL_HANDLE;
    const VkSemaphore signal_handle =
        signal != nullptr ? signal->semaphore_ : VK_NULL_HANDLE;
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
#if defined(_WIN32)
        wait != nullptr || signal != nullptr ? &d3d12_values : nullptr,
#else
        nullptr,
#endif
        wait != nullptr ? 1u : 0u,
        wait != nullptr ? &wait_handle : nullptr,
        wait != nullptr ? &wait_stage : nullptr,
        1,
        &command,
        signal != nullptr ? 1u : 0u,
        signal != nullptr ? &signal_handle : nullptr,
    };
    const VkResult submitted =
        vkQueueSubmit(queue_, 1, &submit, fence);
    if (submitted != VK_SUCCESS) {
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        check(submitted, "vkQueueSubmit");
    }
    VulkanSubmission result;
    result.owner_ = this;
    result.command_ = command;
    result.fence_ = fence;
    return result;
}

void VulkanContext::begin_batch() {
    if (batch_command_ != VK_NULL_HANDLE) {
        throw std::logic_error("nested Vulkan batch");
    }
    batch_command_ = begin_commands();
    batch_has_dispatch_ = false;
    batch_buffer_access_.clear();
    batch_image_access_.clear();
    batch_image_layout_.clear();
}

void VulkanContext::release_batch_resources() noexcept {
    for (const VulkanBatchedDescriptor& descriptor :
         batch_descriptor_sets_) {
        if (descriptor.pipeline && descriptor.set) {
            descriptor.pipeline->cached_descriptor_sets_.push_back(
                descriptor.set);
        }
    }
    batch_descriptor_sets_.clear();
    for (const VulkanDeferredBuffer& buffer : batch_deferred_buffers_) {
        recycle_or_destroy(buffer);
    }
    batch_deferred_buffers_.clear();
    batch_buffer_access_.clear();
    batch_image_access_.clear();
    batch_image_layout_.clear();
}

VulkanSubmission VulkanContext::end_batch_async(
    VulkanSemaphore wait,
    VulkanSemaphore signal) {
    if (batch_command_ == VK_NULL_HANDLE) {
        throw std::logic_error("no active Vulkan batch");
    }
    auto resources = std::make_unique<VulkanSubmission::Resources>();
    resources->descriptor_sets = std::move(batch_descriptor_sets_);
    resources->deferred_buffers = std::move(batch_deferred_buffers_);
    resources->wait = std::move(wait);
    resources->signal = std::move(signal);
    VkCommandBuffer command = batch_command_;
    batch_command_ = VK_NULL_HANDLE;
    batch_has_dispatch_ = false;
    try {
        VulkanSubmission result = submit_commands(
            command, &resources->wait, &resources->signal);
        result.resources_ = resources.release();
        return result;
    } catch (...) {
        for (const VulkanBatchedDescriptor& descriptor :
             resources->descriptor_sets) {
            if (descriptor.pipeline && descriptor.set) {
                descriptor.pipeline->cached_descriptor_sets_.push_back(
                    descriptor.set);
            }
        }
        for (const VulkanDeferredBuffer& buffer :
             resources->deferred_buffers) {
            recycle_or_destroy(buffer);
        }
        throw;
    }
}

void VulkanContext::end_batch() {
    if (batch_command_ == VK_NULL_HANDLE) {
        throw std::logic_error("no active Vulkan batch");
    }
    VkCommandBuffer command = batch_command_;
    batch_command_ = VK_NULL_HANDLE;
    batch_has_dispatch_ = false;
    end_commands(command);
    release_batch_resources();
}

void VulkanContext::cancel_batch() noexcept {
    if (batch_command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(
            device_, command_pool_, 1, &batch_command_);
        batch_command_ = VK_NULL_HANDLE;
    }
    for (VkCommandBuffer command : batch_commands_) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
    }
    batch_commands_.clear();
    segmented_batch_ = false;
    segment_has_commands_ = false;
    batch_has_dispatch_ = false;
    release_batch_resources();
}

void VulkanContext::copy_buffer_raw(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize source_offset,
    VkDeviceSize destination_offset,
    VkDeviceSize bytes) {
    VkCommandBuffer command = segmented_batch_
        ? active_batch_command()
        : begin_commands();
    const VkBufferCopy region{
        source_offset, destination_offset, bytes};
    vkCmdCopyBuffer(command, source, destination, 1, &region);
    if (segmented_batch_) {
        const VkMemoryBarrier release{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &release, 0, nullptr, 0, nullptr);
        segment_has_commands_ = true;
        finish_segment();
    } else {
        end_commands(command);
    }
}

VulkanSubmission VulkanContext::end_segmented_batch_async(
    VulkanSemaphore wait,
    VulkanSemaphore signal) {
    if (!segmented_batch_) {
        throw std::logic_error("no active segmented Vulkan batch");
    }
    finish_segment();
    if (batch_command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &batch_command_);
        batch_command_ = VK_NULL_HANDLE;
    }
    segmented_batch_ = false;
    if (batch_commands_.empty()) {
        throw std::logic_error("empty segmented Vulkan batch");
    }
    auto resources = std::make_unique<VulkanSubmission::Resources>();
    resources->descriptor_sets = std::move(batch_descriptor_sets_);
    resources->deferred_buffers = std::move(batch_deferred_buffers_);
    resources->commands = std::move(batch_commands_);
    resources->wait = std::move(wait);
    resources->signal = std::move(signal);
    if (resources->commands.size() > 1u)
        resources->timeline = create_timeline_semaphore(0u);

    VkFence fence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    check(vkCreateFence(device_, &fence_info, nullptr, &fence),
          "vkCreateFence(segmented)");
    struct SubmitStorage {
        std::array<VkSemaphore, 1> waits{};
        std::array<VkSemaphore, 1> signals{};
        std::array<VkPipelineStageFlags, 1> stages{
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        std::array<std::uint64_t, 1> wait_values{};
        std::array<std::uint64_t, 1> signal_values{};
        VkTimelineSemaphoreSubmitInfo timeline{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
#if defined(_WIN32)
        VkD3D12FenceSubmitInfoKHR d3d12{
            VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR};
#endif
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    };
    std::vector<SubmitStorage> storage(resources->commands.size());
    const std::size_t last = resources->commands.size() - 1u;
    for (std::size_t index = 0; index <= last; ++index) {
        SubmitStorage& item = storage[index];
        const bool first = index == 0u;
        const bool final = index == last;
        item.waits[0] = first
            ? resources->wait.semaphore_
            : resources->timeline.semaphore_;
        item.wait_values[0] = first
            ? resources->wait.value_
            : static_cast<std::uint64_t>(index);
        item.signals[0] = final
            ? resources->signal.semaphore_
            : resources->timeline.semaphore_;
        item.signal_values[0] = final
            ? resources->signal.value_
            : static_cast<std::uint64_t>(index + 1u);
        item.timeline.waitSemaphoreValueCount = 1u;
        item.timeline.pWaitSemaphoreValues = item.wait_values.data();
        item.timeline.signalSemaphoreValueCount = 1u;
        item.timeline.pSignalSemaphoreValues = item.signal_values.data();
#if defined(_WIN32)
        if (first || final) {
            item.d3d12.pNext = &item.timeline;
            item.d3d12.waitSemaphoreValuesCount = 1u;
            item.d3d12.pWaitSemaphoreValues = item.wait_values.data();
            item.d3d12.signalSemaphoreValuesCount = 1u;
            item.d3d12.pSignalSemaphoreValues = item.signal_values.data();
            item.submit.pNext = &item.d3d12;
        } else {
            item.submit.pNext = &item.timeline;
        }
#else
        item.submit.pNext = &item.timeline;
#endif
        item.submit.waitSemaphoreCount = 1u;
        item.submit.pWaitSemaphores = item.waits.data();
        item.submit.pWaitDstStageMask = item.stages.data();
        item.submit.commandBufferCount = 1u;
        item.submit.pCommandBuffers = &resources->commands[index];
        item.submit.signalSemaphoreCount = 1u;
        item.submit.pSignalSemaphores = item.signals.data();
    }
    std::vector<VkSubmitInfo> submits;
    submits.reserve(storage.size());
    for (const SubmitStorage& item : storage) submits.push_back(item.submit);
    VkResult submitted = VK_SUCCESS;
    for (std::size_t index = 0; index < submits.size(); ++index) {
        submitted = vkQueueSubmit(
            queue_, 1u, &submits[index],
            index + 1u == submits.size() ? fence : VK_NULL_HANDLE);
        if (submitted != VK_SUCCESS) break;
    }
    if (submitted != VK_SUCCESS) {
        vkDestroyFence(device_, fence, nullptr);
        for (VkCommandBuffer command : resources->commands)
            vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        resources->commands.clear();
        check(submitted, "vkQueueSubmit(segmented)");
    }
    VulkanSubmission result;
    result.owner_ = this;
    result.fence_ = fence;
    result.resources_ = resources.release();
    return result;
}

void VulkanContext::begin_segmented_batch() {
    if (batch_command_ != VK_NULL_HANDLE || segmented_batch_) {
        throw std::logic_error("nested Vulkan batch");
    }
    segmented_batch_ = true;
    batch_command_ = begin_commands();
    segment_has_commands_ = false;
    batch_has_dispatch_ = false;
    batch_commands_.clear();
    batch_buffer_access_.clear();
    batch_image_access_.clear();
    batch_image_layout_.clear();
}

VkCommandBuffer VulkanContext::active_batch_command() {
    if (batch_command_ == VK_NULL_HANDLE && segmented_batch_) {
        batch_command_ = begin_commands();
        segment_has_commands_ = false;
        if (!batch_commands_.empty()) {
            const VkMemoryBarrier barrier{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            vkCmdPipelineBarrier(
                batch_command_,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
            segment_has_commands_ = true;
        }
    }
    return batch_command_;
}

void VulkanContext::finish_segment() {
    if (!segmented_batch_ || batch_command_ == VK_NULL_HANDLE ||
        !segment_has_commands_) {
        return;
    }
    check(vkEndCommandBuffer(batch_command_), "vkEndCommandBuffer(segment)");
    batch_commands_.push_back(batch_command_);
    batch_command_ = VK_NULL_HANDLE;
    segment_has_commands_ = false;
    batch_has_dispatch_ = false;
    batch_buffer_access_.clear();
    batch_image_access_.clear();
    batch_image_layout_.clear();
}

VulkanSemaphore VulkanContext::create_timeline_semaphore(
    std::uint64_t initial_value) {
    if (!external_capabilities_.timeline_semaphore)
        throw std::runtime_error(
            "Vulkan device has no timeline semaphore support");
    const VkSemaphoreTypeCreateInfo timeline_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        nullptr,
        VK_SEMAPHORE_TYPE_TIMELINE,
        initial_value,
    };
    const VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        &timeline_info,
        0,
    };
    VulkanSemaphore result;
    result.owner_ = this;
    check(vkCreateSemaphore(
        device_, &semaphore_info, nullptr, &result.semaphore_),
        "vkCreateSemaphore(timeline)");
    result.value_ = initial_value;
    return result;
}

void VulkanContext::copy(
    VulkanBuffer& destination,
    VkDeviceSize destination_offset,
    const VulkanBuffer& source,
    VkDeviceSize source_offset,
    VkDeviceSize bytes) {
    if (destination.owner_ != this || source.owner_ != this ||
        destination_offset > destination.size_ ||
        source_offset > source.size_ ||
        bytes > destination.size_ - destination_offset ||
        bytes > source.size_ - source_offset) {
        throw std::invalid_argument("invalid Vulkan buffer copy");
    }
    copy_buffer_raw(
        source.buffer_, destination.buffer_,
        source_offset, destination_offset, bytes);
}

void VulkanContext::upload(
    VulkanBuffer& destination,
    const void* data,
    std::size_t bytes) {
    if (data == nullptr || bytes > destination.size_) {
        throw std::invalid_argument("invalid Vulkan upload");
    }
    tensor_upload_bytes_.fetch_add(
        static_cast<std::uint64_t>(bytes),
        std::memory_order_relaxed);
    g_tensor_upload_bytes.fetch_add(
        static_cast<std::uint64_t>(bytes),
        std::memory_order_relaxed);
    VulkanBuffer staging = create_host_buffer(bytes);
    std::memcpy(staging.mapped_, data, bytes);
    copy_buffer_raw(
        staging.buffer_, destination.buffer_, 0, 0, bytes);
}

void VulkanContext::download(
    const VulkanBuffer& source,
    void* data,
    std::size_t bytes) {
    if (data == nullptr || bytes > source.size_) {
        throw std::invalid_argument("invalid Vulkan download");
    }
    tensor_download_bytes_.fetch_add(
        static_cast<std::uint64_t>(bytes),
        std::memory_order_relaxed);
    g_tensor_download_bytes.fetch_add(
        static_cast<std::uint64_t>(bytes),
        std::memory_order_relaxed);
    VulkanBuffer staging = create_host_buffer(bytes);
    copy_buffer_raw(
        source.buffer_, staging.buffer_, 0, 0, bytes);
    std::memcpy(data, staging.mapped_, bytes);
}

void VulkanContext::transfer_counters(
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes) const {
    upload_bytes =
        tensor_upload_bytes_.load(std::memory_order_relaxed);
    download_bytes =
        tensor_download_bytes_.load(std::memory_order_relaxed);
}

void global_transfer_counters(
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes) {
    upload_bytes = g_tensor_upload_bytes.load(std::memory_order_relaxed);
    download_bytes = g_tensor_download_bytes.load(std::memory_order_relaxed);
}

void VulkanContext::acquire_external_buffer(
    const VulkanBuffer& buffer,
    VkAccessFlags destination_access) {
    if ((!segmented_batch_ && batch_command_ == VK_NULL_HANDLE) ||
        buffer.owner_ != this ||
        buffer.buffer_ == VK_NULL_HANDLE) {
        throw std::invalid_argument(
            "external buffer acquire requires an active batch");
    }
    const VkBufferMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        0,
        destination_access,
        VK_QUEUE_FAMILY_EXTERNAL,
        queue_family_,
        buffer.buffer_,
        0,
        buffer.size_,
    };
    vkCmdPipelineBarrier(
        active_batch_command(),
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &barrier,
        0,
        nullptr);
    segment_has_commands_ = true;
}

void VulkanContext::release_external_buffer(
    const VulkanBuffer& buffer,
    VkAccessFlags source_access) {
    if ((!segmented_batch_ && batch_command_ == VK_NULL_HANDLE) ||
        buffer.owner_ != this ||
        buffer.buffer_ == VK_NULL_HANDLE) {
        throw std::invalid_argument(
            "external buffer release requires an active batch");
    }
    const VkBufferMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        source_access,
        0,
        queue_family_,
        VK_QUEUE_FAMILY_EXTERNAL,
        buffer.buffer_,
        0,
        buffer.size_,
    };
    vkCmdPipelineBarrier(
        active_batch_command(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        1,
        &barrier,
        0,
        nullptr);
    segment_has_commands_ = true;
}

void VulkanContext::acquire_external_image(
    const VulkanImage& image,
    VkImageLayout layout,
    VkAccessFlags destination_access) {
    if ((!segmented_batch_ && batch_command_ == VK_NULL_HANDLE) ||
        image.owner_ != this ||
        image.image_ == VK_NULL_HANDLE) {
        throw std::invalid_argument(
            "external image acquire requires an active batch");
    }
    const VkImageMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        0,
        destination_access,
        VK_IMAGE_LAYOUT_GENERAL,
        layout,
        VK_QUEUE_FAMILY_EXTERNAL,
        queue_family_,
        image.image_,
        {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        },
    };
    vkCmdPipelineBarrier(
        active_batch_command(),
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier);
    segment_has_commands_ = true;
}

void VulkanContext::release_external_image(
    const VulkanImage& image,
    VkImageLayout layout,
    VkAccessFlags source_access) {
    if ((!segmented_batch_ && batch_command_ == VK_NULL_HANDLE) ||
        image.owner_ != this ||
        image.image_ == VK_NULL_HANDLE) {
        throw std::invalid_argument(
            "external image release requires an active batch");
    }
    const VkImageMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        source_access,
        0,
        layout,
        VK_IMAGE_LAYOUT_GENERAL,
        queue_family_,
        VK_QUEUE_FAMILY_EXTERNAL,
        image.image_,
        {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        },
    };
    vkCmdPipelineBarrier(
        active_batch_command(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier);
    segment_has_commands_ = true;
}

VulkanPipeline VulkanContext::create_pipeline(
    const std::uint32_t* spirv,
    std::size_t spirv_bytes,
    std::uint32_t binding_count,
    std::uint32_t push_constant_bytes) {
    std::vector<VkAccessFlags> access(
        binding_count, VK_ACCESS_SHADER_READ_BIT);
    if (!access.empty()) {
        access[0] = VK_ACCESS_SHADER_WRITE_BIT;
    }
    return create_pipeline(
        spirv,
        spirv_bytes,
        std::vector<VkDescriptorType>(
            binding_count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
        access,
        push_constant_bytes);
}

VulkanPipeline VulkanContext::create_pipeline(
    const std::uint32_t* spirv,
    std::size_t spirv_bytes,
    const std::vector<VkDescriptorType>& descriptor_types,
    std::uint32_t push_constant_bytes) {
    std::vector<VkAccessFlags> access(
        descriptor_types.size(), VK_ACCESS_SHADER_READ_BIT);
    if (!access.empty()) {
        access[0] = VK_ACCESS_SHADER_WRITE_BIT;
    }
    return create_pipeline(
        spirv,
        spirv_bytes,
        descriptor_types,
        access,
        push_constant_bytes);
}

VulkanPipeline VulkanContext::create_pipeline(
    const std::uint32_t* spirv,
    std::size_t spirv_bytes,
    const std::vector<VkDescriptorType>& descriptor_types,
    const std::vector<VkAccessFlags>& descriptor_access,
    std::uint32_t push_constant_bytes) {
    if (spirv == nullptr || spirv_bytes == 0 || spirv_bytes % 4 != 0 ||
        descriptor_types.empty() ||
        descriptor_access.size() != descriptor_types.size() ||
        std::any_of(
            descriptor_access.begin(),
            descriptor_access.end(),
            [](VkAccessFlags access) {
                constexpr VkAccessFlags allowed =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                return access == 0 || (access & ~allowed) != 0;
            }) ||
        push_constant_bytes > 128 ||
        push_constant_bytes % 4 != 0) {
        throw std::invalid_argument("invalid compute pipeline description");
    }
    VulkanPipeline result;
    result.owner_ = this;
    result.descriptor_types_ = descriptor_types;
    result.descriptor_access_ = descriptor_access;
    result.push_constant_bytes_ = push_constant_bytes;

    std::vector<VkDescriptorSetLayoutBinding> bindings(
        descriptor_types.size());
    for (std::uint32_t index = 0; index < descriptor_types.size(); ++index) {
        bindings[index] = {
            index,
            descriptor_types[index],
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        };
    }
    const VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        static_cast<std::uint32_t>(bindings.size()),
        bindings.data(),
    };
    check(
        vkCreateDescriptorSetLayout(
            device_, &descriptor_info, nullptr, &result.descriptor_layout_),
        "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        push_constant_bytes,
    };
    const VkPipelineLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &result.descriptor_layout_,
        push_constant_bytes ? 1u : 0u,
        push_constant_bytes ? &push_range : nullptr,
    };
    check(
        vkCreatePipelineLayout(
            device_, &layout_info, nullptr, &result.layout_),
        "vkCreatePipelineLayout");

    const VkShaderModuleCreateInfo shader_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        spirv_bytes,
        spirv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    check(
        vkCreateShaderModule(device_, &shader_info, nullptr, &shader),
        "vkCreateShaderModule");
    const VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader,
        "main",
        nullptr,
    };
    const VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr,
        0,
        stage,
        result.layout_,
        VK_NULL_HANDLE,
        -1,
    };
    try {
        check(
            vkCreateComputePipelines(
                device_,
                VK_NULL_HANDLE,
                1,
                &pipeline_info,
                nullptr,
                &result.pipeline_),
            "vkCreateComputePipelines");
    } catch (...) {
        vkDestroyShaderModule(device_, shader, nullptr);
        throw;
    }
    vkDestroyShaderModule(device_, shader, nullptr);
    return result;
}

void VulkanContext::dispatch(
    const VulkanPipeline& pipeline,
    const std::vector<const VulkanBuffer*>& buffers,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z,
    const VulkanSemaphore* wait) {
    std::vector<VulkanDispatchResource> resources;
    resources.reserve(buffers.size());
    for (const VulkanBuffer* buffer : buffers) {
        resources.push_back({buffer, nullptr});
    }
    dispatch_resources(
        pipeline,
        resources,
        push_constants,
        push_constant_bytes,
        group_x,
        group_y,
        group_z,
        wait);
}

void VulkanContext::dispatch_image_to_buffer(
    const VulkanPipeline& pipeline,
    const VulkanImage& image,
    VulkanBuffer& buffer,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z) {
    dispatch_resources(
        pipeline,
        {{nullptr, &image}, {&buffer, nullptr}},
        push_constants,
        push_constant_bytes,
        group_x,
        group_y,
        group_z,
        nullptr);
}

void VulkanContext::dispatch_buffer_to_image(
    const VulkanPipeline& pipeline,
    const VulkanBuffer& buffer,
    VulkanImage& image,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z) {
    dispatch_resources(
        pipeline,
        {{nullptr, &image}, {&buffer, nullptr}},
        push_constants,
        push_constant_bytes,
        group_x,
        group_y,
        group_z,
        nullptr);
}

void VulkanContext::dispatch_buffers_to_image(
    const VulkanPipeline& pipeline,
    const std::vector<const VulkanBuffer*>& buffers,
    VulkanImage& image,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z) {
    std::vector<VulkanDispatchResource> resources;
    resources.reserve(buffers.size() + 1u);
    resources.push_back({nullptr, &image});
    for (const VulkanBuffer* buffer : buffers)
        resources.push_back({buffer, nullptr});
    dispatch_resources(
        pipeline, resources, push_constants, push_constant_bytes,
        group_x, group_y, group_z, nullptr);
}

void VulkanContext::dispatch_resources(
    const VulkanPipeline& pipeline,
    const std::vector<VulkanDispatchResource>& resources,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z,
    const VulkanSemaphore* wait) {
    if (pipeline.owner_ != this ||
        resources.size() != pipeline.descriptor_types_.size() ||
        push_constant_bytes != pipeline.push_constant_bytes_ ||
        (push_constant_bytes && push_constants == nullptr) ||
        group_x == 0 || group_y == 0 || group_z == 0) {
        throw std::invalid_argument("invalid Vulkan dispatch");
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (!pipeline.cached_descriptor_sets_.empty()) {
        descriptor_set = pipeline.cached_descriptor_sets_.back();
        pipeline.cached_descriptor_sets_.pop_back();
    } else {
        const VkDescriptorSetAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool_,
            1,
            &pipeline.descriptor_layout_,
        };
        check(
            vkAllocateDescriptorSets(
                device_, &allocate_info, &descriptor_set),
            "vkAllocateDescriptorSets");
    }
    std::vector<VkDescriptorBufferInfo> buffer_info(resources.size());
    std::vector<VkDescriptorImageInfo> image_info(resources.size());
    std::vector<VkWriteDescriptorSet> writes(resources.size());
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const VkDescriptorType type =
            pipeline.descriptor_types_[index];
        const bool image_descriptor =
            type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
            type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (image_descriptor) {
            const VulkanImage* image = resources[index].image;
            if (image == nullptr || image->owner_ != this ||
                image->view_ == VK_NULL_HANDLE ||
                (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                 image->sampler_ == VK_NULL_HANDLE)) {
                pipeline.cached_descriptor_sets_.push_back(
                    descriptor_set);
                throw std::invalid_argument(
                    "foreign or invalid Vulkan image");
            }
            image_info[index] = {
                type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    ? image->sampler_
                    : VK_NULL_HANDLE,
                image->view_,
                type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            writes[index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                static_cast<std::uint32_t>(index),
                0,
                1,
                type,
                &image_info[index],
                nullptr,
                nullptr,
            };
            continue;
        }
        const VulkanBuffer* buffer = resources[index].buffer;
        if (buffer == nullptr || buffer->owner_ != this) {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
            throw std::invalid_argument("foreign Vulkan buffer");
        }
        buffer_info[index] = {
            buffer->buffer_,
            0,
            buffer->size_,
        };
        writes[index] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            static_cast<std::uint32_t>(index),
            0,
            1,
            type,
            nullptr,
            &buffer_info[index],
            nullptr,
        };
    }
    vkUpdateDescriptorSets(
        device_,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);

    const bool batched =
        batch_command_ != VK_NULL_HANDLE || segmented_batch_;
    if (batched && wait != nullptr) {
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
        throw std::invalid_argument(
            "external wait is not supported inside a Vulkan batch");
    }
    VkCommandBuffer command =
        batched ? active_batch_command() : begin_commands();
    const bool profile =
        !batched && profile_query_pool_ != VK_NULL_HANDLE;
    if (profile) {
        vkCmdResetQueryPool(
            command, profile_query_pool_, 0, 2);
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            profile_query_pool_,
            0);
    }
    if (batched && batch_has_dispatch_ &&
        !track_resource_hazards_) {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            nullptr,
            0,
            nullptr);
    } else if (batched) {
        std::unordered_map<VkBuffer, VkAccessFlags>
            current_buffers;
        std::unordered_map<VkImage, VkAccessFlags>
            current_images;
        std::unordered_map<VkImage, VkImageLayout> image_layouts;
        for (std::size_t index = 0;
             index < resources.size();
             ++index) {
            const VkAccessFlags access =
                pipeline.descriptor_access_[index];
            if (resources[index].buffer != nullptr) {
                current_buffers[
                    resources[index].buffer->buffer_] |= access;
            } else {
                const VkImage image =
                    resources[index].image->image_;
                current_images[image] |= access;
                image_layouts[image] =
                    pipeline.descriptor_types_[index] ==
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        std::vector<VkBufferMemoryBarrier> buffer_barriers;
        std::vector<VkImageMemoryBarrier> image_barriers;
        for (const auto& [buffer, access] : current_buffers) {
            const auto previous =
                batch_buffer_access_.find(buffer);
            if (previous != batch_buffer_access_.end() &&
                ((previous->second | access) &
                 VK_ACCESS_SHADER_WRITE_BIT) != 0) {
                buffer_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        buffer,
                        0,
                        VK_WHOLE_SIZE,
                    });
            }
            batch_buffer_access_[buffer] = access;
        }
        for (const auto& [image, access] : current_images) {
            const auto previous =
                batch_image_access_.find(image);
            const VkImageLayout layout = image_layouts[image];
            if (previous != batch_image_access_.end()) {
                const VkImageLayout previous_layout =
                    batch_image_layout_.at(image);
                if (((previous->second | access) &
                     VK_ACCESS_SHADER_WRITE_BIT) == 0 &&
                    previous_layout == layout) {
                    batch_image_access_[image] = access;
                    continue;
                }
                image_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        previous_layout,
                        layout,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        image,
                        {
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            0,
                            1,
                            0,
                            1,
                        },
                    });
            }
            batch_image_access_[image] = access;
            batch_image_layout_[image] = layout;
        }
        if (!buffer_barriers.empty() ||
            !image_barriers.empty()) {
            vkCmdPipelineBarrier(
                command,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                static_cast<std::uint32_t>(
                    buffer_barriers.size()),
                buffer_barriers.data(),
                static_cast<std::uint32_t>(
                    image_barriers.size()),
                image_barriers.data());
        }
    }
    vkCmdBindPipeline(
        command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
    vkCmdBindDescriptorSets(
        command,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout_,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    if (push_constant_bytes) {
        vkCmdPushConstants(
            command,
            pipeline.layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            push_constant_bytes,
            push_constants);
    }
    vkCmdDispatch(command, group_x, group_y, group_z);
    if (segmented_batch_) {
        const VkMemoryBarrier release{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &release, 0, nullptr, 0, nullptr);
        segment_has_commands_ = true;
    }
    if (profile) {
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            profile_query_pool_,
            1);
    }
    if (batched) {
        batch_has_dispatch_ = true;
        batch_descriptor_sets_.push_back(
            {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
        if (segmented_batch_) finish_segment();
    } else {
        end_commands(command, wait);
        if (profile) {
            std::uint64_t timestamps[2]{};
            check(
                vkGetQueryPoolResults(
                    device_,
                    profile_query_pool_,
                    0,
                    2,
                    sizeof(timestamps),
                    timestamps,
                    sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT |
                        VK_QUERY_RESULT_WAIT_BIT),
                "vkGetQueryPoolResults");
            record_profile(
                pipeline, timestamps[1] - timestamps[0]);
        }
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
    }
}

void VulkanContext::destroy(VulkanBuffer& buffer) noexcept {
    if (batch_command_ != VK_NULL_HANDLE && !segmented_batch_ &&
        (buffer.buffer_ != VK_NULL_HANDLE ||
         buffer.memory_ != VK_NULL_HANDLE)) {
        batch_deferred_buffers_.push_back(
            {
                buffer.buffer_,
                buffer.memory_,
                buffer.mapped_,
                buffer.size_,
                buffer.cacheable_,
            });
        buffer.owner_ = nullptr;
        buffer.buffer_ = VK_NULL_HANDLE;
        buffer.memory_ = VK_NULL_HANDLE;
        buffer.mapped_ = nullptr;
        buffer.size_ = 0;
        buffer.cacheable_ = false;
        return;
    }
    recycle_or_destroy(
        {
            buffer.buffer_,
            buffer.memory_,
            buffer.mapped_,
            buffer.size_,
            buffer.cacheable_,
        });
    buffer.owner_ = nullptr;
    buffer.buffer_ = VK_NULL_HANDLE;
    buffer.memory_ = VK_NULL_HANDLE;
    buffer.mapped_ = nullptr;
    buffer.size_ = 0;
    buffer.cacheable_ = false;
}

void VulkanContext::destroy(VulkanImage& image) noexcept {
    if (image.sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, image.sampler_, nullptr);
    }
    if (image.view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, image.view_, nullptr);
    }
    if (image.image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_, image.image_, nullptr);
    }
    if (image.memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, image.memory_, nullptr);
    }
    image.owner_ = nullptr;
    image.image_ = VK_NULL_HANDLE;
    image.memory_ = VK_NULL_HANDLE;
    image.view_ = VK_NULL_HANDLE;
    image.sampler_ = VK_NULL_HANDLE;
    image.format_ = VK_FORMAT_UNDEFINED;
    image.width_ = 0;
    image.height_ = 0;
}

void VulkanContext::destroy(VulkanSubmission& submission) noexcept {
    if (submission.fence_ != VK_NULL_HANDLE) {
        (void)vkWaitForFences(
            device_, 1, &submission.fence_, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_, submission.fence_, nullptr);
    }
    if (submission.command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(
            device_, command_pool_, 1, &submission.command_);
    }
    release_submission_resources(submission);
    submission.owner_ = nullptr;
    submission.command_ = VK_NULL_HANDLE;
    submission.fence_ = VK_NULL_HANDLE;
}

void VulkanContext::release_submission_resources(
    VulkanSubmission& submission) noexcept {
    if (submission.resources_ == nullptr) return;
    if (!submission.resources_->commands.empty()) {
        vkFreeCommandBuffers(
            device_, command_pool_,
            static_cast<std::uint32_t>(
                submission.resources_->commands.size()),
            submission.resources_->commands.data());
        submission.resources_->commands.clear();
    }
    for (const VulkanBatchedDescriptor& descriptor :
         submission.resources_->descriptor_sets) {
        if (descriptor.pipeline && descriptor.set) {
            descriptor.pipeline->cached_descriptor_sets_.push_back(
                descriptor.set);
        }
    }
    for (const VulkanDeferredBuffer& buffer :
         submission.resources_->deferred_buffers) {
        recycle_or_destroy(buffer);
    }
    delete submission.resources_;
    submission.resources_ = nullptr;
}

void VulkanContext::destroy(VulkanSemaphore& semaphore) noexcept {
    if (semaphore.semaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(
            device_, semaphore.semaphore_, nullptr);
    }
    semaphore.owner_ = nullptr;
    semaphore.semaphore_ = VK_NULL_HANDLE;
    semaphore.value_ = 0;
}

void VulkanContext::recycle_or_destroy(
    VulkanDeferredBuffer buffer) noexcept {
    constexpr VkDeviceSize maximum_device_pool_bytes =
        VkDeviceSize{2} * 1024 * 1024 * 1024;
    constexpr VkDeviceSize maximum_host_pool_bytes =
        VkDeviceSize{64} * 1024 * 1024;
    if (buffer.cacheable && buffer.buffer && buffer.memory) {
        if (buffer.mapped &&
            buffer.size <= maximum_host_pool_bytes - pooled_host_bytes_) {
            pooled_host_bytes_ += buffer.size;
            host_buffer_pool_.push_back(buffer);
            return;
        }
        if (!buffer.mapped &&
            buffer.size <=
                maximum_device_pool_bytes - pooled_device_bytes_) {
            pooled_device_bytes_ += buffer.size;
            device_buffer_pool_.push_back(buffer);
            return;
        }
    }
    if (buffer.mapped) {
        vkUnmapMemory(device_, buffer.memory);
    }
    if (buffer.buffer) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
    }
    if (buffer.memory) {
        vkFreeMemory(device_, buffer.memory, nullptr);
    }
}

void VulkanContext::discard(VulkanBuffer& buffer) noexcept {
    if (buffer.owner_ != this || buffer.buffer_ == VK_NULL_HANDLE) {
        return;
    }
    if (batch_command_ != VK_NULL_HANDLE) {
        destroy(buffer);
        return;
    }
    if (buffer.mapped_) {
        vkUnmapMemory(device_, buffer.memory_);
    }
    vkDestroyBuffer(device_, buffer.buffer_, nullptr);
    vkFreeMemory(device_, buffer.memory_, nullptr);
    buffer.owner_ = nullptr;
    buffer.buffer_ = VK_NULL_HANDLE;
    buffer.memory_ = VK_NULL_HANDLE;
    buffer.size_ = 0;
    buffer.mapped_ = nullptr;
    buffer.cacheable_ = false;
}

void VulkanContext::destroy(VulkanPipeline& pipeline) noexcept {
    if (!pipeline.cached_descriptor_sets_.empty()) {
        vkFreeDescriptorSets(
            device_,
            descriptor_pool_,
            static_cast<std::uint32_t>(
                pipeline.cached_descriptor_sets_.size()),
            pipeline.cached_descriptor_sets_.data());
        pipeline.cached_descriptor_sets_.clear();
    }
    if (pipeline.pipeline_) {
        vkDestroyPipeline(device_, pipeline.pipeline_, nullptr);
    }
    if (pipeline.layout_) {
        vkDestroyPipelineLayout(device_, pipeline.layout_, nullptr);
    }
    if (pipeline.descriptor_layout_) {
        vkDestroyDescriptorSetLayout(
            device_, pipeline.descriptor_layout_, nullptr);
    }
    pipeline.owner_ = nullptr;
    pipeline.pipeline_ = VK_NULL_HANDLE;
    pipeline.layout_ = VK_NULL_HANDLE;
    pipeline.descriptor_layout_ = VK_NULL_HANDLE;
    pipeline.descriptor_types_.clear();
}

}  // namespace da3_native
