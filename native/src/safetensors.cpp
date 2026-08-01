#include "safetensors.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace da3_native {
namespace {

struct ParsedTensor {
    std::string name;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

class Json {
public:
    Json(
        const char* begin,
        const char* end,
        std::vector<std::pair<std::string, std::string>>& aliases)
        : current_(begin), end_(end), aliases_(aliases) {}

    std::vector<ParsedTensor> parse_root() {
        std::vector<ParsedTensor> result;
        whitespace();
        expect('{');
        whitespace();
        if (consume('}')) {
            fail("safetensors header is empty");
        }
        for (;;) {
            const std::string name = string();
            whitespace();
            expect(':');
            whitespace();
            if (name == "__metadata__") {
                metadata();
            } else {
                result.push_back(tensor(name));
            }
            whitespace();
            if (consume('}')) {
                break;
            }
            expect(',');
            whitespace();
        }
        whitespace();
        if (current_ != end_) {
            fail("trailing safetensors JSON data");
        }
        return result;
    }

private:
    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error(message);
    }

    void whitespace() {
        while (current_ != end_ &&
               (*current_ == ' ' || *current_ == '\t' ||
                *current_ == '\r' || *current_ == '\n')) {
            ++current_;
        }
    }

    bool consume(char value) {
        if (current_ != end_ && *current_ == value) {
            ++current_;
            return true;
        }
        return false;
    }

    void expect(char value) {
        if (!consume(value)) {
            fail("invalid safetensors JSON punctuation");
        }
    }

    static unsigned hex(char value) {
        if (value >= '0' && value <= '9') {
            return static_cast<unsigned>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<unsigned>(value - 'a' + 10);
        }
        if (value >= 'A' && value <= 'F') {
            return static_cast<unsigned>(value - 'A' + 10);
        }
        throw std::runtime_error("invalid safetensors JSON escape");
    }

    void append_utf8(std::string& output, unsigned codepoint) {
        if (codepoint <= 0x7f) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(
                0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    std::string string() {
        whitespace();
        expect('"');
        std::string output;
        while (current_ != end_) {
            const unsigned char value =
                static_cast<unsigned char>(*current_++);
            if (value == '"') {
                return output;
            }
            if (value < 0x20) {
                fail("invalid safetensors JSON string");
            }
            if (value != '\\') {
                output.push_back(static_cast<char>(value));
                continue;
            }
            if (current_ == end_) {
                fail("truncated safetensors JSON escape");
            }
            const char escaped = *current_++;
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    if (end_ - current_ < 4) {
                        fail("truncated safetensors Unicode escape");
                    }
                    unsigned codepoint = 0;
                    for (int index = 0; index < 4; ++index) {
                        codepoint =
                            (codepoint << 4) | hex(*current_++);
                    }
                    if (codepoint >= 0xd800 && codepoint <= 0xdfff) {
                        fail("unsupported safetensors surrogate escape");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default:
                    fail("invalid safetensors JSON escape");
            }
        }
        fail("unterminated safetensors JSON string");
    }

    std::uint64_t number() {
        whitespace();
        if (current_ == end_ ||
            *current_ < '0' || *current_ > '9') {
            fail("invalid safetensors unsigned integer");
        }
        std::uint64_t value = 0;
        do {
            const unsigned digit =
                static_cast<unsigned>(*current_ - '0');
            if (value >
                (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
                fail("safetensors integer overflow");
            }
            value = value * 10 + digit;
            ++current_;
        } while (current_ != end_ &&
                 *current_ >= '0' && *current_ <= '9');
        return value;
    }

    void literal(const char* text) {
        const std::size_t length = std::strlen(text);
        if (static_cast<std::size_t>(end_ - current_) < length ||
            std::memcmp(current_, text, length) != 0) {
            fail("invalid safetensors JSON literal");
        }
        current_ += length;
    }

    void skip_array() {
        expect('[');
        whitespace();
        if (consume(']')) {
            return;
        }
        for (;;) {
            skip_value();
            whitespace();
            if (consume(']')) {
                return;
            }
            expect(',');
            whitespace();
        }
    }

    void skip_object() {
        expect('{');
        whitespace();
        if (consume('}')) {
            return;
        }
        for (;;) {
            (void)string();
            whitespace();
            expect(':');
            whitespace();
            skip_value();
            whitespace();
            if (consume('}')) {
                return;
            }
            expect(',');
            whitespace();
        }
    }

    void skip_value() {
        whitespace();
        if (current_ == end_) {
            fail("truncated safetensors JSON value");
        }
        if (*current_ == '{') {
            skip_object();
        } else if (*current_ == '[') {
            skip_array();
        } else if (*current_ == '"') {
            (void)string();
        } else if (*current_ >= '0' && *current_ <= '9') {
            (void)number();
        } else if (*current_ == 't') {
            literal("true");
        } else if (*current_ == 'f') {
            literal("false");
        } else if (*current_ == 'n') {
            literal("null");
        } else {
            fail("invalid safetensors JSON value");
        }
    }

    void metadata() {
        expect('{');
        whitespace();
        if (consume('}')) {
            return;
        }
        for (;;) {
            std::string alias = string();
            whitespace();
            expect(':');
            whitespace();
            std::string target = string();
            aliases_.emplace_back(
                std::move(alias), std::move(target));
            whitespace();
            if (consume('}')) {
                return;
            }
            expect(',');
            whitespace();
        }
    }

    ParsedTensor tensor(const std::string& name) {
        ParsedTensor result;
        result.name = name;
        bool have_dtype = false;
        bool have_shape = false;
        bool have_offsets = false;
        expect('{');
        whitespace();
        for (;;) {
            const std::string field = string();
            whitespace();
            expect(':');
            whitespace();
            if (field == "dtype") {
                if (have_dtype || string() != "F32") {
                    fail("DA3 requires unique F32 tensors");
                }
                have_dtype = true;
            } else if (field == "shape") {
                if (have_shape) {
                    fail("duplicate safetensors shape");
                }
                expect('[');
                whitespace();
                if (consume(']')) {
                    fail("DA3 scalar tensors are unsupported");
                }
                for (;;) {
                    if (result.rank >= result.dimensions.size()) {
                        fail("DA3 tensor rank exceeds four");
                    }
                    result.dimensions[result.rank++] = number();
                    whitespace();
                    if (consume(']')) {
                        break;
                    }
                    expect(',');
                    whitespace();
                }
                have_shape = true;
            } else if (field == "data_offsets") {
                if (have_offsets) {
                    fail("duplicate safetensors offsets");
                }
                expect('[');
                result.begin = number();
                whitespace();
                expect(',');
                result.end = number();
                whitespace();
                expect(']');
                have_offsets = true;
            } else {
                fail("unknown safetensors tensor field");
            }
            whitespace();
            if (consume('}')) {
                break;
            }
            expect(',');
            whitespace();
        }
        if (!have_dtype || !have_shape || !have_offsets) {
            fail("incomplete safetensors tensor record");
        }
        return result;
    }

    const char* current_;
    const char* end_;
    std::vector<std::pair<std::string, std::string>>& aliases_;
};

#if defined(_WIN32)
std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("model path is empty");
    }
    const int length = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), nullptr, 0);
    if (length <= 0) {
        throw std::invalid_argument("model path is not valid UTF-8");
    }
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
            static_cast<int>(text.size()), result.data(), length) != length) {
        throw std::runtime_error("failed to decode model path");
    }
    return result;
}
#endif

std::uint64_t read_u64(const std::byte* data) {
    std::uint64_t result = 0;
    for (unsigned index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(data[index])) << (index * 8);
    }
    return result;
}

}  // namespace

SafeTensors::SafeTensors(const std::string& path_utf8) {
    try {
#if defined(_WIN32)
        const std::wstring path = utf8_to_wide(path_utf8);
        file_ = CreateFileW(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("failed to open safetensors model");
        }
        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file_, &file_size) || file_size.QuadPart < 0) {
            throw std::runtime_error("failed to query safetensors size");
        }
        size_ = static_cast<std::uint64_t>(file_size.QuadPart);
        mapping_ = CreateFileMappingW(
            file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            throw std::runtime_error("failed to map safetensors model");
        }
        view_ = static_cast<const std::byte*>(
            MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!view_) {
            throw std::runtime_error("failed to view safetensors model");
        }
#else
        if (path_utf8.empty()) {
            throw std::invalid_argument("model path is empty");
        }
        file_descriptor_ = open(path_utf8.c_str(), O_RDONLY | O_CLOEXEC);
        if (file_descriptor_ < 0) {
            throw std::runtime_error("failed to open safetensors model");
        }
        struct stat status {};
        if (fstat(file_descriptor_, &status) != 0 || status.st_size < 0) {
            throw std::runtime_error("failed to query safetensors size");
        }
        size_ = static_cast<std::uint64_t>(status.st_size);
        if (size_ > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("safetensors model is too large");
        }
        void* mapped = mmap(
            nullptr, static_cast<std::size_t>(size_), PROT_READ,
            MAP_PRIVATE, file_descriptor_, 0);
        if (mapped == MAP_FAILED) {
            throw std::runtime_error("failed to map safetensors model");
        }
        view_ = static_cast<const std::byte*>(mapped);
#endif
        if (size_ < 10) {
            throw std::runtime_error("safetensors model is truncated");
        }
        const std::uint64_t header_bytes = read_u64(view_);
        if (header_bytes < 2 || header_bytes > 100 * 1024 * 1024 ||
            header_bytes > size_ - 8) {
            throw std::runtime_error("invalid safetensors header length");
        }
        const std::uint64_t data_offset = 8 + header_bytes;
        const char* json_begin =
            reinterpret_cast<const char*>(view_ + 8);
        std::vector<std::pair<std::string, std::string>> aliases;
        std::vector<ParsedTensor> parsed = Json(
            json_begin, json_begin + header_bytes, aliases).parse_root();
        if (parsed.empty() || parsed.size() > 4096) {
            throw std::runtime_error("invalid safetensors tensor count");
        }
        std::sort(
            parsed.begin(), parsed.end(),
            [](const ParsedTensor& left, const ParsedTensor& right) {
                return left.begin < right.begin;
            });
        std::uint64_t cursor = 0;
        names_.reserve(parsed.size());
        tensors_.reserve(parsed.size());
        for (const ParsedTensor& item : parsed) {
            if (item.name.empty() || item.begin != cursor ||
                item.end < item.begin ||
                item.end > size_ - data_offset) {
                throw std::runtime_error(
                    "invalid or non-contiguous safetensors offsets");
            }
            std::uint64_t elements = 1;
            for (std::uint32_t index = 0; index < item.rank; ++index) {
                const std::uint64_t dimension = item.dimensions[index];
                if (dimension == 0 ||
                    elements >
                        std::numeric_limits<std::uint64_t>::max() /
                            dimension) {
                    throw std::runtime_error(
                        "invalid safetensors tensor dimensions");
                }
                elements *= dimension;
            }
            if (elements >
                    std::numeric_limits<std::uint64_t>::max() / 4 ||
                elements * 4 != item.end - item.begin) {
                throw std::runtime_error(
                    "invalid safetensors tensor byte count");
            }
            TensorView view{
                reinterpret_cast<const float*>(
                    this->view_ + data_offset + item.begin),
                item.dimensions,
                item.rank,
                elements,
            };
            if (!tensors_.emplace(item.name, view).second) {
                throw std::runtime_error(
                    "duplicate safetensors tensor name");
            }
            names_.push_back(item.name);
            cursor = item.end;
        }
        if (cursor != size_ - data_offset) {
            throw std::runtime_error("unclaimed safetensors payload bytes");
        }
        aliases_.reserve(aliases.size());
        for (auto& alias : aliases) {
            // Safetensors metadata is an arbitrary string map.  The upstream
            // DA3 checkpoints additionally use entries whose values name a
            // tensor as shared-tensor aliases.  Preserve those aliases while
            // ignoring ordinary provenance metadata.
            if (tensors_.find(alias.second) == tensors_.end()) {
                continue;
            }
            if (alias.first.empty() ||
                tensors_.find(alias.first) != tensors_.end() ||
                !aliases_.emplace(
                    std::move(alias.first),
                    std::move(alias.second)).second) {
                throw std::runtime_error(
                    "invalid safetensors shared-tensor alias");
            }
        }
    } catch (...) {
        close();
        throw;
    }
}

SafeTensors::~SafeTensors() {
    close();
}

void SafeTensors::close() noexcept {
    tensors_.clear();
    aliases_.clear();
    names_.clear();
#if defined(_WIN32)
    if (view_) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (mapping_) {
        CloseHandle(mapping_);
        mapping_ = nullptr;
    }
    if (file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
    }
#else
    if (view_) {
        munmap(
            const_cast<std::byte*>(view_),
            static_cast<std::size_t>(size_));
        view_ = nullptr;
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
        file_descriptor_ = -1;
    }
#endif
    size_ = 0;
}

const TensorView& SafeTensors::tensor(std::string_view name) const {
    const std::string key(name);
    auto found = tensors_.find(key);
    if (found == tensors_.end()) {
        const auto alias = aliases_.find(key);
        if (alias != aliases_.end()) {
            found = tensors_.find(alias->second);
        }
    }
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "model is missing tensor: " + std::string(name));
    }
    return found->second;
}

bool SafeTensors::contains(std::string_view name) const {
    const std::string key(name);
    return tensors_.find(key) != tensors_.end() ||
        aliases_.find(key) != aliases_.end();
}

}  // namespace da3_native
