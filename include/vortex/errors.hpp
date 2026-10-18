#pragma once

#include <stdexcept>
#include <string>

namespace vortex
{

    class VortexError : public std::runtime_error
    {
        public:
        using std::runtime_error::runtime_error;
    };

    // 参数/前置条件非法。
    class ValueError : public VortexError
    {
        public:
        explicit ValueError(const std::string& msg)
            : VortexError("ValueError: " + msg)
        {
        }
    };

    // 容量已满，插入被拒绝。
    class CapacityError : public VortexError
    {
        public:
        explicit CapacityError(const std::string& msg)
            : VortexError("CapacityError: " + msg)
        {
        }
    };

    // 读写持久化文件失败。
    class IOError : public VortexError
    {
        public:
        explicit IOError(const std::string& msg)
            : VortexError("IOError: " + msg)
        {
        }
    };

    // 文件存在但内容损坏/版本不符。
    class CorruptFileError : public VortexError
    {
        public:
        explicit CorruptFileError(const std::string& msg)
            : VortexError("CorruptFileError: " + msg)
        {
        }
    };

    // 未知
    class NotImplementedError : public VortexError
    {
        public:
        explicit NotImplementedError(const std::string& msg)
            : VortexError("NotImplementedError: " + msg)
        {
        }
    };
}
