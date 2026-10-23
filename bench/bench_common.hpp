#pragma once
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <utility>

namespace bench
{
    inline std::pair<std::string, std::string> stamp()
    {
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        char human[32], fileTag[32];
        std::strftime(human, sizeof human, "%Y-%m-%d %H:%M:%S", &tm);
        std::strftime(fileTag, sizeof fileTag, "%Y%m%d_%H%M%S", &tm);
        return {human, fileTag};
    }

    inline void emit(const std::string& line, bool save, const std::string& fileTag)
    {
        std::fputs(line.c_str(), stdout);
        std::fflush(stdout);
        if (save)
        {
            std::ofstream out("bench/results/" + fileTag + ".jsonl", std::ios::app);
            out << line;
        }
    }
}
