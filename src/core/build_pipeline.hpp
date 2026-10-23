#pragma once

#include <atomic>                    
#include <condition_variable>        
#include <cstddef>                     
#include <cstdint>                     
#include <exception>                   
#include <mutex>                       
#include <vector>                      

#include "core/hnsw_index.hpp"       

namespace vortex
{

    // 计划书：行号 + 每层搜索候选
    struct PlannedRow
    {
        std::size_t row;               
        InsertPlan plan;               // 邻居候选
    };

    // 起线程闸机：全部线程到齐才放行
    class Gate
    {
      public:
        bool arrive();                 // 工人用
        void open(std::size_t total);  // 主线程：等齐 total 个，一起放行
        void abortNow();               // 主线程召回

      private:
        std::mutex mu_;                
        std::condition_variable cv_;   
        std::size_t ready_ = 0;        // 已报到人数
        bool go_ = false;              // 开工令
        bool abort_ = false;           // 解散令
    };

  
    class WorkQueue
    {
      public:
        bool takeTicket(std::size_t kWindow, std::size_t rows, std::size_t& out);
        void pushPlan(PlannedRow pr); 
        bool waitBatch(std::size_t kCommitBatch, std::size_t rows, std::vector<PlannedRow>& out);
        void commitDone(std::size_t n);   
        void recordError();          
        bool failed() const;           
        void wakeAll();                
        std::exception_ptr firstError();  
      private:
        std::mutex mu_;                   
        std::condition_variable cvPlan_;  // 叫醒落笔人
        std::condition_variable cvWindow_;  // 叫工人
        std::vector<PlannedRow> queue_;   // 攒着的计划书
        std::size_t ticketed_ = 0;        // 发出去的工票数
        std::size_t returned_ = 0;        // 交回来的计划书数
        std::size_t committed_ = 0;       // 真正写进图的行数
        std::size_t nextRow_ = 0;         // 下一张票对应的行号
        std::atomic<bool> failed_{false}; // 叫停开关
        std::exception_ptr firstError_;   // 第一个出事的异常
    };

}
