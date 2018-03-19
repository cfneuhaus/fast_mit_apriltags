#ifndef APRILTAG_THREADPOOL_H
#define APRILTAG_THREADPOOL_H

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace AprilTags
{

class ThreadPool;

class ThreadPool
{
public:
    ThreadPool(size_t threads
        = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 2)
        : stop(false)
    {
        numPushedTasks = 0;
        numFinishedTasks = 0;


        for (size_t i = 0; i < threads; ++i)
            workers.push_back(std::thread([this]() {
                std::function<void()> task;
                while (true)
                {
                    { // acquire lock
                        std::unique_lock<std::mutex> lock(queue_mutex);

                        // look for a work item
                        while (!stop && tasks.empty())
                        { // if there are none wait for notification
                            condition.wait(lock);
                        }

                        if (stop) // exit if the pool is stopped
                            return;

                        // get the task from the queue
                        task = tasks.front();
                        tasks.pop_front();

                    } // release lock

                    // execute the task
                    task();

                    {
                        std::unique_lock<std::mutex> lock(taskCountMutex);
                        numFinishedTasks++;
                    }
                    finishcondition.notify_one();
                }
            }));
    }

    // the destructor joins all threads
    ~ThreadPool()
    {
        // stop all threads
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();

        // join them
        for (size_t i = 0; i < workers.size(); ++i)
            workers[i].join();
    }
    template <class F> void pushTask(F f)
    {
        numPushedTasks++;
        { // acquire lock
            std::unique_lock<std::mutex> lock(queue_mutex);

            // add the task
            tasks.push_back(std::function<void()>(f));
        } // release lock

        // wake up one thread
        condition.notify_one();
    }

    void joinAllTasks()
    {
        std::unique_lock<std::mutex> lock(taskCountMutex);

        while (numFinishedTasks != numPushedTasks)
        {
            finishcondition.wait(lock);
        }
    }

    size_t getNumThreads() const { return workers.size(); }

private:
    friend class Worker;

    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;

    // the task queue
    std::deque<std::function<void()>> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    std::condition_variable finishcondition;
    std::mutex taskCountMutex;
    int numPushedTasks;
    int numFinishedTasks;
};
}


#endif
