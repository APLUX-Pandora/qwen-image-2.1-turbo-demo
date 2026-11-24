// SPDX-License-Identifier: Apache-2.0
// Ordinary C++ client of the existing C ABI; no new model SDK.
#include "qwen_image.h"
#include <atomic>
#include <csignal>
#include <pthread.h>
#include <thread>
#include <mutex>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

class StopControl {
    sigset_t signals_{}, previous_{};
    std::atomic<bool> done_{false};
    std::atomic<int> first_{0};
    std::mutex mutex_;
    qwi_engine* engine_ = nullptr; // borrowed only while its ordinary owner lives
    std::thread receiver_;
public:
    StopControl() {
        sigemptyset(&signals_); sigaddset(&signals_, SIGTERM); sigaddset(&signals_, SIGINT);
        if (pthread_sigmask(SIG_BLOCK, &signals_, &previous_)) throw std::runtime_error("cannot block stop signals");
        try {
            receiver_ = std::thread([this] {
                while (!done_.load()) {
                    timespec timeout{0,100000000};
                    int sig = sigtimedwait(&signals_, nullptr, &timeout);
                    if (sig == SIGTERM || sig == SIGINT) {
                        int unset = 0; first_.compare_exchange_strong(unset, sig);
                        std::lock_guard<std::mutex> guard(mutex_);
                        if (engine_) qwi_cancel(engine_);
                    }
                }
            });
        } catch (...) { pthread_sigmask(SIG_SETMASK, &previous_, nullptr); throw; }
    }
    void publish(qwi_engine* engine) { std::lock_guard<std::mutex> guard(mutex_); engine_ = engine; }
    void retire() { publish(nullptr); }
    int received() const { return first_.load(); }
    ~StopControl() {
        retire(); done_.store(true); receiver_.join();
        pthread_sigmask(SIG_SETMASK, &previous_, nullptr);
    }
};
static void progress(qwi_phase, int, int, double, void* user) {
    auto* pair = static_cast<std::pair<StopControl*, qwi_engine*>*>(user);
    if (pair->first->received()) qwi_cancel(pair->second);
}
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: generate MODEL_ROOT OUTPUT.ppm\n"; return 2; }
    try {
        StopControl stop;
        int result = 0;
        try {
            // Retire the cancel borrow before the unique owner destroys the model.
            auto destroy = [&stop](qwi_engine* e) { stop.retire(); qwi_destroy(e); };
            std::unique_ptr<qwi_engine, decltype(destroy)> engine(qwi_create(argv[1], nullptr), destroy);
            if (!engine) throw std::runtime_error(qwi_create_error());
            stop.publish(engine.get());
            if (!stop.received()) {
                constexpr int width = 512, height = 512;
                std::vector<uint8_t> rgb(width * height * 3);
                qwi_request request{"A red fox in the snow, watercolor painting.", width, height, 42};
                std::pair<StopControl*,qwi_engine*> callback{&stop,engine.get()};
                int rc = qwi_generate(engine.get(), &request, progress, &callback, rgb.data(), rgb.size());
                if (rc != QWI_OK && !stop.received()) throw std::runtime_error(qwi_last_error(engine.get()));
                if (!stop.received()) {
                    std::ofstream out(argv[2], std::ios::binary);
                    out.exceptions(std::ios::failbit | std::ios::badbit);
                    out << "P6\n" << width << ' ' << height << "\n255\n";
                    out.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
                    out.close();
                }
            }
        } catch (const std::exception& e) { std::cerr << e.what() << '\n'; result = 1; }
        return stop.received() ? 128 + stop.received() : result;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
