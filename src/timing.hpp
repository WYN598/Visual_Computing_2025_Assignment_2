#pragma once
#include <chrono>
#include <deque>
#include <fstream>
#include <string>

class FpsAverager {
public:
    explicit FpsAverager(size_t window = 120)
        : window_(window), last_(Clock::now()) {
    }

    // 返回当前滑动平均 FPS
    double tick() {
        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        if (dt > 0.000001) {
            double fps = 1.0 / dt;
            dq_.push_back(fps);
            if (dq_.size() > window_) dq_.pop_front();
        }
        // 计算平均
        double sum = 0;
        for (auto v : dq_) sum += v;
        return dq_.empty() ? 0.0 : sum / dq_.size();
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    size_t window_;
    Clock::time_point last_;
    std::deque<double> dq_;
};

class CsvLogger {
public:
    CsvLogger(const std::string& path) {
        file_.open(path, std::ios::out);
        if (file_.is_open()) {
            file_ << "time,mode,filter,resolution,transform,build,fps\n";
        }
    }
    ~CsvLogger() { if (file_.is_open()) file_.close(); }

    void log(const std::string& mode, const std::string& filter,
        const std::string& res, const std::string& transform,
        const std::string& build, double fps) {
        if (!file_.is_open()) return;
        double t = nowSeconds();
        file_ << t << "," << mode << "," << filter << "," << res << ","
            << transform << "," << build << "," << fps << "\n";
    }

private:
    double nowSeconds() const {
        using Clock = std::chrono::steady_clock;
        static auto t0 = Clock::now();
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
    std::ofstream file_;
};
