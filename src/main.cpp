#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "gl_utils.hpp"
#include "gpu_pipeline.hpp"
#include "cv_filters.hpp"
#include "cv_geom.hpp"
#include "timing.hpp"

// -------------------- Synthetic Frame Generator (for benchmarking instead of webcam) --------------------
// Generate a random w×h BGR 8UC3 image; each frame varies slightly to avoid cache optimization
static cv::Mat generateSyntheticFrame(int w, int h, unsigned seedTick) {
    cv::Mat img(h, w, CV_8UC3);
    cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
    // Add a simple shape/gradient overlay to prevent overly ideal randomness that helps GPU caching too much
    int cx = (seedTick * 37) % w;
    int cy = (seedTick * 53) % h;
    cv::circle(img, { cx, cy }, std::max(8, std::min(w, h) / 12), cv::Scalar(20, 20, 220), -1);
    cv::putText(img, std::to_string(seedTick % 10000), { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 240,240,240 }, 2);
    return img;
}

// -------------------- Result Recording --------------------
struct BenchResultRow {
    std::string mode, filter, transform, resolution, build;
    double avg_fps, min_fps, max_fps, std_fps;
    int samples;
};

static void write_summary_csv(const std::vector<BenchResultRow>& rows, const std::string& path) {
    std::ofstream f(path, std::ios::out);
    f << "mode,filter,transform,resolution,build,avg_fps,min_fps,max_fps,std_fps,samples\n";
    for (const auto& r : rows) {
        f << r.mode << "," << r.filter << "," << r.transform << ","
            << r.resolution << "," << r.build << ","
            << r.avg_fps << "," << r.min_fps << "," << r.max_fps << ","
            << r.std_fps << "," << r.samples << "\n";
    }
}

// Simple statistics
static double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
static double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double acc = 0.0; for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / (v.size() - 1));
}

// -------------------- Run One Combination --------------------
static BenchResultRow run_one_combo(GLFWwindow* win,
    GpuPipeline& gpu,
    GLuint passProg, GLint loc_uTex, GLint loc_uAff,
    GLuint vao, GLuint& tex, int& texW, int& texH,
    const std::pair<int, int>& reqRes,
    const std::string& build,
    FilterType filter, bool useGPU, bool useTransform,
    const AffineParams& aff,
    int warmup_sec = 1, int sample_sec = 5)
{
    // 1) Change resolution: recreate texture and resize window
    texW = reqRes.first; texH = reqRes.second;
    glDeleteTextures(1, &tex);
    tex = glutils::createTexture2D(texW, texH, GL_RGB);
    glfwSetWindowSize(win, texW, texH);

    FilterParams fp; fp.pixelBlock = 8; fp.keepBGR = { 20,20,200 }; fp.thresh = 60;
    AffineParams ap = aff;

    std::vector<double> fps_samples;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto elapsed_sec = [&] { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); };

    // 2) Rendering loop: use synthetic frames, not limited by camera FPS
    double last = glfwGetTime();
    unsigned tick = 0;

    while (!glfwWindowShouldClose(win)) {
        // Generate input frame
        cv::Mat frame = generateSyntheticFrame(texW, texH, ++tick);

        // CPU / GPU processing paths
        if (!useGPU) {
            cv::Mat img = frame;
            if (useTransform) warpCpuAffine(img, ap);
            applyCpuFilter(img, filter, fp);
            glutils::uploadFrameToTexture(tex, img);
        }
        else {
            glutils::uploadFrameToTexture(tex, frame);
        }

        // Render
        int fbW, fbH; glfwGetFramebufferSize(win, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClear(GL_COLOR_BUFFER_BIT);

        if (useGPU) {
            gpu.draw(vao, tex, texW, texH, filter, fp, useTransform ? ap : AffineParams{});
        }
        else {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            glUseProgram(passProg);
            if (loc_uTex >= 0) glUniform1i(loc_uTex, 0);

            cv::Matx33f M = affineMatrix(useTransform ? ap : AffineParams{}, texW, texH);
            glm::mat3 gM(
                M(0, 0), M(1, 0), M(2, 0),
                M(0, 1), M(1, 1), M(2, 1),
                M(0, 2), M(1, 2), M(2, 2)
            );
            if (loc_uAff >= 0) glUniformMatrix3fv(loc_uAff, 1, GL_FALSE, glm::value_ptr(gM));

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindVertexArray(0);
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glfwSwapBuffers(win);
        glfwPollEvents();

        // Measure FPS (frame-to-frame interval)
        double now = glfwGetTime();
        double dt = now - last; last = now;
        if (dt > 0.0) {
            double fps = 1.0 / dt;
            if (elapsed_sec() > warmup_sec) fps_samples.push_back(fps);
        }

        if (elapsed_sec() > warmup_sec + sample_sec) break;
    }

    // Collect results
    BenchResultRow row;
    row.mode = useGPU ? "GPU" : "CPU";
    row.filter = filterName(filter);
    row.transform = useTransform ? "On" : "Off";
    row.resolution = std::to_string(texW) + "x" + std::to_string(texH);
    row.build = build;
    row.avg_fps = mean(fps_samples);
    row.min_fps = fps_samples.empty() ? 0.0 : *std::min_element(fps_samples.begin(), fps_samples.end());
    row.max_fps = fps_samples.empty() ? 0.0 : *std::max_element(fps_samples.begin(), fps_samples.end());
    row.std_fps = stdev(fps_samples);
    row.samples = (int)fps_samples.size();
    return row;
}

// -------------------- Automatic Benchmark Pipeline --------------------
static int run_benchmark_mode() {
    // Initialize OpenGL window
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return -1; }
    // Start with an initial window; size will be adjusted later for each test
    GLFWwindow* win = glfwCreateWindow(640, 480, "Synthetic Benchmark", nullptr, nullptr);
    if (!win) { std::cerr << "Create window failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    // Disable VSync to avoid frame rate lock by display refresh rate
    glfwSwapInterval(0);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "glad init failed\n"; return -1;
    }

    // Resources
    GLuint vao = glutils::createFullScreenQuadVAO();
    GpuPipeline gpu; if (!gpu.init("shaders")) { std::cerr << "Shader init failed.\n"; return -1; }
    GLuint passProg = glutils::loadShaderProgram("shaders/passthrough.vert", "shaders/passthrough.frag");
    GLint loc_uTex = glGetUniformLocation(passProg, "uTex");
    GLint loc_uAff = glGetUniformLocation(passProg, "uAffine");

    int texW = 640, texH = 480;
    GLuint tex = glutils::createTexture2D(texW, texH, GL_RGB);

    std::string build =
#ifdef _DEBUG
        "Debug";
#else
        "Release";
#endif

    // Test matrix (can be modified as needed)
    const std::vector<std::pair<int, int>> resolutions = {
        {640, 480},
        {1280, 720},
        {1920, 1080}
    };
    const std::vector<FilterType> filters = {
        FilterType::None, FilterType::Pixelate, FilterType::SinCity
    };
    const std::vector<bool> transforms = { false, true };
    const std::vector<bool> modes = { false /*CPU*/, true /*GPU*/ };

    AffineParams aff; aff.tx = 60.f; aff.ty = 40.f; aff.scale = 1.15f; aff.thetaDeg = 8.f;

    // Clear color
    glClearColor(0.08f, 0.1f, 0.15f, 1.0f);

    // Run all combinations and collect results
    std::vector<BenchResultRow> results;
    for (bool useGPU : modes) {
        for (auto f : filters) {
            for (bool t : transforms) {
                for (auto r : resolutions) {
                    std::cout << "[RUN] " << (useGPU ? "GPU" : "CPU")
                        << " | " << filterName(f)
                        << " | T=" << (t ? "On" : "Off")
                        << " | " << r.first << "x" << r.second << std::endl;

                    auto row = run_one_combo(win, gpu, passProg, loc_uTex, loc_uAff,
                        vao, tex, texW, texH,
                        r, build, f, useGPU, t, aff,
                        /*warmup_sec*/1, /*sample_sec*/5);
                    results.push_back(row);
                }
            }
        }
    }

    // Write CSV (to current working directory)
    std::string out = "perf_summary_" + build + ".csv"; 
    write_summary_csv(results, out);

    // Print summary to console
    std::cout << "\n===== Benchmark Summary (avg_fps) =====\n";
    for (const auto& r : results) {
        std::cout << r.mode << " | " << r.filter << " | " << r.transform
            << " | " << r.resolution << " | " << r.build
            << " => " << r.avg_fps << " FPS (n=" << r.samples << ")\n";
    }

    // Cleanup
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

 //-------------------- main --------------------
//int main() {
//    try {
//        return run_benchmark_mode();
//    }
//    catch (const cv::Exception& e) {
//        std::cerr << "[OpenCV EXCEPTION] " << e.what() << std::endl;
//        return -1;
//    }
//    catch (const std::exception& e) {
//        std::cerr << "[STD EXCEPTION] " << e.what() << std::endl;
//        return -1;
//    }
//}
