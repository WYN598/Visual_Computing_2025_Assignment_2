#include "gl_utils.hpp"
#include "gpu_pipeline.hpp"
#include "cv_filters.hpp"
#include "cv_geom.hpp"
#include "timing.hpp"

#include <opencv2/opencv.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <string>
#include <vector>

// ------------------ Utility Functions ------------------
static void ensureBGR(cv::Mat& img) {
    if (img.empty()) return;
    if (img.type() == CV_8UC3) {
        if (!img.isContinuous()) img = img.clone();
        return;
    }
    cv::Mat out;
    if (img.type() == CV_8UC4)      cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
    else if (img.type() == CV_8UC1) cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else if (img.type() == CV_8UC2) {
        try { cv::cvtColor(img, out, cv::COLOR_YUV2BGR_YUY2); }
        catch (...) { cv::cvtColor(img, out, cv::COLOR_GRAY2BGR); }
    }
    else                           img.convertTo(out, CV_8UC3);
    if (!out.isContinuous()) out = out.clone();
    img = out;
}

static std::string filterName(FilterType f) {
    switch (f) {
    case FilterType::None:     return "None";
    case FilterType::Pixelate: return "Pixelate";
    case FilterType::SinCity:  return "SinCity";
    default:                   return "Unknown";
    }
}

static void setTitle(GLFWwindow* w, bool gpu, FilterType f, bool T, double fps) {
    std::string s = std::string("[Interactive] ")
        + "Mode=" + (gpu ? "GPU" : "CPU")
        + " | Filter=" + filterName(f)
        + " | Transform=" + (T ? "ON" : "OFF")
        + " | FPS=" + std::to_string((int)std::round(fps));
    glfwSetWindowTitle(w, s.c_str());
}

// ---------- Generate HUD texture using OpenCV text drawing ----------
static cv::Mat makeHudBGRA(int w, int h, int scale = 1) {
    // Semi-transparent black background
    cv::Mat bgra(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 140));

    auto put = [&](int y, const std::string& text, double fs = 0.6, int thick = 1) {
        int baseline = 0;
        cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fs, thick, &baseline);
        cv::putText(bgra, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, fs,
            cv::Scalar(255, 255, 255, 255), thick, cv::LINE_AA);
        return y + (int)(sz.height + 10);
    };

    int y = 24;
    y = put(y, "Controls");
    y = put(y, "G: Toggle GPU/CPU");
    y = put(y, "1/2/3: None / Pixelate / SinCity");
    y = put(y, "T: Toggle Transform (Affine)");
    y = put(y, "Arrows: Translate (tx, ty)");
    y = put(y, "Q/E: Rotate");
    y = put(y, "-/=: Zoom in / Zoom out");
    y = put(y, "Z/X: Pixel block size (Pixelate)");
    y = put(y, "C/V: Threshold (SinCity)");
    y = put(y, "ESC: Quit");

    return bgra;
}

static GLuint createHudTextureFromMat(const cv::Mat& bgra) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bgra.cols, bgra.rows, 0, GL_BGRA, GL_UNSIGNED_BYTE, bgra.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

// Draw a small textured quad at the top-left corner (HUD), not affected by affine transform
struct HudQuad {
    GLuint vao = 0, vbo = 0;
    int w = 0, h = 0; // texture pixel size

    // Vertex data: pos(x,y) + uv(u,v), using a GL_TRIANGLE_STRIP quad
    void init() {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)(sizeof(float) * 2));
        glBindVertexArray(0);
    }

    // Update vertex data to place HUD at (x,y) in screen pixels
    void update(int fbW, int fbH, int x, int y, int wpx, int hpx) {
        w = wpx; h = hpx;
        auto toNDC = [&](float px, float py) -> std::pair<float, float> {
            float X = (2.f * px / fbW) - 1.f;
            float Y = 1.f - (2.f * py / fbH);
            return { X, Y };
        };
        auto p0 = toNDC((float)x, (float)y);           // Top-left
        auto p1 = toNDC((float)(x + wpx), (float)y);   // Top-right
        auto p2 = toNDC((float)x, (float)(y + hpx));   // Bottom-left
        auto p3 = toNDC((float)(x + wpx), (float)(y + hpx)); // Bottom-right

        // Note: flip the V coordinate ¡ª top uses v=1, bottom uses v=0
        float verts[16] = {
            p0.first, p0.second,   0.f, 1.f,
            p1.first, p1.second,   1.f, 1.f,
            p2.first, p2.second,   0.f, 0.f,
            p3.first, p3.second,   1.f, 0.f
        };

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
};

// ------------------ Interactive Demonstration ------------------
static void interactive_mode() {
    // Camera initialization (DSHOW + MJPG is more stable on Windows)
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) { std::cerr << "Camera open failed.\n"; return; }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    cv::Mat frame; cap >> frame; if (frame.empty()) { std::cerr << "First frame empty.\n"; return; }
    ensureBGR(frame);
    int texW = frame.cols, texH = frame.rows;

    // Initialize OpenGL context
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(texW, texH, "Interactive Mode", nullptr, nullptr);
    if (!win) { std::cerr << "Create window failed\n"; glfwTerminate(); return; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "glad init failed\n"; return; }

    glClearColor(0.08f, 0.10f, 0.15f, 1.0f);

    // Full-screen quad for displaying frames
    GLuint fsqVAO = glutils::createFullScreenQuadVAO();

    // GPU pipeline + simple pass-through shader
    GpuPipeline gpu; if (!gpu.init("shaders")) { std::cerr << "Shader init failed.\n"; return; }
    GLuint passProg = glutils::loadShaderProgram("shaders/passthrough.vert", "shaders/passthrough.frag");
    GLint loc_uTex_pass = glGetUniformLocation(passProg, "uTex");
    GLint loc_uAff_pass = glGetUniformLocation(passProg, "uAffine");

    // Texture for video frames
    GLuint texVid = glutils::createTexture2D(texW, texH, GL_RGB);
    glBindTexture(GL_TEXTURE_2D, texVid);
    float border[4] = { 0,0,0,1 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glBindTexture(GL_TEXTURE_2D, 0);

    // HUD texture (generated once)
    cv::Mat hudImg = makeHudBGRA(360, 220);
    GLuint texHUD = createHudTextureFromMat(hudImg);
    HudQuad hud; hud.init();

    // State variables
    bool useGPU = true, useTransform = true;
    FilterType curF = FilterType::Pixelate;
    FilterParams fp; fp.pixelBlock = 8; fp.keepBGR = { 20,20,200 }; fp.thresh = 60;
    AffineParams ap; ap.tx = 0; ap.ty = 0; ap.scale = 1.f; ap.thetaDeg = 0.f;

    bool lockG = false, lockT = false, lock1 = false, lock2 = false, lock3 = false;
    FpsAverager fpsAvg(120);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, GLFW_TRUE);

        // Keyboard control logic (toggle switches)
        if (glfwGetKey(win, GLFW_KEY_G) == GLFW_PRESS) { if (!lockG) { useGPU = !useGPU; lockG = true; } }
        else lockG = false;
        if (glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS) { if (!lockT) { useTransform = !useTransform; lockT = true; } }
        else lockT = false;
        if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) { if (!lock1) { curF = FilterType::None; lock1 = true; } }
        else lock1 = false;
        if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) { if (!lock2) { curF = FilterType::Pixelate; lock2 = true; } }
        else lock2 = false;
        if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) { if (!lock3) { curF = FilterType::SinCity; lock3 = true; } }
        else lock3 = false;

        // Translation / rotation / scaling controls
        float tStep = 5.f, rStep = 0.6f, sStep = 0.02f;
        if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS) ap.tx -= tStep;
        if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) ap.tx += tStep;
        if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS) ap.ty -= tStep;
        if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) ap.ty += tStep;
        if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) ap.thetaDeg -= rStep;
        if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) ap.thetaDeg += rStep;
        if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS) ap.scale = std::max(0.1f, ap.scale - sStep);
        if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS) ap.scale += sStep;

        // Filter parameter adjustment
        if (glfwGetKey(win, GLFW_KEY_Z) == GLFW_PRESS) fp.pixelBlock = std::max(2, fp.pixelBlock - 1);
        if (glfwGetKey(win, GLFW_KEY_X) == GLFW_PRESS) fp.pixelBlock = std::min(100, fp.pixelBlock + 1);
        if (glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS) fp.thresh = std::max(0, fp.thresh - 1);
        if (glfwGetKey(win, GLFW_KEY_V) == GLFW_PRESS) fp.thresh = std::min(255, fp.thresh + 1);

        // Capture camera frame
        cap >> frame; if (frame.empty()) continue; ensureBGR(frame);
        if (frame.cols != texW || frame.rows != texH) {
            texW = frame.cols; texH = frame.rows;
            glDeleteTextures(1, &texVid);
            texVid = glutils::createTexture2D(texW, texH, GL_RGB);
            glBindTexture(GL_TEXTURE_2D, texVid);
            float border2[4] = { 0,0,0,1 };
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border2);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // Upload and process
        if (!useGPU) {
            cv::Mat img = frame.clone();
            if (useTransform) warpCpuAffine(img, ap);
            applyCpuFilter(img, curF, fp);
            glutils::uploadFrameToTexture(texVid, img);
        }
        else {
            glutils::uploadFrameToTexture(texVid, frame);
        }

        // Main frame rendering
        int fbW, fbH; glfwGetFramebufferSize(win, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClear(GL_COLOR_BUFFER_BIT);

        if (useGPU) {
            gpu.draw(fsqVAO, texVid, texW, texH, curF, fp, useTransform ? ap : AffineParams{});
        }
        else {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texVid);
            glUseProgram(passProg);
            if (loc_uTex_pass >= 0) glUniform1i(loc_uTex_pass, 0);

            cv::Matx33f M = affineMatrix(useTransform ? ap : AffineParams{}, texW, texH);
            glm::mat3 gM(
                M(0, 0), M(1, 0), M(2, 0),
                M(0, 1), M(1, 1), M(2, 1),
                M(0, 2), M(1, 2), M(2, 2)
            );
            if (loc_uAff_pass >= 0) glUniformMatrix3fv(loc_uAff_pass, 1, GL_FALSE, glm::value_ptr(gM));

            glBindVertexArray(fsqVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindVertexArray(0);
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // Draw HUD (in screen space, top-left, unaffected by affine transform)
        glUseProgram(passProg);
        if (loc_uTex_pass >= 0) glUniform1i(loc_uTex_pass, 0);
        // uAffine = I (identity matrix) so HUD remains static
        glm::mat3 I(1.0f);
        if (loc_uAff_pass >= 0) glUniformMatrix3fv(loc_uAff_pass, 1, GL_FALSE, glm::value_ptr(I));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texHUD);

        hud.update(fbW, fbH, /*x*/8, /*y*/8, /*w*/hudImg.cols, /*h*/hudImg.rows);
        hud.draw();

        // Update window title and FPS counter
        setTitle(win, useGPU, curF, useTransform, fpsAvg.tick());
        glfwSwapBuffers(win);
    }

    glDeleteTextures(1, &texVid);
    glDeleteTextures(1, &texHUD);
    glfwDestroyWindow(win);
    glfwTerminate();
}

int main() {
    interactive_mode();
    return 0;
}
