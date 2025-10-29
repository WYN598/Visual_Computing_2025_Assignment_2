#include "gl_utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace glutils {

    std::string loadFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + path);
        std::stringstream buf;
        buf << file.rdbuf();
        return buf.str();
    }

    GLuint compileShader(GLenum type, const std::string& src) {
        GLuint shader = glCreateShader(type);
        const char* csrc = src.c_str();
        glShaderSource(shader, 1, &csrc, nullptr);
        glCompileShader(shader);

        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            GLint len = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0');
            glGetShaderInfoLog(shader, len, nullptr, log.data());
            std::cerr << "[Shader compile error]\n" << log << std::endl;
            throw std::runtime_error("Shader compile failed");
        }
        return shader;
    }

    GLuint loadShaderProgram(const std::string& vertexPath, const std::string& fragmentPath) {
        std::string vsrc = loadFile(vertexPath);
        std::string fsrc = loadFile(fragmentPath);
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);

        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);

        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            GLint len = 0;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0');
            glGetProgramInfoLog(prog, len, nullptr, log.data());
            std::cerr << "[Program link error]\n" << log << std::endl;
            throw std::runtime_error("Program link failed");
        }

        glDeleteShader(vs);
        glDeleteShader(fs);
        return prog;
    }

    GLuint createTexture2D(int width, int height, GLenum format) {
        GLuint tex; glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        GLfloat borderColor[4] = { 0.f, 0.f, 0.f, 1.f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        glBindTexture(GL_TEXTURE_2D, 0);
        return tex;
    }


    void uploadFrameToTexture(GLuint texID, const cv::Mat& frame) {
        if (frame.empty()) return;

        cv::Mat rgb;
        if (frame.channels() == 3)
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        else if (frame.channels() == 4)
            cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
        else
            rgb = frame;

        glBindTexture(GL_TEXTURE_2D, texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            rgb.cols, rgb.rows, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    GLuint createFullScreenQuadVAO() {
        float verts[] = {
            // pos       // uv
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };

        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
        return vao;
    }

    void checkGLError(const char* tag) {
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "[GL ERROR]";
            if (tag) std::cerr << " (" << tag << ")";
            std::cerr << " code=" << std::hex << err << std::dec << std::endl;
        }
    }

}
