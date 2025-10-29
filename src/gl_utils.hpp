#pragma once
#include <string>
#include <glad/glad.h>
#include <opencv2/opencv.hpp>

namespace glutils {

	/// 读取文件内容（一般用于加载 .vert / .frag 着色器）
	std::string loadFile(const std::string& path);

	/// 编译单个 shader（GL_VERTEX_SHADER / GL_FRAGMENT_SHADER）
	GLuint compileShader(GLenum type, const std::string& src);

	/// 从文件路径编译并链接着色器程序
	GLuint loadShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);

	/// 创建一个空的 2D 纹理
	GLuint createTexture2D(int width, int height, GLenum format = GL_RGB);

	/// 将 OpenCV Mat 上传到已有的 GL 纹理（自动转 RGB）
	void uploadFrameToTexture(GLuint texID, const cv::Mat& frame);

	/// 创建一个全屏 Quad（VAO）用于绘制纹理
	GLuint createFullScreenQuadVAO();

	/// 检查 OpenGL 错误（调试用）
	void checkGLError(const char* tag = nullptr);

}
