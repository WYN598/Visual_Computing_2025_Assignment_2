cmake_minimum_required(VERSION 3.21)
project(VisualComputing_2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad  CONFIG REQUIRED)
find_package(glm   CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

file(GLOB SRC_FILES
    src/*.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glad::glad
    glm::glm
    ${OpenCV_LIBS}
)

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders)
set(SHADER_DST_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_DST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_SRC_DIR} ${SHADER_DST_DIR}
    COMMENT "Copying shaders to build directory"
)
