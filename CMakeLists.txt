cmake_minimum_required(VERSION 3.15)
project(cv_example)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_EXTRA_MODULES_PATH "C:/sers/madha/opencv_contrib-4.x/modules")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(cv_example src/Detect.cpp)  # Ensure cv.cpp path is correct
target_link_libraries(cv_example ${OpenCV_LIBS})
