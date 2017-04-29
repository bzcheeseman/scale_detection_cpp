cmake_minimum_required(VERSION 3.5)
project(Iris)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 2.4 REQUIRED COMPONENTS core highgui imgproc)

set(SOURCE_FILES main.cpp src/ObjectDetector.cpp include/ObjectDetector.hpp include/utils.hpp)
add_executable(Iris ${SOURCE_FILES})
target_link_libraries(Iris dlib ${OpenCV_LIBRARIES} cblas lapack)