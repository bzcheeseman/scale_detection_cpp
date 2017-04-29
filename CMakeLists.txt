cmake_minimum_required(VERSION 3.5)
project(scale_detector_cpp)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 2.4 REQUIRED COMPONENTS core highgui imgproc)

set(SOURCE_FILES main.cpp src/ObjectDetector.cpp include/ObjectDetector.hpp include/utils.hpp)
add_executable(scale_detector_cpp ${SOURCE_FILES})
target_link_libraries(scale_detector_cpp dlib ${OpenCV_LIBRARIES} cblas lapack)