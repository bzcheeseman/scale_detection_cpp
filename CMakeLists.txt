cmake_minimum_required(VERSION 3.8)
project(Iris)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)

#link_directories(/usr/local/Cellar/opencv/2.4.13.2/lib)

find_package(OpenCV 2.4 REQUIRED COMPONENTS core highgui imgproc)

set(SOURCE_FILES main.cpp src/ObjectDetector.cpp include/ObjectDetector.hpp include/utils.hpp)
add_executable(Iris ${SOURCE_FILES})
target_link_libraries(Iris dlib ${OpenCV_LIBRARIES} cblas lapack)