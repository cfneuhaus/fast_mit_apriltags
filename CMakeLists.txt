cmake_minimum_required(VERSION 2.6)
project(fast_mit_apriltags)

option(BUILD_EXAMPLE "Build the fast_mit_apriltags example" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

file(GLOB SOURCE_FILES "src/*.cc")
add_library(apriltags ${SOURCE_FILES})

set_property(TARGET apriltags PROPERTY CXX_STANDARD 11)
set_property(TARGET apriltags PROPERTY CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED core imgproc calib3d highgui)
find_package(Eigen3 REQUIRED NO_MODULE)

target_include_directories(apriltags PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(apriltags PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/AprilTags)
target_link_libraries(apriltags Eigen3::Eigen ${OpenCV_LIBS})

if(BUILD_EXAMPLE)
   add_subdirectory(example)
endif()
