cmake_minimum_required(VERSION 3.22.3)
project(fast_mit_apriltags)

option(BUILD_EXAMPLE "Build the fast_mit_apriltags example" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

file(GLOB SOURCE_FILES "src/*.cc")
add_library(apriltags ${SOURCE_FILES})

set_property(TARGET apriltags PROPERTY CXX_STANDARD 11)
set_property(TARGET apriltags PROPERTY CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED CONFIG)

if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc calib highgui)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

target_compile_definitions(apriltags PUBLIC _USE_MATH_DEFINES)
target_include_directories(apriltags PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(apriltags PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/AprilTags)
target_link_libraries(apriltags Eigen3::Eigen ${OpenCV_LIBS})

if(BUILD_EXAMPLE)
   add_subdirectory(example)
endif()
