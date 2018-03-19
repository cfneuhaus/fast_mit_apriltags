cmake_minimum_required(VERSION 2.6)
project(fast_mit_apriltags)

file(GLOB SOURCE_FILES "src/*.cc")
add_library(apriltags ${SOURCE_FILES})

set_property(TARGET apriltags PROPERTY CXX_STANDARD 11)
set_property(TARGET apriltags PROPERTY CXX_STANDARD_REQUIRED ON)

find_package(OpenCV)
find_package(Eigen3 REQUIRED NO_MODULE)

target_include_directories(apriltags PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(apriltags PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/AprilTags)
target_link_libraries(apriltags Eigen3::Eigen ${OpenCV_LIBS})

add_subdirectory(example)
