cmake_minimum_required (VERSION 2.6)

add_executable(apriltags_demo apriltags_demo.cpp Serial.cpp)
target_link_libraries(apriltags_demo apriltags pthread ${OpenCV_LIBS})
if (UNIX)
target_link_libraries(apriltags_demo v4l2)
endif(UNIX)

set_property(TARGET apriltags_demo PROPERTY CXX_STANDARD 11)
set_property(TARGET apriltags_demo PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(imu imu.cpp Serial.cpp)
