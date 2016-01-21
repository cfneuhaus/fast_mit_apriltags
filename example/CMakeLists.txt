cmake_minimum_required (VERSION 2.6)

link_libraries(apriltags)

add_executable(apriltags_demo apriltags_demo.cpp Serial.cpp)
set_property(TARGET apriltags_demo PROPERTY CXX_STANDARD 11)
set_property(TARGET apriltags_demo PROPERTY CXX_STANDARD_REQUIRED ON)
pods_install_executables(apriltags_demo)

add_executable(imu imu.cpp Serial.cpp)
pods_install_executables(imu)
