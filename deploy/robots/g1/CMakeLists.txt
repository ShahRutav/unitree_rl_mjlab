cmake_minimum_required(VERSION 3.12)
project(g1_controller)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
  ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy
  ${ZMQ_INCLUDE_DIRS}
)

file(GLOB CNPY_SRC ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy/*.cpp)

add_library(cnpy_lib STATIC ${CNPY_SRC})

link_libraries(
  unitree_sdk2
  ddsc
  ddscxx
  rt
  pthread
  Boost::program_options
  yaml-cpp
  fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
  cnpy_lib
  ZLIB::ZLIB
  ${ZMQ_LIBRARIES}
  nlohmann_json::nlohmann_json
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)

# ---- Tests ----
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/v1.14.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

# test_controller: pure math, no hardware/zmq deps.
# Clear the globally-inherited unitree_sdk2/ddsc/etc. first so the test
# binary is standalone and can be executed during ctest without the robot SDK.
add_executable(test_controller tests/test_controller.cpp)
set_target_properties(test_controller PROPERTIES LINK_LIBRARIES "")
target_include_directories(test_controller PRIVATE
  ${PROJECT_SOURCE_DIR}/../../include/
)
target_link_libraries(test_controller PRIVATE
  GTest::gtest_main
  pthread
)
gtest_discover_tests(test_controller DISCOVERY_MODE PRE_TEST)

# test_receiver: ZMQ + nlohmann/json, only tests static parse_message().
add_executable(test_receiver tests/test_receiver.cpp)
set_target_properties(test_receiver PROPERTIES LINK_LIBRARIES "")
target_include_directories(test_receiver PRIVATE
  ${PROJECT_SOURCE_DIR}/../../include/
)
target_link_libraries(test_receiver PRIVATE
  GTest::gtest_main
  ${ZMQ_LIBRARIES}
  nlohmann_json::nlohmann_json
  pthread
)
gtest_discover_tests(test_receiver DISCOVERY_MODE PRE_TEST)
