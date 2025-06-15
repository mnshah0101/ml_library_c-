# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(MLFromScratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

enable_testing()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Core library
add_library(ml_lib ${SOURCES})

target_link_libraries(ml_lib Eigen3::Eigen)

# Demo executable
add_executable(ml_demo src/main.cpp)
target_link_libraries(ml_demo ml_lib)


