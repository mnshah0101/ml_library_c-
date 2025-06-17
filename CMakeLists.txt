# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(ml_library VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen package
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Set source files
set(SOURCES
    src/main.cpp
)

# Set header files
set(HEADERS
    include/KMeans.hpp
    include/PCA.hpp
    include/csv_loader.hpp
    include/dataset.hpp
    include/DecisionTree.hpp
    include/KNearestNeighbors.hpp
    include/LinearRegression.hpp
    include/LogisticRegression.hpp
    include/model.hpp
    include/optimizer.hpp
    include/LearningRateScheduler.hpp
    include/loss.hpp
)

# Create static library
add_library(ml_library STATIC ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(ml_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Install rules
include(GNUInstallDirs)
install(TARGETS ml_library
    EXPORT ml_library-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ml_library
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets
install(EXPORT ml_library-targets
    FILE ml_library-targets.cmake
    NAMESPACE ml_library::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml_library
)

# Create and install config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ml_library-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ml_library-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ml_library-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml_library
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ml_library-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ml_library-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml_library
)


