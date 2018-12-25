cmake_minimum_required(VERSION 3.12)
project(HogBasedObjectDetection)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-DPROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PkgConfig REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)
set(HEADER_FILES
        include/HogVisualization.h
        include/RandomForest.h)

set(SOURCE_FILES
        src/main.cpp
        src/HogVisualization.cpp
        src/RandomForest.cpp)

add_executable(HogBasedObjectDetection ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(HogBasedObjectDetection ${OpenCV_LIBRARIES})

# Debug
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
