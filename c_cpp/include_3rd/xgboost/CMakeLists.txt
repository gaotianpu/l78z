cmake_minimum_required(VERSION 3.10)
project(${PACKAGE_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# xgboost 是使用conda安装地点
if (NOT $ENV{CONDA_PREFIX} STREQUAL "")
    message(STATUS "CURRENT CONDA PREFIX: $ENV{CONDA_PREFIX}")
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
endif ()


add_executable(main.o main.cpp)
target_link_libraries(main.o xgboost)




