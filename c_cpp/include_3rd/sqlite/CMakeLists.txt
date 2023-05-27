cmake_minimum_required(VERSION 3.10)
project(${PACKAGE_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# sqlite3 是使用conda安装地点
# set(CONDA_INCLUDE_DIR /mnt/d/anconda3/include)
# set(CONDA_LIBRARY_DIR /mnt/d/anconda3/lib)
# include_directories(${CONDA_INCLUDE_DIR})
# link_directories(${CONDA_LIBRARY_DIR})

if (NOT $ENV{CONDA_PREFIX} STREQUAL "")
    message(STATUS "CURRENT CONDA PREFIX: $ENV{CONDA_PREFIX}")
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
endif ()


add_executable(main.o main.cpp)
target_link_libraries(main.o sqlite3)


# ~~~~
# g++ -std=c++17 main.cpp -o main.o -I /mnt/d/anconda3/include -L /mnt/d/anconda3/lib -l sqlite3 && ./main.o
# mkdir build && cd build && cmake ..  && make && ./main.o

# clean cmake files
# rm -fR CMakeFiles/ CMakeCache.txt Makefile cmake_install.cmake main.o


