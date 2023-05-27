cmake_minimum_required(VERSION 3.10)
project(${PACKAGE_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Boost_NO_WARN_NEW_VERSIONS 1) #
# set(Boost_INCLUDE_DIR /usr/include/boost)
# set(Boost_LIBRARY_DIR /usr/local/lib)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
# include_directories(${Boost_INCLUDE_DIR})
# link_directories(${Boost_LIBRARY_DIR})


add_executable(main.o main.cpp)
target_link_libraries(main.o ${Boost_LIBRARIES})


# ~~~~
# clean cmake files
# rm -fR CMakeFiles/ CMakeCache.txt Makefile cmake_install.cmake main.o
# cmake .  && make && ./main.o