cmake_minimum_required(VERSION 3.10)
project(${PACKAGE_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(Torch REQUIRED)
include_directories(/mnt/d/github/libtorch/include)
link_directories(/mnt/d/github/libtorch/lib)


add_executable(main.o main.cpp)
target_link_libraries(main.o torch)

# https://pytorch.org/cppdocs/installing.html
# cmake -DCMAKE_PREFIX_PATH=/mnt/d/github/libtorch ..
# or 
# cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
# cmake --build . --config Release


