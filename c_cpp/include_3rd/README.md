# ~~~~
# g++ -std=c++17 main.cpp -o main.o -I /mnt/d/anconda3/include -L /mnt/d/anconda3/lib -l sqlite3 && ./main.o
# 

mkdir build && cd build && cmake ..  && make && ./main.o
# clean cmake files
# rm -fR CMakeFiles/ CMakeCache.txt Makefile cmake_install.cmake main.o