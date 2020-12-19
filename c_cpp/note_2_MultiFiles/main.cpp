// 功能: 汉字字符串截取
// 日期: 2019.9.18
// 编译: 
// g++ -c mylib.cpp && g++ main.cpp -o main.run mylib.o
// g++ -c main.cpp
// g++ mylib.o main.o -o main.run # main.cpp -o main.run mylib.o
// 运行: ./chinese_sub
// 参考: https://sf-zhou.github.io/programming/chinese_encoding.html
//
#include <iostream>
#include <vector>
#include "mylib.h"
int main(int argc, char *argv[]) {
    const int SPLITOR_LEN = 5;
    std::string split[] = {",", "，", "。", "、", "】"};
    const int min = 20;
    const int max = 30;

    std::string cn_str =
        "香港暴露出的问题，为我们去掉了一个错误答案观视频工作室，的秒拍视频";
    std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    return 0;
}


// Undefined symbols for architecture x86_64:
// "cn_sub_str(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >*, int, int, int)", referenced from:
//     _main in main-8aacab.o
// ld: symbol(s) not found for architecture x86_64
// clang: error: linker command failed with exit code 1 (use -v to see invocation)

