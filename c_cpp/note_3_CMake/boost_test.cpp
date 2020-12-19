// https://juejin.im/post/5a6f32e86fb9a01ca6031230
// cmake . && make && ./note_3_CMake.run 4
// cmake .
// make
// ./note_3_CMake.run 4
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <vector>

// void test_0() {
//     if (argc < 2) {
//         fprintf(stdout, "Usage: %s number\n", argv[0]);
//         return 1;
//     }
//     double inputValue = atof(argv[1]);
//     double outputValue = sqrt(inputValue);
//     fprintf(stdout, "The square root of %g is %g\n", inputValue, outputValue);
// }

void test_1();
void test_2();
void test_3();
int is_zh_cn(char p);  //判断字符是否为汉字

int main(int argc, char *argv[]) {

    std::string str_1 = "我叫王晓明。你呢，boost今天天气不错哦";
    is_zh_cn(str_1.at(0));
    
    // test_1();
    // test_2();
    // test_3();
    return 0;
}

int is_zh_cn(char p) {
    
    // std::cout << p << ":";
    if (~(p >> 8) == 0) {
        // std::cout << 1 << std::endl;
        return 1;
    }
    // std::cout << -1 << std::endl;
    return -1;
}

void test_1(){
    std::string str_1 = "hello,world.te st";
    std::vector<std::string> vecSegTag;
    boost::split(vecSegTag, str_1, boost::is_any_of(",. "));
    for (size_t i = 0; i < vecSegTag.size(); i++) {
        std::cout << vecSegTag[i] << std::endl;
    }
}

void test_2(){
    std::string str_1 = "假如目标程序foo需要链接Boost库regex和system，编写如下的CMakeLists文件";
    std::vector<std::string> vecSegTag;
    boost::split(vecSegTag, str_1, boost::is_any_of("，"));
    for (size_t i = 0; i < vecSegTag.size(); i++) {
        std::cout << vecSegTag[i] << std::endl;
    }
}

void test_3(){
    std::string str_1 = "我叫王晓明。你呢，boost今天天气不错哦";
    std::vector<std::string> vecSegTag;
    boost::split(vecSegTag, str_1, boost::is_any_of("， 。"));
    for (size_t i = 0; i < vecSegTag.size(); i++) {
        std::cout << vecSegTag[i] << std::endl;
    }
}