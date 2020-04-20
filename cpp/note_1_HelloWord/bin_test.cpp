// 功能: 各种类型的二进制表示
// 日期: 2019.9.29
// 编译: g++ bin_test.cpp -o bin_test && ./bin_test
// 运行: ./hello_word
//
//

#include <bitset>
#include <iostream>

int main(int argc, char *argv[]) {
    std::string test = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,./;'[]";
    std::size_t len = test.length();
    for(std::size_t i=0;i<len;++i){
        std::cout << test[i] ;
        std::cout << "\tbin:" << std::bitset<sizeof(char)*10>( test[i] ) ;
        std::cout << "\toct:" << std::oct << test[i];
        std::cout << "\thex:" << std::hex << test[i];
        std::cout << std::endl;
    }
    

    // int b = 1;
    // std::cout << std::bitset<sizeof(int)>(b) << std::endl;
}
