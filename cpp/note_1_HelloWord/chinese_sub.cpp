// 功能: 汉字字符串截取
// 日期: 2019.9.18
// 编译: g++ chinese_sub.cpp -o chinese_sub
// 运行: ./chinese_sub
// 参考: https://sf-zhou.github.io/programming/chinese_encoding.html
//
#include <iostream>

int is_zh_cn(char p);  //判断字符是否为汉字

int main() {
    //一个char多少个字节？ 1个
    std::cout << "sizeof(char)=" << sizeof(char) << std::endl;
    std::cout << "sizeof('a')=" << sizeof('a') << std::endl;  // 1
    // error,中文不能用单引号
    // std::cout << "sizeof(高)=" << sizeof('高') << std::endl;

    //字符串
    std::cout << "sizeof(a)=" << sizeof("a") << std::endl;    // 1+1 '\0'
    std::cout << "sizeof(ab)=" << sizeof("ab") << std::endl;  // 2+1

    //一个中文3个字节？
    std::cout << "sizeof(高)=" << sizeof("高") << std::endl;      // 3+1 ?
    std::cout << "sizeof(高天)=" << sizeof("高天") << std::endl;  // 3+3+1 ?
    std::string cn_str = "欢迎welcome来到我的世界";
    std::cout << "cn_str length:" << cn_str.length() << std::endl;
    std::cout << cn_str.substr(0, 9) << std::endl;
    std::cout << cn_str.c_str() << std::endl;

    //不同的编码对汉字长度的影响？

    //截取n个字符长度？
}

int is_zh_ch(char p) {
    if (~(p >> 8) == 0) {
        return 1;
    }
    return -1;
}