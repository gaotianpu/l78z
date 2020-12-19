// 功能: 十进制转2进制，2进制转十进制， 位运算符
// 日期: 2019.9.18
// 编译: g++ binary_test.cpp -o binary_test
// 运行: ./binary_test
//
//
#include <math.h>
#include <stdio.h>
#include <iostream>

// 10进制转2进制
long long convertDecimalToBinary(int n);
// 2进制转10进制
int convertBinaryToDecimal(long long n);

int main() {
    // int n;
    // printf("输入一个十进制数: ");
    // scanf("%d", &n);
    // printf("十进制数 %d 转换为二进制位 %lld", n, convertDecimalToBinary(n));

    int a = 10;
    std::cout << "a=" << a << ", bin=" << convertDecimalToBinary(a)
              << std::endl;

    int b = a << 4;
    std::cout << "b=" << b << ", bin=" << convertDecimalToBinary(b)
              << std::endl;

    int c = a | b;
    std::cout << "c=" << c << ", bin=" << convertDecimalToBinary(c)
              << std::endl;

    std::cout << std::endl;
    return 0;
}

int convertBinaryToDecimal(long long n) {
    int decimalNumber = 0, i = 0, remainder;
    while (n != 0) {
        remainder = n % 10;
        n /= 10;
        decimalNumber += remainder * pow(2, i);
        ++i;
    }
    return decimalNumber;
}

long long convertDecimalToBinary(int n) {
    //需改进：根据n的大小，前面补0，返回string？

    long long binaryNumber = 0;
    int remainder, i = 1, step = 1;

    while (n != 0) {
        remainder = n % 2;
        // printf("Step %d: %d/2, 余数 = %d, 商 = %d\n", step++, n, remainder,
        //        n / 2);
        n /= 2;
        binaryNumber += remainder * i;
        i *= 10;
    }
    return binaryNumber;
}