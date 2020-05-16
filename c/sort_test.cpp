#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

bool compare(int a, int b) {
    return a > b;  //升序排列，如果改为return a>b，则为降序
}

int main() {
    int count = 400;
    float rate = 0.8;
    int a = count * rate;

    // int x[a] = {3, 2, 1, 4, 8};
    int *x = new int[a];
    std::sort(x, x + 5);

    for (int i = 0; i < 5; ++i) {
        std::cout << x[i] << ",";
    }
    std::cout << std::endl;
    delete[] x;

    // std::cout << x << std::endl;
}

// g++ sort_test.cpp && ./a.out