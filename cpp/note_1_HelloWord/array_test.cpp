// 功能: hello world
// 日期: 2019.9.18
// 编译: g++ array_test.cpp -o array_test && ./array_test
// 运行: ./array_test
//
//
#include <iostream>

int has_item(int a, int& index);

int main() {
    int index = -1;
    int ret = has_item(28243, index);
    std::cout << "ret:" << ret << ",index:" << index << std::endl;
    return 0;
}

int has_item(int a, int& index) {
    const int items_length = 6;
    uint32_t srcid_list[items_length] = {23, 85, 28243, 28204, 28236, 28532};
    for (int i = 0; i < items_length; i++) {
        if (a == srcid_list[i]) {
            index = i;
            return 1;
        }
    }
    return -1;
}