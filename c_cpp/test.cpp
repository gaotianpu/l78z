/*
字符串替换，去除两边的空格
gcc test.cpp -lstdc++ -o bin/test && bin/test
*/
#include <iostream>
#include <vector>
using namespace std;
int main() { 
    // uint64_t x2 = 62813999801040896;
    uint64_t x2 = 62813999799992320;
    uint16_t site_id_2 =  (x2 << 3) >> (64 - 6); //6bit 站点
    std::cout << site_id_2 << std::endl;

    // uint64_t x = 44209163803312128;
    // uint16_t a =  (x << 3) >> (64 - 6); //6bit 站点 
    // cout << a << endl;

    // uint64_t x1 = 51396122948681728;
    // uint16_t a1 =  (x1 << 3) >> (64 - 6); //6bit 站点

    // cout << a1 << endl;

    // int  num = 0x00636261;
    // int* pnum = &num;
    // // unsigned char* p_value = nullptr;
    // char* pstr = reinterpret_cast<char*>(&pnum);
    // std::cout << pstr << std::endl;

}