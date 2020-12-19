//
//
#include "mylib.h"
#include <iostream>
std::string cn_sub_str(std::string cn_str, std::string split[],
                       int split_char_size, int min, int max) {
    size_t len = cn_str.length();
    size_t char_idx = 0;
    int cn_en_len = 0;

    while (char_idx < len) {
        std::string current_chars;
        int is_cn = 0;
        if (~(cn_str.at(char_idx) >> 8) == 0) {
            //中文
            is_cn = 1;
            current_chars = cn_str.substr(char_idx, 3);
            char_idx = char_idx + 3;
        } else {
            //非中文
            current_chars = cn_str.substr(char_idx, 1);
            ++char_idx;
        }

        for (int i = 0; i < split_char_size; i++) {
            size_t split_len = split[i].length();
            if (split[i] == current_chars) {
                if (cn_en_len > min) {
                    std::cout << "1:" << cn_en_len << std::endl;
                    return cn_str.substr(0, char_idx);
                }

                break;
            }
        }

        if (cn_en_len > max) {
            std::cout << "2:" << cn_en_len  << std::endl;
            return cn_str.substr(0, char_idx);
        }

        std::cout << char_idx << ":" << cn_en_len << ",";
        ++cn_en_len;
    }
    std::cout << "3" << std::endl;
    return cn_str;
}