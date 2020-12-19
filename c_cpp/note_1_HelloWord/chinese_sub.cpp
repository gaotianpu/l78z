// 功能: 汉字字符串截取
// 日期: 2019.9.18
// 编译: g++ chinese_sub.cpp -o chinese_sub  && ./chinese_sub
// 运行: ./chinese_sub
// 参考: https://sf-zhou.github.io/programming/chinese_encoding.html
//
#include <iostream>
#include <vector>

int is_zh_cn(char p);  //判断字符是否为汉字
std::string sub_str(std::string str, int start, int end);  //截取

std::string cn_sub_str(std::string cn_str, std::string split[], int size,
                       int min, int max);

int main(int argc, char *argv[]) {
    // std::string cn_str_1 = "我爱你，中国";
    // int ret ;
    // char c = cn_str_1.at(0);
    // ret = is_zh_cn(c) ;

    // std::cout << cn_str_1.length() << std::endl;
    // std::cout << cn_str_1.at(0) << std::endl;
    // std::cout << cn_str_1.at(1) << std::endl;
    // std::cout << cn_str_1.at(2) << std::endl;
    // std::cout << typedef( cn_str_1.at(2) ) << std::endl;

    // std::cout << cn_str_1.at(0) << cn_str_1.at(1) << cn_str_1.at(2) <<
    // std::endl; return 0;

    // std::vector<std::string> split;

    const int SPLITOR_LEN = 6;
    std::string split[] = {",", "，", "。", "、", "】","！"};
    const int min = 20;
    const int max = 30;

    std::string cn_str ;
    cn_str = "#香港首家房企宣布无偿捐地#【记住这家企业！香港首家房企宣布无偿捐地[赞]】9月25日，新世界发展覿";
    std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    // cn_str =
    //     "【苹果宣布新Mac "
    //     "Pro生产地迁回美国德州】苹果表示，将在德克萨斯州奥斯汀的工厂生产新的Mac"
    //     " Pro台式计算机，新Mac "
    //     "Pro将包含十多家美国公司设计、开发和制造的组件，并分发给美国客户。";
    // std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    // cn_str =
    //     "【这是他们#为祖国热泪盈眶的瞬间#[泪]"
    //     "】“2008年汶川地震，15名空降兵从5000米高空纵身一跃”“北京申奥成功的那一"
    //     "天，我们一家人紧紧抱在了一起”“刘翔的那次奔跑，让世界知道了中国速度”……"
    //     "​​​​#30天表白祖国#今日话题#"
    //     "为祖国热泪盈眶的瞬间#"
    //     "，你有什么难忘的记忆？恭喜";
    // std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    // cn_str =
    //     "“今年双十一，直播将成为淘宝内容生态划时代的节点，其意义可对标2015年的"
    //     "淘宝无线化”—淘宝内容电商事业部总经理玄德";
    // std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    // cn_str =
    //     "坐下，基本操作！#台湾节目吹爆北京大兴国际机场#"
    //     "】“全球最大！‘新世界第七大奇迹’！” "
    //     "近期的一档台湾节目上，主持人和嘉宾一谈起月底即将通航的北京大兴国际机场"
    //     "就坐不住了，“机场里没有一根柱子！";
    // std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    // cn_str = "国庆临近，机场全线停飞… 伟大祖国";
    // std::cout << cn_sub_str(cn_str, split, SPLITOR_LEN, min, max) << std::endl;

    

    // std::cout << is_zh_cn(cn_str.at(0)) << std::endl;
    // std::cout << is_zh_cn(cn_str.at(1)) << std::endl;
    // std::cout << is_zh_cn(cn_str.at(2)) << std::endl;
    // std::cout << is_zh_cn(cn_str.at(6)) << std::endl;
    // std::cout << "sub_str:" << sub_str(cn_str, 0, 6) << std::endl;

    //一个char多少个字节？ 1个
    // std::cout << "sizeof(char)=" << sizeof(char) << std::endl;
    // std::cout << "sizeof('a')=" << sizeof('a') << std::endl;  // 1
    // // error,中文不能用单引号
    // // std::cout << "sizeof(高)=" << sizeof('高') << std::endl;

    // //字符串
    // std::cout << "sizeof(a)=" << sizeof("a") << std::endl;    // 1+1 '\0'
    // std::cout << "sizeof(ab)=" << sizeof("ab") << std::endl;  // 2+1

    // //一个中文3个字节？
    // std::cout << "sizeof(高)=" << sizeof("高") << std::endl;      // 3+1
    // ? std::cout << "sizeof(高天)=" << sizeof("高天") << std::endl;  //
    // 3+3+1 ?

    // std::cout << "cn_str length:" << cn_str.length() << std::endl;
    // std::cout << "cn_str.substr(0, 9):" << cn_str.substr(0, 9) <<
    // std::endl;

    //不同的编码对汉字长度的影响？

    //截取n个字符长度？

    std::cout << std::endl;
    return 0;
}

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

int is_zh_cn(char p) {
    // std::cout << p << ":";
    if (~(p >> 8) == 0) {
        // std::cout << 1 << std::endl;
        return 1;
    }
    // std::cout << -1 << std::endl;
    return -1;
}

std::string sub_str(std::string str, int start, int end = -1) {
    //实现的还是有些问题？
    if (typeid(str) == typeid(std::string) && str.length() > 0) {
        int len = str.length();

        std::string current_chars = "";

        //先把str里的汉字和英文分开
        std::vector<std::string> dump;
        int i = 0;
        while (i < len) {
            if (is_zh_cn(str.at(i)) == 1) {
                // std::cout << str.at(i) << str.at(i + 1) << str.at(i + 2)
                //           << std::endl;

                dump.push_back(str.substr(i, 3));
                i = i + 3;

            } else {
                dump.push_back(str.substr(i, 1));
                // std::cout << str.at(i) << std::endl;
                i = i + 1;
            }
        }

        end = end > 0 ? end : dump.size();  // end默认为dump.size
        if (start < 0 || start > end) {
            printf("start is wrong");
        }

        //直接从dump里取即可
        for (i = start; i <= end; i++) {
            current_chars += dump[i - 1];
        }

        return current_chars;
    } else {
        printf("str is not string\n");
        return "";
    }
}