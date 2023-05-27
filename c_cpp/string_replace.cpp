/*
字符串替换，去除两边的空格
gcc string_replace.cpp -lstdc++ -o bin/string_replace && bin/string_replace
*/
#include <iostream>
#include <vector>
using namespace std;
int main() {
    // std::string terms[] = {"电影"};
    // std::vector<std::string> other_terms = {"电影", "电视剧", "游戏", "影评",
    // "评价", "观后感"};

    string str="X战警:天启2";
    std::transform(str.begin(),str.end(),str.begin(),::tolower);
    cout << str << endl;

    std::vector<std::string> other_terms;
    other_terms.push_back("电影");

    char originalQuery[] = {"X战警:天启2"};
    // char originalQuery[] = {""};
    std::string str_originalQuery = std::string(originalQuery);

    std::string str_reyi_entity_query = std::string("X战警");
    // std::string str_left;
    // std::string str_left =
    
    std::string str_left = str_originalQuery.replace(
        str_originalQuery.find(str_reyi_entity_query),
        str_originalQuery.find(str_reyi_entity_query) +
            str_reyi_entity_query.size(),
        "");

    str_left.erase(0, str_left.find_first_not_of(" "));
    str_left.erase(str_left.find_last_not_of(" ") + 1);

    bool ret = str_left.compare("电影");
    std::cout << "ret:" << ret << endl;

    if (std::count(other_terms.begin(), other_terms.end(), str_left)) {
        std::cout << "exist" << endl;
    }

    printf("%s | %s | %s | %s\n", originalQuery, str_originalQuery.c_str(),
           str_reyi_entity_query.c_str(), str_left.c_str());

    // std::string t = "盛夏未来";
    // std::string line = "盛夏未来电影";
    // std::string line2 = std::string(line);
    // // std::string line1 = line2.replace(line2.find(t), t.length(), "");
    // std::string line1 = line2.replace(line2.find(t), t.size(), "");
    // std::cout << line << endl;
    // std::cout << line1 << endl;
    return 0;
}
