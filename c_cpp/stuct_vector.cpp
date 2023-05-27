/*
字符串替换，去除两边的空格
gcc stuct_vector.cpp -lstdc++ -o bin/stuct_vector && bin/stuct_vector
*/
#include <iostream>
#include <vector>
using namespace std;

#define MAX_QUERYWORD_LEN 256

struct Entity {
    char entity[MAX_QUERYWORD_LEN];
    char etype[MAX_QUERYWORD_LEN];
    int offset;
    int length;
};

struct InputData {
    std::vector<Entity> entity_list;  //
    std::vector<float> multi_classification_res;

    struct Entity stu[5];
};

int main() {
    InputData input;

    Entity entity;
    std::string name = "hello";
    std::string etype = "test";
    snprintf(entity.entity, MAX_QUERYWORD_LEN, "%s", name.c_str());
    snprintf(entity.etype, MAX_QUERYWORD_LEN, "%s", etype.c_str());
    input.entity_list.push_back(entity);

    input.multi_classification_res.push_back(0.1);
    input.multi_classification_res.push_back(0.2);
    input.multi_classification_res.push_back(0.3);

    for (int i = 0; i < 5; i++) {
        Entity entity;
        std::string name = "hello";
        std::string etype = "test";
        snprintf(entity.entity, MAX_QUERYWORD_LEN, "%s", name.c_str());
        snprintf(entity.etype, MAX_QUERYWORD_LEN, "%s", etype.c_str());
        input.stu[i] = entity;
    }

    return 0;
}