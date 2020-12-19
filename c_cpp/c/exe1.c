// https://www.runoob.com/cprogramming/c-exercise-example1.html
// 题目：有1、2、3、4个数字，能组成多少个互不相同且无重复数字的三位数？都是多少？
// gcc exe1.c -o out/exe1 && out/exe1
// 问题：1.python用久了，单引号双引号部分；2.不带分号
#include <stdio.h>
int main() {
    int i, j, k;
    for (i = 1; i < 5; i++) {
        for (j = 1; j < 5; j++) {
            for (k = 1; k < 5; k++) {
                if (i != k && i != j && k != j) {
                    printf("%d%d%d\n", i, j, k);
                }
            }
        }
    }
}