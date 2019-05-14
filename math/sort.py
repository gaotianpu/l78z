#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
各种排序算法
"""
import numpy as np
sys.setrecursionlimit(1500)  #设置最大递归调用次数

# quick_sort_times = 0


def quick_sort(nums):
    """快速排序算法"""
    if len(nums) <= 1:
        return nums

    # quick_sort_times = quick_sort_times + 1

    # 左子数组
    less = []
    # 右子数组
    greater = []
    # 基准数
    base = nums.pop()

    # 对原数组进行划分
    for x in nums:
        if x < base:
            less.append(x)
        else:
            greater.append(x)

    # 递归调用
    return quick_sort(less) + [base] + quick_sort(greater)


def run_assert():
    nums = [6, 1, 2, 7, 9, 3, 4, 5, 10, 8]
    sorted_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sorted_nums == quick_sort(nums), "quick_sort wrong"


def main():
    nums = np.random.randint(1, 50000, 2000).tolist()
    # nums = [6,1,2,7,9,3,4,5,10,8]
    print quick_sort(nums)
    #计算递归调用次数？


if __name__ == "__main__":
    run_assert()
    main()
