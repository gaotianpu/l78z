#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

def load_vali(fname):
    with open(fname, 'r') as f:
        l = []
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            key = parts[0]+parts[1]
            l.append(key)
        return set(l)


def process_all(left_file, right_file):
    vali_keys = load_vali(right_file)
    with open(left_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            key = parts[0]+parts[1]
            if key in vali_keys:
                continue
            else:
                print(line.strip())


if __name__ == "__main__":
    left_file = sys.argv[1]
    right_file = sys.argv[2]
    process_all(left_file, right_file)
    # process_all('data/all_data.txt', 'data/all_validate.txt')
