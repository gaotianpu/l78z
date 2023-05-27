#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

def gen_feature(count=45):
    for i in range(count):
        print("f_%d      DECIMAL(10,3) NOT NULL," % (i) )


if __name__ == "__main__":
    gen_feature(45)