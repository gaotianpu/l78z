#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

def _fullmatch(regex, string, flags=0):
    if hasattr(re, 'fullmatch'):
        return re.fullmatch(regex, string)
    return re.match("(?:" + regex + r")\Z", string, flags=flags)

with open('data/re_test.data','r') as f:
    for line in f:
        matchObj = re.match(r"^[\W]+$",line.strip())   #_fullmatch( r'[\W]+$', q, re.M|re.I)
        if matchObj:
            continue
        print(line.strip())
