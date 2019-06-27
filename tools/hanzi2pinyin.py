import os
import sys
from pypinyin import lazy_pinyin

for line in sys.stdin:
    cn = line.strip()
    p = lazy_pinyin(cn)
    pinyin =  "".join(p)
    print(cn,pinyin)

# print(lazy_pinyin('中心'))

# pip install pypinyin
