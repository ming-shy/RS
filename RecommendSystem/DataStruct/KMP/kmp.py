'''
KMP算法：首先求取模式串的最长前缀表的长度；然后与文本串进行匹配
'''
def prefixTable(pattern):
    prefix = [0] * len(pattern)
    i = 1
    len_ = 0
    while i < len(pattern):
        if pattern[i] == pattern[len_]:
            len_ += 1
            prefix[i] = len_
            i += 1
        else:  # 对于不匹配的情况，获取斜匹配对应的下标
            if len_ - 1 < 0:
                prefix[i] = 0
                i += 1
            else:
                len_ = prefix[len_ - 1]
    # prefix.insert(0, -1)
    return prefix[:len(pattern)]


def kmpSearch(text, pattern):
    prefix = prefixTable(pattern)
    print(prefix)
    m = len(text)
    n = len(pattern)
    i, j = 0, 0
    while i < m:
        # 结束条件
        if j == n-1 and text[i] == pattern[j]:
            return i-n+1
        if text[i] == pattern[j]:
            i += 1
            j += 1
        else:
            if prefix[j] == -1:
                i += 1
            else:
                j = prefix[j]
    return -1


if __name__ == '__main__':
    print(prefixTable("ababcaabc"))
    # print(kmpSearch('daddabc', "abc"))