# @Time : 2023/7/4 14:56
# @Author : Li Jiaqi
# @Description :
import random


def product(*args, repeat=1, shuffle=False):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    if shuffle:
        random.shuffle(result)
    return result
