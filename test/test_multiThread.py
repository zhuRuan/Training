# coding=utf-8
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time


def test1(abc):
    time.sleep(abc)
    print(abc)
    return abc

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=None) as executor:
        aaa = executor.map(test1,[6,4,3,1,4])
    for a in aaa:
        print(a)

