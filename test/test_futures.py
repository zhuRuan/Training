from concurrent.futures import ThreadPoolExecutor


def inddd(x):
    a, b, c, d = x
    return a,c


if __name__ == '__main__':
    input_list = []
    for i in range(5):
        input_list.append((1, 2, 3, 4))
    with ThreadPoolExecutor(max_workers=None) as executor:
        res = executor.map(inddd, input_list)
    for re in res:
        print(re)