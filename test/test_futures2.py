from concurrent import futures


def test(num):
    import time
    time.sleep(num)
    return time.ctime(), num


data = [3, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1]
with futures.ThreadPoolExecutor(max_workers=None) as executor:
    for future in executor.map(test, data):
        print(future)