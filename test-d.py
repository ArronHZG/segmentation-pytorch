import multiprocessing
import time


def func(msg):
    for i in range(3):
        print(msg)
        time.sleep(1)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=10)
    for i in range(10):
        msg = "hello %d" % (i)
        pool.apply_async(func, (msg,))
    pool.close()
    pool.join()
