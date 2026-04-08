import os
import sys
import blosc
import numpy as np

from time import perf_counter as time

def write_numpy(arr, file_name):
    np.save(f"{file_name}.npy", arr)
    os.sync()


def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    os.sync()


def read_numpy(file_name):
    return np.load(f"{file_name}.npy")


def read_blosc(file_name):
    with open(f"{file_name}.bl", "rb") as r:
        b_arr = r.read()
    return blosc.unpack_array(b_arr)


if __name__ == "__main__":
    # read N as command line arg
    N = int(sys.argv[1])
    
    arr = np.random.randint(0, 256, size=(N,N,N), dtype=np.uint8)

    # Time write_numpy
    write_time_np = time()
    write_numpy(arr, "test_numpy")
    write_time_np = time() - write_time_np
    print(write_time_np)

    # Time write blosc
    write_time_blosc = time()
    write_blosc(arr, "test_blosc")
    write_time_blosc = time() - write_time_blosc
    print(write_time_blosc)

    # Time read numpy
    read_time_np = time()
    read_numpy("test_numpy")
    read_time_np = time() - read_time_np
    print(read_time_np)

    # Time read blosc
    read_time_blosc = time()
    read_blosc("test_blosc")
    read_time_blosc = time() - read_time_blosc  
    print(read_time_blosc)

    