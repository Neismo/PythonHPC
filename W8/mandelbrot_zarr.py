import sys
import multiprocessing
import zarr
import numpy as np

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2.0:
            return i
    return 100

def process_chunk(args):
    i, j, i_end, j_end, x_slice, y_slice = args

    points = [complex(x, y) for y in y_slice for x in x_slice]
    escape_times = [mandelbrot_escape_time(p) for p in points]
    chunk_data = np.array(escape_times, dtype=np.int32).reshape((len(y_slice), len(x_slice)))

    return i, j, i_end, j_end, chunk_data

if __name__ == "__main__":
    N = int(sys.argv[1])
    C = int(sys.argv[2])
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0
    num_proc = 4

    x_values = np.linspace(xmin, xmax, N)
    y_values = np.linspace(ymin, ymax, N)

    mandelbrot_set = zarr.open('mandelbrot.zarr', mode='w', shape=(N, N), chunks=(C, C), dtype=np.int32)

    tasks = []
    for i in range(0, N, C):
        for j in range(0, N, C):
            i_end = min(i + C, N)
            j_end = min(j + C, N)
            
            y_slice = y_values[i:i_end]
            x_slice = x_values[j:j_end]
            
            tasks.append((i, j, i_end, j_end, x_slice, y_slice))

    with multiprocessing.Pool(processes=num_proc) as pool:
        for i, j, i_end, j_end, chunk_data in pool.imap_unordered(process_chunk, tasks):
            mandelbrot_set[i:i_end, j:j_end] = chunk_data

    print("Done! Data saved to 'mandelbrot.zarr'.")