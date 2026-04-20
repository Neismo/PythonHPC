import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import sys

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2.0:
            return i
    return 100

def generate_mandelbrot_set(points, num_processes):
    chunk_size = max(1, len(points) // num_processes) # Chunks of of the points for porcesses to work with. 

    # Pool and I use map, it is easier to then have the chunking done.
    with multiprocessing.Pool(processes=num_processes) as pool:
        escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
    return np.array(escape_times)

def generate_mandelbrot_set_chunks(points, num_processes):
    # Fixed number of chunks, but larger than the number of processers!
    chunk_size = 50
    with multiprocessing.Pool(processes=num_processes) as pool:
        escape_times = pool.map(mandelbrot_escape_time, points, chunksize=chunk_size)
    return np.array(escape_times)

def plot_mandelbrot(escape_times):
    plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    N = int(sys.argv[1])
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    num_proc = 4

    # Precompute points
    x_values = np.linspace(xmin, xmax, N)
    y_values = np.linspace(ymin, ymax, N)
    points = [complex(x, y) for y in y_values for x in x_values]

    # Compute set
    mandelbrot_set = np.memmap('mandelbrot.dat', dtype=np.int32, mode='w+', shape=(N, N))
    mandelbrot_set[:] = generate_mandelbrot_set(points, num_proc).reshape((N, N)).T

    mandelbrot_set.flush()

    # Save set as image
    # mandelbrot_set = mandelbrot_set.reshape((N, N))
    # plot_mandelbrot(mandelbrot_set)