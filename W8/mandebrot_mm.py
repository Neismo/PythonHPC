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
    width, height = int(sys.argv[1]), int(sys.argv[1]) # N x N
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    num_proc = 4

    # Precompute points
    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)
    points = np.memmap('points.dat', dtype=np.int32, mode='w+', shape=(width * height,))
    #for i, x in enumerate(x_values):
    #    for j, y in enumerate(y_values):
    #        points[i * width + j] = complex(x, y)

    # Compute set
    # mandelbrot_set = generate_mandelbrot_set(points, num_proc)

    # Save set as image
    # mandelbrot_set = mandelbrot_set.reshape((height, width))
    # plot_mandelbrot(mandelbrot_set)