import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    mandelbrot_set_path = sys.argv[1]
    N = int(sys.argv[2])
    step_length = int(sys.argv[3])

    mandelbrot_set = np.memmap(mandelbrot_set_path, dtype=np.int32, mode='r', shape=(N, N))

    downsampled_set = mandelbrot_set[::step_length, ::step_length]

    # Save as PNG
    plt.imshow(downsampled_set, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot_downsampled.png', bbox_inches='tight', pad_inches=0)