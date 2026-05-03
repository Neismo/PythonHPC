from os.path import join
import sys
from time import perf_counter as time

import numpy as np
from numba import cuda


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_kernel(old, new, interior_mask):
    cols, rows = old.shape
    col, row = cuda.grid(2)

    if row < rows - 2 and col < cols - 2:
        i, j = row + 1, col + 1

        if interior_mask[row, col]:
            val = 0.25 * (
                old[i, j - 1] + 
                old[i, j + 1] +
                old[i - 1, j] + 
                old[i + 1, j]
            )
            diff = abs(val - old[i, j])
            new[i, j] = val
        else:
            new[i, j] = old[i, j]

def jacobi_cuda(u, interior_mask, max_iter):
    old_device = cuda.to_device(u)
    new_device = cuda.to_device(u.copy())
    mask_device = cuda.to_device(interior_mask)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (u.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (u.shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
    )

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](
            old_device,
            new_device,
            mask_device,
        )
        old_device, new_device = new_device, old_device

    cuda.synchronize()
    return old_device.copy_to_host()


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


if __name__ == "__main__":
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    if len(sys.argv) < 3:
        MAX_ITER = 20_000
    else:
        MAX_ITER = int(sys.argv[2])

    building_ids = building_ids[:N]

    # Compile the kernel before timing the actual floorplans.
    warm_u = np.zeros((514, 514), dtype=np.float64)
    warm_mask = np.ones((512, 512), dtype=np.bool_)
    jacobi_cuda(warm_u, warm_mask, 1)

    all_u0 = np.empty((N, 514, 514), dtype=np.float64)
    all_interior_mask = np.empty((N, 512, 512), dtype=np.bool_)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    start_total = time()

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        start_t = time()
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u
        print(f"Finished {building_ids[i]} in {time() - start_t:.2f} seconds")

    print(f"Total time for {N} floor plans: {time() - start_total:.2f} seconds")

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
