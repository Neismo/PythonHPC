from os.path import join
import sys
from time import perf_counter as time

import numpy as np
from numba import cuda

DTYPE = np.float64

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=DTYPE)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(DTYPE)
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_step_kernel(old, new, mask, block_max):
    # kernel for a single jacobi step, started from the numba implementation
    rows, cols = old.shape
    
    # get the cuda indexes and dims 
    col, row    = cuda.grid(2)
    tx, ty      = cuda.threadIdx.x, cuda.threadIdx.y # threadIds
    bx, by      = cuda.blockIdx.x, cuda.blockIdx.y # blockIds
    bdx, bdy    = cuda.blockDim.x, cuda.blockDim.y # blockDims
    gdx, gdy    = cuda.gridDim.x, cuda.gridDim.y # gridDims
    tid         = ty * bdx + tx # thread id
    bid         = by * gdx + bx # block id
    
    # shared mem for storing diffs, 1 slot per thread (16x16)
    smem = cuda.shared.array(256, dtype=DTYPE)
    
    diff = 0.0
    
    # if row and col from cuda grid is within u we do the ops
    if row < rows - 2 and col < cols - 2:
        i, j = row + 1, col + 1

        if mask[row, col]:
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

    # store the diff in shared mem for this thread id 
    smem[tid] = diff
    cuda.syncthreads()

    # following is for the convergence check 
    # find the max diff in each block, later cpu checks if global max diff < atol

    # reduce the 256 diffs in smem down to one max value in 8 reductions (256 = 2^8)
    stride = cuda.blockDim.x * cuda.blockDim.y // 2
    while stride:
        if tid < stride and smem[tid + stride] > smem[tid]:
            smem[tid] = smem[tid + stride]
        cuda.syncthreads()
        # halve for next step
        stride //= 2

    # smem[0] has max diff for this block
    if tid == 0:
        block_max[bid] = smem[0]


def jacobi_cuda(u, interior_mask, max_iter, atol=1e-4, check_interval=100):

    # init old u, new u and mask on gpu
    u               = np.ascontiguousarray(u, dtype=DTYPE)
    old_u_device    = cuda.to_device(u)
    new_u_device    = cuda.to_device(u.copy())
    interior_mask   = np.ascontiguousarray(interior_mask.astype(np.bool_))
    mask_device     = cuda.to_device(interior_mask)

    # calc blocks based on shape of u and the threads
    threads = (16, 16)
    blocks = (
        (u.shape[1] - 2 + threads[0] - 1) // threads[0],
        (u.shape[0] - 2 + threads[1] - 1) // threads[1],
    ) # (32,32)

    # for the max in smem from each block
    block_max_device = cuda.device_array(blocks[0] * blocks[1], dtype=DTYPE)

    for i in range(max_iter):
        # (32,32) and (16,16)
        jacobi_step_kernel[blocks, threads](
            old_u_device, new_u_device, mask_device, block_max_device
        )
        old_u_device, new_u_device = new_u_device, old_u_device

        if i % check_interval == 0:
            # get the max from the parallel recutions
            if block_max_device.copy_to_host().max() < atol:
                break

    return old_u_device.copy_to_host()


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
    building_ids = building_ids[:N]

    # compile the kernel before timing 
    u = np.zeros((514, 514), dtype=DTYPE)
    mask = np.ones((512, 512), dtype="bool")
    jacobi_cuda(u, mask, 1)

    # Load floor plans
    all_u0 = np.empty((N, 514, 514), dtype=DTYPE)
    all_interior_mask = np.empty((N, 512, 512), dtype="bool")
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    start_total = time()

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        start_t = time()
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u
        print(f"Finished {building_ids[i]} in {time() - start_t:.2f} seconds")

    print(f"Total time for {N} floor plans: {time() - start_total:.2f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
