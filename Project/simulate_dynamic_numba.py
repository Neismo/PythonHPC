import multiprocessing
from os.path import join
import sys
from numba import jit
from time import perf_counter as time

import numpy as np

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@jit(nopython=True, cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol):
    old = u.copy()
    new = u.copy()

    rows, cols = old.shape

    for _ in range(max_iter):
        delta = 0.0

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if interior_mask[i - 1, j - 1]:
                    val = 0.25 * (
                        old[i, j - 1] +
                        old[i, j + 1] +
                        old[i - 1, j] +
                        old[i + 1, j]
                    )

                    diff = abs(old[i, j] - val)
                    if diff > delta:
                        delta = diff

                    new[i, j] = val

        old, new = new, old

        if delta < atol:
            break
    return old


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

# Top-level helper function for multiprocessing
def process_floorplan(i, u0, interior_mask, max_iter, atol, bid):
    start_t = time()
    u = jacobi_numba(u0, interior_mask, max_iter, atol)
    print(f"Finished {bid} in {time() - start_t:.2f} seconds", flush=True)
    return i, u


if __name__ == '__main__':

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    jacobi_numba(np.zeros((514, 514)), np.ones((512, 512), dtype='bool'), MAX_ITER, ABS_TOL)  # warm-up numba

    start_time = time()

    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    NUM_PROCS = 1
    if len(sys.argv) < 2:
        N = 1
    elif len(sys.argv) < 3:
        N = int(sys.argv[1])
    else:
        N, NUM_PROCS = int(sys.argv[1]), int(sys.argv[2])
    building_ids = building_ids[:N]

    # STATIC SCHEDULING
    chunk_size = 1  # force dynamic scheduling

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    all_u = np.empty_like(all_u0)
    tasks = [(i, all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL, building_ids[i]) 
             for i in range(N)]
    
    # Delegate to the multiprocessing pool using starmap and chunk_size
    with multiprocessing.Pool(processes=NUM_PROCS) as pool:
        results = pool.starmap(process_floorplan, tasks, chunksize=chunk_size)
    
    # Slot the results back into the array in the correct order
    for i, u in results:
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    print(f"Total time for {N} floor plans: {time() - start_time:.2f} seconds")