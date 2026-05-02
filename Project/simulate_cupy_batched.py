from os.path import join
import sys

import cupy as cp
from time import perf_counter as time

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_batched(u_batch, interior_mask_batch, max_iter, atol=1e-4, check_interval=100):
    u = cp.copy(u_batch)

    for i in range(max_iter):
        u_new = 0.25 * (u[:, 1:-1, :-2] + 
                        u[:, 1:-1, 2:] + 
                        u[:, :-2, 1:-1] + 
                        u[:, 2:, 1:-1])
    
        u_updated = cp.where(interior_mask_batch, u_new, u[:, 1:-1, 1:-1])
        
        if i % check_interval == 0:  # continue until "worst" floor converged
            delta = cp.abs(u[:, 1:-1, 1:-1] - u_updated).max()
            if delta < atol:
                u[:, 1:-1, 1:-1] = u_updated
                break

        u[:, 1:-1, 1:-1] = u_updated

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    start_total = time()
    # print(f"Running Jacobi iterations for {N} floor plans...")
    all_u = jacobi_batched(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)
    print(f"Total time for {N} floor plans: {time() - start_total:.2f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))