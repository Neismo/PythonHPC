from os.path import join
import sys
from time import perf_counter as time

import cupy as cp


SIZE = 512


def load_data(load_dir, bid, dtype=cp.float32):
    u = cp.zeros((SIZE + 2, SIZE + 2), dtype=dtype)

    domain = cp.load(join(load_dir, f"{bid}_domain.npy")).astype(dtype, copy=False)
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy")).astype(cp.bool_, copy=False)

    u[1:-1, 1:-1] = domain

    return u, interior_mask


@cp.fuse()
def jacobi_update(center, left, right, up, down, mask):
    new_val = 0.25 * (left + right + up + down)
    return cp.where(mask, new_val, center)


def jacobi_batched(u_batch, interior_mask_batch, max_iter, atol=1e-4, check_interval=100):
    current = cp.array(u_batch, copy=True)
    next_u = cp.array(u_batch, copy=True)

    for i in range(max_iter):
        center = current[:, 1:-1, 1:-1]

        next_u[:, 1:-1, 1:-1] = jacobi_update(
            center,
            current[:, 1:-1, :-2],
            current[:, 1:-1, 2:],
            current[:, :-2, 1:-1],
            current[:, 2:, 1:-1],
            interior_mask_batch,
        )

        if i % check_interval == 0:
            delta = cp.abs(center - next_u[:, 1:-1, 1:-1]).max()

            if delta < atol:
                current = next_u
                break

        current, next_u = next_u, current

    return current


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]

    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100

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

    # Use cp.float64 if you need the same precision as the original version.
    dtype = cp.float32

    all_u0 = cp.empty((N, SIZE + 2, SIZE + 2), dtype=dtype)
    all_interior_mask = cp.empty((N, SIZE, SIZE), dtype=cp.bool_)

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid, dtype=dtype)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    CHECK_INTERVAL = 100

    cp.cuda.Stream.null.synchronize()
    start_total = time()

    all_u = jacobi_batched(
        all_u0,
        all_interior_mask,
        MAX_ITER,
        ABS_TOL,
        CHECK_INTERVAL,
    )

    cp.cuda.Stream.null.synchronize()
    print(f"Total time for {N} floor plans: {time() - start_total:.2f} seconds")

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    #print("building_id, " + ", ".join(stat_keys))

    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        stats_cpu = {k: float(stats[k].get()) for k in stat_keys}

        #print(f"{bid},", ", ".join(str(stats_cpu[k]) for k in stat_keys))