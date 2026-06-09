# Inject this as a new code cell after notebook Cell 8:
# "Data collection" / after encode_shot(), collect_one_game(), and jobs are defined,
# but before Cell 9 runs the full multiprocessing data collection.
#
# Goal: cheaply triage proposed features with a tiny pilot sample before collecting
# the full dataset or training the validator.

import math
import time
import numpy as np
from pathlib import Path
from kaggle_environments import make


PROPOSED_FEATURE_NAMES = [
    # Local pressure / map geometry.
    "nearest_my_to_target",
    "nearest_enemy_to_target",
    "ally_ships_near_target",
    "enemy_ships_near_target",
    "ally_ships_near_source",
    "enemy_ships_near_source",
    # Capture margin / source safety.
    "capture_margin_growth",
    "target_growth_until_eta",
    "post_launch_source_reserve",
    "send_to_target_ships",
    # Global economy shares.
    "my_ship_share",
    "my_prod_share",
    "my_planet_share",
    "target_prod_share",
    # Angle quality.
    "cos_shot_error",
    "abs_sin_shot_error",
]


def _nearest_planet_distance(planets, x, y, owner_pred):
    vals = []
    for p in planets:
        if owner_pred(int(p[1])):
            vals.append(max(math.hypot(float(p[2]) - x, float(p[3]) - y) - float(p[4]), 0.0))
    return min(vals) if vals else BOARD


def _ships_near_planets(planets, x, y, me, radius=25.0):
    ally = 0.0
    enemy = 0.0
    for p in planets:
        d = math.hypot(float(p[2]) - x, float(p[3]) - y)
        if d > radius:
            continue
        owner = int(p[1])
        ships = float(p[5])
        if owner == me:
            ally += ships
        elif owner >= 0:
            enemy += ships
    return ally, enemy


def encode_proposed_features(obs, src_id, target_id, ships_sent, shot_angle):
    """Return proposed feature vector for one shot, or None if target recovery failed."""
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict:
        return None

    planets = obs["planets"]
    src = pdict[src_id]
    tgt = pdict[target_id]
    me = int(obs.get("player", 0))

    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), float(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), float(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(math.hypot(dx, dy) - sr - tr, 0.0)
    speed = fleet_speed(ships_sent)
    eta = dist / max(speed, 0.5)

    my_ships = sum(float(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships = sum(float(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_prod = sum(float(p[6]) for p in planets if int(p[1]) == me)
    enemy_prod = sum(float(p[6]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)

    nearest_my_to_target = _nearest_planet_distance(planets, tx, ty, lambda owner: owner == me)
    nearest_enemy_to_target = _nearest_planet_distance(
        planets, tx, ty, lambda owner: owner >= 0 and owner != me
    )
    ally_near_t, enemy_near_t = _ships_near_planets(planets, tx, ty, me)
    ally_near_s, enemy_near_s = _ships_near_planets(planets, sx, sy, me)

    target_owner = int(tgt[1])
    target_growth_until_eta = 0.0 if target_owner == me else tprod * max(eta, 0.0)
    capture_margin_growth = ships_sent - (tships + target_growth_until_eta)
    post_launch_source_reserve = max(sships - ships_sent, 0.0)
    send_to_target_ships = ships_sent / max(tships + 1.0, 1.0)

    total_ships = my_ships + enemy_ships
    total_prod = my_prod + enemy_prod
    total_planets = my_planets + enemy_planets

    target_angle = math.atan2(dy, dx)
    err = shot_angle - target_angle
    cos_shot_error = math.cos(err)
    abs_sin_shot_error = abs(math.sin(err))

    return np.array(
        [
            nearest_my_to_target / BOARD,
            nearest_enemy_to_target / BOARD,
            ally_near_t / 200.0,
            enemy_near_t / 200.0,
            ally_near_s / 200.0,
            enemy_near_s / 200.0,
            capture_margin_growth / 100.0,
            target_growth_until_eta / 100.0,
            post_launch_source_reserve / 100.0,
            send_to_target_ships,
            my_ships / max(total_ships, 1.0),
            my_prod / max(total_prod, 1.0),
            my_planets / max(total_planets, 1.0),
            tprod / max(total_prod, 1.0),
            cos_shot_error,
            abs_sin_shot_error,
        ],
        dtype=np.float32,
    )


def _auc_1d(x, y):
    """Small dependency-free ROC AUC for one feature."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)

    # Average tied ranks.
    xs = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and xs[end] == xs[start]:
            end += 1
        if end - start > 1:
            avg = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg
        start = end

    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg)
    return float(max(auc, 1.0 - auc))  # direction-free usefulness


def collect_preflight_one_game(args):
    teacher_path, opponent_path, seed, side, game_id = args
    paths = [teacher_path, opponent_path] if side == 0 else [opponent_path, teacher_path]
    env = make("orbit_wars", configuration={"randomSeed": seed}, debug=False)
    try:
        env.run(paths)
    except Exception as e:
        return [], game_id, str(e)

    rows = []
    for step_idx, st in enumerate(env.steps):
        obs = st[side].observation
        action = st[side].action or []
        if obs is None or not action:
            continue
        planets = obs["planets"]
        src_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
        for mv in action:
            try:
                src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
            except Exception:
                continue
            if src_id not in src_xy:
                continue
            tgt_id = find_target_via_ray(src_xy[src_id], ang, planets)
            if tgt_id < 0 or tgt_id == src_id:
                continue
            tgt_owner = next((int(p[1]) for p in planets if int(p[0]) == tgt_id), -2)
            if tgt_owner == side:
                continue

            proposed = encode_proposed_features(obs, src_id, tgt_id, ships, ang)
            if proposed is None:
                continue

            tx, ty, tr = next(
                ((float(p[2]), float(p[3]), float(p[4])) for p in planets if int(p[0]) == tgt_id),
                (0, 0, 0),
            )
            sx, sy = src_xy[src_id]
            sr = next((float(p[4]) for p in planets if int(p[0]) == src_id), 0)
            dist = max(math.hypot(tx - sx, ty - sy) - sr - tr, 0.0)
            eta_turns = max(int(math.ceil(dist / max(fleet_speed(ships), 0.5))), 1)
            label = label_outcome(env.steps, tgt_id, side, step_idx + eta_turns, window=10)
            rows.append((proposed, label, game_id, step_idx))
    return rows, game_id, None


# Tiny pilot: change these knobs if you want a stronger preflight signal.
PREFLIGHT_SEEDS = list(range(701, 703))
PREFLIGHT_OPPONENTS = OPPONENT_PATHS[: min(3, len(OPPONENT_PATHS))]
PREFLIGHT_INCLUDE_SELFPLAY = True

preflight_jobs = []
gid = 0
for opp in PREFLIGHT_OPPONENTS:
    for seed in PREFLIGHT_SEEDS:
        for side in (0, 1):
            gid += 1
            preflight_jobs.append((TEACHER, opp, seed, side, gid))
if PREFLIGHT_INCLUDE_SELFPLAY:
    for seed in PREFLIGHT_SEEDS:
        for side in (0, 1):
            gid += 1
            preflight_jobs.append((TEACHER, TEACHER, seed + 1000, side, gid))

print(f"Preflight jobs: {len(preflight_jobs)} games")
t0 = time.time()
preflight_rows = []
failed = 0
for args in preflight_jobs:
    rows, gid_, err = collect_preflight_one_game(args)
    if err is not None:
        failed += 1
        print(f"  [WARN] game {gid_} failed: {err[:80]}")
    else:
        preflight_rows.extend(rows)

print(f"Collected {len(preflight_rows)} preflight shots in {time.time() - t0:.1f}s ({failed} failed games)")
assert preflight_rows, "No preflight rows collected; check opponents/env setup."

Xpf = np.stack([r[0] for r in preflight_rows]).astype(np.float32)
ypf = np.asarray([r[1] for r in preflight_rows], dtype=np.int64)
print(f"positive rate: {ypf.mean() * 100:.1f}%")

summary = []
for j, name in enumerate(PROPOSED_FEATURE_NAMES):
    x = Xpf[:, j]
    pos_x = x[ypf == 1]
    neg_x = x[ypf == 0]
    std = float(np.std(x))
    mean_gap = float(np.mean(pos_x) - np.mean(neg_x)) if len(pos_x) and len(neg_x) else np.nan
    corr = float(np.corrcoef(x, ypf)[0, 1]) if std > 1e-8 and np.std(ypf) > 1e-8 else np.nan
    auc = _auc_1d(x, ypf)
    summary.append(
        {
            "feature": name,
            "std": std,
            "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
            "auc_dirfree": auc,
            "mean_gap": mean_gap,
        }
    )

# Redundancy against current 24 baseline features on the same shot rows.
# Rebuild baseline features for these rows would be heavier, so this preflight
# only checks redundancy among proposed features. Full redundancy can be checked
# after Cell 9 with np.corrcoef(np.c_[feats, proposed_feats]).
corr_mat = np.corrcoef(Xpf, rowvar=False)
for j, row in enumerate(summary):
    others = np.delete(np.abs(corr_mat[j]), j)
    row["max_abs_corr_with_proposed"] = float(np.nanmax(others)) if len(others) else 0.0

summary = sorted(
    summary,
    key=lambda r: (
        0 if np.isnan(r["auc_dirfree"]) else r["auc_dirfree"],
        0 if np.isnan(r["abs_corr"]) else r["abs_corr"],
        r["std"],
    ),
    reverse=True,
)

print("\nProposed feature preflight ranking:")
print("feature                      std      |corr|   auc*    gap      max_redund")
for r in summary:
    keep_hint = (
        "KEEP"
        if r["std"] > 1e-4 and (r["auc_dirfree"] >= 0.56 or r["abs_corr"] >= 0.06) and r["max_abs_corr_with_proposed"] < 0.97
        else "WATCH"
    )
    print(
        f"{r['feature']:<28} {r['std']:>7.4f} "
        f"{r['abs_corr']:>7.3f} {r['auc_dirfree']:>7.3f} "
        f"{r['mean_gap']:>8.4f} {r['max_abs_corr_with_proposed']:>10.3f}  {keep_hint}"
    )

print("\n* auc is direction-free: 0.50 means random; larger means the feature separates success/failure in either direction.")
print("Rule of thumb: KEEP if std is nontrivial, auc >= 0.56 or |corr| >= 0.06, and redundancy < 0.97.")

