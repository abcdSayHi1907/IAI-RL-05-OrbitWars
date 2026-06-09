

from __future__ import annotations

import dataclasses
import os
import sys
import math as _math_nn
from dataclasses import dataclass

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor
import numpy as _np_nn

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),                                                 # [S, 1]
        target_idx.unsqueeze(0),                                                 # [1, T]
        sizes,                                                                    # [S, T]
        active=active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr                                         # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]

    # --- pack one candidate per (source, target); contributor axis L = 1 --------
    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]                                  # [C]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )                                                                            # [C]
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0
        or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
        owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
        prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        prod_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype))))
        ships_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype))))
        strength = prod_by_owner + 0.025 * ships_by_owner
        my_strength = strength[pid].detach()

        target_owner = owner[target_idx.clamp(0, P - 1)].clamp(min=0, max=max(int(player_count) - 1, 0))
        target_owned_enemy = (
            target_exists
            & obs.is_enemy[target_idx.clamp(0, P - 1)]
            & (obs.owner_abs[target_idx.clamp(0, P - 1)] >= 0)
        )
        owner_strength = strength[target_owner]
        leader_delta = (owner_strength - my_strength).clamp(min=0.0)
        target_bonus_short = torch.where(
            target_owned_enemy,
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * prod[target_idx.clamp(0, P - 1)].to(dtype),
            torch.zeros_like(owner_strength),
        )
        score = score + target_bonus_short[cand_tgt_short]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_offensive_targets=7,
    max_defensive_targets=2,
    roi_threshold=1.55,
    min_ships_to_launch=5.0,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    ffa_leader_attack_bonus=0.035,
    ffa_target_prod_bonus=0.08,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _submission_c_agent_internal(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)


# ---------------------------------------------------------------------------
# Old-style final-move throttle: baseline first, then top-k final attacks.
# ---------------------------------------------------------------------------

_OW_WEIGHTS_NAME = "badmove_mlp_weights.npz"
_OW_TOPK_FINAL = 2
_OW_RANK_MODE = "validator"
_OW_USE_NN_FILTER = False
_OW_VETO_THRESHOLD = 1.01
_OW_KEEP_SUPPORT = True
_OW_RAW_WEIGHTS = None
_OW_MISSING = False


def _ow_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _ow_find_weights_path():
    candidates = []
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), _OW_WEIGHTS_NAME))
    except NameError:
        pass
    candidates.append(os.path.join(os.getcwd(), _OW_WEIGHTS_NAME))
    candidates.append(os.path.join('/kaggle_simulations/agent', _OW_WEIGHTS_NAME))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _ow_load_weights():
    global _OW_RAW_WEIGHTS, _OW_MISSING
    if _OW_MISSING:
        return None
    if _OW_RAW_WEIGHTS is None:
        path = _ow_find_weights_path()
        if path is None:
            _OW_MISSING = True
            return None
        try:
            with _np_nn.load(path) as data:
                _OW_RAW_WEIGHTS = {k: data[k].astype('float32') for k in data.files}
        except Exception:
            _OW_MISSING = True
            return None
    return _OW_RAW_WEIGHTS


def _ow_nn_badmove_proba_np(x):
    weights = _ow_load_weights()
    if weights is None or x.shape[-1] != int(weights['x_mean'].shape[-1]):
        return None
    z = (x.astype('float32') - weights['x_mean']) / _np_nn.maximum(weights['x_std'], 1.0e-6)
    h = _np_nn.maximum(0.0, z @ weights['l0_w'].T + weights['l0_b'])
    h = _np_nn.maximum(0.0, h @ weights['l1_w'].T + weights['l1_b'])
    logit = h @ weights['l2_w'].T + weights['l2_b']
    return 1.0 / (1.0 + _np_nn.exp(-logit.reshape(-1)))


def _ow_fleet_speed(ships):
    s = max(float(ships), 1.0)
    return 1.0 + 5.0 * min(_math_nn.log(s) / _math_nn.log(1000.0), 1.0) ** 1.5


def _ow_player_count(obs, player_id):
    owners = [int(player_id)]
    for p in _ow_get(obs, 'planets', []) or []:
        try:
            owner = int(p[1])
            if owner >= 0:
                owners.append(owner)
        except Exception:
            pass
    for f in _ow_get(obs, 'fleets', []) or []:
        try:
            owner = int(f[1])
            if owner >= 0:
                owners.append(owner)
        except Exception:
            pass
    return max(max(owners) + 1, 2) if owners else 2


def _ow_planet_maps(obs):
    planets = list(_ow_get(obs, 'planets', []) or [])
    by_id = {}
    for p in planets:
        try:
            by_id[int(p[0])] = p
        except Exception:
            pass
    return planets, by_id


def _ow_find_target_ray(src_xy, angle, planets, source_id, ray_horizon=240.0, perp_margin=1.0):
    sx, sy = src_xy
    fx, fy = _math_nn.cos(float(angle)), _math_nn.sin(float(angle))
    best = None
    for p in planets:
        try:
            pid = int(p[0])
            if pid == int(source_id):
                continue
            px, py, pr = float(p[2]), float(p[3]), float(p[4])
        except Exception:
            continue
        dx, dy = px - sx, py - sy
        along = dx * fx + dy * fy
        if along <= 0.0 or along > float(ray_horizon):
            continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + float(perp_margin):
            key = (along, perp)
            if best is None or key < best[0]:
                best = (key, pid)
    return -1 if best is None else int(best[1])


def _ow_pressure_proxy(planets, player_id, horizon):
    pressure = {}
    for tgt in planets:
        try:
            tid = int(tgt[0]); tx = float(tgt[2]); ty = float(tgt[3])
        except Exception:
            continue
        acc = 0.0
        for src in planets:
            try:
                owner = int(src[1])
                if owner < 0 or owner == int(player_id):
                    continue
                sx = float(src[2]); sy = float(src[3]); ships = float(src[5])
            except Exception:
                continue
            dist = _math_nn.hypot(tx - sx, ty - sy)
            reach = max(_ow_fleet_speed(ships) * float(horizon), 1.0e-6)
            acc += ships * max(0.0, 1.0 - dist / reach)
        pressure[tid] = acc
    return pressure


def _ow_encode_final_move(obs, move, target_id, player_id, player_count, planets, by_id):
    try:
        sid, _angle, send = int(move[0]), float(move[1]), int(move[2])
        src = by_id[sid]; tgt = by_id[int(target_id)]
    except Exception:
        return None
    src_sh = float(src[5]); tgt_sh = float(tgt[5]); tgt_prod = float(tgt[6])
    sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
    tx, ty, tr = float(tgt[2]), float(tgt[3]), float(tgt[4])
    dist = max(_math_nn.hypot(tx - sx, ty - sy) - sr - tr, 0.0)
    eta = dist / max(_ow_fleet_speed(send), 0.5)
    owner = int(tgt[1])
    target_mine = 1.0 if owner == int(player_id) else 0.0
    target_neutral = 1.0 if owner < 0 else 0.0
    target_enemy = 1.0 if owner >= 0 and owner != int(player_id) else 0.0
    approx_floor = 1.0 if target_mine else max(tgt_sh + 1.0, 1.0)
    cap_margin = float(send) - approx_floor
    cap_ratio = float(send) / max(approx_floor, 1.0)
    src_after = max(src_sh - float(send), 0.0)

    alive = []
    owned = []
    owned_abs = []
    for p in planets:
        try:
            if float(p[5]) <= 0.0:
                continue
            alive.append(p)
            if int(p[1]) == int(player_id):
                owned.append(p)
            if int(p[1]) >= 0:
                owned_abs.append(p)
        except Exception:
            pass
    total_prod = max(sum(float(p[6]) for p in alive), 1.0e-6)
    total_owned_ships = max(sum(float(p[5]) for p in owned_abs), 1.0e-6)
    my_prod_share = sum(float(p[6]) for p in owned) / total_prod
    my_ship_share = sum(float(p[5]) for p in owned) / total_owned_ships

    horizon = 13.0 if int(player_count) >= 4 else 18.0
    pressure = _ow_pressure_proxy(planets, int(player_id), horizon)
    p_src = pressure.get(int(sid), 0.0)
    p_tgt = pressure.get(int(target_id), 0.0)
    roi = 1.55 if int(player_count) >= 4 else 1.50
    score_base = roi + max(min(cap_margin, 50.0), -50.0) / 25.0 + tgt_prod / 10.0 - eta / 100.0

    return _np_nn.array([
        score_base / 20.0,
        (score_base - roi) / 10.0,
        eta / 20.0,
        float(send) / 100.0,
        float(send) / max(src_sh, 1.0),
        src_sh / 100.0,
        src_after / 100.0,
        tgt_sh / 100.0,
        tgt_prod / 5.0,
        target_neutral,
        target_enemy,
        cap_margin / 50.0,
        cap_ratio / 3.0,
        float(int(_ow_get(obs, 'step', 0) or 0)) / 500.0,
        1.0 if int(player_count) >= 4 else 0.0,
        my_prod_share,
        my_ship_share,
        p_src / 100.0,
        p_tgt / 100.0,
        (p_tgt - p_src) / 100.0,
    ], dtype='float32')


def _ow_filter_final_moves(obs, moves):
    if not moves or int(_OW_TOPK_FINAL) <= 0:
        return moves
    player_id = int(_ow_get(obs, 'player', 0) or 0)
    player_count = _ow_player_count(obs, player_id)
    planets, by_id = _ow_planet_maps(obs)
    if not planets or not by_id:
        return moves

    attack_idxs = []
    protected_idxs = []
    for i, move in enumerate(moves):
        try:
            sid, angle, ships = int(move[0]), float(move[1]), int(move[2])
            src = by_id[sid]
            tid = _ow_find_target_ray((float(src[2]), float(src[3])), angle, planets, sid)
        except Exception:
            protected_idxs.append(i)
            continue
        if tid < 0 or tid not in by_id:
            protected_idxs.append(i)
            continue
        owner = int(by_id[tid][1])
        if owner == player_id:
            if bool(_OW_KEEP_SUPPORT):
                protected_idxs.append(i)
            else:
                attack_idxs.append((i, ships, tid))
            continue
        p_bad = None
        if bool(_OW_USE_NN_FILTER) or str(_OW_RANK_MODE).lower() == 'validator':
            feat = _ow_encode_final_move(obs, move, tid, player_id, player_count, planets, by_id)
            if feat is not None:
                probs = _ow_nn_badmove_proba_np(feat.reshape(1, -1))
                if probs is not None:
                    p_bad = float(probs[0])
                    if bool(_OW_USE_NN_FILTER) and p_bad > float(_OW_VETO_THRESHOLD):
                        continue
        attack_idxs.append((i, ships, tid, p_bad))

    k = min(int(_OW_TOPK_FINAL), len(attack_idxs))
    keep_attack = set()
    if k > 0:
        if str(_OW_RANK_MODE).lower() == 'validator':
            ranked = sorted(
                attack_idxs,
                key=lambda item: (
                    1 if item[3] is None else 0,
                    item[3] if item[3] is not None else 999.0,
                    -int(item[1]),
                    int(item[0]),
                ),
            )
        else:
            ranked = sorted(attack_idxs, key=lambda item: (int(item[1]), -int(item[0])), reverse=True)
        keep_attack = {int(i) for i, _ships, _tid, _p_bad in ranked[:k]}
    keep_protected = set(protected_idxs) if bool(_OW_KEEP_SUPPORT) else set()
    keep = keep_attack | keep_protected
    return [move for i, move in enumerate(moves) if i in keep]


def agent(obs):
    moves = _submission_c_agent_internal(obs)
    return _ow_filter_final_moves(obs, moves)

