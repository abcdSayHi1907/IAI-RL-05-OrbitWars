from __future__ import annotations

import copy
import dataclasses
import importlib.util
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)


def load_heuristic_module(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("whole_plan_heuristic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import heuristic from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    max_planets: int = 64
    max_fleets: int = 256
    hidden_size: int = 192
    rollout_steps: int = 128
    num_envs: int = 4
    total_updates: int = 1000
    ppo_epochs: int = 8
    minibatch_size: int = 128
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    learning_rate: float = 1e-3
    max_grad_norm: float = 0.5
    shaping_coef: float = 0.03
    checkpoint_every: int = 25
    eval_every: int = 25
    eval_games: int = 20
    save_dir: str = "/kaggle/working/whole_plan_artifacts"
    alternate_sides: bool = True


@dataclass(slots=True)
class PlanProposal:
    name: str
    moves: list[list[float | int]]
    memory: Any
    valid: bool = True


@dataclass(slots=True)
class EncodedDecision:
    planet_features: np.ndarray
    planet_mask: np.ndarray
    fleet_features: np.ndarray
    fleet_mask: np.ndarray
    global_features: np.ndarray
    plan_features: np.ndarray
    plan_mask: np.ndarray


@dataclass(slots=True)
class Transition:
    encoded: EncodedDecision
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass(slots=True)
class Batch:
    planet_features: torch.Tensor
    planet_mask: torch.Tensor
    fleet_features: torch.Tensor
    fleet_mask: torch.Tensor
    global_features: torch.Tensor
    plan_features: torch.Tensor
    plan_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WholePlanLibrary:
    """Generates complete turn plans while committing only the selected planner state."""

    def __init__(self, heuristic_module: Any):
        self.h = heuristic_module
        base = self.h.ProducerLiteConfig()
        self.variant_configs = [
            ("baseline", base),
            ("aggressive", dataclasses.replace(base, roi_threshold=0.5)),
            ("very_aggressive", dataclasses.replace(base, roi_threshold=-0.5)),
            ("conservative", dataclasses.replace(base, roi_threshold=3.0)),
            ("no_regroup", dataclasses.replace(base, enable_regroup=False)),
            (
                "risk_regroup",
                dataclasses.replace(
                    base,
                    enable_potential_risk=True,
                    risk_blend_weight=0.5,
                ),
            ),
            ("short_horizon", dataclasses.replace(base, horizon=10)),
            ("long_horizon", dataclasses.replace(base, horizon=24)),
            ("no_focus_fire", dataclasses.replace(base, enable_focus_fire=False)),
        ]
        self.names = ["pass"] + [name for name, _ in self.variant_configs]
        self.memory = self.h.ProducerLiteMemory()

    @property
    def action_count(self) -> int:
        return len(self.names)

    def reset(self) -> None:
        self.memory = self.h.ProducerLiteMemory()

    def propose(self, observation: Any) -> list[PlanProposal]:
        player = int(obs_get(observation, "player", 0))
        obs_tensors = self.h.single_obs_to_tensor(observation, player_id=player)
        player_count = int(obs_tensors["player_count"].item())
        proposals = [
            PlanProposal(
                name="pass",
                moves=[],
                memory=copy.deepcopy(self.memory),
                valid=True,
            )
        ]
        seen = {()}
        for name, configured in self.variant_configs:
            config = configured
            if player_count >= 4:
                config = dataclasses.replace(
                    configured,
                    horizon=min(configured.horizon, 13),
                    max_sources_per_lane=min(configured.max_sources_per_lane, 6),
                    max_defensive_targets=min(configured.max_defensive_targets, 2),
                    max_strike_sources=min(configured.max_strike_sources, 3),
                )
            candidate_memory = copy.deepcopy(self.memory)
            with torch.inference_mode():
                row = self.h.run_turn(
                    obs_tensors,
                    config=config,
                    player_count=player_count,
                    memory=candidate_memory,
                )
            moves = self.h.sparse_action_row_to_moves(
                row,
                observation,
                player_id=player,
            )
            key = tuple(
                (int(move[0]), round(float(move[1]), 5), int(move[2]))
                for move in moves
            )
            valid = key not in seen
            seen.add(key)
            proposals.append(
                PlanProposal(
                    name=name,
                    moves=moves,
                    memory=candidate_memory,
                    valid=valid,
                )
            )
        return proposals

    def commit(self, proposal: PlanProposal) -> None:
        self.memory = proposal.memory


PLANET_DIM = 11
FLEET_DIM = 9
GLOBAL_DIM = 12
PLAN_STATS_DIM = 16


def encode_decision(
    observation: Any,
    proposals: list[PlanProposal],
    variant_count: int,
    cfg: TrainConfig,
) -> EncodedDecision:
    player = int(obs_get(observation, "player", 0))
    planets = list(obs_get(observation, "planets", []))
    fleets = list(obs_get(observation, "fleets", []))
    step = int(obs_get(observation, "step", 0))
    episode_steps = int(obs_get(observation, "episode_steps", 500))

    planet_features = np.zeros((cfg.max_planets, PLANET_DIM), dtype=np.float32)
    planet_mask = np.zeros((cfg.max_planets,), dtype=bool)
    planet_by_id: dict[int, list[float]] = {}
    for idx, row in enumerate(planets[: cfg.max_planets]):
        pid, owner, x, y, radius, ships, production = row[:7]
        pid = int(pid)
        owner = int(owner)
        planet_by_id[pid] = row
        mine = owner == player
        neutral = owner == -1
        enemy = not mine and not neutral
        dx = float(x) - 50.0
        dy = float(y) - 50.0
        planet_features[idx] = np.asarray(
            [
                float(mine),
                float(enemy),
                float(neutral),
                float(x) / 100.0,
                float(y) / 100.0,
                float(radius) / 5.0,
                math.log1p(max(float(ships), 0.0)) / math.log(1001.0),
                float(production) / 5.0,
                math.hypot(dx, dy) / 70.71,
                math.sin(math.atan2(dy, dx)),
                math.cos(math.atan2(dy, dx)),
            ],
            dtype=np.float32,
        )
        planet_mask[idx] = True

    fleet_features = np.zeros((cfg.max_fleets, FLEET_DIM), dtype=np.float32)
    fleet_mask = np.zeros((cfg.max_fleets,), dtype=bool)
    for idx, row in enumerate(fleets[: cfg.max_fleets]):
        _, owner, x, y, angle, from_pid, ships = row[:7]
        from_row = planet_by_id.get(int(from_pid))
        source_mine = from_row is not None and int(from_row[1]) == player
        fleet_features[idx] = np.asarray(
            [
                float(int(owner) == player),
                float(int(owner) != player),
                float(x) / 100.0,
                float(y) / 100.0,
                math.sin(float(angle)),
                math.cos(float(angle)),
                math.log1p(max(float(ships), 0.0)) / math.log(1001.0),
                float(source_mine),
                math.hypot(float(x) - 50.0, float(y) - 50.0) / 70.71,
            ],
            dtype=np.float32,
        )
        fleet_mask[idx] = True

    mine = [p for p in planets if int(p[1]) == player]
    neutral = [p for p in planets if int(p[1]) == -1]
    enemy = [p for p in planets if int(p[1]) not in {-1, player}]
    my_fleets = [f for f in fleets if int(f[1]) == player]
    enemy_fleets = [f for f in fleets if int(f[1]) != player]
    my_ships = sum(float(p[5]) for p in mine)
    enemy_ships = sum(float(p[5]) for p in enemy)
    my_prod = sum(float(p[6]) for p in mine)
    enemy_prod = sum(float(p[6]) for p in enemy)
    global_features = np.asarray(
        [
            step / max(episode_steps, 1),
            len(mine) / max(cfg.max_planets, 1),
            len(enemy) / max(cfg.max_planets, 1),
            len(neutral) / max(cfg.max_planets, 1),
            math.log1p(my_ships) / math.log(10001.0),
            math.log1p(enemy_ships) / math.log(10001.0),
            my_prod / max(5.0 * cfg.max_planets, 1.0),
            enemy_prod / max(5.0 * cfg.max_planets, 1.0),
            len(my_fleets) / max(cfg.max_fleets, 1),
            len(enemy_fleets) / max(cfg.max_fleets, 1),
            math.log1p(sum(float(f[6]) for f in my_fleets)) / math.log(10001.0),
            math.log1p(sum(float(f[6]) for f in enemy_fleets)) / math.log(10001.0),
        ],
        dtype=np.float32,
    )

    plan_features = np.zeros(
        (variant_count, variant_count + PLAN_STATS_DIM),
        dtype=np.float32,
    )
    plan_mask = np.zeros((variant_count,), dtype=bool)
    for idx, proposal in enumerate(proposals):
        plan_features[idx, idx] = 1.0
        plan_features[idx, variant_count:] = plan_statistics(
            proposal.moves,
            planets,
            player,
            my_ships,
            my_prod,
        )
        plan_mask[idx] = proposal.valid
    plan_mask[0] = True
    return EncodedDecision(
        planet_features=planet_features,
        planet_mask=planet_mask,
        fleet_features=fleet_features,
        fleet_mask=fleet_mask,
        global_features=global_features,
        plan_features=plan_features,
        plan_mask=plan_mask,
    )


def plan_statistics(
    moves: list[list[float | int]],
    planets: list[list[float]],
    player: int,
    my_ships: float,
    my_prod: float,
) -> np.ndarray:
    if not moves:
        return np.zeros((PLAN_STATS_DIM,), dtype=np.float32)
    by_id = {int(p[0]): p for p in planets}
    sent = np.asarray([float(move[2]) for move in moves], dtype=np.float32)
    source_ids = [int(move[0]) for move in moves]
    angles = np.asarray([float(move[1]) for move in moves], dtype=np.float32)
    source_ships = []
    source_prod = []
    target_types = []
    target_prod = []
    distances = []
    for source_id, angle in zip(source_ids, angles):
        source = by_id.get(source_id)
        if source is None:
            continue
        source_ships.append(float(source[5]))
        source_prod.append(float(source[6]))
        target = infer_target(source, angle, planets)
        if target is None:
            continue
        owner = int(target[1])
        target_types.append(
            (
                float(owner == player),
                float(owner == -1),
                float(owner not in {-1, player}),
            )
        )
        target_prod.append(float(target[6]))
        distances.append(math.hypot(float(target[2]) - float(source[2]), float(target[3]) - float(source[3])))
    total_sent = float(sent.sum())
    source_ships_arr = np.asarray(source_ships or [1.0], dtype=np.float32)
    type_arr = np.asarray(target_types or [(0.0, 0.0, 0.0)], dtype=np.float32)
    return np.asarray(
        [
            len(moves) / max(len(planets), 1),
            total_sent / max(my_ships, 1.0),
            float(sent.mean()) / 400.0,
            float(sent.max()) / 400.0,
            len(set(source_ids)) / max(len(moves), 1),
            float(np.mean(source_ships_arr)) / 400.0,
            float(np.mean(source_prod or [0.0])) / 5.0,
            float(np.mean(np.sin(angles))),
            float(np.mean(np.cos(angles))),
            float(type_arr[:, 0].mean()),
            float(type_arr[:, 1].mean()),
            float(type_arr[:, 2].mean()),
            float(np.mean(target_prod or [0.0])) / 5.0,
            float(np.mean(distances or [0.0])) / 100.0,
            total_sent / max(my_prod * 10.0, 1.0),
            float(len(moves) > 1),
        ],
        dtype=np.float32,
    )


def infer_target(
    source: list[float],
    angle: float,
    planets: list[list[float]],
) -> list[float] | None:
    best = None
    best_score = float("inf")
    for target in planets:
        if int(target[0]) == int(source[0]):
            continue
        dx = float(target[2]) - float(source[2])
        dy = float(target[3]) - float(source[3])
        target_angle = math.atan2(dy, dx)
        delta = abs(math.atan2(math.sin(angle - target_angle), math.cos(angle - target_angle)))
        distance = math.hypot(dx, dy)
        score = delta + 0.0005 * distance
        if score < best_score:
            best_score = score
            best = target
    return best


class SetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.item = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.item(x)
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask_f).sum(dim=1) / denom
        neg_inf = torch.finfo(hidden.dtype).min
        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
        empty = ~mask.any(dim=1)
        max_pool[empty] = 0.0
        return self.out(torch.cat([mean, max_pool], dim=-1))


@dataclass(slots=True)
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


class WholePlanPolicy(nn.Module):
    def __init__(self, action_count: int, hidden_size: int = 192):
        super().__init__()
        self.action_count = action_count
        self.planet_encoder = SetEncoder(PLANET_DIM, hidden_size)
        self.fleet_encoder = SetEncoder(FLEET_DIM, hidden_size)
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.plan_encoder = nn.Sequential(
            nn.Linear(action_count + PLAN_STATS_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        planet_features: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_features: torch.Tensor,
        fleet_mask: torch.Tensor,
        global_features: torch.Tensor,
        plan_features: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> PolicyOutput:
        state = self.state_encoder(
            torch.cat(
                [
                    self.planet_encoder(planet_features, planet_mask),
                    self.fleet_encoder(fleet_features, fleet_mask),
                    self.global_encoder(global_features),
                ],
                dim=-1,
            )
        )
        plans = self.plan_encoder(plan_features)
        expanded = state.unsqueeze(1).expand(-1, self.action_count, -1)
        logits = self.policy_head(torch.cat([expanded, plans], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~plan_mask, torch.finfo(logits.dtype).min)
        value = self.value_head(state).squeeze(-1)
        return PolicyOutput(logits=logits, value=value)


def encoded_to_tensors(encoded: EncodedDecision, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planet_features": torch.from_numpy(encoded.planet_features).unsqueeze(0).to(device),
        "planet_mask": torch.from_numpy(encoded.planet_mask).unsqueeze(0).to(device),
        "fleet_features": torch.from_numpy(encoded.fleet_features).unsqueeze(0).to(device),
        "fleet_mask": torch.from_numpy(encoded.fleet_mask).unsqueeze(0).to(device),
        "global_features": torch.from_numpy(encoded.global_features).unsqueeze(0).to(device),
        "plan_features": torch.from_numpy(encoded.plan_features).unsqueeze(0).to(device),
        "plan_mask": torch.from_numpy(encoded.plan_mask).unsqueeze(0).to(device),
    }


def act(
    policy: WholePlanPolicy,
    encoded: EncodedDecision,
    device: torch.device,
    deterministic: bool,
) -> tuple[int, float, float]:
    with torch.inference_mode():
        output = policy(**encoded_to_tensors(encoded, device))
        dist = Categorical(logits=output.logits)
        action = output.logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(output.value.item())


def potential(observation: Any) -> float:
    player = int(obs_get(observation, "player", 0))
    planets = list(obs_get(observation, "planets", []))
    fleets = list(obs_get(observation, "fleets", []))
    mine = [p for p in planets if int(p[1]) == player]
    enemy = [p for p in planets if int(p[1]) not in {-1, player}]
    my_planets = len(mine)
    enemy_planets = len(enemy)
    my_prod = sum(float(p[6]) for p in mine)
    enemy_prod = sum(float(p[6]) for p in enemy)
    my_ships = sum(float(p[5]) for p in mine) + sum(float(f[6]) for f in fleets if int(f[1]) == player)
    enemy_ships = sum(float(p[5]) for p in enemy) + sum(float(f[6]) for f in fleets if int(f[1]) != player)
    scale = max(len(planets), 1)
    return (
        0.35 * (my_planets - enemy_planets) / scale
        + 0.45 * (my_prod - enemy_prod) / max(5.0 * scale, 1.0)
        + 0.20 * (my_ships - enemy_ships) / max(my_ships + enemy_ships, 100.0)
    )


class HeuristicOpponent:
    def __init__(self, heuristic_module: Any):
        self.h = heuristic_module
        self.memory = self.h.ProducerLiteMemory()

    def reset(self) -> None:
        self.memory = self.h.ProducerLiteMemory()

    def act(self, observation: Any) -> list[list[float | int]]:
        player = int(obs_get(observation, "player", 0))
        obs_tensors = self.h.single_obs_to_tensor(observation, player_id=player)
        player_count = int(obs_tensors["player_count"].item())
        config = self.h._config_for(player_count)
        with torch.inference_mode():
            row = self.h.run_turn(
                obs_tensors,
                config=config,
                player_count=player_count,
                memory=self.memory,
            )
        return self.h.sparse_action_row_to_moves(
            row,
            observation,
            player_id=player,
        )


class OrbitWarsWholePlanEnv:
    def __init__(
        self,
        cfg: TrainConfig,
        heuristic_module: Any,
        env_index: int,
        make_fn: Any | None = None,
    ):
        self.cfg = cfg
        self.h = heuristic_module
        self.env_index = env_index
        self.make_fn = make_fn
        self.library = WholePlanLibrary(heuristic_module)
        self.opponent = HeuristicOpponent(heuristic_module)
        self.env = None
        self.learner_player = 0
        self.episode_index = 0
        self.observation = None
        self.last_potential = 0.0

    def reset(self, seed: int) -> tuple[list[PlanProposal], EncodedDecision]:
        if self.make_fn is None:
            from kaggle_environments import make

            make_fn = make
        else:
            make_fn = self.make_fn
        self.learner_player = (
            (self.env_index + self.episode_index) % 2
            if self.cfg.alternate_sides
            else 0
        )
        self.episode_index += 1
        self.library.reset()
        self.opponent.reset()
        self.env = make_fn(
            "orbit_wars",
            configuration={"seed": int(seed), "randomSeed": int(seed)},
            debug=False,
        )
        self.env.reset(num_agents=2)
        states = self.env.step([[], []])
        self.observation = extract_observation(states[self.learner_player])
        self.last_potential = potential(self.observation)
        return self.current_decision()

    def current_decision(self) -> tuple[list[PlanProposal], EncodedDecision]:
        proposals = self.library.propose(self.observation)
        encoded = encode_decision(
            self.observation,
            proposals,
            self.library.action_count,
            self.cfg,
        )
        return proposals, encoded

    def step(
        self,
        proposal: PlanProposal,
    ) -> tuple[float, bool, list[PlanProposal] | None, EncodedDecision | None, dict[str, Any]]:
        self.library.commit(proposal)
        opponent_obs = extract_observation(
            self.env.steps[-1][1 - self.learner_player]
        )
        opponent_action = self.opponent.act(opponent_obs)
        actions = [None, None]
        actions[self.learner_player] = proposal.moves
        actions[1 - self.learner_player] = opponent_action
        states = self.env.step(actions)
        player_state = states[self.learner_player]
        opponent_state = states[1 - self.learner_player]
        self.observation = extract_observation(player_state)
        done = extract_status(player_state) != "ACTIVE"
        next_potential = potential(self.observation)
        shaped = self.cfg.shaping_coef * (
            self.cfg.gamma * next_potential - self.last_potential
        )
        self.last_potential = next_potential
        terminal = terminal_outcome(player_state, opponent_state) if done else 0.0
        reward = terminal + shaped
        info = {
            "terminal": terminal,
            "raw_reward": extract_reward(player_state),
            "opponent_raw_reward": extract_reward(opponent_state),
        }
        if done:
            return reward, True, None, None, info
        proposals, encoded = self.current_decision()
        return reward, False, proposals, encoded, info


def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    value = state.get("reward", 0.0) if isinstance(state, dict) else getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def terminal_outcome(player_state: Any, opponent_state: Any) -> float:
    player_reward = extract_reward(player_state)
    opponent_reward = extract_reward(opponent_state)
    if player_reward > opponent_reward:
        return 1.0
    if player_reward < opponent_reward:
        return -1.0
    return 0.0


def stack_batch(
    trajectories: list[list[Transition]],
    bootstrap_values: list[float],
    cfg: TrainConfig,
) -> Batch:
    flat: list[Transition] = []
    all_returns: list[float] = []
    all_advantages: list[float] = []
    for trajectory, bootstrap in zip(trajectories, bootstrap_values):
        advantages = [0.0] * len(trajectory)
        last_gae = 0.0
        next_value = bootstrap
        for idx in reversed(range(len(trajectory))):
            transition = trajectory[idx]
            nonterminal = 0.0 if transition.done else 1.0
            delta = (
                transition.reward
                + cfg.gamma * next_value * nonterminal
                - transition.value
            )
            last_gae = (
                delta
                + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
            )
            advantages[idx] = last_gae
            next_value = transition.value
        flat.extend(trajectory)
        all_advantages.extend(advantages)
        all_returns.extend(
            advantage + transition.value
            for advantage, transition in zip(advantages, trajectory)
        )
    return Batch(
        planet_features=torch.from_numpy(np.stack([t.encoded.planet_features for t in flat])),
        planet_mask=torch.from_numpy(np.stack([t.encoded.planet_mask for t in flat])),
        fleet_features=torch.from_numpy(np.stack([t.encoded.fleet_features for t in flat])),
        fleet_mask=torch.from_numpy(np.stack([t.encoded.fleet_mask for t in flat])),
        global_features=torch.from_numpy(np.stack([t.encoded.global_features for t in flat])),
        plan_features=torch.from_numpy(np.stack([t.encoded.plan_features for t in flat])),
        plan_mask=torch.from_numpy(np.stack([t.encoded.plan_mask for t in flat])),
        actions=torch.tensor([t.action for t in flat], dtype=torch.long),
        old_log_probs=torch.tensor([t.log_prob for t in flat], dtype=torch.float32),
        old_values=torch.tensor([t.value for t in flat], dtype=torch.float32),
        returns=torch.tensor(all_returns, dtype=torch.float32),
        advantages=torch.tensor(all_advantages, dtype=torch.float32),
    )


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y) < 1e-12:
        return float("nan")
    return float((1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y).cpu())


def ppo_update(
    policy: WholePlanPolicy,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    size = batch.actions.shape[0]
    metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
    }
    updates = 0
    for _ in range(cfg.ppo_epochs):
        order = torch.randperm(size)
        for start in range(0, size, cfg.minibatch_size):
            idx = order[start : start + cfg.minibatch_size]
            output = policy(
                batch.planet_features[idx].to(device),
                batch.planet_mask[idx].to(device),
                batch.fleet_features[idx].to(device),
                batch.fleet_mask[idx].to(device),
                batch.global_features[idx].to(device),
                batch.plan_features[idx].to(device),
                batch.plan_mask[idx].to(device),
            )
            dist = Categorical(logits=output.logits)
            new_log_prob = dist.log_prob(batch.actions[idx].to(device))
            entropy = dist.entropy().mean()
            old_log_prob = batch.old_log_probs[idx].to(device)
            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()
            adv = advantages[idx].to(device)
            policy_loss = torch.maximum(
                -adv * ratio,
                -adv * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
            ).mean()
            value_loss = 0.5 * (
                output.value - batch.returns[idx].to(device)
            ).pow(2).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy.detach().cpu())
            metrics["approx_kl"] += float(approx_kl.cpu())
            metrics["clipfrac"] += float(clipfrac.cpu())
            updates += 1
    metrics = {key: value / max(updates, 1) for key, value in metrics.items()}
    metrics["explained_variance"] = explained_variance(
        batch.old_values,
        batch.returns,
    )
    return metrics


def bootstrap_value(
    policy: WholePlanPolicy,
    encoded: EncodedDecision | None,
    device: torch.device,
) -> float:
    if encoded is None:
        return 0.0
    with torch.inference_mode():
        return float(policy(**encoded_to_tensors(encoded, device)).value.item())


def collect_rollout(
    envs: list[OrbitWarsWholePlanEnv],
    current: list[tuple[list[PlanProposal], EncodedDecision]],
    policy: WholePlanPolicy,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
) -> tuple[Batch, list[tuple[list[PlanProposal], EncodedDecision]], int, dict[str, float]]:
    trajectories: list[list[Transition]] = [[] for _ in envs]
    episode_outcomes: list[float] = []
    action_counts = np.zeros((envs[0].library.action_count,), dtype=np.int64)
    optional_pass_count = 0
    optional_decision_count = 0
    for _ in range(cfg.rollout_steps):
        for env_idx, env in enumerate(envs):
            proposals, encoded = current[env_idx]
            action, log_prob, value = act(policy, encoded, device, deterministic=False)
            action_counts[action] += 1
            if bool(encoded.plan_mask[1:].any()):
                optional_decision_count += 1
                optional_pass_count += int(action == 0)
            reward, done, next_proposals, next_encoded, info = env.step(proposals[action])
            trajectories[env_idx].append(
                Transition(
                    encoded=encoded,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                )
            )
            if done:
                episode_outcomes.append(float(info["terminal"]))
                next_seed += 1
                current[env_idx] = env.reset(next_seed)
            else:
                current[env_idx] = (next_proposals, next_encoded)
    bootstraps = [
        bootstrap_value(policy, encoded, device)
        for _, encoded in current
    ]
    batch = stack_batch(trajectories, bootstraps, cfg)
    stats = {
        "episodes": float(len(episode_outcomes)),
        "mean_outcome": float(np.mean(episode_outcomes)) if episode_outcomes else 0.0,
        "win_rate": float(np.mean(np.asarray(episode_outcomes) > 0.0)) if episode_outcomes else 0.0,
        "optional_pass_frac": float(
            optional_pass_count / max(optional_decision_count, 1)
        ),
        "optional_decisions": float(optional_decision_count),
    }
    for idx, count in enumerate(action_counts):
        stats[f"action_{idx}_frac"] = float(count / max(action_counts.sum(), 1))
    return batch, current, next_seed, stats


def evaluate(
    policy: WholePlanPolicy,
    cfg: TrainConfig,
    heuristic_module: Any,
    device: torch.device,
    games: int,
    seed_start: int,
) -> dict[str, float]:
    outcomes = []
    lengths = []
    for game_idx in range(games):
        env = OrbitWarsWholePlanEnv(cfg, heuristic_module, env_index=game_idx)
        proposals, encoded = env.reset(seed_start + game_idx)
        done = False
        steps = 0
        while not done:
            action, _, _ = act(policy, encoded, device, deterministic=True)
            reward, done, proposals_next, encoded_next, info = env.step(proposals[action])
            steps += 1
            if not done:
                proposals, encoded = proposals_next, encoded_next
        outcomes.append(float(info["terminal"]))
        lengths.append(steps)
    arr = np.asarray(outcomes)
    return {
        "eval_win_rate": float(np.mean(arr > 0.0)),
        "eval_loss_rate": float(np.mean(arr < 0.0)),
        "eval_draw_rate": float(np.mean(arr == 0.0)),
        "eval_mean_outcome": float(arr.mean()),
        "eval_mean_length": float(np.mean(lengths)),
    }


def train(
    cfg: TrainConfig,
    heuristic_path: str | Path,
) -> tuple[WholePlanPolicy, list[dict[str, float]]]:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    heuristic_module = load_heuristic_module(heuristic_path)
    action_library = WholePlanLibrary(heuristic_module)
    action_count = action_library.action_count
    action_names = action_library.names
    policy = WholePlanPolicy(action_count, cfg.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-4,
    )
    envs = [
        OrbitWarsWholePlanEnv(cfg, heuristic_module, env_index=idx)
        for idx in range(cfg.num_envs)
    ]
    next_seed = cfg.seed
    current = []
    for env in envs:
        current.append(env.reset(next_seed))
        next_seed += 1
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_eval = -float("inf")
    for update in range(1, cfg.total_updates + 1):
        batch, current, next_seed, rollout_stats = collect_rollout(
            envs,
            current,
            policy,
            cfg,
            device,
            next_seed,
        )
        metrics = ppo_update(policy, optimizer, batch, cfg, device)
        row = {"update": float(update), **rollout_stats, **metrics}
        if update % cfg.eval_every == 0 or update == 1:
            eval_stats = evaluate(
                policy,
                cfg,
                heuristic_module,
                device,
                cfg.eval_games,
                seed_start=100_000 + update * cfg.eval_games,
            )
            row.update(eval_stats)
            if eval_stats["eval_mean_outcome"] > best_eval:
                best_eval = eval_stats["eval_mean_outcome"]
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "config": dataclasses.asdict(cfg),
                        "action_names": action_names,
                        "update": update,
                    },
                    save_dir / "best.pt",
                )
        history.append(row)
        if update % cfg.checkpoint_every == 0 or update == cfg.total_updates:
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": dataclasses.asdict(cfg),
                    "action_names": action_names,
                    "update": update,
                    "history": history,
                },
                save_dir / "last.pt",
            )
        action_fractions = [
            row.get(f"action_{idx}_frac", 0.0)
            for idx in range(action_count)
        ]
        top_action_idx = int(np.argmax(action_fractions))
        progress = (
            f"update={update:04d} outcome={row['mean_outcome']:+.3f} "
            f"loss={row['loss']:.4f} ev={row['explained_variance']:+.3f} "
            f"kl={row['approx_kl']:.5f} clipfrac={row['clipfrac']:.3f} "
            f"entropy={row['entropy']:.3f} "
            f"pass={action_fractions[0]:.3f} "
            f"optional_pass={row['optional_pass_frac']:.3f} "
            f"baseline={action_fractions[1]:.3f} "
            f"top={action_names[top_action_idx]}:{action_fractions[top_action_idx]:.3f}"
        )
        if "eval_win_rate" in row:
            progress += (
                f" eval_w/l/d={row['eval_win_rate']:.3f}/"
                f"{row['eval_loss_rate']:.3f}/"
                f"{row['eval_draw_rate']:.3f}"
            )
        if (
            row["optional_decisions"] > 0
            and row["optional_pass_frac"] >= 0.75
        ):
            progress += " WARNING=pass_collapse"
        print(progress)
    return policy, history
