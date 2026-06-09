# Paste-ready notebook snippets for the compact 23-dim feature experiment.
#
# There are TWO cells below.
#
# Cell A goes immediately after current notebook Cell 8, before Cell 9 full data
# collection. It overrides encode_shot() from 24 dims to 23 dims.
#
# Cell B goes immediately after current notebook Cell 18, before Cell 20 sanity
# check. It patches generated main.py/main_topk*.py so submission-time validator
# inference uses the same 23-dim feature vector.


# =========================
# CELL A: 23-dim train-time encoder
# Inject after Cell 8 and before Cell 9.
# =========================

COMPACT_EXTRA_FEATURE_NAMES = [
    "target_prod_share",
    "my_ship_share",
    "send_to_target_ships",
    "capture_margin_growth",
]

REMOVED_BASE_FEATURE_NAMES = {
    "tgt_is_self",
    "speed",
    "my_ships_total",
    "my_planets",
    "enemy_planets",
}

BASE_FEATURE_NAMES_24 = [
    "src_ships", "src_prod", "src_radius",
    "tgt_ships", "tgt_prod", "tgt_radius",
    "tgt_is_self", "tgt_is_neutral", "tgt_is_enemy",
    "ships_sent", "ship_frac", "dist", "eta", "speed",
    "ally_fleet_n", "ally_fleet_ships", "enemy_fleet_n", "enemy_fleet_ships",
    "turn", "my_ships_total", "enemy_ships_total", "ship_delta",
    "my_planets", "enemy_planets",
]
KEPT_BASE_FEATURE_NAMES = [
    name for name in BASE_FEATURE_NAMES_24
    if name not in REMOVED_BASE_FEATURE_NAMES
]
FEATURE_NAMES = KEPT_BASE_FEATURE_NAMES + COMPACT_EXTRA_FEATURE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)

def encode_shot(obs, src_id, target_id, ships_sent):
    """23 dims: 19 retained baseline features + 4 cheap extra features."""
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict:
        return None

    src = pdict[src_id]
    tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]

    # One planet pass computes all global totals needed by retained/extra features.
    my_ships_total = 0.0
    enemy_ships_total = 0.0
    my_prod_total = 0.0
    enemy_prod_total = 0.0
    for p in planets:
        owner = int(p[1])
        ships = float(p[5])
        prod = float(p[6])
        if owner == me:
            my_ships_total += ships
            my_prod_total += prod
        elif owner >= 0:
            enemy_ships_total += ships
            enemy_prod_total += prod

    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(math.hypot(dx, dy) - sr - tr, 0.0)
    speed = fleet_speed(ships_sent)
    eta = dist / max(speed, 0.5)

    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    ship_frac = ships_sent / max(sships, 1)

    # One fleet pass replaces four separate scans.
    ally_n = 0
    ally_s = 0.0
    enemy_n = 0
    enemy_s = 0.0
    for f in fleets:
        if int(f[1]) == me:
            ally_n += 1
            ally_s += float(f[6])
        else:
            enemy_n += 1
            enemy_s += float(f[6])
    turn = int(obs.get("step", 0))

    total_ships = my_ships_total + enemy_ships_total
    total_prod = my_prod_total + enemy_prod_total
    target_growth_until_eta = 0.0 if int(tgt[1]) == me else tprod * max(eta, 0.0)

    extra = [
        tprod / max(total_prod, 1.0),
        my_ships_total / max(total_ships, 1.0),
        ships_sent / max(tships + 1.0, 1.0),
        (ships_sent - (tships + target_growth_until_eta)) / 100.0,
    ]

    values = np.array(
        [
            sships / 100.0,
            sprod / 5.0,
            sr / 4.0,
            tships / 100.0,
            tprod / 5.0,
            tr / 4.0,
            own_neutral,
            own_enemy,
            ships_sent / 100.0,
            ship_frac,
            dist / BOARD,
            eta / 60.0,
            ally_n / 10.0,
            ally_s / 100.0,
            enemy_n / 10.0,
            enemy_s / 100.0,
            turn / 500.0,
            enemy_ships_total / 200.0,
            (my_ships_total - enemy_ships_total) / 200.0,
            *extra,
        ],
        dtype=np.float32,
    )
    assert values.shape[0] == FEATURE_DIM, (values.shape, FEATURE_DIM)
    return values


print(f"Using compact feature set: {len(BASE_FEATURE_NAMES_24)} -> {FEATURE_DIM} dims")
print("Removed:", ", ".join(sorted(REMOVED_BASE_FEATURE_NAMES)))
print("Added:", ", ".join(COMPACT_EXTRA_FEATURE_NAMES))


# =========================
# CELL B: 23-dim submission-wrapper patch
# Inject after Cell 18 and before Cell 20.
# =========================

COMPACT_WRAPPER_FEATURE_BLOCK = r'''
_FEATURE_DIM_H = 23

def _encode_shot_h(obs, src_id, target_id, ships_sent):
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict: return None
    src = pdict[src_id]; tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_t = 0.0; en_t = 0.0; my_prod = 0.0; en_prod = 0.0
    for p in planets:
        owner = int(p[1]); ships = float(p[5]); prod = float(p[6])
        if owner == me:
            my_t += ships; my_prod += prod
        elif owner >= 0:
            en_t += ships; en_prod += prod
    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(_math_h.hypot(dx, dy) - sr - tr, 0.0)
    speed = _fleet_speed_h(ships_sent); eta = dist / max(speed, 0.5)
    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    sf = ships_sent / max(sships, 1)
    an = 0; a_s = 0.0; en = 0; e_s = 0.0
    for f in fleets:
        if int(f[1]) == me:
            an += 1; a_s += float(f[6])
        else:
            en += 1; e_s += float(f[6])
    turn = int(obs.get("step", 0))
    total_ships = my_t + en_t
    total_prod = my_prod + en_prod
    target_growth = 0.0 if int(tgt[1]) == me else tprod * max(eta, 0.0)
    values = _np_h.array([
        sships/100.0, sprod/5.0, sr/4.0,
        tships/100.0, tprod/5.0, tr/4.0,
        own_neutral, own_enemy,
        ships_sent/100.0, sf,
        dist/_BOARD_H, eta/60.0,
        an/10.0, a_s/100.0, en/10.0, e_s/100.0,
        turn/500.0, en_t/200.0,
        (my_t - en_t)/200.0,
        tprod / max(total_prod, 1.0),
        my_t / max(total_ships, 1.0),
        ships_sent / max(tships + 1.0, 1.0),
        (ships_sent - (tships + target_growth)) / 100.0,
    ], dtype=_np_h.float32)
    if values.shape[0] != _FEATURE_DIM_H:
        raise ValueError(
            f"validator encoder produced {values.shape[0]} features; "
            f"expected {_FEATURE_DIM_H}"
        )
    return values
'''


def patch_compact_wrapper_features(agent_path):
    agent_path = Path(agent_path)
    text = agent_path.read_text()
    start = text.index("def _encode_shot_h(")
    end = text.index("\ndef agent(", start)
    patched = text[:start] + COMPACT_WRAPPER_FEATURE_BLOCK.strip() + text[end:]
    agent_path.write_text(patched)
    print(f"patched compact 23-dim wrapper features: {agent_path}")


patch_compact_wrapper_features(FINAL_AGENT_PATH)
for _path in TOPK_AGENT_PATHS.values():
    patch_compact_wrapper_features(_path)

with np.load(WORK / "weights.npz") as _check_weights:
    WEIGHTS_FEATURE_DIM = int(_check_weights["l0_w"].shape[1])

assert FEATURE_DIM == 23, f"train-time FEATURE_DIM is {FEATURE_DIM}, expected 23"
assert WEIGHTS_FEATURE_DIM == FEATURE_DIM, (
    f"weights.npz expects {WEIGHTS_FEATURE_DIM} features but the encoder uses "
    f"{FEATURE_DIM}. Rerun model creation, training, and weight export after Cell A."
)

print(
    "Wrapper patch complete:",
    f"train={FEATURE_DIM}, weights={WEIGHTS_FEATURE_DIM}, wrapper=23",
)
