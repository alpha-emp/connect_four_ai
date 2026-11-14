"""NumPy-only DQN inference (no torch needed, so it fits on Vercel).
Supports two saved models, detected from the weight shapes:
  v1: 42 inputs  -> 128 -> 128 -> 7
  v2: 222 inputs (board + 69 four-in-a-row windows + playable squares) -> 256 -> 256 -> 128 -> 7
Weights are stored as w0,b0,w1,b1,... in an .npz written by `python dqn_train.py train/export`."""
import re, time
import numpy as np
from .engine import ROWS, COLS, EMPTY, get_valid_locations


def encode_state(board, piece):
    """v1 input. Board from `piece`'s point of view: own = 1, opponent = -1, empty = 0."""
    s = np.zeros((ROWS, COLS), dtype=np.float32)
    s[board == piece] = 1.0
    s[(board != piece) & (board != EMPTY)] = -1.0
    return s.flatten()


def _windows():
    """All 69 four-in-a-row lines as flat cell indices (24 horizontal, 21 vertical, 12 + 12 diagonal)."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= y < ROWS and 0 <= x < COLS for y, x in cells):
                    out.append([y * COLS + x for y, x in cells])
    return np.array(out)

WINDOWS = _windows()                                   # shape (69, 4)
N_FEATURES_V2 = ROWS * COLS + 2 * len(WINDOWS) + ROWS * COLS      # 42 + 138 + 42 = 222


def encode_state_v2(board, piece):
    """v2 input: [board (own=1/opp=-1) | own pieces per window /4 | opp pieces per window /4 | playable squares].
    The first 42 values are the v1 encoding, so 'column is full' is still readable from the top row."""
    base = encode_state(board, piece)
    own, opp = (base == 1).astype(np.float32), (base == -1).astype(np.float32)
    playable = np.zeros(ROWS * COLS, dtype=np.float32)
    heights = (board != EMPTY).sum(axis=0)
    for c in range(COLS):
        if heights[c] < ROWS:
            playable[heights[c] * COLS + c] = 1.0      # the square a piece dropped in column c lands on
    return np.concatenate([base, own[WINDOWS].sum(1) / 4, opp[WINDOWS].sum(1) / 4, playable]).astype(np.float32)


ENCODERS = {ROWS * COLS: encode_state, N_FEATURES_V2: encode_state_v2}


class NumpyDQN:
    def __init__(self, params, meta=None):
        n = 0
        while f"w{n}" in params:
            n += 1
        if n < 2:
            raise ValueError("Model file has no usable weights (expected w0,b0,w1,b1,...)")
        prev = None
        for i in range(n):
            w, b = params[f"w{i}"], params.get(f"b{i}")
            if b is None or w.ndim != 2 or b.shape != (w.shape[0],) or (prev is not None and w.shape[1] != prev):
                raise ValueError(f"Bad shape for layer {i}")
            if not (np.isfinite(w).all() and np.isfinite(b).all()):
                raise ValueError(f"Layer {i} contains NaN/inf")
            prev = w.shape[0]
        n_in = params["w0"].shape[1]
        if prev != COLS or n_in not in ENCODERS:
            raise ValueError(f"Unsupported model (inputs={n_in}, outputs={prev})")
        self.layers = [(params[f"w{i}"], params[f"b{i}"]) for i in range(n)]
        self.encode = ENCODERS[n_in]
        self.version = "v2" if n_in == N_FEATURES_V2 else "v1"
        self.meta = dict(meta or {}, version=self.version)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k].astype(np.float32) for k in z.files if re.fullmatch(r"[wb]\d+", k)}
            meta = {k: float(z[k]) for k in ("training_steps", "epsilon", "episodes") if k in z.files}
        return cls(params, meta)

    def q_values(self, state):
        h = state
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return h


def dqn_move(model, board, piece):
    """Greedy move with full columns masked; same result format as engine.get_move."""
    t0 = time.perf_counter()
    valid = get_valid_locations(board)
    q = model.q_values(model.encode(board, piece))
    col = max(valid, key=lambda c: q[c])
    return {"column": int(col), "score": None, "nodes": None, "time": time.perf_counter() - t0,
            "q_values": [float(v) for v in q], "valid": valid}
