"""NumPy-only DQN inference (no torch needed, so it fits on Vercel).
Weights come from models/dqn_connect4.npz, written by `python dqn_train.py train` or `export`."""
import time
import numpy as np
from .engine import ROWS, COLS, EMPTY, get_valid_locations

KEYS = ("w0", "b0", "w1", "b1", "w2", "b2")
SHAPES = {"w0": (128, ROWS * COLS), "b0": (128,), "w1": (128, 128), "b1": (128,), "w2": (COLS, 128), "b2": (COLS,)}


def encode_state(board, piece):
    """Board from `piece`'s point of view: own = 1, opponent = -1, empty = 0."""
    s = np.zeros((ROWS, COLS), dtype=np.float32)
    s[board == piece] = 1.0
    s[(board != piece) & (board != EMPTY)] = -1.0
    return s.flatten()


class NumpyDQN:
    def __init__(self, params, meta=None):
        for k in KEYS:
            if k not in params or tuple(params[k].shape) != SHAPES[k]:
                raise ValueError(f"Bad or missing weight '{k}' (expected shape {SHAPES[k]})")
            if not np.isfinite(params[k]).all():
                raise ValueError(f"Weight '{k}' contains NaN/inf")
        self.p, self.meta = params, meta or {}

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k].astype(np.float32) for k in KEYS if k in z}
            meta = {k: float(z[k]) for k in ("training_steps", "epsilon", "episodes") if k in z}
        return cls(params, meta)

    def q_values(self, state):
        p = self.p
        h = np.maximum(state @ p["w0"].T + p["b0"], 0)
        h = np.maximum(h @ p["w1"].T + p["b1"], 0)
        return h @ p["w2"].T + p["b2"]


def dqn_move(model, board, piece):
    """Greedy move with full columns masked; same result format as engine.get_move."""
    t0 = time.perf_counter()
    valid = get_valid_locations(board)
    q = model.q_values(encode_state(board, piece))
    col = max(valid, key=lambda c: q[c])
    return {"column": int(col), "score": None, "nodes": None, "time": time.perf_counter() - t0,
            "q_values": [float(v) for v in q], "valid": valid}
