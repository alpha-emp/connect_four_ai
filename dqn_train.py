#!/usr/bin/env python3
"""Connect Four DQN: learn, play and export.

  python dqn_train.py train --episodes 5000            # MODE 1: learns, saves the model
  python dqn_train.py train --episodes 5000 --resume   # keep learning from the saved model
  python dqn_train.py play --opponent human            # MODE 2: plays only (greedy, no learning)
  python dqn_train.py play --opponent alphabeta --depth 2 --games 20
  python dqn_train.py export                           # .pth -> models/dqn_connect4.npz (used by the website)

Files written to models/:
  dqn_connect4.pth       latest checkpoint (resume training from this)
  dqn_connect4_best.pth  best checkpoint by evaluation vs Random / Alpha-Beta
  dqn_connect4.npz       weights for the website (NumPy only, no torch at serving time)
"""
import argparse, copy, os, random, time
from collections import deque
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

from c4.engine import (ROWS, COLS, PLAYER, AI, create_board, get_valid_locations, make_move,
                       winning_move, is_draw, get_move, random_ai_move)
from c4.dqn import encode_state

MODEL_DIR = "models"
LATEST = f"{MODEL_DIR}/dqn_connect4.pth"
BEST = f"{MODEL_DIR}/dqn_connect4_best.pth"
WEB = f"{MODEL_DIR}/dqn_connect4.npz"

GAMMA, EPS_START, EPS_MIN, EPS_DECAY = 0.99, 1.0, 0.05, 0.9995   # epsilon decays per episode
BATCH, LR, TARGET_UPDATE, MEMORY = 64, 1e-3, 500, 50000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):                      # same architecture as the notebook
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(ROWS * COLS, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, COLS))

    def forward(self, x):
        return self.network(x)


def greedy_action(model, board, piece, epsilon=0.0):
    valid = get_valid_locations(board)
    if random.random() < epsilon:
        return random.choice(valid)
    x = torch.as_tensor(encode_state(board, piece), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q = model(x).squeeze(0).cpu().numpy()
    return max(valid, key=lambda c: q[c])


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path, trainer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model_state_dict": trainer.model.state_dict(),
                "target_state_dict": trainer.target.state_dict(),
                "optimizer_state_dict": trainer.opt.state_dict(),
                "training_steps": trainer.steps, "epsilon": trainer.eps, "episodes": trainer.episodes}, tmp)
    os.replace(tmp, path)                  # atomic: a crash never leaves a half-written model


def load_model(path):
    """Load weights only (for play/export). Uses weights_only=True, so no arbitrary pickle code runs."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = DQN().to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def export_npz(pth, npz=WEB):
    _, ckpt = load_model(pth)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    g = lambda k: sd[k].detach().cpu().numpy().astype(np.float32)
    np.savez(npz, w0=g("network.0.weight"), b0=g("network.0.bias"), w1=g("network.2.weight"),
             b1=g("network.2.bias"), w2=g("network.4.weight"), b2=g("network.4.bias"),
             training_steps=ckpt.get("training_steps", 0), epsilon=ckpt.get("epsilon", 0.0),
             episodes=ckpt.get("episodes", 0))
    print(f"Exported {pth} -> {npz}")


# ------------------------------------------------------------------ training
class Trainer:
    def __init__(self):
        self.model = DQN().to(DEVICE)
        self.target = copy.deepcopy(self.model).eval()
        self.opt = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=MEMORY)
        self.eps, self.steps, self.episodes = EPS_START, 0, 0

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.target.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.eps = ckpt.get("epsilon", EPS_START)
        self.steps, self.episodes = ckpt.get("training_steps", 0), ckpt.get("episodes", 0)

    def optimize(self):
        if len(self.memory) < BATCH:
            return
        s, a, r, ns, d = zip(*random.sample(self.memory, BATCH))
        t = lambda x, dt=torch.float32: torch.as_tensor(np.array(x), dtype=dt, device=DEVICE)
        s, ns, r, d, a = t(s), t(ns), t(r), t(d), t(a, torch.int64)
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(ns)
            nq[ns[:, (ROWS - 1) * COLS:] != 0] = -1e9          # top row occupied -> column is full
            y = r + GAMMA * nq.max(1).values * (1 - d)
        loss = self.loss_fn(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.model.state_dict())


def make_opponent(kind, trainer):
    if kind == "random":
        return lambda b, p: random_ai_move(b)
    if kind.startswith("alphabeta"):
        depth = int(kind[9:] or 1)
        return lambda b, p: get_move(b, p, "Alpha-Beta", depth)["column"]
    if kind == "self":
        return lambda b, p: greedy_action(trainer.model, b, p, 0.1)
    raise SystemExit(f"Unknown opponent '{kind}'")


def run_episode(tr, opponent):
    board, me = create_board(), random.choice([PLAYER, AI])
    other = AI if me == PLAYER else PLAYER
    if me == AI:                                   # opponent (Red) moves first
        make_move(board, opponent(board, other), other)
    while True:
        s = encode_state(board, me)
        a = greedy_action(tr.model, board, me, tr.eps)
        make_move(board, a, me)
        if winning_move(board, me): r, done = 1.0, True
        elif is_draw(board): r, done = 0.0, True
        else:
            make_move(board, opponent(board, other), other)
            if winning_move(board, other): r, done = -1.0, True
            elif is_draw(board): r, done = 0.0, True
            else: r, done = 0.0, False
        tr.memory.append((s, a, r, encode_state(board, me), float(done)))
        tr.optimize()
        if done:
            return r


def play_game(pick_red, pick_yellow, show=False):
    """Returns PLAYER, AI or 0 (draw)."""
    board, turn = create_board(), PLAYER
    while True:
        col = (pick_red if turn == PLAYER else pick_yellow)(board, turn)
        make_move(board, col, turn)
        if show: show_board(board)
        if winning_move(board, turn): return turn
        if is_draw(board): return 0
        turn = AI if turn == PLAYER else PLAYER


def evaluate(model, games=20):
    """Greedy model vs Random and Alpha-Beta(depth 1), colours alternating. Returns a score in [0, 2]."""
    dqn = lambda b, p: greedy_action(model, b, p)
    total = 0.0
    for name in ("Random", "Alpha-Beta"):
        opp = lambda b, p, n=name: get_move(b, p, n, 1)["column"]
        pts = 0.0
        for g in range(games):
            as_red = g % 2 == 0
            w = play_game(dqn if as_red else opp, opp if as_red else dqn)
            me = PLAYER if as_red else AI
            pts += 1.0 if w == me else 0.5 if w == 0 else 0.0
        print(f"   eval vs {name:<10}: {pts / games:.0%}", end="")
        total += pts / games
    print()
    return total


def cmd_train(a):
    tr = Trainer()
    if a.resume and os.path.exists(LATEST):
        tr.load(LATEST)
        print(f"Resumed from {LATEST} (episodes={tr.episodes}, steps={tr.steps}, epsilon={tr.eps:.3f})")
    best, t0 = -1.0, time.time()
    try:
        for i in range(1, a.episodes + 1):
            if a.minutes and time.time() - t0 > a.minutes * 60:
                print(f"Time limit of {a.minutes} minutes reached - stopping.")
                break
            kind = a.opponent if a.opponent != "mixed" else random.choices(
                ["random", "self", "alphabeta1", "alphabeta2"], [0.3, 0.4, 0.2, 0.1])[0]
            run_episode(tr, make_opponent(kind, tr))
            tr.eps, tr.episodes = max(EPS_MIN, tr.eps * EPS_DECAY), tr.episodes + 1
            if i % a.eval_every == 0:
                tr.model.eval()
                score = evaluate(tr.model, a.eval_games)
                tr.model.train()
                print(f"[{i}/{a.episodes}] eps={tr.eps:.3f} steps={tr.steps} score={score:.2f} ({time.time() - t0:.0f}s)")
                if score > best:
                    best = score
                    save_checkpoint(BEST, tr)
            if i % a.save_every == 0:
                save_checkpoint(LATEST, tr)
    except KeyboardInterrupt:
        print("\nInterrupted - saving progress...")
    save_checkpoint(LATEST, tr)
    export_npz(BEST if os.path.exists(BEST) else LATEST)
    print(f"Saved {LATEST}" + (f", {BEST}" if os.path.exists(BEST) else "") + f", {WEB}")


# ------------------------------------------------------------------ play (no learning)
def show_board(board):
    print("\n".join(" ".join(".RY"[v] for v in row) for row in board[::-1]))
    print(" ".join(str(c + 1) for c in range(COLS)) + "\n")


def human_pick(board, piece):
    valid = get_valid_locations(board)
    while True:
        try:
            col = int(input(f"Your move (column 1-{COLS}): ")) - 1
            if col in valid:
                return col
        except ValueError:
            pass
        print("Invalid or full column.")


def cmd_play(a):
    path = a.model or (BEST if os.path.exists(BEST) else LATEST)
    if not os.path.exists(path):
        raise SystemExit(f"No saved model at {path}. Train one first: python dqn_train.py train")
    model, ckpt = load_model(path)                 # eval mode, no optimizer, no weight updates
    print(f"Loaded {path} (episodes={ckpt.get('episodes', '?')}, steps={ckpt.get('training_steps', '?')})")
    dqn = lambda b, p: greedy_action(model, b, p, 0.0)
    if a.opponent == "human": opp = human_pick
    elif a.opponent == "random": opp = lambda b, p: random_ai_move(b)
    else: opp = lambda b, p: get_move(b, p, "Minimax" if a.opponent == "minimax" else "Alpha-Beta", a.depth)["column"]
    wins = losses = draws = 0
    for g in range(a.games):
        dqn_red = (g % 2 == 0) if a.opponent != "human" else (a.you == "yellow")
        if a.opponent == "human": print(f"--- Game {g + 1}: you are {'Yellow' if dqn_red else 'Red'} ---")
        w = play_game(dqn if dqn_red else opp, opp if dqn_red else dqn, show=a.opponent == "human" or a.show)
        me = PLAYER if dqn_red else AI
        wins, losses, draws = wins + (w == me), losses + (w not in (0, me)), draws + (w == 0)
        if a.opponent == "human":
            print("DQN wins!" if w == me else "Draw." if w == 0 else "You win!")
    print(f"DQN result vs {a.opponent}: {wins} wins, {losses} losses, {draws} draws")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="learning mode")
    t.add_argument("--episodes", type=int, default=5000)
    t.add_argument("--opponent", default="mixed", help="mixed | random | self | alphabeta1 | alphabeta2")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--minutes", type=float, default=None, help="stop training after this many minutes")
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--eval-games", type=int, default=20)
    t.add_argument("--save-every", type=int, default=250)
    pl = sub.add_parser("play", help="playing mode (no learning)")
    pl.add_argument("--opponent", default="human", choices=["human", "random", "minimax", "alphabeta"])
    pl.add_argument("--depth", type=int, default=2)
    pl.add_argument("--games", type=int, default=1)
    pl.add_argument("--you", default="red", choices=["red", "yellow"], help="your colour vs the DQN")
    pl.add_argument("--model", help="path to a .pth checkpoint")
    pl.add_argument("--show", action="store_true", help="print boards for bot-vs-bot games")
    e = sub.add_parser("export", help="convert a .pth checkpoint to the website's .npz")
    e.add_argument("--model", default=None)
    a = p.parse_args()
    if a.cmd == "train": cmd_train(a)
    elif a.cmd == "play": cmd_play(a)
    else: export_npz(a.model or (BEST if os.path.exists(BEST) else LATEST))


if __name__ == "__main__":
    main()
