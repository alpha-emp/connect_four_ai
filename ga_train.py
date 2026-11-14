#!/usr/bin/env python3
"""Connect Four -- Genetic Algorithm weight trainer.

Evolves 4 evaluation weights [center, two-in-row, three-in-row, block-opponent-three]
by having each chromosome play games against a hand-tuned baseline using Alpha-Beta.
Best weights are saved to models/ga_weights.json for the web app to load.

Usage:
  python ga_train.py                                   # defaults: pop=16, gen=15, games=10, depth=2
  python ga_train.py --pop 20 --gen 20 --games 12 --depth 3
"""
import argparse, json, math, os, random, time

from c4.engine import (AI, PLAYER,
                       create_board, make_move, winning_move, is_draw,
                       get_valid_locations, score_position_w, _ab_weighted,
                       swap_perspective)

MODEL_DIR    = "models"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "ga_weights.json")

# Hand-tuned baseline weights (matches the original score_position heuristic exactly)
HAND_WEIGHTS = [6, 5, 10, 12]   # [center, two-in-row, three-in-row, block-opponent-three]
GENE_MIN, GENE_MAX = 0, 40


# ------------------------------------------------------------------ game helpers
def _weighted_move(board, piece, w, depth):
    """Best move for `piece` using Alpha-Beta with evaluation weights `w`."""
    b = board.copy() if piece == AI else swap_perspective(board)
    counter = [0]
    col, _ = _ab_weighted(b, depth, -math.inf, math.inf, True, counter, w)
    return col


def _play_game(w_a, w_b, depth, a_first, seed):
    """Play one full game. AI uses w_a, opponent uses w_b. Returns 'A', 'B', or 'Draw'."""
    board = create_board()
    rng   = random.Random(seed)
    turn  = AI if a_first else PLAYER

    # Reproducible random opening (4 moves) so every chromosome sees the same position
    for _ in range(4):
        valid = get_valid_locations(board)
        if not valid:
            break
        col = rng.choice(valid)
        make_move(board, col, turn)
        if winning_move(board, turn):
            return "A" if turn == AI else "B"
        turn = PLAYER if turn == AI else AI

    while not (winning_move(board, AI) or winning_move(board, PLAYER) or is_draw(board)):
        w   = w_a if turn == AI else w_b
        col = _weighted_move(board, turn, w, depth)
        if col is None:
            break
        make_move(board, col, turn)
        if winning_move(board, turn):
            return "A" if turn == AI else "B"
        turn = PLAYER if turn == AI else AI

    return "Draw"


def fitness(w, baseline, games, depth):
    """Win rate of chromosome `w` vs `baseline` over `games` reproducible games."""
    points = 0.0
    for g in range(games):
        result = _play_game(w, baseline, depth, a_first=(g % 2 == 0), seed=10000 + g)
        if result == "A":      points += 1.0
        elif result == "Draw": points += 0.5
    return points / games


# ------------------------------------------------------------------ GA operators
def _random_chromosome():
    return [random.randint(GENE_MIN, GENE_MAX) for _ in range(4)]


def _tournament_select(population, fits, k=3):
    picks = random.sample(range(len(population)), k)
    return population[max(picks, key=lambda i: fits[i])]


def _crossover(p1, p2):
    cut = random.randint(1, 3)
    return p1[:cut] + p2[cut:]


def _mutate(ch, rate):
    return [
        max(GENE_MIN, min(GENE_MAX, v + (random.randint(-5, 5) if random.random() < rate else 0)))
        for v in ch
    ]


# ------------------------------------------------------------------ main training loop
def train(pop_size=16, generations=15, games_per_fitness=10,
          depth=2, mutation_rate=0.25, seed=42):

    random.seed(seed)
    population   = [_random_chromosome() for _ in range(pop_size)]
    best_weights, best_fit = None, -1.0
    best_history, avg_history = [], []
    t0 = time.time()

    print(f"GA training: pop={pop_size}, gen={generations}, games={games_per_fitness}, depth={depth}")
    print(f"Baseline (hand-tuned) weights: {HAND_WEIGHTS}\n")

    for gen in range(generations):
        fits  = [fitness(ch, HAND_WEIGHTS, games_per_fitness, depth) for ch in population]
        order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        gen_best = fits[order[0]]

        if gen_best > best_fit:
            best_fit     = gen_best
            best_weights = population[order[0]][:]

        avg = sum(fits) / pop_size
        best_history.append(best_fit)
        avg_history.append(avg)

        print(f"Gen {gen+1:>2}/{generations} | "
              f"gen_best={gen_best:.3f} | best_so_far={best_fit:.3f} | "
              f"avg={avg:.3f} | weights={population[order[0]]}")

        # Elitism: carry top 2 unchanged
        new_pop = [population[order[0]][:], population[order[1]][:]]
        while len(new_pop) < pop_size:
            p1 = _tournament_select(population, fits)
            p2 = _tournament_select(population, fits)
            new_pop.append(_mutate(_crossover(p1, p2), mutation_rate))
        population = new_pop

    elapsed = round(time.time() - t0)
    print(f"\nGA finished in {elapsed}s")
    print(f"Best weights : {best_weights}  (fitness vs baseline: {best_fit:.3f})")
    print(f"Hand-tuned   : {HAND_WEIGHTS}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "weights":           best_weights,
        "fitness":           best_fit,
        "hand_weights":      HAND_WEIGHTS,
        "pop_size":          pop_size,
        "generations":       generations,
        "games_per_fitness": games_per_fitness,
        "search_depth":      depth,
        "best_history":      best_history,
        "avg_history":       avg_history,
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {WEIGHTS_PATH}")
    return best_weights


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GA weight trainer for Connect Four")
    p.add_argument("--pop",   type=int,   default=16,   help="Population size (default: 16)")
    p.add_argument("--gen",   type=int,   default=15,   help="Generations (default: 15)")
    p.add_argument("--games", type=int,   default=10,   help="Games per fitness eval (default: 10)")
    p.add_argument("--depth", type=int,   default=2,    help="AB search depth during training (default: 2)")
    p.add_argument("--mut",   type=float, default=0.25, help="Mutation rate (default: 0.25)")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed (default: 42)")
    args = p.parse_args()
    train(args.pop, args.gen, args.games, args.depth, args.mut, args.seed)
