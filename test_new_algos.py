"""Quick smoke-test for the two new algorithms: Iterative Deepening and Genetic."""
import numpy as np
from c4.engine import (create_board, make_move, get_move, ALGORITHMS, PLAYER, AI, winning_move)

print("ALGORITHMS:", ALGORITHMS)
assert "Iterative Deepening" in ALGORITHMS, "Missing Iterative Deepening"
assert "Genetic" in ALGORITHMS, "Missing Genetic"

# ── Test 1: empty board ───────────────────────────────────────────────────────
board = create_board()

print("\n--- Iterative Deepening (depth=4, empty board) ---")
res = get_move(board, AI, "Iterative Deepening", depth=4)
print(f"  column={res['column']}, score={res['score']}, nodes={res['nodes']}, time={res['time']:.3f}s")
assert res["column"] in range(7), "Column out of range"

print("\n--- Genetic Algorithm (empty board) ---")
res = get_move(board, AI, "Genetic")
print(f"  column={res['column']}, score={res['score']}, time={res['time']:.3f}s")
assert res["column"] in range(7), "Column out of range"

# ── Test 2: AI has 3-in-a-col, should complete the win ───────────────────────
board2 = create_board()
for _ in range(3):
    make_move(board2, 3, AI)       # 3 AI pieces stacked in col 3
make_move(board2, 0, PLAYER)
make_move(board2, 0, PLAYER)
make_move(board2, 0, PLAYER)

print("\n--- IDDLS near-win board (AI 3-in-col-3, expects col 3) ---")
res = get_move(board2, AI, "Iterative Deepening", depth=4)
print(f"  column={res['column']} (expected 3), score={res['score']}")
assert res["column"] == 3, f"IDDLS missed forced win: got col {res['column']}"

print("\n--- Genetic near-win board (stochastic, just check valid column) ---")
res = get_move(board2, AI, "Genetic")
print(f"  column={res['column']}, score={res['score']}")
assert res["column"] in range(7), f"Genetic returned invalid column: {res['column']}"

# ── Test 3: block player's immediate win ──────────────────────────────────────
board3 = create_board()
for _ in range(3):
    make_move(board3, 2, PLAYER)   # player about to win in col 2
make_move(board3, 6, AI)
make_move(board3, 6, AI)

print("\n--- IDDLS block player win (expects col 2) ---")
res = get_move(board3, AI, "Iterative Deepening", depth=4)
print(f"  column={res['column']} (expected 2)")
assert res["column"] == 2, f"IDDLS failed to block: got col {res['column']}"

print("\nAll tests passed!")
