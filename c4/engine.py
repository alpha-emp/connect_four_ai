"""Game logic, heuristic and search AIs (ported from the notebook, no plotting/UI)."""
import math, random, time
import numpy as np

ROWS, COLS = 6, 7
EMPTY, PLAYER, AI = 0, 1, 2          # PLAYER = Red (moves first), AI = Yellow
WIN_SCORE = 100000


def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def get_valid_locations(board):
    return [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]

def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def make_move(board, col, piece):
    if col not in get_valid_locations(board):
        return False
    drop_piece(board, get_next_open_row(board, col), col, piece)
    return True

def find_win_cells(board, piece):
    """Return the 4 (row, col) cells of a winning line for `piece`, or []."""
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == piece for rr, cc in cells):
                    return cells
    return []

def winning_move(board, piece):
    return bool(find_win_cells(board, piece))

def is_draw(board):
    return not get_valid_locations(board)


# ---------------- heuristic ----------------
def evaluate_window(window, piece):
    opp = PLAYER if piece == AI else AI
    score = 0
    if window.count(piece) == 4: score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1: score += 10
    elif window.count(piece) == 2 and window.count(EMPTY) == 2: score += 5
    if window.count(opp) == 3 and window.count(EMPTY) == 1: score -= 12
    return score

def score_position(board, piece):
    score = list(board[:, COLS // 2]).count(piece) * 6
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += evaluate_window(list(board[r, c:c + 4]), piece)
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += evaluate_window(list(board[r:r + 4, c]), piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += evaluate_window([board[r + i][c + i] for i in range(4)], piece)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            score += evaluate_window([board[r - i][c + i] for i in range(4)], piece)
    return score


# ---------------- search AIs ----------------
def random_ai_move(board):
    valid = get_valid_locations(board)
    return random.choice(valid) if valid else None

def swap_perspective(board):
    swapped = board.copy()
    swapped[board == AI] = PLAYER
    swapped[board == PLAYER] = AI
    return swapped

def _search(board, depth, alpha, beta, maximizing, counter, prune):
    """Minimax (prune=False) / Alpha-Beta (prune=True). The searching side is always AI."""
    counter[0] += 1
    valid = get_valid_locations(board)
    if winning_move(board, AI): return None, WIN_SCORE
    if winning_move(board, PLAYER): return None, -WIN_SCORE
    if not valid: return None, 0
    if depth == 0: return None, score_position(board, AI)
    piece = AI if maximizing else PLAYER
    value, best = (-math.inf if maximizing else math.inf), random.choice(valid)
    for col in valid:
        child = board.copy()
        drop_piece(child, get_next_open_row(child, col), col, piece)
        s = _search(child, depth - 1, alpha, beta, not maximizing, counter, prune)[1]
        if maximizing and s > value: value, best = s, col
        if not maximizing and s < value: value, best = s, col
        if prune:
            if maximizing: alpha = max(alpha, value)
            else: beta = min(beta, value)
            if alpha >= beta: break
    return best, value

def get_move(board, piece, algorithm, depth=4):
    """Move for either colour. Returns dict(column, score, nodes, time)."""
    b = board.copy() if piece == AI else swap_perspective(board)
    counter, t0 = [0], time.perf_counter()
    if algorithm == "Random":
        col, score = random_ai_move(b), score_position(b, AI)
    elif algorithm == "Minimax":
        col, score = _search(b, depth, -math.inf, math.inf, True, counter, False)
    elif algorithm == "Alpha-Beta":
        col, score = _search(b, depth, -math.inf, math.inf, True, counter, True)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return {"column": col, "score": score, "nodes": counter[0], "time": time.perf_counter() - t0}
