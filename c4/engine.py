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

def score_position_w(board, piece, w):
    """Weighted version of score_position with learnable coefficients w = [center, two, three, block].
    When w = [6, 5, 10, 12] the result is identical to score_position."""
    center_w, two_w, three_w, block_w = w
    opp = PLAYER if piece == AI else AI
    score = list(board[:, COLS // 2]).count(piece) * center_w
    for r in range(ROWS):
        for c in range(COLS - 3):
            window = list(board[r, c:c + 4])
            mine, theirs, empty = window.count(piece), window.count(opp), window.count(EMPTY)
            if mine == 4:                      score += 100
            elif mine == 3 and empty == 1:     score += three_w
            elif mine == 2 and empty == 2:     score += two_w
            if theirs == 3 and empty == 1:     score -= block_w
    for c in range(COLS):
        for r in range(ROWS - 3):
            window = list(board[r:r + 4, c])
            mine, theirs, empty = window.count(piece), window.count(opp), window.count(EMPTY)
            if mine == 4:                      score += 100
            elif mine == 3 and empty == 1:     score += three_w
            elif mine == 2 and empty == 2:     score += two_w
            if theirs == 3 and empty == 1:     score -= block_w
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            mine, theirs, empty = window.count(piece), window.count(opp), window.count(EMPTY)
            if mine == 4:                      score += 100
            elif mine == 3 and empty == 1:     score += three_w
            elif mine == 2 and empty == 2:     score += two_w
            if theirs == 3 and empty == 1:     score -= block_w
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r - i][c + i] for i in range(4)]
            mine, theirs, empty = window.count(piece), window.count(opp), window.count(EMPTY)
            if mine == 4:                      score += 100
            elif mine == 3 and empty == 1:     score += three_w
            elif mine == 2 and empty == 2:     score += two_w
            if theirs == 3 and empty == 1:     score -= block_w
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

# ---- Alpha-Beta variants: move ordering, and move ordering + transposition table ----
CENTER_ORDER = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))      # [3, 2, 4, 1, 5, 0, 6]
EXACT, LOWER, UPPER = 0, 1, 2

def _search_ordered(board, depth, alpha, beta, maximizing, counter, tt):
    """Alpha-Beta trying centre columns first. If `tt` (a dict) is given, also uses a transposition table."""
    counter[0] += 1
    alpha0, key, tt_col = alpha, None, None
    if tt is not None:
        key = (board.tobytes(), depth, maximizing)
        hit = tt.get(key)
        if hit:
            val, flag, tt_col = hit
            if flag == EXACT: return tt_col, val
            if flag == LOWER: alpha = max(alpha, val)
            else: beta = min(beta, val)
            if alpha >= beta: return tt_col, val
    valid = get_valid_locations(board)
    if winning_move(board, AI): return None, WIN_SCORE
    if winning_move(board, PLAYER): return None, -WIN_SCORE
    if not valid: return None, 0
    if depth == 0: return None, score_position(board, AI)
    order = [c for c in CENTER_ORDER if c in valid]
    if tt_col in order:                                   # best move from an earlier visit goes first
        order.remove(tt_col); order.insert(0, tt_col)
    piece = AI if maximizing else PLAYER
    value, best = (-math.inf if maximizing else math.inf), order[0]
    for col in order:
        child = board.copy()
        drop_piece(child, get_next_open_row(child, col), col, piece)
        s = _search_ordered(child, depth - 1, alpha, beta, not maximizing, counter, tt)[1]
        if maximizing and s > value: value, best = s, col
        if not maximizing and s < value: value, best = s, col
        if maximizing: alpha = max(alpha, value)
        else: beta = min(beta, value)
        if alpha >= beta: break
    if tt is not None:
        tt[key] = (value, UPPER if value <= alpha0 else LOWER if value >= beta else EXACT, best)
    return best, value


# ---- Iterative Deepening Alpha-Beta (IDDLS) ----
def _iddls(board, max_depth, counter):
    """Iterative Deepening: runs Alpha-Beta + Move Ordering + TT from depth 1 up to max_depth,
    keeping the best move from the deepest completed iteration."""
    best_col = random.choice(get_valid_locations(board))
    best_score = -math.inf
    for d in range(1, max_depth + 1):
        col, score = _search_ordered(board, d, -math.inf, math.inf, True, counter, {})
        if col is not None:
            best_col, best_score = col, score
        if score >= WIN_SCORE:          # forced win found — no need to go deeper
            break
    return best_col, best_score


# ---- Alpha-Beta with learnable weights (used by the Genetic (Tuned) agent) ----
def _ab_weighted(board, depth, alpha, beta, maximizing, counter, w):
    """Alpha-Beta with move ordering and weighted evaluation. `w` = [center, two, three, block]."""
    counter[0] += 1
    valid = get_valid_locations(board)
    if winning_move(board, AI):     return None, WIN_SCORE
    if winning_move(board, PLAYER): return None, -WIN_SCORE
    if not valid:                   return None, 0
    if depth == 0:                  return None, score_position_w(board, AI, w)
    order = [c for c in CENTER_ORDER if c in valid]
    piece = AI if maximizing else PLAYER
    value, best = (-math.inf if maximizing else math.inf), order[0]
    for col in order:
        child = board.copy()
        drop_piece(child, get_next_open_row(child, col), col, piece)
        s = _ab_weighted(child, depth - 1, alpha, beta, not maximizing, counter, w)[1]
        if maximizing and s > value: value, best = s, col
        if not maximizing and s < value: value, best = s, col
        if maximizing: alpha = max(alpha, value)
        else:          beta  = min(beta, value)
        if alpha >= beta: break
    return best, value


ALGORITHMS = ("Random", "Minimax", "Alpha-Beta", "Alpha-Beta + Move Ordering",
              "Alpha-Beta + Move Ordering + TT", "Iterative Deepening", "Genetic (Tuned)")

def get_move(board, piece, algorithm, depth=4, weights=None):
    """Move for either colour. Returns dict(column, score, nodes, time).
    `weights` is required when algorithm == 'Genetic (Tuned)'."""
    b = board.copy() if piece == AI else swap_perspective(board)
    counter, t0 = [0], time.perf_counter()
    if algorithm == "Random":
        col, score = random_ai_move(b), score_position(b, AI)
    elif algorithm == "Minimax":
        col, score = _search(b, depth, -math.inf, math.inf, True, counter, False)
    elif algorithm == "Alpha-Beta":
        col, score = _search(b, depth, -math.inf, math.inf, True, counter, True)
    elif algorithm == "Alpha-Beta + Move Ordering":
        col, score = _search_ordered(b, depth, -math.inf, math.inf, True, counter, None)
    elif algorithm == "Alpha-Beta + Move Ordering + TT":
        col, score = _search_ordered(b, depth, -math.inf, math.inf, True, counter, {})
    elif algorithm == "Iterative Deepening":
        col, score = _iddls(b, depth, counter)
    elif algorithm == "Genetic (Tuned)":
        if weights is None:
            raise ValueError("Genetic (Tuned) requires pre-trained weights. Run: python ga_train.py")
        col, score = _ab_weighted(b, depth, -math.inf, math.inf, True, counter, weights)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return {"column": col, "score": score, "nodes": counter[0], "time": time.perf_counter() - t0}
