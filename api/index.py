"""Flask API (Vercel serverless function). Stateless: the browser sends the board, gets a move back."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from c4.engine import ROWS, COLS, PLAYER, AI, get_valid_locations, winning_move, is_draw, get_move
from c4.dqn import NumpyDQN, dqn_move

app = Flask(__name__)
PUBLIC = os.path.join(ROOT, "public")
MODEL_PATH = os.path.join(ROOT, "models", "dqn_connect4.npz")
MAX_DEPTH = {"Minimax": 4, "Alpha-Beta": 5}          # keeps pure-Python search within serverless time limits
_model = {"m": None}


def get_model():
    if _model["m"] is None and os.path.isfile(MODEL_PATH):
        try:
            _model["m"] = NumpyDQN.load(MODEL_PATH)
        except Exception as e:
            print("Could not load DQN model:", e)
    return _model["m"]


@app.get("/api/agents")
def agents():
    m = get_model()
    return jsonify([
        {"id": "Random", "label": "Random", "depth": None, "available": True},
        {"id": "Minimax", "label": "Minimax", "depth": MAX_DEPTH["Minimax"], "available": True},
        {"id": "Alpha-Beta", "label": "Alpha-Beta", "depth": MAX_DEPTH["Alpha-Beta"], "available": True},
        {"id": "DQN", "label": "Trained DQN", "depth": None, "available": m is not None,
         "info": m.meta if m else None},
    ])


@app.post("/api/ai_move")
def ai_move():
    d = request.get_json(silent=True) or {}
    try:
        board = np.array(d["board"], dtype=int)
        piece, agent = int(d["piece"]), str(d["agent"])
        depth = int(d.get("depth") or 3)
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Invalid request."), 400
    if board.shape != (ROWS, COLS) or not np.isin(board, (0, 1, 2)).all() or piece not in (PLAYER, AI):
        return jsonify(error="Invalid board or piece."), 400
    if winning_move(board, PLAYER) or winning_move(board, AI) or is_draw(board):
        return jsonify(error="The game is already over."), 400
    try:
        if agent == "DQN":
            m = get_model()
            if m is None:
                return jsonify(error="No trained DQN model found. Run: python dqn_train.py train"), 503
            res = dqn_move(m, board, piece)
        elif agent in ("Random", "Minimax", "Alpha-Beta"):
            res = get_move(board, piece, agent, max(1, min(depth, MAX_DEPTH.get(agent, depth))))
        else:
            return jsonify(error="Unknown agent."), 400
    except Exception as e:
        return jsonify(error=f"AI error: {e}"), 500
    if res["column"] not in get_valid_locations(board):
        return jsonify(error="AI returned an illegal column."), 500
    res.update(agent=agent, depth=depth if agent in MAX_DEPTH else None)
    return jsonify(res)


# Local development only (on Vercel, files in public/ are served directly).
@app.get("/")
def index():
    return send_from_directory(PUBLIC, "index.html")

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
