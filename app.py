"""Flask API (Vercel serverless function). Stateless: the browser sends the board, gets a move back."""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request, send_from_directory

_import_error = None
try:
    import numpy as np
    from c4.engine import ROWS, COLS, PLAYER, AI, get_valid_locations, winning_move, is_draw, get_move, ALGORITHMS
    from c4.dqn import NumpyDQN, dqn_move
except Exception as e:          # report the real cause through the API instead of a blank 500
    _import_error = f"{type(e).__name__}: {e}"

app = Flask(__name__)

PUBLIC          = os.path.join(ROOT, "public")
GA_WEIGHTS_FILE = os.path.join(ROOT, "models", "ga_weights.json")
GA_PATH         = GA_WEIGHTS_FILE
# Max search depth per algorithm (keeps pure-Python search within serverless time limits)
MAX_DEPTH = {"Minimax": 4, "Alpha-Beta": 5, "Alpha-Beta + Move Ordering": 6,
             "Alpha-Beta + Move Ordering + TT": 6, "Iterative Deepening": 6, "Genetic (Tuned)": 5}
MODEL_FILES = {"DQN": "dqn_connect4.npz", "DQN v2": "dqn_connect4_v2.npz"}
_models = {}                                         # name -> {"m": model or None, "error": str or None}
_ga    = {"w": None, "meta": None, "error": None}


def _ls(p):
    try: return sorted(os.listdir(p))[:40]
    except Exception as e: return str(e)


def get_model(name="DQN"):
    s = _models.setdefault(name, {"m": None, "error": None})
    path = os.path.join(ROOT, "models", MODEL_FILES[name])
    if s["m"] is None and os.path.isfile(path):
        try:
            s["m"] = NumpyDQN.load(path)
        except Exception as e:
            s["error"] = f"{type(e).__name__}: {e}"
            print(f"Could not load {name} model:", s["error"])
    return s["m"]


def get_ga_weights():
    if _ga["w"] is None and os.path.isfile(GA_PATH):
        try:
            import json
            with open(GA_PATH) as f:
                data = json.load(f)
            _ga["w"]    = data["weights"]
            _ga["meta"] = {k: data[k] for k in ("fitness", "pop_size", "generations",
                                                  "games_per_fitness", "search_depth")
                           if k in data}
        except Exception as e:
            _ga["error"] = f"{type(e).__name__}: {e}"
            print("Could not load GA weights:", _ga["error"])
    return _ga["w"]


@app.get("/api/health")
def health():
    return jsonify(python=sys.version, root=ROOT, root_files=_ls(ROOT), c4_files=_ls(os.path.join(ROOT, "c4")),
                   model_files=_ls(os.path.join(ROOT, "models")), import_error=_import_error,
                   dqn={n: {"loaded": get_model(n) is not None, "error": _models[n]["error"]} for n in MODEL_FILES})


@app.before_request
def _guard():
    if _import_error and request.path.startswith("/api/") and request.path != "/api/health":
        return jsonify(error="Server failed to start: " + _import_error), 500


@app.get("/api/agents")
def agents():
    m, m2 = get_model("DQN"), get_model("DQN v2")
    gw = get_ga_weights()
    return jsonify([
        {"id": "Random", "label": "Random", "depth": None, "available": True},
        {"id": "Minimax", "label": "Minimax", "depth": MAX_DEPTH["Minimax"], "available": True},
        {"id": "Alpha-Beta", "label": "Alpha-Beta", "depth": MAX_DEPTH["Alpha-Beta"], "available": True},
        {"id": "Alpha-Beta + Move Ordering", "label": "Alpha-Beta + Move Ordering",
         "depth": MAX_DEPTH["Alpha-Beta + Move Ordering"], "available": True},
        {"id": "Alpha-Beta + Move Ordering + TT", "label": "Alpha-Beta + Move Ordering + TT",
         "depth": MAX_DEPTH["Alpha-Beta + Move Ordering + TT"], "available": True},
        {"id": "Iterative Deepening", "label": "Iterative Deepening (IDDLS)",
         "depth": MAX_DEPTH["Iterative Deepening"], "available": True},
        {"id": "Genetic (Tuned)", "label": "Genetic (Tuned weights)",
         "depth": MAX_DEPTH["Genetic (Tuned)"], "available": gw is not None,
         "info": _ga["meta"], "error": _ga["error"]},
        {"id": "DQN", "label": "Trained DQN", "depth": None, "available": m is not None,
         "info": m.meta if m else None},
        {"id": "DQN v2", "label": "Trained DQN v2 (4-in-a-row features)", "depth": None, "available": m2 is not None,
         "info": m2.meta if m2 else None},
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
        if agent in MODEL_FILES:
            m = get_model(agent)
            if m is None:
                return jsonify(error=f"No trained '{agent}' model file found in models/."), 503
            res = dqn_move(m, board, piece)
        elif agent == "Genetic (Tuned)":
            w = get_ga_weights()
            if w is None:
                return jsonify(error="No GA weights found. Run: python ga_train.py"), 503
            res = get_move(board, piece, agent, max(1, min(depth, MAX_DEPTH["Genetic (Tuned)"])), weights=w)
        elif agent in ALGORITHMS:
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
