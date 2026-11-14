# Connect Four AI

Two modes on the website: **Human vs AI** (pick one AI) and **AI vs AI** (pick two).
AIs: Random, Minimax, Alpha-Beta, and a trained DQN.

## Layout
- `c4/engine.py` game logic, heuristic, Random / Minimax / Alpha-Beta
- `c4/dqn.py` NumPy-only DQN inference (no torch on the server)
- `dqn_train.py` DQN **learn** mode, **play** mode and `.pth` -> `.npz` export
- `api/index.py` Flask API (Vercel serverless function)
- `public/index.html` frontend

## Run locally
    pip install -r requirements.txt
    python api/index.py        # open http://localhost:5000

## DQN: learn mode, play mode, saved model
    pip install -r requirements-train.txt
    python dqn_train.py train --episodes 5000            # learns and saves to models/
    python dqn_train.py train --episodes 5000 --resume   # continue from the saved model
    python dqn_train.py play --opponent human            # play mode: greedy, no learning
    python dqn_train.py play --opponent alphabeta --depth 2 --games 20

Training writes `models/dqn_connect4.pth` (latest), `models/dqn_connect4_best.pth` (best by evaluation)
and `models/dqn_connect4.npz` (what the website loads). Commit the `.npz` so Vercel can serve the DQN.
Trained in Colab already? `python dqn_train.py export --model your.pth` creates the `.npz`.

## Deploy on Vercel
Push to GitHub, import the repo in Vercel, deploy. No build settings needed.
Search depth is capped (Minimax 4, Alpha-Beta 5) to stay within serverless time limits.
