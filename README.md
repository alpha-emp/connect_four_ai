# Connect Four AI

A Connect Four web app and AI playground built with Python and Flask. It includes several search-based agents, a genetic algorithm optimizer, and trained DQN models for playing against the computer or watching AIs battle each other.

## Features

- Human vs AI mode
- AI vs AI mode with start, pause, next-move, and reset controls
- Multiple agents:
  - Random
  - Minimax
  - Alpha-Beta
  - Alpha-Beta + Move Ordering
  - Alpha-Beta + Move Ordering + TT
  - Iterative Deepening
  - Genetic (Tuned) weights
  - MCTS
  - DQN and DQN v2
- Stateless Flask API backend
- Static HTML/JS frontend served from `public/`
- Optional training scripts for DQN and genetic tuning

## Repository layout

```text
app.py                 Flask API and local entry point
c4/
  __init__.py           Package marker
  engine.py             Core game logic, heuristics, and AI search algorithms
  dqn.py                NumPy-based DQN inference and feature preprocessing
public/index.html      Frontend UI (HTML/CSS/JS)
models/                Trained model weights and GA output
  ga_weights.json      Genetic algorithm tuned weights (created by ga_train.py)
  dqn_connect4.npz      DQN v1 model for the web app
  dqn_connect4_v2.npz  DQN v2 model for the web app
requirements.txt       Runtime dependencies
requirements-train.txt Training dependencies (adds PyTorch)
dqn_train.py           DQN training, evaluation, export, and play scripts
ga_train.py            Genetic algorithm tuning for Alpha-Beta weights
test_new_algos.py      Experimental algorithm tests
LICENSE                MIT license
README.md              Project documentation
```

## Requirements

Python 3.10+ is recommended.

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For DQN training, install the training stack:

```bash
pip install -r requirements-train.txt
```

## Run locally

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

The app serves the frontend at `/` and exposes the move API at `/api/ai_move` and health checks at `/api/health`.

## Agent overview

### Classical search agents

Implemented in `c4/engine.py`:

- Random
- Minimax
- Alpha-Beta pruning
- Alpha-Beta with move ordering
- Alpha-Beta with move ordering + transposition table
- Iterative deepening
- Genetic algorithm tuned heuristic weights
- MCTS

### Deep learning agents

The project includes DQN support for Connect Four:

- `DQN` (v1)
- `DQN v2` (with 4-in-a-row feature windows)

These models are loaded from `models/` and used by the Flask API for inference.

## Genetic algorithm tuning

The genetic algorithm trainer tunes the evaluation weights used by the weighted Alpha-Beta heuristic.

Run training:

```bash
python ga_train.py
```

Optional arguments:

```bash
python ga_train.py --pop 20 --gen 20 --games 12 --depth 3
```

This produces:

```text
models/ga_weights.json
```

The web app automatically loads these weights for the `Genetic (Tuned)` agent when available.

## DQN training and export

Train the original model:

```bash
python dqn_train.py train --minutes 110 --episodes 1000000
```

Train the v2 architecture:

```bash
python dqn_train.py train --arch v2 --minutes 110 --episodes 1000000
```

Resume training:

```bash
python dqn_train.py train --arch v2 --minutes 60 --episodes 1000000 --resume
```

Play against a trained model:

```bash
python dqn_train.py play --opponent human
python dqn_train.py play --arch v2 --opponent alphabeta --depth 3 --games 50
```

Export a trained checkpoint for web use:

```bash
python dqn_train.py export --arch v2 --model models/dqn_connect4_v2_best.pth
```

## Deployment

This project is designed to deploy to Vercel.

1. Ensure the project root contains `app.py`.
2. Commit the required model files in `models/`.
3. Push to GitHub.
4. Import the repository in Vercel and set the root directory to the folder containing `app.py`.
5. Deploy.

Vercel installs only `requirements.txt`, so heavy training dependencies like PyTorch are not needed in production.

## Health checks and troubleshooting

Open the health endpoint after deployment:

```text
https://YOUR-SITE.vercel.app/api/health
```

This reports:

- whether the Flask app booted correctly
- files present in the root and `c4/`
- model loading status
- import errors

Common problems:

- `API problem` or HTTP 404: the server did not start or the Vercel root directory is incorrect
- `import_error` is not null: required files are missing or outdated
- a DQN option is disabled: the matching `.npz` file was not committed to `models/`

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Notes

This repository mixes classical game-search approaches with machine-learning-based play, making it useful both as a game demo and as a research/experimentation playground for Connect Four strategies.
