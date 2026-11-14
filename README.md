# Connect Four AI

A Connect Four website with two modes:

- **Human vs AI**: you pick one AI to play against.
- **AI vs AI**: you pick two AIs and watch them play (Start, Pause, Next move, Reset).

Available AIs: Random, Minimax, Alpha-Beta, Alpha-Beta + Move Ordering, Alpha-Beta + Move Ordering + TT (transposition table), and two trained DQN (reinforcement learning) models, **DQN** and **DQN v2**.

## Project layout

```
app.py                 Flask API (also the Vercel entry point)
dqn_train.py           DQN training, playing and export (needs PyTorch)
c4/engine.py           Game rules, heuristic, Random / Minimax / Alpha-Beta variants
c4/dqn.py              NumPy-only DQN inference + feature encoders (no PyTorch on the server)
public/index.html      Frontend (HTML, CSS and JavaScript in one file)
models/                Trained models (.pth for training, .npz for the website)
requirements.txt       Website dependencies (flask, numpy)
requirements-train.txt Training dependencies (adds torch)
```

The server is stateless: the browser sends the board and the chosen AI, and the API returns a column.

## Run the website locally

```
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

## The two DQN models

| | DQN (v1) | DQN v2 |
|---|---|---|
| Inputs | 42 (the board) | 222 (board + 69 four-in-a-row windows + playable squares) |
| Hidden layers | 128, 128 | 256, 256, 128 |
| Website file | `models/dqn_connect4.npz` | `models/dqn_connect4_v2.npz` |

The v2 inputs give the network the state of every possible four-in-a-row line, so it doesn't have to work out diagonals itself. The two models use separate files and never overwrite each other.

## Train the DQN (learning mode)

```
pip install -r requirements-train.txt

python dqn_train.py train --minutes 110 --episodes 1000000               # v1
python dqn_train.py train --arch v2 --minutes 110 --episodes 1000000     # v2
python dqn_train.py train --arch v2 --minutes 60 --episodes 1000000 --resume   # continue training
```

- `--minutes` stops training after that time. Ctrl+C also saves safely.
- Training opponents (default `--opponent mixed`): 30% self-play, 20% Alpha-Beta+MO+TT depth 1, 25% depth 2, 15% depth 3, 10% random. A single opponent can be chosen, for example `--opponent abtt3`.
- Every `--eval-every` episodes (default 500) the model plays test games against Random and Alpha-Beta depth 1 and 2. The score is out of 3.00, and the best model is kept.

Files written to `models/` (v2 files have a `_v2` suffix):

- `dqn_connect4.pth`: latest checkpoint, used by `--resume`
- `dqn_connect4_best.pth`: best checkpoint by evaluation
- `dqn_connect4.npz`: weights the website loads

## Play or test the DQN (play mode, no learning)

```
python dqn_train.py play --opponent human                                         # you vs v1
python dqn_train.py play --arch v2 --opponent alphabeta --depth 3 --games 50      # v2 vs Alpha-Beta depth 3
```

`--opponent` can be `human`, `random`, `minimax` or `alphabeta`. Colours alternate between games. To convert an existing checkpoint for the website: `python dqn_train.py export --arch v2 --model path/to/file.pth`.

## Deploy on Vercel

1. Train a model and commit the `.npz` file(s) in `models/`.
2. Push the repo to GitHub.
3. In Vercel, choose **Add New, then Project**, and import the repo. The Root Directory must be the folder that contains `app.py`. Leave the build settings empty.
4. Deploy. Each `git push` redeploys automatically.

Vercel installs only `requirements.txt`, so PyTorch is not needed on the server. Search depth is capped (Minimax 4, Alpha-Beta 5, the two faster variants 6) to stay within serverless time limits.

## Troubleshooting

Open `https://YOUR-SITE.vercel.app/api/health`. It shows whether the server started, which files were deployed, and whether each DQN model loaded.

- **"API problem" banner or HTTP 404:** the API isn't running. Check that `app.py` is in the repo root and the Vercel Root Directory is correct.
- **`import_error` is not null:** a file is missing or outdated on GitHub. Check that `c4/__init__.py`, `c4/engine.py` and `c4/dqn.py` are committed with the latest code.
- **A DQN option is greyed out:** its `.npz` file isn't in `models/` on GitHub, or `dqn_error` in `/api/health` explains why it failed to load.
- **Changes don't appear after a push:** check that the newest deployment matches your latest commit, then redeploy without the build cache.
