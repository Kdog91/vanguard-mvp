"""
Deep learning with TensorFlow / Keras (the CS-370 stack), for customers' own data:

  1. Tabular neural network  - dense layers, dropout, optimizer and learning rate you choose,
                               early stopping, compared with classical baselines, fairness by group
  2. Image classifier (CNN)  - Conv2D + MaxPooling + Dropout + data augmentation, from a zip of
                               folders (one folder per class) or the built-in handwritten digits
  3. Deep Q-learning (DQN)   - an agent that learns a maze (treasure hunt) or Gymnasium CartPole
                               with epsilon-greedy exploration, experience replay and a discount factor
  4. Export                  - trained model + preprocessing + a FastAPI service + Dockerfile

TensorFlow is imported only when needed, so the rest of the app starts quickly.
"""
import io
import json
import os
import random
import time
import zipfile

import numpy as np
import pandas as pd

SEED = 42


def tf_available() -> tuple[bool, str]:
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf  # noqa: F401
        return True, tf.__version__
    except Exception as e:  # not installed or no wheel for this Python
        return False, str(e).split("\n")[0]


def _seed():
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def _optimizer(name: str, lr: float):
    import keras
    return {"Adam": keras.optimizers.Adam, "SGD": keras.optimizers.SGD,
            "RMSprop": keras.optimizers.RMSprop}[name](learning_rate=lr)


# =================================================================== 1. tabular
class TabularPrep:
    """Turns a DataFrame into model inputs with plain numbers, so the serving script needs no scikit-learn."""

    def __init__(self, num_cols, cat_cols, date_cols):
        self.num_cols, self.cat_cols, self.date_cols = list(num_cols), list(cat_cols), list(date_cols)
        self.means, self.stds, self.cats = {}, {}, {}

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self.date_cols:
            d = pd.to_datetime(df[c], errors="coerce")
            out[f"{c}__month"] = d.dt.month
            out[f"{c}__dow"] = d.dt.dayofweek
        return out

    def fit(self, df: pd.DataFrame):
        f = self._frame(df)
        for c in f.columns:
            self.means[c] = float(f[c].mean()) if f[c].notna().any() else 0.0
            sd = float(f[c].std()) if f[c].notna().sum() > 1 else 1.0
            self.stds[c] = sd if sd and np.isfinite(sd) else 1.0
        for c in self.cat_cols:
            self.cats[c] = [str(v) for v in df[c].astype(str).value_counts().head(30).index]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        f = self._frame(df)
        parts = []
        for c in f.columns:
            parts.append(((f[c].fillna(self.means[c]) - self.means[c]) / self.stds[c]).to_numpy()[:, None])
        for c in self.cat_cols:
            s = df[c].astype(str)
            parts.append(np.stack([(s == v).to_numpy() for v in self.cats[c]], axis=1).astype("float32"))
        return np.hstack(parts).astype("float32") if parts else np.zeros((len(df), 0), "float32")

    @property
    def feature_names(self) -> list:
        names = list(self.means)
        for c in self.cat_cols:
            names += [f"{c}={v}" for v in self.cats[c]]
        return names

    def to_json(self) -> dict:
        return {"num_cols": self.num_cols, "cat_cols": self.cat_cols, "date_cols": self.date_cols,
                "means": self.means, "stds": self.stds, "cats": self.cats}


def build_dense(n_in: int, n_out: int, task: str, layers: list, dropout: float, activation: str,
                optimizer: str, lr: float):
    import keras
    m = keras.Sequential([keras.Input(shape=(n_in,))])
    for units in layers:
        m.add(keras.layers.Dense(units, activation=activation))
        if dropout:
            m.add(keras.layers.Dropout(dropout))
    if task == "binary":
        m.add(keras.layers.Dense(1, activation="sigmoid"))
        loss, metrics = "binary_crossentropy", ["accuracy"]
    elif task == "multiclass":
        m.add(keras.layers.Dense(n_out, activation="softmax"))
        loss, metrics = "sparse_categorical_crossentropy", ["accuracy"]
    else:
        m.add(keras.layers.Dense(1))
        loss, metrics = "mse", ["mae"]
    m.compile(optimizer=_optimizer(optimizer, lr), loss=loss, metrics=metrics)
    return m


class _Progress:
    """Keras callback that reports each epoch to a function (used for live charts)."""

    def __new__(cls, fn):
        import keras

        class CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                fn(epoch + 1, dict(logs or {}))
        return CB()


def train_tabular(df: pd.DataFrame, target: str, features: list, profile: dict, layers: list, dropout: float,
                  activation: str, optimizer: str, lr: float, epochs: int, batch: int, patience: int,
                  max_rows: int = 50_000, on_epoch=None) -> dict:
    import keras
    from sklearn.model_selection import train_test_split
    _seed()
    d = df[features + [target]].dropna(subset=[target])
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=SEED)
    y_raw = d[target]
    kind = profile.get(target)
    classes = None
    if kind == "categorical" or (y_raw.nunique() <= 10 and not pd.api.types.is_float_dtype(y_raw)):
        classes = sorted(y_raw.astype(str).unique())
        if len(classes) < 2:
            raise ValueError("The target has only one value.")
        y = y_raw.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
        task = "binary" if len(classes) == 2 else "multiclass"
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype="float64")
        task = "regression"
    prep = TabularPrep([c for c in features if profile.get(c) == "numeric"],
                       [c for c in features if profile.get(c) == "categorical"],
                       [c for c in features if profile.get(c) == "datetime"])
    strat = y if task != "regression" and np.bincount(y.astype(int)).min() >= 3 else None
    i_tr, i_te = train_test_split(np.arange(len(d)), test_size=0.15, random_state=SEED, stratify=strat)
    i_tr, i_va = train_test_split(i_tr, test_size=0.1765, random_state=SEED,   # 15% of the whole for validation
                                  stratify=strat[i_tr] if strat is not None else None)
    prep.fit(d.iloc[i_tr])
    X = prep.transform(d)
    if X.shape[1] == 0:
        raise ValueError("No usable input columns (numbers, categories or dates).")
    y_mean, y_std = 0.0, 1.0
    if task == "regression":
        y_mean, y_std = float(y[i_tr].mean()), float(y[i_tr].std() or 1.0)
    yt = (y - y_mean) / y_std if task == "regression" else y
    model = build_dense(X.shape[1], len(classes or []), task, layers, dropout, activation, optimizer, lr)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if on_epoch:
        cbs.append(_Progress(on_epoch))
    t0 = time.time()
    hist = model.fit(X[i_tr], yt[i_tr], validation_data=(X[i_va], yt[i_va]), epochs=epochs, batch_size=batch,
                     callbacks=cbs, verbose=0)
    secs = time.time() - t0
    pred = model.predict(X[i_te], verbose=0)
    out = {"model": model, "prep": prep, "task": task, "classes": classes, "y_mean": y_mean, "y_std": y_std,
           "history": pd.DataFrame(hist.history).assign(epoch=lambda h: np.arange(1, len(h) + 1)),
           "seconds": secs, "n_train": len(i_tr), "n_val": len(i_va), "n_test": len(i_te),
           "params": int(model.count_params()), "test_index": d.index[i_te]}
    out["metrics"], out["test_pred"] = _score(task, y[i_te], pred, y_mean, y_std)
    out["baselines"] = _baselines(task, X[i_tr], y[i_tr], X[i_te], y[i_te])
    out["y_test"] = y[i_te]
    return out


def _score(task, y_true, pred, y_mean, y_std):
    from sklearn import metrics as M
    if task == "regression":
        p = pred.ravel() * y_std + y_mean
        return ({"RMSE": float(np.sqrt(M.mean_squared_error(y_true, p))), "MAE": float(M.mean_absolute_error(y_true, p)),
                 "R²": float(M.r2_score(y_true, p))}, p)
    p = (pred.ravel() >= 0.5).astype(int) if task == "binary" else pred.argmax(axis=1)
    avg = "binary" if task == "binary" else "macro"
    out = {"Accuracy": float(M.accuracy_score(y_true, p)),
           "Balanced accuracy": float(M.balanced_accuracy_score(y_true, p)),
           "F1": float(M.f1_score(y_true, p, average=avg, zero_division=0))}
    if task == "binary":
        out["ROC AUC"] = float(M.roc_auc_score(y_true, pred.ravel())) if len(set(y_true)) == 2 else float("nan")
    return out, p


def _baselines(task, Xtr, ytr, Xte, yte) -> pd.DataFrame:
    """Classical models on the same inputs, so the neural network has something honest to beat."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    rows = []
    if task == "regression":
        models = {"Guess the average": DummyRegressor(), "Linear regression": LinearRegression(),
                  "Random forest": RandomForestRegressor(200, random_state=SEED, n_jobs=-1)}
    else:
        models = {"Guess the most common class": DummyClassifier(strategy="most_frequent"),
                  "Logistic regression": LogisticRegression(max_iter=2000),
                  "Random forest": RandomForestClassifier(200, random_state=SEED, n_jobs=-1)}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        if task == "regression":
            sc, _ = _score(task, yte, m.predict(Xte), 0.0, 1.0)
        elif task == "binary":
            sc, _ = _score(task, yte, m.predict_proba(Xte)[:, 1], 0, 1)
        else:
            sc, _ = _score(task, yte, m.predict_proba(Xte), 0, 1)
        rows.append({"Model": name, **sc})
    return pd.DataFrame(rows)


def fairness(group: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, task: str, positive: int = 1) -> pd.DataFrame:
    """Results by group (CS-370 bias analysis): accuracy, how often the model says 'yes', recall,
    and the ratio of each group's 'yes' rate to the highest group's (below 0.8 fails the four-fifths rule)."""
    g = pd.Series(group.values).astype(str)
    rows = []
    for name in g.value_counts().index:
        m = (g == name).to_numpy()
        if m.sum() < 5:
            continue
        if task == "regression":
            err = y_pred[m] - y_true[m]
            rows.append({"Group": name, "Rows": int(m.sum()), "MAE": float(np.abs(err).mean()),
                         "Average error (bias)": float(err.mean())})
        else:
            yt, yp = y_true[m], y_pred[m]
            pos = yt == positive
            rows.append({"Group": name, "Rows": int(m.sum()), "Accuracy": float((yt == yp).mean()),
                         "Predicted positive rate": float((yp == positive).mean()),
                         "Recall (true positives found)": float((yp[pos] == positive).mean()) if pos.any() else np.nan})
    out = pd.DataFrame(rows)
    if task != "regression" and not out.empty:
        top = out["Predicted positive rate"].max()
        out["Ratio to highest rate"] = (out["Predicted positive rate"] / top).round(3) if top else np.nan
        out["Four-fifths rule"] = np.where(out["Rows"] < 30, "too few rows to judge",
                                           np.where(out["Ratio to highest rate"] >= 0.8, "pass", "check"))
    return out.round(4)


# =================================================================== 2. images (CNN)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def load_digits_demo():
    """scikit-learn's 1,797 handwritten digits (8x8 grayscale), an offline stand-in for MNIST."""
    from sklearn.datasets import load_digits
    d = load_digits()
    X = (d.images / 16.0).astype("float32")[..., None]
    return X, d.target.astype(int), [str(i) for i in range(10)]


def load_image_zip(data: bytes, size: int = 32, gray: bool = False, max_per_class: int = 2000):
    """A zip with one folder per class (cats/..., dogs/...). Returns X (n, size, size, ch) in 0-1, y, class names."""
    from PIL import Image
    z = zipfile.ZipFile(io.BytesIO(data))
    by_class = {}
    for name in z.namelist():
        if name.endswith("/") or not name.lower().endswith(IMG_EXT) or "__MACOSX" in name:
            continue
        parts = [p for p in name.split("/") if p]
        if len(parts) < 2:
            continue
        by_class.setdefault(parts[-2], []).append(name)
    if len(by_class) < 2:
        raise ValueError("The zip needs at least two folders, one per class, each holding images.")
    classes = sorted(by_class)
    X, y, skipped = [], [], 0
    for i, c in enumerate(classes):
        for name in by_class[c][:max_per_class]:
            try:
                img = Image.open(io.BytesIO(z.read(name))).convert("L" if gray else "RGB").resize((size, size))
                X.append(np.asarray(img, dtype="float32") / 255.0)
                y.append(i)
            except Exception:
                skipped += 1
    X = np.stack(X)
    if gray:
        X = X[..., None]
    return X, np.array(y), classes, skipped


def build_cnn(shape, n_classes: int, blocks: int, filters: int, dense: int, dropout: float, augment: bool,
              optimizer: str, lr: float):
    import keras
    L = keras.layers
    m = keras.Sequential([keras.Input(shape=shape)])
    if augment:
        m.add(L.RandomFlip("horizontal"))
        m.add(L.RandomRotation(0.08))
        m.add(L.RandomZoom(0.1))
    f, side = filters, shape[0]
    for _ in range(blocks):
        m.add(L.Conv2D(f, 3, padding="same", activation="relu"))
        m.add(L.Conv2D(f, 3, padding="same", activation="relu"))
        if side >= 2:
            m.add(L.MaxPooling2D())
            side //= 2
        if dropout:
            m.add(L.Dropout(dropout))
        f *= 2
    m.add(L.Flatten())
    m.add(L.Dense(dense, activation="relu"))
    if dropout:
        m.add(L.Dropout(dropout))
    m.add(L.Dense(n_classes, activation="softmax"))
    m.compile(optimizer=_optimizer(optimizer, lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train_cnn(X, y, classes, blocks, filters, dense, dropout, augment, optimizer, lr, epochs, batch, patience,
              on_epoch=None) -> dict:
    import keras
    from sklearn.model_selection import train_test_split
    _seed()
    strat = y if np.bincount(y).min() >= 3 else None
    i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED, stratify=strat)
    i_tr, i_va = train_test_split(i_tr, test_size=0.1765, random_state=SEED,
                                  stratify=y[i_tr] if strat is not None else None)
    model = build_cnn(X.shape[1:], len(classes), blocks, filters, dense, dropout, augment, optimizer, lr)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if on_epoch:
        cbs.append(_Progress(on_epoch))
    t0 = time.time()
    hist = model.fit(X[i_tr], y[i_tr], validation_data=(X[i_va], y[i_va]), epochs=epochs, batch_size=batch,
                     callbacks=cbs, verbose=0)
    pred = model.predict(X[i_te], verbose=0)
    p = pred.argmax(axis=1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y[i_te], p, labels=list(range(len(classes))))
    per = pd.DataFrame({"Class": classes, "Test images": cm.sum(axis=1),
                        "Correct (%)": np.round(np.diag(cm) / np.maximum(cm.sum(axis=1), 1) * 100, 1)})
    return {"model": model, "classes": classes, "history": pd.DataFrame(hist.history).assign(
                epoch=lambda h: np.arange(1, len(h) + 1)),
            "accuracy": float((p == y[i_te]).mean()), "chance": 1 / len(classes),
            "majority": float(np.bincount(y[i_te]).max() / len(i_te)),
            "confusion": pd.DataFrame(cm, index=classes, columns=classes), "per_class": per,
            "test_X": X[i_te], "test_y": y[i_te], "test_p": p, "test_conf": pred.max(axis=1),
            "seconds": time.time() - t0, "params": int(model.count_params()),
            "n": (len(i_tr), len(i_va), len(i_te))}


# =================================================================== 3. deep Q-learning
DEFAULT_MAZE = """1 0 1 1 1 1 1 1
1 0 1 1 1 0 1 1
1 1 1 1 0 1 0 1
1 1 1 0 1 1 1 1
1 1 0 1 0 0 0 0
1 1 1 0 1 1 1 1
1 1 1 1 1 0 1 1
1 0 1 1 1 1 1 1"""
SMALL_MAZE = """1 1 1 0 1
1 0 1 1 1
1 1 1 0 1
0 1 0 1 1
1 1 1 1 1"""
ACTIONS = ["left", "up", "right", "down"]
MOVES = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}


def parse_maze(text: str) -> np.ndarray:
    rows = [[float(v) for v in line.replace(",", " ").split()] for line in text.strip().splitlines() if line.strip()]
    m = np.array(rows, dtype="float32")
    if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] < 3:
        raise ValueError("The maze must be a grid of at least 3 × 3 (1 = open, 0 = wall).")
    if m[-1, -1] != 1 or m.sum() < 3:
        raise ValueError("The bottom-right cell is the treasure and must be open (1).")
    return m


class TreasureMaze:
    """The CS-370 pirate maze: start on any open cell, find the treasure at the bottom-right.
    Rewards: +1 treasure, -0.04 per move, -0.25 for revisiting, -0.75 for hitting a wall or edge.
    A game is lost when the total reward drops below -0.5 x the number of cells."""

    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.free = [tuple(c) for c in np.argwhere(maze == 1) if tuple(c) != (maze.shape[0] - 1, maze.shape[1] - 1)]
        self.target = (maze.shape[0] - 1, maze.shape[1] - 1)
        self.min_reward = -0.5 * maze.size

    @property
    def n_states(self):
        return self.maze.size

    def reset(self, start=None):
        self.pos = start or random.choice(self.free)
        self.visited, self.total = {self.pos}, 0.0
        return self.observe()

    def observe(self):
        s = self.maze.copy()
        s[self.pos] = 0.5
        return s.reshape(1, -1)

    def valid(self, pos=None):
        r, c = pos or self.pos
        out = []
        for a, (dr, dc) in MOVES.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.maze.shape[0] and 0 <= nc < self.maze.shape[1] and self.maze[nr, nc] == 1:
                out.append(a)
        return out

    def step(self, a):
        if a in self.valid():
            dr, dc = MOVES[a]
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)
            reward = 1.0 if self.pos == self.target else (-0.25 if self.pos in self.visited else -0.04)
        else:
            reward = -0.75
        self.visited.add(self.pos)
        self.total += reward
        status = "win" if self.pos == self.target else ("lose" if self.total < self.min_reward else "playing")
        return self.observe(), reward, status


def build_q_net(n_in: int, n_actions: int, units: int, lr: float):
    import keras
    m = keras.Sequential([keras.Input(shape=(n_in,)), keras.layers.Dense(units), keras.layers.PReLU(),
                          keras.layers.Dense(units), keras.layers.PReLU(), keras.layers.Dense(n_actions)])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m


def _q(model, s):
    return model(s, training=False).numpy()[0]


def train_maze(maze: np.ndarray, episodes: int, epsilon: float, eps_min: float, eps_decay: float, gamma: float,
               lr: float, memory_size: int, batch: int, time_limit: float, on_episode=None) -> dict:
    """Deep Q-learning with experience replay (the training loop from Project Two)."""
    _seed()
    env = TreasureMaze(maze)
    model = build_q_net(env.n_states, 4, env.n_states, lr)
    memory, rows, history_wins = [], [], []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        s = env.reset()
        steps, status = 0, "playing"
        while status == "playing":
            a = random.randrange(4) if random.random() < epsilon else int(np.argmax(_q(model, s)))
            s2, r, status = env.step(a)
            memory.append((s, a, r, s2, status != "playing"))
            if len(memory) > memory_size:
                memory.pop(0)
            s = s2
            steps += 1
            if len(memory) >= batch:
                sample = random.sample(memory, batch)
                S = np.vstack([m[0] for m in sample])
                S2 = np.vstack([m[3] for m in sample])
                targets = model(S, training=False).numpy()
                nxt = model(S2, training=False).numpy().max(axis=1)
                for i, (_, a_, r_, _, done) in enumerate(sample):
                    targets[i, a_] = r_ if done else r_ + gamma * nxt[i]
                model.train_on_batch(S, targets)
        history_wins.append(1 if status == "win" else 0)
        epsilon = max(eps_min, epsilon * eps_decay)
        win_rate = float(np.mean(history_wins[-20:]))
        rows.append({"Episode": ep, "Result": status, "Steps": steps, "Win rate (last 20)": win_rate,
                     "Epsilon": round(epsilon, 3)})
        if on_episode:
            on_episode(ep, rows[-1])
        if time.time() - t0 > time_limit:
            break
        if ep % 10 == 0 and win_rate == 1.0 and completion_check(model, env)[0] == 1.0:
            break
    done_rate, results = completion_check(model, env)
    return {"model": model, "env": env, "log": pd.DataFrame(rows), "completion": done_rate,
            "per_start": results, "seconds": time.time() - t0}


def completion_check(model, env: TreasureMaze, max_steps: int | None = None):
    """Play greedily (no exploration) from every open cell: the share of starts that reach the treasure."""
    max_steps = max_steps or env.maze.size * 2
    results = {}
    for start in env.free:
        s, status, n = env.reset(start), "playing", 0
        while status == "playing" and n < max_steps:
            s, _, status = env.step(int(np.argmax(_q(model, s))))
            n += 1
        results[start] = status == "win"
    return (float(np.mean(list(results.values()))) if results else 0.0), results


def greedy_path(model, env: TreasureMaze, start):
    s, path, status = env.reset(start), [start], "playing"
    for _ in range(env.maze.size * 2):
        s, _, status = env.step(int(np.argmax(_q(model, s))))
        path.append(env.pos)
        if status != "playing":
            break
    return path, status


def train_cartpole(episodes: int, epsilon: float, eps_min: float, eps_decay: float, gamma: float, lr: float,
                   memory_size: int, batch: int, time_limit: float, on_episode=None) -> dict:
    """DQN on Gymnasium CartPole-v1 (Module 5): keep the pole up; reward +1 per step, 500 is the maximum."""
    import gymnasium as gym
    import keras
    _seed()
    env = gym.make("CartPole-v1")
    n_in, n_act = int(env.observation_space.shape[0]), int(env.action_space.n)
    model = keras.Sequential([keras.Input(shape=(n_in,)), keras.layers.Dense(24, activation="relu"),
                              keras.layers.Dense(24, activation="relu"), keras.layers.Dense(n_act)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    memory, rows, t0 = [], [], time.time()
    for ep in range(1, episodes + 1):
        s, _ = env.reset(seed=SEED + ep)
        s = s.reshape(1, -1).astype("float32")
        total, done = 0, False
        while not done:
            a = env.action_space.sample() if random.random() < epsilon else int(np.argmax(_q(model, s)))
            s2, r, term, trunc, _ = env.step(a)
            s2 = s2.reshape(1, -1).astype("float32")
            done = term or trunc
            memory.append((s, a, r if not term else -10.0, s2, term))
            if len(memory) > memory_size:
                memory.pop(0)
            s, total = s2, total + r
            if len(memory) >= batch:
                sample = random.sample(memory, batch)
                S, S2 = np.vstack([m[0] for m in sample]), np.vstack([m[3] for m in sample])
                tq = model(S, training=False).numpy()
                nq = model(S2, training=False).numpy().max(axis=1)
                for i, (_, a_, r_, _, d_) in enumerate(sample):
                    tq[i, a_] = r_ if d_ else r_ + gamma * nq[i]
                model.train_on_batch(S, tq)
        epsilon = max(eps_min, epsilon * eps_decay)
        rows.append({"Episode": ep, "Reward": total, "Epsilon": round(epsilon, 3)})
        if on_episode:
            on_episode(ep, rows[-1])
        if time.time() - t0 > time_limit:
            break
    env.close()
    log = pd.DataFrame(rows)
    log["Average (last 10)"] = log["Reward"].rolling(10, min_periods=1).mean()
    return {"model": model, "log": log, "seconds": time.time() - t0}


# =================================================================== 4. export
SERVE_TABULAR = '''"""
FastAPI service for a model trained in Vanguard Analytics.
Run:   pip install -r requirements.txt
       uvicorn serve:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs
Call:  POST /predict  with  {"records": [{"column": value, ...}, ...]}
"""
import json
import numpy as np
import pandas as pd
import keras
from fastapi import FastAPI
from pydantic import BaseModel

model = keras.saving.load_model("model.keras")
P = json.load(open("preprocess.json"))
app = FastAPI(title="Vanguard Analytics model: " + P["target"])


class Records(BaseModel):
    records: list[dict]


def col(df, c):
    return df[c] if c in df else pd.Series([None] * len(df), index=df.index)


def transform(df):
    parts = []
    for c in P["num_cols"]:
        v = pd.to_numeric(col(df, c), errors="coerce").fillna(P["means"][c])
        parts.append(((v - P["means"][c]) / P["stds"][c]).to_numpy()[:, None])
    for c in P["date_cols"]:
        d = pd.to_datetime(col(df, c), errors="coerce")
        for part, val in (("month", d.dt.month), ("dow", d.dt.dayofweek)):
            k = f"{c}__{part}"
            parts.append(((val.fillna(P["means"][k]) - P["means"][k]) / P["stds"][k]).to_numpy()[:, None])
    for c in P["cat_cols"]:
        s = col(df, c).astype(str)
        parts.append(np.stack([(s == v).to_numpy() for v in P["cats"][c]], axis=1).astype("float32"))
    return np.hstack(parts).astype("float32")


@app.get("/")
def info():
    return {"target": P["target"], "task": P["task"], "inputs": P["num_cols"] + P["cat_cols"] + P["date_cols"]}


@app.post("/predict")
def predict(body: Records):
    df = pd.DataFrame(body.records)
    out = model.predict(transform(df), verbose=0)
    if P["task"] == "regression":
        return {"predictions": (out.ravel() * P["y_std"] + P["y_mean"]).round(4).tolist()}
    if P["task"] == "binary":
        p = out.ravel()
        return {"predictions": [P["classes"][int(v >= 0.5)] for v in p], "probability": p.round(4).tolist()}
    return {"predictions": [P["classes"][i] for i in out.argmax(axis=1)], "probability": out.max(axis=1).round(4).tolist()}
'''

SERVE_IMAGE = '''"""
FastAPI service for an image classifier trained in Vanguard Analytics.
Run:   pip install -r requirements.txt python-multipart pillow
       uvicorn serve:app --host 0.0.0.0 --port 8000
Call:  POST /predict  with an image file (form field "file")
"""
import io, json
import numpy as np
import keras
from fastapi import FastAPI, UploadFile
from PIL import Image

model = keras.saving.load_model("model.keras")
P = json.load(open("preprocess.json"))
app = FastAPI(title="Vanguard Analytics image classifier")


@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("L" if P["gray"] else "RGB").resize((P["size"], P["size"]))
    x = np.asarray(img, dtype="float32")[None] / 255.0
    if P["gray"]:
        x = x[..., None]
    p = model.predict(x, verbose=0)[0]
    order = p.argsort()[::-1][:3]
    return {"prediction": P["classes"][int(order[0])],
            "top3": [{"class": P["classes"][int(i)], "probability": round(float(p[i]), 4)} for i in order]}
'''

DOCKERFILE = '''FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
'''


def export_bundle(model, preprocess: dict, kind: str = "tabular") -> bytes:
    import tempfile
    import tensorflow as tf
    path = os.path.join(tempfile.mkdtemp(), "model.keras")
    model.save(path)
    reqs = f"tensorflow-cpu=={tf.__version__}\nfastapi\nuvicorn\npandas\nnumpy\n" + \
           ("python-multipart\npillow\n" if kind == "image" else "")
    readme = ("Vanguard Analytics model export\n\n"
              "1. pip install -r requirements.txt\n2. uvicorn serve:app --port 8000\n"
              "3. Open http://localhost:8000/docs to try it.\n\n"
              "Docker: docker build -t vanguard-model . && docker run -p 8000:8000 vanguard-model\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "model.keras")
        z.writestr("preprocess.json", json.dumps(preprocess, indent=2, default=str))
        z.writestr("serve.py", SERVE_IMAGE if kind == "image" else SERVE_TABULAR)
        z.writestr("requirements.txt", reqs)
        z.writestr("Dockerfile", DOCKERFILE)
        z.writestr("README.txt", readme)
    os.remove(path)
    return buf.getvalue()
