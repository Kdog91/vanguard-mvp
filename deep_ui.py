"""Streamlit screen: Deep learning lab (TensorFlow / Keras) — tabular neural network, image CNN, deep Q-learning."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import deep_learning as dl

BLUE, ORANGE, RED, GREEN = "#2a78d6", "#eda100", "#e34948", "#2e9e6a"


def _curves(hist: pd.DataFrame, metric: str, title: str):
    cols = [c for c in (metric, f"val_{metric}") if c in hist]
    long = hist[["epoch"] + cols].melt(id_vars="epoch", var_name="Set", value_name="value")
    long["Set"] = long["Set"].map(lambda s: "Validation" if s.startswith("val_") else "Training")
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="Epoch"), y=alt.Y("value:Q", title=metric),
        color=alt.Color("Set:N", scale=alt.Scale(domain=["Training", "Validation"], range=[BLUE, ORANGE])),
        tooltip=["epoch", "Set", alt.Tooltip("value:Q", title=metric, format=".4f")]).properties(height=240, title=title)


def _live(placeholder, rows, metric):
    h = pd.DataFrame(rows)
    if len(h):
        placeholder.altair_chart(_curves(h, metric, "Training progress"), width="stretch")


def render_deep(df: pd.DataFrame, profile: dict, label: str):
    ok, ver = dl.tf_available()
    st.write("Train deep learning models with **TensorFlow / Keras**: a neural network on your table, a "
             "**convolutional neural network (CNN)** on your images, or a **deep Q-learning** agent. Every model is "
             "checked on data it never saw, compared with simpler models, and can be downloaded with a ready-to-run "
             "**FastAPI** service and **Dockerfile**.")
    if not ok:
        st.error("TensorFlow is not installed where this app is running, so this mode is off. On your own computer: "
                 "`py -m pip install tensorflow` (Python 3.10 to 3.13). Details: " + ver)
        return
    st.caption(f"TensorFlow {ver} on CPU. Training runs here, so keep data and epochs modest on the public demo "
               "(tables are capped at 50,000 rows).")
    t1, t2, t3 = st.tabs(["Neural network on this table", "Image classifier (CNN)", "Deep Q-learning (agent)"])
    with t1:
        _tabular(df, profile, label)
    with t2:
        _images(label)
    with t3:
        _rl()


# ------------------------------------------------------------------ tabular
def _tabular(df, profile, label):
    targets = [c for c, t in profile.items() if t in ("numeric", "categorical")]
    if not targets:
        st.warning("No number or category column to predict.")
        return
    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("Predict (target)", targets, key="dl_target",
                          index=next((i for i, c in enumerate(targets) if c.lower() in ("y", "churned", "target", "label")), 0))
    usable = [c for c, t in profile.items() if c != target and t in ("numeric", "categorical", "datetime")]
    feats = c2.multiselect("Inputs", usable, default=usable, key=f"dl_feats_{target}",
                           help="Leave out anything that is the answer in another form or is only known afterwards.")
    if not feats:
        st.info("Pick at least one input.")
        return
    st.markdown("**Network design**")
    a1, a2, a3, a4 = st.columns(4)
    n_layers = a1.slider("Hidden layers", 1, 4, 2, key="dl_nl")
    units = a2.select_slider("Neurons in the first layer", [8, 16, 32, 64, 128, 256], 64, key="dl_units",
                             help="Each next layer has half as many.")
    dropout = a3.slider("Dropout", 0.0, 0.6, 0.2, 0.05, key="dl_drop", help="Randomly switches off neurons while training, "
                                                                           "to reduce overfitting.")
    activation = a4.selectbox("Activation", ["relu", "tanh", "elu"], key="dl_act")
    b1, b2, b3, b4, b5 = st.columns(5)
    opt = b1.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], key="dl_opt")
    lr = b2.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], 0.001, key="dl_lr")
    epochs = b3.number_input("Max epochs", 5, 500, 100, key="dl_ep")
    batch = b4.select_slider("Batch size", [16, 32, 64, 128, 256], 64, key="dl_batch")
    patience = b5.number_input("Early stop after", 2, 50, 10, key="dl_pat", help="Stop when validation loss has not "
                                                                                 "improved for this many epochs, and keep the best weights.")
    layers = [max(4, units // (2 ** i)) for i in range(n_layers)]
    st.caption(f"Layers: input → {' → '.join(map(str, layers))} → output. Data is split 70% training, 15% validation "
               "(for early stopping), 15% test (scored once at the end).")
    if st.button("Train neural network", type="primary", key="dl_go"):
        rows, ph, txt = [], st.empty(), st.empty()
        metric = "loss"

        def on_epoch(ep, logs):
            rows.append({"epoch": ep, **logs})
            txt.caption(f"Epoch {ep}: loss {logs.get('loss', 0):.4f}, validation loss {logs.get('val_loss', 0):.4f}")
            if ep % 2 == 0 or ep == 1:
                _live(ph, rows, metric)
        try:
            with st.spinner("Training..."):
                r = dl.train_tabular(df, target, feats, profile, layers, dropout, activation, opt, lr, int(epochs),
                                     int(batch), int(patience), on_epoch=on_epoch)
            ph.empty()
            txt.empty()
            st.session_state["dl_tab"] = (target, tuple(feats), r)
        except Exception as e:
            st.error(f"Training failed: {e}")
    saved = st.session_state.get("dl_tab")
    if not saved or saved[0] != target or saved[1] != tuple(feats):
        return
    r = saved[2]
    h = r["history"]
    best_ep = int(h["val_loss"].idxmin()) + 1 if "val_loss" in h else len(h)
    st.success(f"Trained {len(h)} epochs in {r['seconds']:.1f} s ({r['params']:,} weights). Best validation loss at "
               f"epoch {best_ep}; those weights were kept. Training {r['n_train']:,}, validation {r['n_val']:,}, "
               f"test {r['n_test']:,} rows.")
    m = r["metrics"]
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(f"{k} (test)", f"{v:,.3f}")
    g1, g2 = st.columns(2)
    g1.altair_chart(_curves(h, "loss", "Loss"), width="stretch")
    second = "accuracy" if "accuracy" in h else "mae"
    g2.altair_chart(_curves(h, second, second.capitalize() if second == "accuracy" else "MAE (scaled)"), width="stretch")
    gap = (h["val_loss"].iloc[best_ep - 1] - h["loss"].iloc[best_ep - 1]) if "val_loss" in h else 0
    if gap > 0.25 * max(h["loss"].iloc[best_ep - 1], 1e-9):
        st.caption("Validation loss sits well above training loss: some overfitting. Try more dropout or a smaller network.")

    st.markdown("**Compared with simpler models** (same inputs, same test rows)")
    comp = pd.concat([r["baselines"], pd.DataFrame([{"Model": "Neural network (Keras)", **m}])], ignore_index=True)
    key = "R²" if r["task"] == "regression" else ("ROC AUC" if r["task"] == "binary" else "Balanced accuracy")
    comp = comp.sort_values(key, ascending=False).reset_index(drop=True)
    st.dataframe(comp.round(4), hide_index=True, width="stretch")
    winner = comp.iloc[0]["Model"]
    st.caption(f"Ranked by {key} on the test rows. "
               + ("The neural network did best." if winner.startswith("Neural") else
                  f"**{winner}** did as well or better. On tables this is common: neural networks shine on images, "
                  "audio and text, while tree models often win on spreadsheets. Use the simpler model if it is as good."))

    if r["task"] != "regression":
        st.markdown("**Fairness check: results by group**")
        groups = [c for c, t in profile.items() if t == "categorical" and c != target and 2 <= df[c].nunique() <= 20]
        if groups:
            gc = st.selectbox("Group column (for example gender, age band, region)", groups, key="dl_fair")
            pos = len(r["classes"]) - 1 if r["task"] == "binary" else 0
            if r["task"] == "binary" and "Yes" in r["classes"]:
                pos = r["classes"].index("Yes")
            fair = dl.fairness(df.loc[r["test_index"], gc], r["y_test"], r["test_pred"], r["task"], pos)
            st.dataframe(fair, hide_index=True, width="stretch")
            st.caption(f"'Positive' = {r['classes'][pos]}. If one group is predicted positive at under 80% of the "
                       "highest group's rate, the four-fifths rule flags it for review. A gap can reflect real "
                       "differences or bias in past data; check before using the model for decisions about people.")

    pre = {**r["prep"].to_json(), "target": target, "task": r["task"], "classes": r["classes"],
           "y_mean": r["y_mean"], "y_std": r["y_std"]}
    st.download_button("Download model + FastAPI service + Dockerfile (.zip)", dl.export_bundle(r["model"], pre),
                       file_name=f"{label}_{target}_keras_model.zip", mime="application/zip", key="dl_export")


# ------------------------------------------------------------------ images
def _images(label):
    st.write("Upload a **zip with one folder per class** (for example `cats/…`, `dogs/…`). Or try the built-in "
             "handwritten digits (1,797 small images, like MNIST).")
    src = st.radio("Images", ["Built-in handwritten digits", "Upload a zip"], horizontal=True, key="cnn_src")
    if src.startswith("Upload"):
        up = st.file_uploader("Zip of image folders", type=["zip"], key="cnn_zip")
        c1, c2 = st.columns(2)
        size = c1.select_slider("Resize images to", [32, 48, 64, 96], 32, key="cnn_size")
        gray = c2.checkbox("Grayscale", key="cnn_gray")
        if up is None:
            return
        try:
            X, y, classes, skipped = dl.load_image_zip(up.getvalue(), size, gray)
        except Exception as e:
            st.error(f"Could not read the images: {e}")
            return
        if skipped:
            st.caption(f"{skipped} files could not be opened and were skipped.")
    else:
        X, y, classes = dl.load_digits_demo()
        size, gray = 8, True
    counts = pd.Series(y).value_counts().sort_index()
    st.caption(f"{len(y):,} images, {len(classes)} classes, {X.shape[1]}×{X.shape[2]} pixels, "
               f"{'grayscale' if X.shape[-1] == 1 else 'color'}. Smallest class: {counts.min()} images.")
    show = st.columns(min(8, len(y)))
    for i, c in enumerate(show):
        k = int(np.random.default_rng(i).integers(len(y)))
        c.image(X[k].squeeze(), caption=classes[y[k]], clamp=True, width=70)

    st.markdown("**CNN design**")
    a1, a2, a3, a4 = st.columns(4)
    blocks = a1.slider("Conv blocks", 1, 3, 1 if X.shape[1] <= 8 else 2, key="cnn_blocks",
                       help="Each block: two Conv2D layers, then MaxPooling (halves the image size).")
    filters = a2.select_slider("Filters in first block", [8, 16, 32, 64], 32, key="cnn_filters")
    dense = a3.select_slider("Dense layer", [32, 64, 128, 256], 128, key="cnn_dense")
    dropout = a4.slider("Dropout", 0.0, 0.6, 0.25, 0.05, key="cnn_drop")
    b1, b2, b3, b4, b5 = st.columns(5)
    augment = b1.checkbox("Data augmentation", value=not src.startswith("Built"), key="cnn_aug",
                          help="Random flips, small rotations and zooms while training. Off for digits (a flipped 2 is not a 2).")
    opt = b2.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], key="cnn_opt")
    lr = b3.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], 0.001, key="cnn_lr")
    epochs = b4.number_input("Max epochs", 2, 200, 30, key="cnn_ep")
    batch = b5.select_slider("Batch size", [16, 32, 64, 128], 32, key="cnn_batch")
    if st.button("Train CNN", type="primary", key="cnn_go"):
        rows, ph, txt = [], st.empty(), st.empty()

        def on_epoch(ep, logs):
            rows.append({"epoch": ep, **logs})
            txt.caption(f"Epoch {ep}: accuracy {logs.get('accuracy', 0):.3f}, validation {logs.get('val_accuracy', 0):.3f}")
            _live(ph, rows, "accuracy")
        try:
            with st.spinner("Training the CNN..."):
                r = dl.train_cnn(X, y, classes, blocks, filters, dense, dropout, augment, opt, lr, int(epochs),
                                 int(batch), 6, on_epoch=on_epoch)
            ph.empty()
            txt.empty()
            st.session_state["dl_cnn"] = (src, len(y), r, size, gray)
        except Exception as e:
            st.error(f"Training failed: {e}")
    saved = st.session_state.get("dl_cnn")
    if not saved or saved[0] != src or saved[1] != len(y):
        return
    r = saved[2]
    st.success(f"Test accuracy **{r['accuracy']:.1%}** on {r['n'][2]:,} unseen images (guessing the largest class "
               f"would score {r['majority']:.1%}). {len(r['history'])} epochs in {r['seconds']:.0f} s, "
               f"{r['params']:,} weights.")
    g1, g2 = st.columns(2)
    g1.altair_chart(_curves(r["history"], "accuracy", "Accuracy"), width="stretch")
    g2.altair_chart(_curves(r["history"], "loss", "Loss"), width="stretch")
    st.markdown("**Confusion matrix** (rows = true class, columns = predicted)")
    st.dataframe(r["confusion"], width="stretch")
    st.dataframe(r["per_class"], hide_index=True, width="stretch")
    wrong = np.where(r["test_p"] != r["test_y"])[0][:8]
    if len(wrong):
        st.markdown("**Some mistakes**")
        cols = st.columns(len(wrong))
        for c, i in zip(cols, wrong):
            c.image(r["test_X"][i].squeeze(), clamp=True, width=70,
                    caption=f"true {r['classes'][r['test_y'][i]]}, said {r['classes'][r['test_p'][i]]}")
    st.download_button("Download model + FastAPI service + Dockerfile (.zip)",
                       dl.export_bundle(r["model"], {"classes": r["classes"], "size": int(saved[3]),
                                                     "gray": bool(saved[4])}, kind="image"),
                       file_name=f"{label}_cnn_model.zip", mime="application/zip", key="cnn_export")
    st.caption("Images of people (faces) raise privacy and bias concerns: check accuracy for each group and get "
               "consent before training on them.")


# ------------------------------------------------------------------ reinforcement learning
def _rl():
    st.write("An **agent** learns by trial and error: it tries actions, gets rewards, and learns which action is "
             "best in each situation. This is for decisions made step by step (routes, games, control), not for "
             "predicting a column in a table.")
    env_pick = st.radio("Environment", ["Treasure maze (pirate agent)", "CartPole (Gymnasium)"], horizontal=True, key="rl_env")
    st.markdown("**Training settings**")
    a1, a2, a3, a4 = st.columns(4)
    eps = a1.slider("Starting epsilon (exploration)", 0.05, 1.0, 0.5 if env_pick.startswith("Treasure") else 1.0, 0.05,
                    key="rl_eps", help="Chance of a random move. It decays each episode so the agent explores, then exploits.")
    decay = a2.select_slider("Epsilon decay per episode", [0.9, 0.95, 0.97, 0.98, 0.99, 0.995], 0.97, key="rl_decay")
    gamma = a3.slider("Discount factor (gamma)", 0.5, 0.99, 0.95, 0.01, key="rl_gamma",
                      help="How much future rewards count. Near 1 = plans ahead.")
    lr = a4.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], 0.001, key="rl_lr")
    b1, b2, b3, b4 = st.columns(4)
    episodes = b1.number_input("Max episodes", 10, 2000, 300 if env_pick.startswith("Treasure") else 150, key="rl_episodes")
    memory = b2.select_slider("Replay memory", [500, 1000, 5000, 10000], 1000 if env_pick.startswith("Treasure") else 5000,
                              key="rl_mem", help="Past moves stored and re-sampled for training (experience replay).")
    batch = b3.select_slider("Replay batch", [16, 32, 64], 32, key="rl_batch")
    limit = b4.number_input("Time limit (seconds)", 15, 900, 120, key="rl_time")

    if env_pick.startswith("Treasure"):
        st.session_state.setdefault("rl_maze", dl.SMALL_MAZE)
        m1, m2 = st.columns(2)
        m1.button("Small 5×5 maze (fast)", key="rl_small", on_click=lambda: st.session_state.update(rl_maze=dl.SMALL_MAZE))
        m2.button("8×8 course maze (slower)", key="rl_big", on_click=lambda: st.session_state.update(rl_maze=dl.DEFAULT_MAZE))
        maze_txt = st.text_area("Maze (1 = open, 0 = wall; treasure at bottom-right)", height=180, key="rl_maze")
        try:
            maze = dl.parse_maze(maze_txt)
        except Exception as e:
            st.error(str(e))
            return
        if st.button("Train the agent", type="primary", key="rl_go"):
            rows, ph = [], st.empty()

            def on_ep(ep, row):
                rows.append(row)
                if ep % 5 == 0:
                    ph.line_chart(pd.DataFrame(rows).set_index("Episode")[["Win rate (last 20)", "Epsilon"]], height=220)
            with st.spinner("Training (deep Q-learning)..."):
                r = dl.train_maze(maze, int(episodes), eps, 0.05, decay, gamma, lr, int(memory), int(batch), float(limit), on_ep)
            ph.empty()
            st.session_state["dl_maze"] = (maze_txt, r)
        saved = st.session_state.get("dl_maze")
        if saved and saved[0] == maze_txt:
            r = saved[1]
            log = r["log"]
            (st.success if r["completion"] == 1 else st.info)(
                f"From every open starting cell, the trained agent reaches the treasure **{r['completion']:.0%}** of the "
                f"time ({len(log)} episodes, {r['seconds']:.0f} s)."
                + ("" if r["completion"] == 1 else " Train longer or raise the time limit to reach 100%."))
            st.line_chart(log.set_index("Episode")[["Win rate (last 20)", "Epsilon"]], height=220)
            free = r["env"].free
            start = st.selectbox("Show the learned route from", free, format_func=lambda p: f"row {p[0] + 1}, column {p[1] + 1}",
                                 key="rl_start")
            path, status = dl.greedy_path(r["model"], r["env"], start)
            st.altair_chart(_maze_chart(maze, path), width="stretch")
            st.caption(f"Route: {len(path) - 1} moves, result: {status}. Blue = open, dark = wall, orange = route, "
                       "green = treasure.")
    else:
        try:
            import gymnasium  # noqa: F401
        except Exception:
            st.warning("Gymnasium is not installed: `py -m pip install gymnasium`.")
            return
        st.caption("Keep a pole balanced on a moving cart. +1 reward per step; 500 is a perfect episode. "
                   "Learning is noisy and can take several minutes.")
        if st.button("Train the agent", type="primary", key="rl_cp"):
            rows, ph = [], st.empty()

            def on_ep(ep, row):
                rows.append(row)
                if ep % 3 == 0:
                    ph.line_chart(pd.DataFrame(rows).set_index("Episode")[["Reward"]], height=220)
            with st.spinner("Training (deep Q-learning)..."):
                r = dl.train_cartpole(int(episodes), eps, 0.01, decay, gamma, lr, int(memory), int(batch), float(limit), on_ep)
            ph.empty()
            st.session_state["dl_cp"] = r
        r = st.session_state.get("dl_cp")
        if r:
            log = r["log"]
            st.success(f"{len(log)} episodes in {r['seconds']:.0f} s. Best episode: {int(log['Reward'].max())} steps; "
                       f"average of the last 10: {log['Reward'].tail(10).mean():.0f}.")
            st.line_chart(log.set_index("Episode")[["Reward", "Average (last 10)"]], height=240)


def _maze_chart(maze, path):
    rows = []
    order = {p: i for i, p in enumerate(path)}
    for (r, c), v in np.ndenumerate(maze):
        kind = "Wall" if v == 0 else "Open"
        if (r, c) in order:
            kind = "Route"
        if (r, c) == (maze.shape[0] - 1, maze.shape[1] - 1):
            kind = "Treasure"
        rows.append({"row": r, "col": c, "Cell": kind, "step": order.get((r, c), "")})
    d = pd.DataFrame(rows)
    base = alt.Chart(d).encode(x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None))
    rect = base.mark_rect(stroke="white", strokeWidth=2).encode(color=alt.Color(
        "Cell:N", scale=alt.Scale(domain=["Open", "Wall", "Route", "Treasure"], range=["#cfe0f5", "#26303d", ORANGE, GREEN])))
    text = base.mark_text(color="#14263d").encode(text="step:N")
    return (rect + text).properties(width=40 * maze.shape[1], height=40 * maze.shape[0])
