#!/usr/bin/env python3
"""Small Streamlit web app for annotating Xenium crops.

Run locally:
    streamlit run assess/annotation_webapp.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile


REPO = Path(__file__).resolve().parents[1]
ANNOT_DIR = REPO / "assess" / "annotations"
SAMPLE_PATH = ANNOT_DIR / "sample_100_per_channel.csv"
LABEL_OPTIONS = ["", "signal", "noise", "unclear", "skip"]
GLOBAL_SCALE_CACHE = ANNOT_DIR / "protein_global_scale_webapp.json"


def demo_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "cell_id": ["demo-1", "demo-2", "demo-3"],
            "channel": ["demo", "demo", "demo"],
            "crop_path": ["", "", ""],
            "label": ["", "", ""],
            "confidence": [np.nan, np.nan, np.nan],
            "notes": ["", "", ""],
            "annotated_at": ["", "", ""],
        }
    )


@st.cache_data(show_spinner=False)
def load_table(csv_path: str) -> pd.DataFrame:
    ann = pd.read_csv(csv_path)
    for col in ["label", "notes", "annotated_at"]:
        if col not in ann.columns:
            ann[col] = ""
    if "confidence" not in ann.columns:
        ann["confidence"] = np.nan
    ann["label"] = ann["label"].fillna("").astype(str)
    ann["notes"] = ann["notes"].fillna("").astype(str)
    ann["annotated_at"] = ann["annotated_at"].fillna("").astype(str)
    ann["confidence"] = pd.to_numeric(ann["confidence"], errors="coerce")
    return ann


def ensure_schema(ann: pd.DataFrame) -> pd.DataFrame:
    ann = ann.copy()
    required = ["sample_id", "cell_id", "channel", "crop_path", "label", "confidence", "notes", "annotated_at"]
    for col in required:
        if col not in ann.columns:
            ann[col] = "" if col not in {"sample_id", "confidence"} else np.nan
    ann["label"] = ann["label"].fillna("").astype(str)
    ann["notes"] = ann["notes"].fillna("").astype(str)
    ann["annotated_at"] = ann["annotated_at"].fillna("").astype(str)
    ann["confidence"] = pd.to_numeric(ann["confidence"], errors="coerce")
    ann["channel"] = ann["channel"].fillna("unknown").astype(str)
    ann["cell_id"] = ann["cell_id"].fillna("").astype(str)
    ann["crop_path"] = ann["crop_path"].fillna("").astype(str)
    ann["sample_id"] = pd.to_numeric(ann["sample_id"], errors="coerce").fillna(0).astype(int)
    return ann


def save_table(ann: pd.DataFrame, csv_path: Path) -> None:
    ann.to_csv(csv_path, index=False)


def mask_boundary(mask: np.ndarray) -> np.ndarray | None:
    m = np.asarray(mask) > 0
    if not m.any():
        return None
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    eroded = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & (~eroded)


def compute_global_scale(paths: list[str]) -> tuple[float, float]:
    vals = []
    for p in paths:
        try:
            a = tifffile.imread(p)
            prot = np.asarray(a[1], dtype=np.float32)
            vals.append(prot[::2, ::2].reshape(-1))
        except Exception:
            continue
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    lo, hi = np.percentile(x, [1.0, 99.8])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def get_global_scale(ann: pd.DataFrame) -> tuple[float, float]:
    paths = ann["crop_path"].dropna().astype(str).unique().tolist()
    key = f"n={len(paths)}"
    if GLOBAL_SCALE_CACHE.exists():
        try:
            cached = pd.read_json(GLOBAL_SCALE_CACHE, typ="series")
            if cached.get("key") == key:
                return float(cached["lo"]), float(cached["hi"])
        except Exception:
            pass
    lo, hi = compute_global_scale(paths)
    try:
        pd.Series({"key": key, "lo": lo, "hi": hi}).to_json(GLOBAL_SCALE_CACHE)
    except Exception:
        pass
    return lo, hi


def render_triptych(crop_path: str, global_lo: float, global_hi: float) -> np.ndarray:
    if not crop_path or not Path(crop_path).exists():
        h, w = 240, 720
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, :] = 32
        img[:, : w // 3, :] = [44, 62, 80]
        img[:, w // 3 : 2 * w // 3, :] = [72, 57, 38]
        img[:, 2 * w // 3 :, :] = [57, 72, 38]
        return img
    a = tifffile.imread(crop_path)
    dapi = np.asarray(a[0], dtype=np.float32)
    prot = np.asarray(a[1], dtype=np.float32)

    boundary = None
    if np.asarray(a).ndim >= 3 and a.shape[0] > 2:
        boundary = mask_boundary(a[2])

    d1, d99 = np.percentile(dapi, [1, 99])
    if d99 <= d1:
        d99 = d1 + 1e-6
    dapi_n = np.clip((dapi - d1) / (d99 - d1), 0.0, 1.0)

    p1, p99 = np.percentile(prot, [1, 99])
    if p99 <= p1:
        p99 = p1 + 1e-6
    prot_local = np.clip((prot - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    prot_global = np.clip((prot - global_lo) / (global_hi - global_lo + 1e-6), 0.0, 1.0)

    # Build RGB triptych without matplotlib dependency at runtime.
    dapi_rgb = np.stack([dapi_n, dapi_n, dapi_n], axis=-1)
    local_rgb = np.stack([prot_local, prot_local**0.8, np.zeros_like(prot_local)], axis=-1)
    global_rgb = np.stack([prot_global, prot_global**0.8, np.zeros_like(prot_global)], axis=-1)

    if boundary is not None:
        cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        dapi_rgb[boundary] = cyan
        local_rgb[boundary] = cyan
        global_rgb[boundary] = cyan

    trip = np.concatenate([dapi_rgb, local_rgb, global_rgb], axis=1)
    trip = (np.clip(trip, 0, 1) * 255).astype(np.uint8)
    return trip


def next_unlabeled_idx(df: pd.DataFrame, start_pos: int) -> int:
    if len(df) == 0:
        return 0
    start_pos = int(np.clip(start_pos, 0, len(df) - 1))
    labels = df["label"].fillna("").to_numpy()
    unlabeled = np.where(labels == "")[0]
    if len(unlabeled) == 0:
        return 0
    after = unlabeled[unlabeled >= start_pos]
    return int(after[0] if len(after) else unlabeled[0])


def label_badge(label: str) -> str:
    v = str(label).strip().lower()
    if v == "signal":
        return "S"
    if v == "noise":
        return "N"
    if v == "unclear":
        return "?"
    if v == "skip":
        return "-"
    return "o"


def main() -> None:
    st.set_page_config(page_title="Xenium Annotation", layout="wide")
    st.title("Xenium Crop Annotation")

    csv_path = st.sidebar.text_input("Annotation CSV", str(SAMPLE_PATH))
    csv_file = Path(csv_path)
    uploaded = st.sidebar.file_uploader("Or upload a CSV", type=["csv"])
    if csv_file.exists():
        ann = load_table(str(csv_file))
        source_csv = csv_file
    elif uploaded is not None:
        ann = pd.read_csv(uploaded)
        source_csv = None
    else:
        st.warning("No local annotation CSV was found, so the app is showing a small demo table.")
        ann = demo_table()
        source_csv = None
    ann = ensure_schema(ann)

    channels = ["(all)"] + sorted(ann["channel"].dropna().astype(str).unique().tolist())
    channel = st.sidebar.selectbox("Channel", channels)
    unlabeled_only = st.sidebar.checkbox("Show unlabeled only", value=False)

    view = ann.copy()
    if channel != "(all)":
        view = view[view["channel"].astype(str) == channel]
    if unlabeled_only:
        view = view[view["label"].fillna("") == ""]
    view = view.reset_index(drop=False).rename(columns={"index": "orig_idx"})

    if len(view) == 0:
        st.warning("No rows match current filters.")
        st.stop()

    done = int((ann["label"].fillna("") != "").sum())
    st.sidebar.metric("Progress", f"{done}/{len(ann)}")

    if "pos" not in st.session_state:
        st.session_state.pos = next_unlabeled_idx(view, 0)
    st.session_state.pos = int(np.clip(st.session_state.pos, 0, len(view) - 1))

    st.sidebar.markdown("### Image Picker")
    min_sid = int(view["sample_id"].min())
    max_sid = int(view["sample_id"].max())
    default_sid = int(view.iloc[st.session_state.pos]["sample_id"])
    jump_sid = st.sidebar.number_input(
        "Jump to sample_id",
        min_value=min_sid,
        max_value=max_sid,
        value=default_sid,
        step=1,
    )
    if st.sidebar.button("Go", use_container_width=True):
        matches = np.where(view["sample_id"].astype(int).to_numpy() == int(jump_sid))[0]
        if len(matches):
            st.session_state.pos = int(matches[0])
            st.rerun()
        st.sidebar.warning("sample_id not in current filter view")

    picker_window = st.sidebar.slider("Picker size", min_value=12, max_value=60, value=24, step=12)
    window_start = (st.session_state.pos // picker_window) * picker_window
    window_end = min(len(view), window_start + picker_window)
    st.sidebar.caption(f"Showing rows {window_start + 1} to {window_end} of {len(view)}")

    picker_cols = st.sidebar.columns(3)
    for i in range(window_start, window_end):
        r = view.iloc[i]
        sid = int(r["sample_id"])
        badge = label_badge(r["label"])
        txt = f"{sid} {badge}"
        key = f"pick_{channel}_{int(unlabeled_only)}_{i}"
        if picker_cols[(i - window_start) % 3].button(txt, key=key, use_container_width=True):
            st.session_state.pos = i
            st.rerun()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("Prev"):
            st.session_state.pos = max(0, st.session_state.pos - 1)
    with c2:
        if st.button("Next"):
            st.session_state.pos = min(len(view) - 1, st.session_state.pos + 1)
    with c3:
        if st.button("Next Unlabeled"):
            st.session_state.pos = next_unlabeled_idx(view, st.session_state.pos + 1)
    with c4:
        st.write(f"Row {st.session_state.pos + 1} / {len(view)}")

    row = view.iloc[st.session_state.pos]
    orig_idx = int(row["orig_idx"])

    st.write(
        f"sample_id={int(row['sample_id'])} | channel={row['channel']} | cell_id={row['cell_id']}"
    )

    try:
        global_lo, global_hi = get_global_scale(ann)
        trip = render_triptych(str(row["crop_path"]), global_lo, global_hi)
        st.image(trip, caption="DAPI | Protein local | Protein global (cyan boundary)")
    except Exception as e:
        st.error(f"Failed to render crop: {e}")

    current_label = str(ann.loc[orig_idx, "label"])
    current_conf = ann.loc[orig_idx, "confidence"]
    current_conf = 0.8 if pd.isna(current_conf) else float(current_conf)
    current_notes = str(ann.loc[orig_idx, "notes"])

    label = st.selectbox("Label", LABEL_OPTIONS, index=LABEL_OPTIONS.index(current_label) if current_label in LABEL_OPTIONS else 0)
    conf = st.slider("Confidence", min_value=0.0, max_value=1.0, value=float(current_conf), step=0.05)
    notes = st.text_input("Notes", value=current_notes)

    b1, b2 = st.columns(2)
    with b1:
        save_clicked = st.button("Save")
    with b2:
        save_next_clicked = st.button("Save + Next")

    if save_clicked or save_next_clicked:
        ann.loc[orig_idx, "label"] = label
        ann.loc[orig_idx, "confidence"] = float(conf) if label else np.nan
        ann.loc[orig_idx, "notes"] = notes
        ann.loc[orig_idx, "annotated_at"] = dt.datetime.now().isoformat(timespec="seconds") if label else ""
        if source_csv is not None:
            save_table(ann, source_csv)
            load_table.clear()
        st.success("Saved")

        if save_next_clicked:
            st.session_state.pos = min(len(view) - 1, st.session_state.pos + 1)
        st.rerun()


if __name__ == "__main__":
    main()
