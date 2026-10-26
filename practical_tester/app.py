from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import LabelEncoder, StandardScaler


st.set_page_config(
    page_title="DeepKey | Keystroke Intelligence",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "fixed-text.csv"
LSTM_PATH = ROOT / "models" / "deepkey_lstm_repetition_w10.keras"
METRICS_DIR = ROOT / "results" / "metrics"
SEED_RUNS_PATH = METRICS_DIR / "model_stability_session1_seed_runs_20261022.csv"
SEED_SUMMARY_PATH = METRICS_DIR / "model_stability_session1_summary_20261022.csv"
HISTORICAL_MLP_PATH = METRICS_DIR / "mlp_validation_experiments.csv"
LSTM_SINGLE_PATH = METRICS_DIR / "lstm_repetition_w10_validation.csv"
GRU_SINGLE_PATH = METRICS_DIR / "gru_repetition_w10_validation.csv"

WINDOW_LENGTH = 10
FIT_MAX_REPETITION = 70
EXPECTED_FULL_ROW_COUNT = 19_772
EXPECTED_SESSION_1_ROW_COUNT = 9_878
EXPECTED_CORRECTED_ROW_COUNT = 9_876
EXPECTED_FIT_ROW_COUNT = 6_928
EXPECTED_FIT_WINDOW_COUNT = 6_021
EXPECTED_FIT_WINDOW_SHA256 = (
    "fddb824bb58a983dfa3b4e6eef9905a215e53db82fce0dfa204530c82e599b85"
)
SESSION_1_EXCLUSIONS = {("p071", 20), ("p087", 8)}
FEATURES = [
    "DU.v.v", "DD.v.p", "DU.v.p", "UD.v.p", "UU.v.p", "DU.p.p",
    "DD.p.w", "DU.p.w", "UD.p.w", "UU.p.w", "DU.w.w", "DD.w.j",
    "DU.w.j", "UD.w.j", "UU.w.j", "DU.j.j", "DD.j.k", "DU.j.k",
    "UD.j.k", "UU.j.k", "DU.k.k", "DD.k.e", "DU.k.e", "UD.k.e",
    "UU.k.e", "DU.e.e", "DD.e.u", "DU.e.u", "UD.e.u", "UU.e.u",
    "DU.u.u", "DD.u.r", "DU.u.r", "UD.u.r", "UU.u.r", "DU.r.r",
    "DD.r.k", "DU.r.k", "UD.r.k", "UU.r.k", "DU.k.k.1", "DD.k.b",
    "DU.k.b", "UD.k.b", "UU.k.b", "DU.b.b",
]
MODEL_IDS = {
    "MLP": "mlp_single_attempt",
    "Repetition-window LSTM": "lstm_10_attempt",
    "Repetition-window GRU": "gru_10_attempt",
}
MODEL_COLORS = {
    "MLP": "#3B82F6",
    "Repetition-window LSTM": "#22D3EE",
    "Repetition-window GRU": "#34D399",
}
COLORS = {
    "background": "#0B1020",
    "sidebar": "#111827",
    "card": "#172033",
    "border": "#263550",
    "blue": "#3B82F6",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "amber": "#FBBF24",
    "red": "#F87171",
    "text": "#F8FAFC",
    "secondary": "#CBD5E1",
    "muted": "#94A3B8",
}
METRIC_COLUMNS = {
    "Accuracy": "accuracy",
    "Balanced accuracy": "balanced_accuracy",
    "Macro precision": "macro_precision",
    "Macro recall": "macro_recall",
    "Macro-F1": "macro_f1",
}


@st.cache_data
def count_raw_dataset_rows() -> int:
    """Count CSV records without parsing or analyzing any session values."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    with DATA_PATH.open("rb") as source:
        line_count = sum(1 for _ in source)
    if line_count < 2:
        raise ValueError("Dataset CSV does not contain any data rows.")
    return line_count - 1


def window_fingerprint(windows: list[tuple[int, ...]]) -> str:
    serialized = "\n".join(",".join(map(str, window)) for window in windows)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def build_contiguous_windows(rows: pd.DataFrame) -> list[tuple[int, ...]]:
    windows: list[tuple[int, ...]] = []
    ordered = rows.sort_values(["participant", "repetition"])
    for _, participant_rows in ordered.groupby("participant", sort=True):
        row_ids = participant_rows["_raw_row_id"].tolist()
        repetitions = participant_rows["repetition"].astype(int).tolist()
        run_start = 0
        for position in range(1, len(row_ids) + 1):
            end_of_run = (
                position == len(row_ids)
                or repetitions[position] != repetitions[position - 1] + 1
            )
            if not end_of_run:
                continue
            run_ids = row_ids[run_start:position]
            windows.extend(
                tuple(run_ids[start : start + WINDOW_LENGTH])
                for start in range(len(run_ids) - WINDOW_LENGTH + 1)
            )
            run_start = position
    return windows


@st.cache_data
def load_dataset() -> dict[str, object]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw_row_count = count_raw_dataset_rows()
    session_1_chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=2048):
        if "session" not in chunk.columns:
            raise ValueError("Dataset is missing required column: session")
        # Discard other-session rows at once; only Session 1 rows are retained.
        selected = chunk.loc[chunk["session"].eq(1)].copy()
        if not selected.empty:
            session_1_chunks.append(selected)
    if not session_1_chunks:
        raise ValueError("Dataset contains no Session 1 records.")
    session_1 = pd.concat(session_1_chunks, axis=0)
    del session_1_chunks
    required = {"participant", "session", "repetition", "total time"}
    missing = required.difference(session_1.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(session_1) != EXPECTED_SESSION_1_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SESSION_1_ROW_COUNT} Session 1 rows; found {len(session_1)}."
        )
    session_1["_raw_row_id"] = session_1.index
    feature_columns = [
        column for column in session_1.columns
        if column not in {"participant", "session", "repetition", "total time", "_raw_row_id"}
    ]
    if feature_columns != FEATURES:
        raise ValueError("Dataset timing feature columns/order differ from the verified model schema.")
    for column in FEATURES:
        session_1[column] = pd.to_numeric(session_1[column], errors="raise")
    non_finite_count = int(
        np.isinf(session_1[FEATURES].to_numpy(dtype=np.float64)).sum()
    )

    modeling_rows = session_1.copy()
    for participant, repetition in SESSION_1_EXCLUSIONS:
        excluded = (
            modeling_rows["participant"].eq(participant)
            & modeling_rows["repetition"].eq(repetition)
        )
        if int(excluded.sum()) != 1:
            raise ValueError(
                f"Expected one documented modeling exclusion for {participant} repetition {repetition}."
            )
        modeling_rows = modeling_rows.loc[~excluded].copy()
    if len(modeling_rows) != EXPECTED_CORRECTED_ROW_COUNT:
        raise ValueError(
            f"Corrected Session 1 modeling rows expected {EXPECTED_CORRECTED_ROW_COUNT}; "
            f"found {len(modeling_rows)}."
        )
    participants = sorted(session_1["participant"].astype(str).unique())
    if len(participants) != 99:
        raise ValueError(f"Expected 99 Session 1 participants; found {len(participants)}.")

    return {
        "raw_row_count": raw_row_count,
        "raw_column_count": len(session_1.columns) - 1,
        "session_1": session_1,
        "modeling_rows": modeling_rows,
        "features": feature_columns,
        "participants": participants,
        "non_finite_count": non_finite_count,
    }


@st.cache_resource
def load_lstm_pipeline() -> dict[str, object]:
    dataset = load_dataset()
    modeling_rows = dataset["modeling_rows"]
    fit_rows = modeling_rows.loc[
        modeling_rows["repetition"].le(FIT_MAX_REPETITION)
    ].copy()
    if len(fit_rows) != EXPECTED_FIT_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_FIT_ROW_COUNT} fit rows; found {len(fit_rows)}.")

    fit_values = fit_rows[FEATURES].to_numpy(dtype=np.float64)
    if fit_values.shape != (EXPECTED_FIT_ROW_COUNT, len(FEATURES)):
        raise ValueError(f"Unexpected fit feature shape: {fit_values.shape}.")
    if not np.isfinite(fit_values).all():
        raise ValueError("Training fit rows contain missing or non-finite features.")
    scaler = StandardScaler().fit(fit_values)
    if int(scaler.n_samples_seen_) != EXPECTED_FIT_ROW_COUNT:
        raise ValueError("Scaler does not match the documented training fit-row count.")
    encoder = LabelEncoder().fit(fit_rows["participant"])
    if len(encoder.classes_) != 99 or not np.array_equal(
        encoder.classes_, np.sort(fit_rows["participant"].unique())
    ):
        raise ValueError("Fit-partition participant encoding differs from the expected class order.")

    windows = build_contiguous_windows(fit_rows)
    fingerprint = window_fingerprint(windows)
    if len(windows) != EXPECTED_FIT_WINDOW_COUNT:
        raise ValueError(f"Expected {EXPECTED_FIT_WINDOW_COUNT} fit windows; found {len(windows)}.")
    if fingerprint != EXPECTED_FIT_WINDOW_SHA256:
        raise ValueError("Fit-window raw-row IDs differ from the audited training design.")
    by_id = fit_rows.set_index("_raw_row_id")
    windows_by_participant: dict[str, list[tuple[int, ...]]] = {}
    for row_ids in windows:
        window = by_id.loc[list(row_ids)]
        repetitions = window["repetition"].astype(int).tolist()
        if (
            window["participant"].nunique() != 1
            or len(repetitions) != WINDOW_LENGTH
            or any(right != left + 1 for left, right in zip(repetitions, repetitions[1:]))
        ):
            raise ValueError("Audited candidate window crosses a participant or repetition gap.")
        label = str(window["participant"].iloc[0])
        windows_by_participant.setdefault(label, []).append(row_ids)

    if not LSTM_PATH.is_file():
        raise FileNotFoundError(f"Saved repetition-window LSTM not found: {LSTM_PATH}")
    try:
        model = tf.keras.models.load_model(LSTM_PATH, compile=False)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError(f"Could not load the saved LSTM model: {exc}") from exc

    expected_input_shape = (None, WINDOW_LENGTH, len(FEATURES))
    expected_output_shape = (None, len(encoder.classes_))
    if tuple(model.input_shape) != expected_input_shape:
        raise ValueError(f"LSTM input shape {model.input_shape} != {expected_input_shape}.")
    if tuple(model.output_shape) != expected_output_shape:
        raise ValueError(f"LSTM output shape {model.output_shape} != {expected_output_shape}.")
    return {
        "model": model,
        "fit_rows": fit_rows,
        "scaler": scaler,
        "encoder": encoder,
        "windows_by_participant": windows_by_participant,
        "window_count": len(windows),
        "fingerprint": fingerprint,
    }


@st.cache_data
def load_metric_files() -> dict[str, pd.DataFrame]:
    files = {
        "seed_runs": SEED_RUNS_PATH,
        "seed_summary": SEED_SUMMARY_PATH,
        "historical_mlp": HISTORICAL_MLP_PATH,
        "lstm_single": LSTM_SINGLE_PATH,
        "gru_single": GRU_SINGLE_PATH,
    }
    return {name: pd.read_csv(path) for name, path in files.items() if path.is_file()}


def get_summary_value(
    summary: pd.DataFrame,
    model_id: str,
    metric: str,
    field: str = "mean",
) -> float | None:
    expected = {"model", "metric", field}
    if summary.empty or not expected.issubset(summary.columns):
        return None
    values = summary.loc[
        summary["model"].eq(model_id) & summary["metric"].eq(metric), field
    ]
    if values.empty or pd.isna(values.iloc[0]):
        return None
    return float(values.iloc[0])


def percent(value: float | None) -> str:
    return f"{value:.2%}" if value is not None and np.isfinite(value) else "—"


def validate_window(
    attempts: pd.DataFrame,
    participant: str,
    feature_columns: list[str],
) -> tuple[bool, list[tuple[str, bool]]]:
    feature_order_ok = feature_columns == FEATURES and all(
        column in attempts.columns for column in FEATURES
    )
    repetitions = (
        attempts["repetition"].astype(int).tolist()
        if "repetition" in attempts.columns
        else []
    )
    participant_ok = (
        "participant" in attempts.columns
        and attempts["participant"].nunique() == 1
        and not attempts.empty
        and str(attempts["participant"].iloc[0]) == participant
    )
    session_ok = "session" in attempts.columns and attempts["session"].eq(1).all()
    consecutive_ok = len(repetitions) == WINDOW_LENGTH and all(
        right == left + 1 for left, right in zip(repetitions, repetitions[1:])
    )
    excluded_ok = not any(
        (str(row.participant), int(row.repetition)) in SESSION_1_EXCLUSIONS
        for row in attempts.itertuples()
    ) if {"participant", "repetition"}.issubset(attempts.columns) else False
    features_finite = False
    if feature_order_ok:
        values = attempts[FEATURES].to_numpy(dtype=np.float64)
        features_finite = values.shape == (WINDOW_LENGTH, len(FEATURES)) and np.isfinite(values).all()
    checks = [
        ("Exactly 10 attempts", len(attempts) == WINDOW_LENGTH),
        ("One participant", participant_ok),
        ("Session 1 only", session_ok),
        ("Consecutive repetitions", consecutive_ok),
        ("No documented excluded rows", excluded_ok),
        ("46 features in verified order", feature_order_ok),
        ("All feature values finite", features_finite),
    ]
    return all(passed for _, passed in checks), checks


def styled_figure(figsize: tuple[float, float] = (8.0, 3.4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["card"])
    ax.set_facecolor(COLORS["card"])
    ax.tick_params(colors=COLORS["secondary"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    ax.grid(axis="y", color=COLORS["border"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def present_chart(fig, ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, color=COLORS["text"], fontsize=12, loc="left", pad=12, weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS["secondary"], fontsize=9)
    ax.tick_params(axis="x", colors=COLORS["secondary"])
    ax.tick_params(axis="y", colors=COLORS["secondary"])
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def clear_prediction() -> None:
    st.session_state.pop("model_tester_result", None)


def reset_model_tester() -> None:
    for key in ("model_tester_participant", "model_tester_window", "model_tester_result"):
        st.session_state.pop(key, None)


st.markdown(
    """
    <style>
    :root {
      --dk-bg:#0B1020; --dk-sidebar:#111827; --dk-card:#172033;
      --dk-border:#263550; --dk-blue:#3B82F6; --dk-cyan:#22D3EE;
      --dk-green:#34D399; --dk-amber:#FBBF24; --dk-red:#F87171;
      --dk-text:#F8FAFC; --dk-secondary:#CBD5E1; --dk-muted:#94A3B8;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background:var(--dk-bg)!important; color:var(--dk-text)!important;
    }
    [data-testid="stHeader"] { background:rgba(11,16,32,.92)!important; }
    [data-testid="stSidebar"] {
      background:var(--dk-sidebar)!important; border-right:1px solid var(--dk-border);
    }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color:var(--dk-text)!important; }
    [data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4 {
      color:var(--dk-text)!important;
    }
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li {
      color:var(--dk-secondary);
    }
    [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] * {
      color:var(--dk-muted)!important;
    }
    [data-testid="stMetric"] {
      background:var(--dk-card); border:1px solid var(--dk-border);
      border-radius:13px; padding:12px 15px;
    }
    [data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],[data-testid="stMetricValue"] * {
      color:var(--dk-text)!important;
    }
    [data-testid="stMetricDelta"] { color:var(--dk-cyan)!important; }
    [data-testid="stSelectbox"] label,[data-testid="stMultiSelect"] label,
    [data-testid="stTextInput"] label,[data-testid="stSlider"] label,
    [data-testid="stRadio"] label { color:var(--dk-secondary)!important; }
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stMultiSelect"] [role="combobox"],
    [data-testid="stTextInput"] input {
      color:var(--dk-text)!important; background:#111A2D!important;
      border-color:var(--dk-border)!important; -webkit-text-fill-color:var(--dk-text)!important;
    }
    [role="listbox"],[role="option"],[data-baseweb="popover"],
    [data-baseweb="popover"] *,[data-baseweb="menu"],[data-baseweb="menu"] * {
      color:var(--dk-text)!important; background:#111A2D!important;
    }
    [role="option"]:hover,[role="option"][aria-selected="true"] {
      background:#1E3A5F!important; color:#FFFFFF!important;
    }
    [data-testid="stButton"] button {
      color:#FFFFFF!important; background:var(--dk-blue)!important;
      border:1px solid #5793F7!important; border-radius:9px!important; font-weight:650;
    }
    [data-testid="stButton"] button:disabled {
      color:var(--dk-muted)!important; background:#263247!important;
      border-color:var(--dk-border)!important;
    }
    [data-testid="stExpander"] {
      background:var(--dk-card)!important; border:1px solid var(--dk-border)!important;
      border-radius:11px!important; color:var(--dk-text)!important;
    }
    [data-testid="stExpander"] summary,[data-testid="stExpander"] summary * {
      color:var(--dk-text)!important;
    }
    [data-testid="stAlert"] { border-radius:10px; }
    [data-testid="stAlert"] p,[data-testid="stAlert"] span {
      color:var(--dk-text)!important;
    }
    [data-testid="stDataFrame"] { color-scheme:dark; }
    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="columnheader"] { color:var(--dk-text)!important; }
    [data-testid="stTooltipIcon"] { color:var(--dk-muted)!important; }
    .dk-hero {
      border:1px solid #294C70; border-radius:17px; padding:22px 25px;
      margin:3px 0 18px; background:linear-gradient(115deg,#13213D 0%,#123B63 68%,#075E70 100%);
    }
    .dk-hero-kicker { color:#67E8F9; font-size:.69rem; font-weight:750; letter-spacing:.14em; }
    .dk-hero h1 { color:#F8FAFC!important; font-size:2.05rem; margin:.24rem 0 .15rem; }
    .dk-hero p { color:#D5E7F4!important; margin:0; font-size:.96rem; }
    .dk-kpi {
      background:var(--dk-card); border:1px solid var(--dk-border);
      border-radius:13px; padding:14px 16px; min-height:102px;
    }
    .dk-kpi-label { color:var(--dk-muted); font-size:.7rem; font-weight:700; letter-spacing:.07em; }
    .dk-kpi-value { color:var(--dk-text); font-size:1.5rem; font-weight:760; margin:.32rem 0 .15rem; }
    .dk-kpi-note { color:var(--dk-muted); font-size:.7rem; }
    .dk-live-note {
      background:#302718; border:1px solid #785A20; border-radius:10px;
      padding:12px 14px; color:#FDE68A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="dk-hero"><div class="dk-hero-kicker">DEEPKEY · AI RESEARCH DEMO</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_kpi(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="dk-kpi"><div class="dk-kpi-label">{label}</div>'
        f'<div class="dk-kpi-value">{value}</div><div class="dk-kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## ⌨️ DeepKey")
    st.caption("Keystroke behavioral biometrics")
    page_icons = {
        "Front Page / Overview": "◈  Overview",
        "Model Testing": "⌁  Model Testing",
        "Dataset Explorer": "▦  Dataset Explorer",
        "Model Performance": "▤  Model Performance",
        "Project Methodology": "ⓘ  Project Methodology",
    }
    page = st.radio(
        "Navigate",
        list(page_icons),
        format_func=lambda item: page_icons[item],
        label_visibility="collapsed",
        key="deepkey_page",
    )
    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.markdown("🟢 Session 1 pipeline")
    st.markdown("🟢 Saved LSTM model")
    st.markdown("🟡 Live feature extraction pending")
    st.caption("Local inference · no model training")

try:
    dataset = load_dataset()
    dataset_error = None
except (FileNotFoundError, ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
    dataset = None
    dataset_error = str(exc)

try:
    metric_files = load_metric_files()
    metrics_error = None
except (OSError, ValueError, pd.errors.ParserError) as exc:
    metric_files = {}
    metrics_error = str(exc)

if dataset_error:
    st.error(f"Dataset not ready: {dataset_error}")
if metrics_error:
    st.error(f"Metric files could not be read: {metrics_error}")

seed_runs = metric_files.get("seed_runs", pd.DataFrame())
seed_summary = metric_files.get("seed_summary", pd.DataFrame())


if page == "Front Page / Overview":
    show_hero(
        "DeepKey",
        "Deep Learning-Based Continuous User Authentication Using Keystroke Behavioral Biometrics",
    )
    st.markdown(
        "A B.Tech research prototype that explores identifying participants from "
        "keystroke timing patterns. Current reported model metrics are Session 1 validation results."
    )
    raw_rows = dataset["raw_row_count"] if dataset else None
    participant_count = len(dataset["participants"]) if dataset else None
    lstm_mean_accuracy = get_summary_value(
        seed_summary, MODEL_IDS["Repetition-window LSTM"], "accuracy"
    )
    kpis = st.columns(4)
    with kpis[0]:
        show_kpi("RAW DATASET ROWS", f"{raw_rows:,}" if raw_rows is not None else "—", "CSV record count")
    with kpis[1]:
        show_kpi("PARTICIPANT CLASSES", str(participant_count or "—"), "Session 1 participants")
    with kpis[2]:
        feature_count = len(dataset["features"]) if dataset else None
        show_kpi("TIMING FEATURES", str(feature_count or "—"), "Verified feature columns")
    with kpis[3]:
        show_kpi("LSTM MEAN ACCURACY", percent(lstm_mean_accuracy), "Session 1 · five-seed chronological validation")

    st.subheader("From typing attempts to participant prediction")
    st.markdown(
        '<div class="dk-live-note" style="background:#172033;border-color:#263550;color:#CBD5E1">'
        '<b style="color:#F8FAFC">Keystroke timing data</b> &nbsp;→&nbsp; '
        '<b style="color:#F8FAFC">Preprocessing</b> &nbsp;→&nbsp; '
        '<b style="color:#F8FAFC">10-attempt windows</b> &nbsp;→&nbsp; '
        '<b style="color:#F8FAFC">LSTM / GRU</b> &nbsp;→&nbsp; '
        '<b style="color:#F8FAFC">Participant class</b></div>',
        unsafe_allow_html=True,
    )
    st.subheader("Five-seed model accuracy")
    st.caption(
        "Session 1 chronological validation, averaged over five seeds. This is not "
        "authentication accuracy or a measure of real-world performance."
    )
    if seed_summary.empty:
        st.warning("Five-seed metric summary is missing; comparison charts are unavailable.")
    elif not {"model", "metric", "mean", "std_sample"}.issubset(seed_summary.columns):
        st.warning("Five-seed metric summary has an unexpected schema; comparison charts are unavailable.")
    else:
        chart_models = [
            (model_id, label) for label, model_id in MODEL_IDS.items()
            if not seed_summary.loc[seed_summary["model"].eq(model_id)].empty
        ]
        fig, ax = styled_figure(figsize=(9, 3.6))
        labels = [label for _, label in chart_models]
        means = [get_summary_value(seed_summary, model_id, "accuracy") for model_id, _ in chart_models]
        deviations = [
            get_summary_value(seed_summary, model_id, "accuracy", "std_sample") or 0.0
            for model_id, _ in chart_models
        ]
        ax.bar(labels, means, yerr=deviations, capsize=6,
               color=[MODEL_COLORS[label] for label in labels], width=.5)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis="x", rotation=5)
        present_chart(fig, ax, "Mean accuracy ± sample standard deviation", "Five-seed mean accuracy")

    st.subheader("Model summary")
    st.dataframe(
        pd.DataFrame([
            {"Model": "MLP", "Prediction unit": "One attempt", "Input": "46 timing features"},
            {"Model": "LSTM", "Prediction unit": "10-attempt window", "Input": "(10, 46)"},
            {"Model": "GRU", "Prediction unit": "10-attempt window", "Input": "(10, 46)"},
        ]),
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Five seeds measure training randomness on a fixed chronological partition, not "
        "uncertainty across data splits. The sequence models use ten attempts per prediction; "
        "their scores are not a perfectly equivalent head-to-head comparison with the single-attempt MLP."
    )

    st.markdown(
        "**What this dashboard demonstrates:** recorded model validation, a local "
        "in-sample dataset inference demonstration, and Session 1 feature exploration. "
        "It does not currently provide live typing authentication."
    )


elif page == "Model Performance":
    show_hero(
        "Model Performance",
        "Compare saved Session 1 validation results with their protocols kept explicit.",
    )
    st.info(
        "The five-seed study uses one fixed chronological Session 1 partition "
        "(fit ≤70, embargo 71–79, validation ≥80). MLP predicts individual attempts; "
        "the repetition LSTM and GRU predict ten-attempt windows."
    )

    required_run_columns = {
        "model", "seed", "prediction_unit", *METRIC_COLUMNS.values()
    }
    required_summary_columns = {
        "model", "prediction_unit", "metric", "mean", "std_sample", "n_seeds"
    }
    if seed_runs.empty or not required_run_columns.issubset(seed_runs.columns):
        st.warning("Five-seed run metrics are missing or do not match the expected CSV schema.")
    elif seed_summary.empty or not required_summary_columns.issubset(seed_summary.columns):
        st.warning("Five-seed summary metrics are missing or do not match the expected CSV schema.")
    else:
        selected_metric = st.selectbox(
            "Metric for the per-seed chart",
            list(METRIC_COLUMNS),
            key="performance_metric_selector",
        )
        metric_column = METRIC_COLUMNS[selected_metric]
        filtered_runs = seed_runs.loc[seed_runs["model"].isin(MODEL_IDS.values())].copy()
        filtered_summary = seed_summary.loc[
            seed_summary["model"].isin(MODEL_IDS.values())
        ].copy()
        if filtered_runs.empty:
            st.warning("No recognized MLP, LSTM, or GRU runs are present in the seed CSV.")
        else:
            st.subheader("Mean accuracy ± sample standard deviation")
            accuracy_summary = filtered_summary.loc[
                filtered_summary["metric"].eq("accuracy")
            ]
            fig, ax = styled_figure(figsize=(9, 3.4))
            accuracy_labels = []
            accuracy_means = []
            accuracy_deviations = []
            for label, model_id in MODEL_IDS.items():
                mean_values = accuracy_summary.loc[
                    accuracy_summary["model"].eq(model_id), "mean"
                ]
                if mean_values.empty:
                    continue
                deviation_values = accuracy_summary.loc[
                    accuracy_summary["model"].eq(model_id), "std_sample"
                ]
                accuracy_labels.append(label)
                accuracy_means.append(float(mean_values.iloc[0]))
                accuracy_deviations.append(
                    float(deviation_values.iloc[0]) if not deviation_values.empty else 0.0
                )
            ax.bar(
                accuracy_labels,
                accuracy_means,
                yerr=accuracy_deviations,
                capsize=6,
                width=.55,
                color=[MODEL_COLORS[label] for label in accuracy_labels],
            )
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.tick_params(axis="x", rotation=6)
            present_chart(fig, ax, "Accuracy across five fixed seeds", "Mean accuracy")

            st.subheader("Per-seed comparison")
            fig, ax = styled_figure(figsize=(9, 3.7))
            for label, model_id in MODEL_IDS.items():
                model_runs = filtered_runs.loc[filtered_runs["model"].eq(model_id)].sort_values("seed")
                if model_runs.empty:
                    continue
                ax.plot(
                    model_runs["seed"].astype(str),
                    model_runs[metric_column],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=label,
                    color=MODEL_COLORS[label],
                )
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("Random seed", color=COLORS["secondary"])
            ax.legend(frameon=False, labelcolor=COLORS["text"])
            present_chart(fig, ax, f"{selected_metric} by seed")

            readable_runs = filtered_runs.copy()
            readable_runs["Model"] = readable_runs["model"].map(
                {model_id: label for label, model_id in MODEL_IDS.items()}
            ).fillna(readable_runs["model"])
            readable_runs = readable_runs.rename(columns={
                "seed": "Seed",
                "prediction_unit": "Prediction unit",
                "accuracy": "Accuracy",
                "balanced_accuracy": "Balanced accuracy",
                "macro_precision": "Macro precision",
                "macro_recall": "Macro recall",
                "macro_f1": "Macro-F1",
            })
            run_columns = [
                "Model", "Seed", "Prediction unit", "Accuracy", "Balanced accuracy",
                "Macro precision", "Macro recall", "Macro-F1",
            ]
            st.dataframe(
                readable_runs[[column for column in run_columns if column in readable_runs.columns]]
                .style.format({
                    column: "{:.2%}" for column in METRIC_COLUMNS if column in readable_runs.columns
                }),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Five-seed mean results")
            mean_rows = []
            for label, model_id in MODEL_IDS.items():
                row: dict[str, object] = {"Model": label}
                for display, metric_id in (
                    ("Accuracy", "accuracy"),
                    ("Balanced accuracy", "balanced_accuracy"),
                    ("Macro-F1", "macro_f1"),
                ):
                    mean_value = get_summary_value(filtered_summary, model_id, metric_id)
                    deviation = get_summary_value(
                        filtered_summary, model_id, metric_id, "std_sample"
                    )
                    row[display] = (
                        f"{mean_value:.2%} ± {deviation:.2%}"
                        if mean_value is not None and deviation is not None
                        else "—"
                    )
                mean_rows.append(row)
            st.dataframe(pd.DataFrame(mean_rows), width="stretch", hide_index=True)
            st.caption(
                "Standard deviation is the sample SD across seeds on the same fixed partition; "
                "it does not estimate variation across data splits."
            )
            lstm_runs = filtered_runs.loc[
                filtered_runs["model"].eq(MODEL_IDS["Repetition-window LSTM"]),
                ["seed", "accuracy"],
            ].rename(columns={"accuracy": "lstm_accuracy"})
            gru_runs = filtered_runs.loc[
                filtered_runs["model"].eq(MODEL_IDS["Repetition-window GRU"]),
                ["seed", "accuracy"],
            ].rename(columns={"accuracy": "gru_accuracy"})
            paired = lstm_runs.merge(gru_runs, on="seed", how="inner")
            gru_seed_wins = int((paired["gru_accuracy"] > paired["lstm_accuracy"]).sum())
            lstm_mean = get_summary_value(
                filtered_summary, MODEL_IDS["Repetition-window LSTM"], "accuracy"
            )
            st.info(
                f"The repetition-window LSTM has the highest mean accuracy "
                f"({percent(lstm_mean)}) in this five-seed study. The GRU is higher "
                f"than the LSTM on {gru_seed_wins} of {len(paired)} paired seeds. "
                "This is Session 1 validation, not authentication accuracy."
            )

    st.divider()
    st.subheader("Historical MLP validation · separate row-holdout protocol")
    historical = metric_files.get("historical_mlp", pd.DataFrame())
    if historical.empty:
        st.caption("Historical MLP validation CSV is not available.")
    else:
        if "session_2_evaluated" in historical.columns:
            historical = historical.loc[historical["session_2_evaluated"].eq(False)].copy()
        historical_columns = [
            column for column in (
                "experiment", "alpha", "split_random_state", "fit_samples",
                "validation_samples", "validation_accuracy",
                "validation_balanced_accuracy", "validation_macro_precision",
                "validation_macro_recall", "validation_macro_f1",
            ) if column in historical.columns
        ]
        historical = historical[historical_columns].rename(columns={
            "experiment": "Configuration",
            "alpha": "Alpha",
            "split_random_state": "Split seed",
            "fit_samples": "Fit attempts",
            "validation_samples": "Validation attempts",
            "validation_accuracy": "Accuracy",
            "validation_balanced_accuracy": "Balanced accuracy",
            "validation_macro_precision": "Macro precision",
            "validation_macro_recall": "Macro recall",
            "validation_macro_f1": "Macro-F1",
        })
        st.dataframe(
            historical.style.format({
                metric: "{:.2%}" for metric in METRIC_COLUMNS
                if metric in historical.columns
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Historical stratified row-holdout values are not combined with the "
            "five-seed chronological results above."
        )

    st.subheader("Single-run sequence results · seed 42")
    single_rows = []
    for key, label, prefix in (
        ("lstm_single", "Repetition-window LSTM", "validation_"),
        ("gru_single", "Repetition-window GRU", "gru_validation_"),
    ):
        table = metric_files.get(key, pd.DataFrame())
        if table.empty:
            continue
        row = table.iloc[0]
        result = {"Model": label, "Seed": row.get("seed", 42)}
        for display_name, source_column in METRIC_COLUMNS.items():
            result[display_name] = row.get(prefix + source_column)
        single_rows.append(result)
    if single_rows:
        st.dataframe(
            pd.DataFrame(single_rows).style.format(
                {column: "{:.2%}" for column in METRIC_COLUMNS}
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("Historical single-run sequence metrics are not available.")
    st.caption(
        "Single-run values are shown for context only. No confusion matrix is presented "
        "because no verified saved confusion-matrix artifact is available."
    )


elif page == "Model Testing":
    show_hero("Model Tester", "Run an audited, local inference demonstration on a dataset window.")
    st.selectbox(
        "Verified inference model",
        ["Repetition-window LSTM"],
        help="Only the repetition-window LSTM has this verified dataset inference pipeline.",
    )
    st.markdown(
        '<div class="dk-live-note"><b>Live typing authentication · Feature extraction pending</b><br>'
        'Browser key events cannot yet be converted and parity-checked against all 46 training features. '
        'Live predictions remain disabled.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "**In-sample dataset demonstration:** examples are drawn from the Session 1 fit partition "
        "(repetitions 1–70). A matching label is not proof of identity or generalization."
    )
    if dataset_error:
        st.error("Model tester is unavailable until the dataset integrity checks pass.")
    else:
        try:
            pipeline = load_lstm_pipeline()
            pipeline_error = None
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as exc:
            pipeline = None
            pipeline_error = str(exc)
        if pipeline_error:
            st.error(f"LSTM pipeline is not ready: {pipeline_error}")
        elif pipeline is not None:
            participant_options = sorted(pipeline["windows_by_participant"])
            left, right = st.columns(2)
            with left:
                participant = st.selectbox(
                    "Participant",
                    participant_options,
                    key="model_tester_participant",
                    on_change=clear_prediction,
                )
            windows = pipeline["windows_by_participant"][participant]
            by_id = pipeline["fit_rows"].set_index("_raw_row_id")
            indexed_windows = [
                (int(by_id.loc[ids[0], "repetition"]), ids) for ids in windows
            ]
            indexed_windows.sort(key=lambda item: item[0])
            starts = [start for start, _ in indexed_windows]
            with right:
                selected_start = st.selectbox(
                    "Valid ten-attempt sequence",
                    starts,
                    format_func=lambda start: f"Repetitions {start}–{start + WINDOW_LENGTH - 1}",
                    key="model_tester_window",
                    on_change=clear_prediction,
                )
            selected_ids = dict(indexed_windows)[selected_start]
            attempts = by_id.loc[list(selected_ids)].sort_values("repetition")
            is_valid, checks = validate_window(attempts, participant, FEATURES)

            st.markdown("#### Input validation")
            if is_valid:
                st.success("All required checks passed; inference is available.")
            else:
                st.error("Input checks failed; inference is disabled.")
            check_columns = st.columns(4)
            for index, (label, passed) in enumerate(checks):
                check_columns[index % 4].markdown(
                    f"{'🟢' if passed else '🔴'} {label}"
                )
            check_columns[0].caption(
                f"Scaler fit rows: {int(pipeline['scaler'].n_samples_seen_):,}"
            )
            check_columns[1].caption(
                f"Label classes: {len(pipeline['encoder'].classes_)}"
            )
            check_columns[2].caption(
                f"Verified windows: {pipeline['window_count']:,}"
            )
            check_columns[3].caption("Model input: (1, 10, 46)")

            with st.expander("Inspect selected attempts and ordered feature values"):
                st.dataframe(
                    attempts.drop(columns=["_raw_row_id"], errors="ignore"),
                    width="stretch",
                    hide_index=True,
                )
            display_feature = st.selectbox(
                "Timing feature to visualize",
                FEATURES,
                key="model_tester_feature",
            )
            fig, ax = styled_figure(figsize=(9, 2.8))
            ax.plot(
                attempts["repetition"],
                attempts[display_feature],
                marker="o",
                color=COLORS["cyan"],
                linewidth=2,
            )
            ax.set_xlabel("Repetition", color=COLORS["secondary"])
            ax.set_ylabel(f"{display_feature} · units unspecified", color=COLORS["secondary"])
            present_chart(fig, ax, "Selected feature across the sequence")

            if st.button("Run Model Prediction", type="primary", disabled=not is_valid):
                raw_features = attempts[FEATURES].to_numpy(dtype=np.float64)
                scaled = pipeline["scaler"].transform(raw_features)
                model_input = np.expand_dims(scaled, axis=0)
                if model_input.shape != (1, WINDOW_LENGTH, len(FEATURES)):
                    st.error(f"Unexpected input tensor shape: {model_input.shape}.")
                elif not np.isfinite(model_input).all():
                    st.error("Scaled model input contains non-finite values.")
                else:
                    try:
                        probabilities = np.asarray(
                            pipeline["model"].predict(model_input, verbose=0)[0]
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(f"Model inference failed: {exc}")
                    else:
                        if probabilities.shape != (len(pipeline["encoder"].classes_),):
                            st.error(f"Unexpected output probability shape: {probabilities.shape}.")
                        elif (
                            not np.isfinite(probabilities).all()
                            or not np.isclose(probabilities.sum(), 1.0)
                        ):
                            st.error("The model returned invalid class probabilities.")
                        else:
                            st.session_state["model_tester_result"] = {
                                "participant": participant,
                                "window_ids": selected_ids,
                                "probabilities": probabilities,
                            }

            result = st.session_state.get("model_tester_result")
            if result and result["window_ids"] == selected_ids:
                probabilities = result["probabilities"]
                prediction_index = int(np.argmax(probabilities))
                predicted = str(
                    pipeline["encoder"].inverse_transform([prediction_index])[0]
                )
                actual = result["participant"]
                result_columns = st.columns(3)
                result_columns[0].metric("Actual participant", actual)
                result_columns[1].metric("Predicted participant", predicted)
                result_columns[2].metric(
                    "Prediction confidence", f"{probabilities[prediction_index]:.2%}"
                )
                if predicted == actual:
                    st.success("Dataset label match · in-sample demonstration only.")
                else:
                    st.warning("Dataset label mismatch · in-sample demonstration only.")
                top_indices = np.argsort(probabilities)[::-1][:5]
                top_table = pd.DataFrame({
                    "Rank": np.arange(1, len(top_indices) + 1),
                    "Participant class": pipeline["encoder"].classes_[top_indices],
                    "Probability": probabilities[top_indices],
                })
                st.dataframe(
                    top_table.style.format({"Probability": "{:.2%}"}),
                    width="stretch",
                    hide_index=True,
                )
            reset_col, note_col = st.columns([1, 3])
            with reset_col:
                st.button("Reset tester", on_click=reset_model_tester)
            with note_col:
                st.caption("Inference runs locally; no model training or file writes occur.")


elif page == "Dataset Explorer":
    show_hero("Dataset Explorer", "Explore raw Session 1 records and timing-feature patterns.")
    if dataset_error:
        st.warning("Dataset explorer is unavailable until the dataset integrity checks pass.")
    else:
        session_1 = dataset["session_1"]
        modeling_rows = dataset["modeling_rows"]
        st.info(
            "For integrity, this dashboard analyzes Session 1 only. The raw dataset row KPI "
            "uses the CSV record count; Session 2 rows are not used in any chart, filter, "
            "model input, or metric."
        )
        kpi_columns = st.columns(4)
        with kpi_columns[0]:
            show_kpi("RAW DATASET ROWS", f"{dataset['raw_row_count']:,}", "CSV rows, before exclusions")
        with kpi_columns[1]:
            show_kpi("SESSION 1 ROWS", f"{len(session_1):,}", "Raw Session 1 records")
        with kpi_columns[2]:
            show_kpi("PARTICIPANTS", str(session_1["participant"].nunique()), "Session 1 participants")
        with kpi_columns[3]:
            show_kpi(
                "TIMING FEATURES",
                str(len(dataset["features"])),
                "Verified ordered features",
            )
        st.caption(
            "The corrected modeling view contains "
            f"{len(modeling_rows):,} Session 1 rows after excluding "
            "`p071/repetition 20` and `p087/repetition 8`. These exclusions are not applied "
            "to the raw-data explorer."
        )

        missing = session_1[FEATURES].isna().sum()
        missing_count = int(missing.sum())
        missing_left, missing_right = st.columns([1, 1])
        with missing_left:
            st.metric("Missing timing-feature values", f"{missing_count:,}")
            st.metric("Non-finite timing-feature values", f"{dataset['non_finite_count']:,}")
            if missing_count == 0:
                if dataset["non_finite_count"] == 0:
                    st.success("No missing or non-finite feature values in the raw Session 1 view.")
                else:
                    st.warning("Non-finite values exist; affected charts and model input are guarded.")
            elif dataset["non_finite_count"]:
                st.warning("Missing and non-finite values exist; affected charts/model inputs are guarded.")
            else:
                st.warning("Missing values are present. Charts and summaries use non-null values explicitly.")
        with missing_right:
            missing_table = missing.rename_axis("Feature").rename("Missing values").reset_index()
            st.dataframe(missing_table, width="stretch", height=220, hide_index=True)
        with st.expander("Timing-feature summary statistics"):
            st.dataframe(
                session_1[FEATURES].describe().T.reset_index().rename(
                    columns={"index": "Feature"}
                ),
                width="stretch",
                height=320,
                hide_index=True,
            )

        st.divider()
        st.subheader("Filter records")
        participants = dataset["participants"]
        filter_columns = st.columns([1.0, 1.1, 1.2, .9])
        with filter_columns[0]:
            session_filter = st.selectbox("Session", ["Session 1"], help="Session 2 is not available in this explorer.")
        with filter_columns[1]:
            participant_filter = st.selectbox("Participant", ["All participants", *participants])
        with filter_columns[2]:
            participant_search = st.text_input("Search participant", placeholder="Type a participant ID")
        with filter_columns[3]:
            page_size = st.selectbox("Rows", [50, 100, 250], index=1)
        max_repetition = int(session_1["repetition"].max())
        repetition_range = st.slider(
            "Repetition range",
            min_value=int(session_1["repetition"].min()),
            max_value=max_repetition,
            value=(int(session_1["repetition"].min()), max_repetition),
        )
        filtered = session_1.loc[
            session_1["repetition"].between(*repetition_range)
        ].copy()
        if participant_filter != "All participants":
            filtered = filtered.loc[filtered["participant"].eq(participant_filter)]
        if participant_search.strip():
            filtered = filtered.loc[
                filtered["participant"].astype(str).str.contains(
                    participant_search.strip(), case=False, regex=False
                )
            ]
        st.caption(
            f"{len(filtered):,} matching Session 1 rows · showing first "
            f"{min(len(filtered), page_size):,}."
        )
        st.dataframe(
            filtered.drop(columns=["_raw_row_id"]).head(page_size),
            width="stretch",
            height=350,
            hide_index=True,
        )

        st.divider()
        st.subheader("Timing-feature analysis")
        feature = st.selectbox("Feature", FEATURES, key="explorer_feature")
        chart_columns = st.columns(2)
        with chart_columns[0]:
            values = session_1[feature].dropna().to_numpy(dtype=float)
            fig, ax = styled_figure()
            ax.hist(values, bins=36, color=COLORS["blue"], edgecolor=COLORS["card"])
            ax.set_xlabel(f"{feature} · source units unspecified", color=COLORS["secondary"])
            ax.set_ylabel("Attempt count", color=COLORS["secondary"])
            present_chart(fig, ax, f"Feature distribution · {feature}")
        with chart_columns[1]:
            participant_choices = st.multiselect(
                "Participants to compare",
                participants,
                default=participants[:4],
                max_selections=8,
            )
            comparison = session_1.loc[session_1["participant"].isin(participant_choices)]
            groups = [
                comparison.loc[comparison["participant"].eq(person), feature].dropna().to_numpy()
                for person in participant_choices
            ]
            labels = [person for person, group in zip(participant_choices, groups) if len(group)]
            groups = [group for group in groups if len(group)]
            fig, ax = styled_figure()
            if groups:
                box = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False)
                palette = [COLORS["blue"], COLORS["cyan"], COLORS["green"], "#A78BFA",
                           "#F472B6", "#F97316", "#2DD4BF", "#FACC15"]
                for index, patch in enumerate(box["boxes"]):
                    patch.set_facecolor(palette[index % len(palette)])
                    patch.set_alpha(.76)
                    patch.set_edgecolor(COLORS["border"])
            ax.set_ylabel(f"{feature} · units unspecified", color=COLORS["secondary"])
            ax.tick_params(axis="x", rotation=12)
            present_chart(fig, ax, "Feature spread by participant")

        st.subheader("Feature correlation")
        corr_features = st.multiselect(
            "Choose features for the heatmap",
            FEATURES,
            default=FEATURES[:10],
            max_selections=20,
            key="explorer_corr_features",
        )
        if len(corr_features) < 2:
            st.info("Select at least two timing features.")
        else:
            corr_values = session_1[corr_features].to_numpy(dtype=float)
            if not np.isfinite(corr_values).all():
                st.warning("Heatmap unavailable: selected features contain non-finite values.")
            else:
                correlation = session_1[corr_features].corr()
                fig, ax = plt.subplots(
                    figsize=(max(7, len(corr_features) * .55), max(5, len(corr_features) * .45)),
                    facecolor=COLORS["card"],
                )
                ax.set_facecolor(COLORS["card"])
                image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks(range(len(corr_features)), corr_features, rotation=65, ha="right",
                              color=COLORS["secondary"], fontsize=8)
                ax.set_yticks(range(len(corr_features)), corr_features,
                              color=COLORS["secondary"], fontsize=8)
                ax.set_title("Pearson correlation · Session 1", color=COLORS["text"], loc="left")
                colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.03)
                colorbar.ax.tick_params(colors=COLORS["secondary"])
                colorbar.outline.set_edgecolor(COLORS["border"])
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

        coverage_columns = st.columns(2)
        with coverage_columns[0]:
            participant_counts = session_1.groupby("participant").size().sort_values()
            fig, ax = styled_figure()
            ax.bar(np.arange(len(participant_counts)), participant_counts.values,
                   color=COLORS["blue"], width=.8)
            ax.set_xlabel("Participants, sorted", color=COLORS["secondary"])
            ax.set_ylabel("Attempt rows", color=COLORS["secondary"])
            present_chart(fig, ax, "Session 1 attempts by participant")
        with coverage_columns[1]:
            repetition_counts = session_1.groupby("repetition").size()
            fig, ax = styled_figure()
            ax.plot(repetition_counts.index, repetition_counts.values,
                    color=COLORS["cyan"], linewidth=2)
            ax.fill_between(repetition_counts.index, repetition_counts.values,
                            color=COLORS["cyan"], alpha=.12)
            ax.set_xlabel("Repetition ID", color=COLORS["secondary"])
            ax.set_ylabel("Participants with a record", color=COLORS["secondary"])
            present_chart(fig, ax, "Participant coverage by repetition")


elif page == "Project Methodology":
    show_hero(
        "Project Methodology",
        "A concise view of the problem, data pipeline, models, and current limits.",
    )
    st.subheader("Project objective")
    st.write(
        "DeepKey investigates whether keystroke timing behavior can support continuous "
        "user identification. It is a B.Tech research prototype, not a production security system."
    )
    st.subheader("Keystroke dynamics")
    st.write(
        "Keystroke dynamics describe timing patterns around typing, including durations "
        "and transitions between keys. The available CSV contains precomputed features "
        "for complete typing attempts, not the original browser keydown/keyup traces."
    )
    st.subheader("Models and prediction units")
    st.dataframe(
        pd.DataFrame([
            {
                "Model": "MLP",
                "Input": "46 features for one attempt",
                "Role": "Feed-forward identification baseline",
            },
            {
                "Model": "LSTM",
                "Input": "(10, 46) consecutive attempts",
                "Role": "Repetition-window sequence model",
            },
            {
                "Model": "GRU",
                "Input": "(10, 46) consecutive attempts",
                "Role": "Gated recurrent sequence comparison",
            },
        ]),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Dataset and validation")
    if dataset:
        st.markdown(
            f"The dataset has **{dataset['raw_row_count']:,} rows × "
            f"{dataset['raw_column_count']} columns**. Session 1 contains "
            f"**{len(dataset['session_1']):,} attempts** from **{len(dataset['participants'])} "
            f"participants**, with **{len(dataset['features'])} ordered timing features**. "
            "Only Session 1 is analyzed; documented modeling exclusions are applied only "
            "in the corrected modeling workflow."
        )
    st.markdown(
        "**Modeling workflow:** apply the documented attempt exclusions "
        "`p071/repetition 20` and `p087/repetition 8`; split Session 1 chronologically "
        "(fit through repetition 70, embargo 71–79, validate from 80); fit scaling on "
        "the fit partition only; and create LSTM/GRU windows of ten consecutive attempts."
    )
    st.write(
        "The five-seed results use one fixed chronological Session 1 partition. Seed "
        "variation is not data-split uncertainty. MLP uses one-attempt predictions, "
        "whereas LSTM and GRU use ten-attempt windows."
    )
    st.subheader("Identification is not authentication")
    st.write(
        "The classifiers select one of 99 participant labels. Genuine-versus-impostor "
        "verification with decision thresholds and FAR/FRR/EER is not implemented; "
        "therefore the current models do not make an access-grant decision."
    )
    st.subheader("Limitations and live-typing status")
    st.markdown(
        "- Dataset tester examples come from the model fit partition. A correct prediction is an in-sample demonstration, not proof of identity or generalization.\n"
        "- The dataset does not document units for its timing columns.\n"
        "- Live typing is pending authoritative formulas for the 46 features and a tested parity check from browser events to those features.\n"
        "- No saved model is retrained by this app; only the verified repetition-window LSTM is enabled for dataset-based inference."
    )
