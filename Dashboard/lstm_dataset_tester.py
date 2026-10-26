from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler


st.set_page_config(
    page_title="DeepKey Dataset Tester",
    page_icon="🔐",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fixed-text.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "deepkey_lstm_repetition_w10.keras"

WINDOW_LENGTH = 10
FIT_MAX_REPETITION = 70
EXPECTED_ATTEMPT_ROWS = 6928
EXPECTED_WINDOW_COUNT = 6021
EXPECTED_MODEL_INPUT_SHAPE = (None, WINDOW_LENGTH, 46)
EXPECTED_MODEL_OUTPUT_SHAPE = (None, 99)
EXPECTED_FIT_WINDOW_IDS_SHA256 = (
    "fddb824bb58a983dfa3b4e6eef9905a215e53db82fce0dfa204530c82e599b85"
)

EXPECTED_FEATURE_COLUMNS = [
    "DU.v.v",
    "DD.v.p",
    "DU.v.p",
    "UD.v.p",
    "UU.v.p",
    "DU.p.p",
    "DD.p.w",
    "DU.p.w",
    "UD.p.w",
    "UU.p.w",
    "DU.w.w",
    "DD.w.j",
    "DU.w.j",
    "UD.w.j",
    "UU.w.j",
    "DU.j.j",
    "DD.j.k",
    "DU.j.k",
    "UD.j.k",
    "UU.j.k",
    "DU.k.k",
    "DD.k.e",
    "DU.k.e",
    "UD.k.e",
    "UU.k.e",
    "DU.e.e",
    "DD.e.u",
    "DU.e.u",
    "UD.e.u",
    "UU.e.u",
    "DU.u.u",
    "DD.u.r",
    "DU.u.r",
    "UD.u.r",
    "UU.u.r",
    "DU.r.r",
    "DD.r.k",
    "DU.r.k",
    "UD.r.k",
    "UU.r.k",
    "DU.k.k.1",
    "DD.k.b",
    "DU.k.b",
    "UD.k.b",
    "UU.k.b",
    "DU.b.b",
]

EXCLUSIONS = [
    ("p020", 2, 4),
    ("p071", 1, 20),
    ("p087", 1, 8),
]


def ordered_window_ids(fit_rows: pd.DataFrame) -> list[tuple[int, ...]]:
    windows: list[tuple[int, ...]] = []
    ordered = fit_rows.sort_values(["participant", "repetition"])

    for _, participant_rows in ordered.groupby("participant", sort=True):
        row_ids = participant_rows["_raw_row_id"].tolist()
        repetitions = participant_rows["repetition"].astype(int).tolist()
        run_start = 0

        for position in range(1, len(participant_rows) + 1):
            at_run_end = (
                position == len(participant_rows)
                or repetitions[position] != repetitions[position - 1] + 1
            )
            if not at_run_end:
                continue

            run_ids = row_ids[run_start:position]
            windows.extend(
                tuple(run_ids[start : start + WINDOW_LENGTH])
                for start in range(len(run_ids) - WINDOW_LENGTH + 1)
            )
            run_start = position

    return windows


def window_ids_sha256(windows: list[tuple[int, ...]]) -> str:
    payload = "\n".join(",".join(map(str, row_ids)) for row_ids in windows)
    import hashlib

    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@st.cache_resource
def load_dataset_pipeline():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw fixed-text dataset not found: {RAW_DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Repetition-window LSTM model not found: {MODEL_PATH}")

    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_session_1 = df_raw.loc[df_raw["session"].eq(1)].copy()
    del df_raw

    # The p020/session-2 exclusion is outside this Session 1-only pipeline.
    for participant, session, repetition in EXCLUSIONS:
        if session != 1:
            continue
        matching = (
            df_session_1["participant"].eq(participant)
            & df_session_1["session"].eq(session)
            & df_session_1["repetition"].eq(repetition)
        )
        if int(matching.sum()) != 1:
            raise ValueError(
                f"Expected exactly one corrected-data exclusion: "
                f"{(participant, session, repetition)}"
            )
        df_session_1 = df_session_1.loc[~matching].copy()

    excluded_columns = {"participant", "session", "repetition", "total time"}
    feature_columns = [
        column for column in df_session_1.columns if column not in excluded_columns
    ]
    if feature_columns != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            "Raw feature columns or their order differ from the audited LSTM schema."
        )

    df_session_1["_raw_row_id"] = df_session_1.index
    if len(df_session_1) != 9876 or df_session_1["participant"].nunique() != 99:
        raise ValueError("Corrected Session 1 data has an unexpected shape.")

    fit_rows = df_session_1.loc[
        df_session_1["repetition"].le(FIT_MAX_REPETITION)
    ].copy()
    if len(fit_rows) != EXPECTED_ATTEMPT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ATTEMPT_ROWS} fit attempts, found {len(fit_rows)}."
        )

    X_fit = fit_rows[feature_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(X_fit).all():
        raise ValueError("Fit-partition feature data contains missing or infinite values.")

    scaler = StandardScaler()
    scaler.fit(X_fit)
    if int(scaler.n_samples_seen_) != EXPECTED_ATTEMPT_ROWS:
        raise ValueError("Scaler was not fitted on exactly the documented fit attempts.")

    label_encoder = LabelEncoder().fit(fit_rows["participant"])
    if len(label_encoder.classes_) != EXPECTED_MODEL_OUTPUT_SHAPE[-1]:
        raise ValueError("Fit-partition label mapping does not contain 99 classes.")

    windows = ordered_window_ids(fit_rows)
    fingerprint = window_ids_sha256(windows)
    if len(windows) != EXPECTED_WINDOW_COUNT:
        raise ValueError(f"Expected {EXPECTED_WINDOW_COUNT} fit windows, found {len(windows)}.")
    if fingerprint != EXPECTED_FIT_WINDOW_IDS_SHA256:
        raise ValueError("Fit window row IDs differ from the audited training definition.")

    participant_by_id = fit_rows.set_index("_raw_row_id")["participant"].to_dict()
    repetition_by_id = (
        fit_rows.set_index("_raw_row_id")["repetition"].astype(int).to_dict()
    )
    window_by_participant: dict[str, list[tuple[int, ...]]] = {}
    for row_ids in windows:
        participants = {participant_by_id[row_id] for row_id in row_ids}
        repetitions = [repetition_by_id[row_id] for row_id in row_ids]
        if len(participants) != 1:
            raise AssertionError("A training window crosses participant boundaries.")
        if any(
            right != left + 1
            for left, right in zip(repetitions, repetitions[1:])
        ):
            raise AssertionError("A training window crosses a missing repetition.")
        participant = next(iter(participants))
        window_by_participant.setdefault(participant, []).append(row_ids)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if tuple(model.input_shape) != EXPECTED_MODEL_INPUT_SHAPE:
        raise ValueError(f"Unexpected LSTM input shape: {model.input_shape}")
    if tuple(model.output_shape) != EXPECTED_MODEL_OUTPUT_SHAPE:
        raise ValueError(f"Unexpected LSTM output shape: {model.output_shape}")

    return (
        model,
        df_session_1,
        fit_rows,
        feature_columns,
        scaler,
        label_encoder,
        window_by_participant,
    )


def clear_demo_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("prediction_", "participant_", "window_")):
            del st.session_state[key]
    st.session_state["selection_version"] = (
        st.session_state.get("selection_version", 0) + 1
    )


st.title("DeepKey — Keystroke Behavioral Biometrics")
st.subheader("Dataset-Based Repetition-Window LSTM Tester")
st.info(
    "This local demonstration selects existing Session 1 dataset attempts. "
    "It is not a live typing test, held-out evaluation, or authentication system. "
    "A predicted participant label does not establish a typist's identity."
)

try:
    (
        model,
        session_1_rows,
        fit_rows,
        feature_columns,
        scaler,
        label_encoder,
        window_by_participant,
    ) = load_dataset_pipeline()
except (FileNotFoundError, ValueError, AssertionError) as error:
    st.error(f"DeepKey could not load a verified inference pipeline: {error}")
    st.stop()

st.caption(
    "Input source: corrected Session 1 fit partition only (repetitions 1–70). "
    "Session 2 is excluded. No new features are derived from raw keystroke events."
)

model_col, sequence_col, feature_col, class_col = st.columns(4)
model_col.metric("Model", "LSTM (64 units)")
sequence_col.metric("Sequence length", WINDOW_LENGTH)
feature_col.metric("Timing features", len(feature_columns))
class_col.metric("Participant classes", len(label_encoder.classes_))

st.markdown(
    "Choose a participant and one available sequence of **10 consecutive attempts**. "
    "Windows are grouped by participant, sorted by repetition, and broken at every "
    "missing repetition. The scaler and class mapping are reconstructed from the "
    "documented fit attempts only."
)

if "selection_version" not in st.session_state:
    st.session_state["selection_version"] = 0

reset_col, _ = st.columns([1, 5])
if reset_col.button("Reset / discard selection"):
    clear_demo_state()
    st.rerun()

version = st.session_state["selection_version"]
participants = sorted(window_by_participant)
participant = st.selectbox(
    "Participant",
    participants,
    key=f"participant_{version}",
)
participant_windows = window_by_participant[participant]
repetition_by_row = fit_rows.set_index("_raw_row_id")["repetition"].to_dict()
window_by_start: dict[int, tuple[int, ...]] = {
    int(repetition_by_row[window[0]]): window
    for window in participant_windows
}
window_starts = sorted(window_by_start)
start_repetition = st.selectbox(
    "Sequence start repetition",
    window_starts,
    format_func=lambda start: (
        f"Repetitions {start}–{start + WINDOW_LENGTH - 1}"
    ),
    key=f"window_{version}",
)
selected_row_ids = window_by_start[start_repetition]
selected_attempts = fit_rows.loc[list(selected_row_ids)].sort_values("repetition")

checks = {
    "Exactly 10 attempts selected": len(selected_attempts) == WINDOW_LENGTH,
    "One participant throughout": (
        selected_attempts["participant"].nunique() == 1
        and selected_attempts["participant"].iloc[0] == participant
    ),
    "Session 1 only": bool(selected_attempts["session"].eq(1).all()),
    "Fit partition only (repetition ≤70)": bool(
        selected_attempts["repetition"].le(FIT_MAX_REPETITION).all()
    ),
    "Consecutive repetition IDs": (
        selected_attempts["repetition"].astype(int).diff().dropna().eq(1).all()
    ),
    "Expected feature names and order": (
        list(selected_attempts[feature_columns].columns) == EXPECTED_FEATURE_COLUMNS
    ),
    "All feature values finite": bool(
        np.isfinite(
            selected_attempts[feature_columns]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        ).all()
    ),
}
checks_passed = all(checks.values())

st.subheader("Selected attempts")
st.dataframe(
    selected_attempts[["participant", "session", "repetition"]].reset_index(drop=True),
    width="stretch",
    hide_index=True,
)

with st.expander("Show the 46 dataset timing features for all 10 attempts"):
    st.dataframe(
        selected_attempts[["repetition", *feature_columns]].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

st.subheader("Input validation")
for description, passed in checks.items():
    st.write(f"{'✅' if passed else '❌'} {description}")
st.write(
    "Scaler fitted rows:",
    int(scaler.n_samples_seen_),
    "| Fit-partition attempts:",
    len(fit_rows),
    "| Encoder classes:",
    len(label_encoder.classes_),
)

input_features = selected_attempts[feature_columns].apply(
    pd.to_numeric, errors="raise"
).to_numpy(dtype=np.float64)
scaled_features = scaler.transform(input_features)
model_input = scaled_features[np.newaxis, :, :].astype(np.float32)
tensor_shape_ok = model_input.shape == (1, WINDOW_LENGTH, len(feature_columns))
scaled_values_finite = bool(np.isfinite(model_input).all())
if tensor_shape_ok:
    st.write(f"Model input tensor shape: `{model_input.shape}`")
else:
    st.error(f"Unexpected model input tensor shape: {model_input.shape}")
st.write(f"{'✅' if scaled_values_finite else '❌'} Scaled model input contains only finite values")
checks_passed = checks_passed and tensor_shape_ok and scaled_values_finite

selection_signature = (participant, int(start_repetition))
prediction_key = "prediction_result"
if (
    prediction_key in st.session_state
    and st.session_state[prediction_key]["selection"] != selection_signature
):
    del st.session_state[prediction_key]

if st.button("Predict participant", type="primary", disabled=not checks_passed):
    probabilities = model.predict(model_input, verbose=0)[0]
    if probabilities.shape != (len(label_encoder.classes_),):
        st.error(f"Unexpected model output shape: {probabilities.shape}")
        st.stop()
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
        st.error("Model returned invalid class probabilities.")
        st.stop()
    predicted_class_index = int(np.argmax(probabilities))
    st.session_state[prediction_key] = {
        "selection": selection_signature,
        "true_participant": participant,
        "predicted_participant": str(
            label_encoder.inverse_transform([predicted_class_index])[0]
        ),
        "probabilities": probabilities,
        "class_names": label_encoder.classes_.copy(),
    }

if prediction_key in st.session_state:
    result = st.session_state[prediction_key]
    predicted = result["predicted_participant"]
    true_label = result["true_participant"]
    st.subheader("Dataset inference result")
    true_col, prediction_col, confidence_col = st.columns(3)
    true_col.metric("True dataset participant", true_label)
    prediction_col.metric("Predicted participant", predicted)
    confidence_col.metric(
        "Top-class confidence",
        f"{float(np.max(result['probabilities'])):.1%}",
    )
    if predicted == true_label:
        st.success("Prediction matches the selected dataset participant.")
    else:
        st.warning("Prediction does not match the selected dataset participant.")

    top_indices = np.argsort(result["probabilities"])[::-1][:5]
    top_predictions = pd.DataFrame(
        {
            "Rank": np.arange(1, len(top_indices) + 1),
            "Participant": result["class_names"][top_indices],
            "Probability": result["probabilities"][top_indices],
        }
    )
    st.markdown("**Top participant probabilities**")
    st.dataframe(
        top_predictions.style.format({"Probability": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )

st.caption(
    "This is an in-sample dataset inference demonstration using attempts from the "
    "model's Session 1 fit partition. It does not verify a live person and is not "
    "a genuine-user/impostor authentication decision."
)
