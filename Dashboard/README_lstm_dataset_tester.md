# DeepKey dataset-based LSTM tester

This separate Streamlit page demonstrates the saved repetition-window LSTM using existing corrected **Session 1 training attempts**. It is not a live typing test. Each input window is ten consecutive dataset attempts from one participant; it is an in-sample demonstration, not a held-out evaluation or identity verification.

The app uses the corrected exclusions and the training notebook's fit partition (Session 1 repetitions 1–70). It recreates `StandardScaler` from the 6,928 fit-partition attempt rows and recreates the alphabetically ordered `LabelEncoder` from those same participants. It checks the 46 ordered feature columns, model input/output shapes, finite values, consecutive repetitions, and the audited fit-window raw-row fingerprint before inference. The MLP scaler is not used.

The source CSV contains precomputed timing features, not individual key events. The app does not derive features from typing, retrain the model, write selected data to disk, or evaluate Session 2. Streamlit keeps the interaction local; the reset button discards the current selection and displayed prediction from the session.

## Run from the project root

In PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\Dashboard\lstm_dataset_tester.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit (normally `http://127.0.0.1:8501`). The project environment already declares Streamlit, TensorFlow, NumPy, pandas, and scikit-learn; this demo does not require a dependency-manifest change.

## Limitations

- The output is a 99-class participant identification prediction, not a genuine-user/impostor authentication result.
- The demo uses Session 1 fit-partition examples seen during model development; matching the true dataset label is not evidence of generalization.
- The model has no persisted class-label encoder or LSTM scaler artifact. The app reconstructs both according to the documented training pipeline and verifies expected counts, feature order, class count, model shapes, and fit-window IDs.
- No participant class or prediction proves who is typing.
