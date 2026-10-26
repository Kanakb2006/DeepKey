# DeepKey Premium Dashboard

A local Streamlit dashboard with five pages in this order: **Front Page / Overview**,
**Model Testing**, **Dataset Explorer**, **Model Performance**, and **Project Methodology**.
It reuses saved metrics and the saved repetition-window LSTM; it does not train or
modify models.

## Launch

From the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run practical_tester\app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`. To bind only to
the local machine:

```powershell
.\venv\Scripts\python.exe -m streamlit run practical_tester\app.py --server.address 127.0.0.1
```

## Pages

- **Front Page / Overview:** project objective, four dataset/model KPIs, the modeling
  workflow, and a five-seed model accuracy comparison.
- **Model Testing:** selects an audited ten-attempt Session 1 fit window, applies the
  fit-only LSTM scaler and label mapping, validates all input requirements, and shows
  prediction probabilities. Use **Run Model Prediction** for this in-sample demonstration.
- **Dataset Explorer:** Session 1 filters, summary statistics, feature histogram,
  participant comparison, correlation heatmap, and coverage charts. Modeling exclusions
  are identified separately and are not applied to raw explorer statistics.
- **Model Performance:** per-seed results for all five metrics, mean/sample-SD summary,
  historical MLP row-holdout results, and single-run sequence metrics shown separately.
- **Project Methodology:** objective, models, dataset, validation protocol, and limitations.

Matplotlib is already available in the project environment; no packages need to be
installed.

## Evaluation and safety notes

- Data analysis and model development are limited to Session 1. The dashboard does not
  use Session 2 for evaluation or model selection.
- The LSTM scaler is reconstructed from the documented 6,928 fit attempts only; the
  MLP scaler is not used.
- The corrected LSTM modeling view excludes `p071` repetition `20` and `p087`
  repetition `8`. Raw dataset counts remain separate.
- Five-seed results use a fixed chronological partition (fit through repetition 70,
  embargo 71–79, validation from repetition 80). MLP predicts individual attempts;
  LSTM/GRU predict ten-attempt windows, so their prediction units differ.
- Historical row-holdout MLP and single-run metrics are displayed separately from the
  five-seed comparison.
- The tester's examples come from the training fit partition. A matching participant
  label does not establish generalization, identity, or authentication.
- Live typing is pending verified browser-event-to-feature extraction for all 46
  ordered model inputs. No raw keyboard capture or fabricated live prediction is used.
- Feature units are not documented in the dataset, and no genuine-user/impostor
  authentication threshold is implemented.
