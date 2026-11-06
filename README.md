# 🔐 DeepKey – Keystroke Dynamics Authentication Dashboard

**DeepKey** is a machine learning-based keystroke dynamics analysis and authentication research project that explores behavioral biometrics through typing patterns.

It uses statistical features extracted from keyboard interaction patterns and sequence-based deep learning models to analyze typing behavior. The project includes an interactive Streamlit dashboard for dataset exploration, model testing, performance evaluation, and understanding the complete methodology.

> **Note:** DeepKey is currently a research and demonstration project. It does not implement production-ready identity authentication or live keyboard-based authentication.

## 🚀 Project Overview

DeepKey explores how typing behavior can be analyzed using machine learning and deep learning techniques.

The project focuses on:
- Analyzing keystroke dynamics and behavioral patterns.
- Exploring and preprocessing typing-related datasets.
- Applying machine learning and sequence-based deep learning models.
- Evaluating model performance using multiple metrics and random seeds.
- Visualizing model predictions and dataset insights through an interactive dashboard.

## ✨ Key Features

- **Interactive Dashboard:** Five organized pages for exploring and evaluating the project.
- **Model Testing:** Test the saved repetition-window LSTM using audited examples.
- **Dataset Explorer:** Explore feature distributions, participant comparisons, correlations, and dataset coverage.
- **Model Performance:** Compare five-seed results using multiple evaluation metrics.
- **Model Comparison:** Analyze MLP, LSTM, and GRU approaches.
- **Project Methodology:** Understand the dataset, preprocessing, validation strategy, and limitations.
- **No Retraining Required:** Uses saved models and previously calculated metrics.

## 🖥️ Dashboard Pages

### 1. Front Page / Overview
- Project objective and introduction.
- Dataset and model KPIs.
- Machine learning workflow.
- Five-seed model accuracy comparison.

### 2. Model Testing
- Select an audited ten-attempt Session 1 fit window.
- Apply the saved LSTM scaler and label mapping.
- Validate all required model inputs.
- Display prediction probabilities.

### 3. Dataset Explorer
- Session 1 dataset filtering.
- Statistical summaries.
- Feature histograms.
- Participant comparisons.
- Correlation heatmap.
- Dataset coverage visualizations.

### 4. Model Performance
- Five-seed evaluation results.
- Accuracy and other performance metrics.
- Mean and sample standard deviation.
- Historical MLP results.
- Single-run sequence model results.

### 5. Project Methodology
- Project objectives.
- Dataset description.
- Model architectures.
- Validation methodology.
- Evaluation strategy.
- Limitations and future improvements.

## 🧠 Models and Technologies

**Models explored:**
- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

**Technology stack:**

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Development Environment | VS Code |

## 📊 Evaluation Methodology

The evaluation follows a controlled Session 1 experimental protocol.

- Five-seed evaluation with a fixed chronological partition.
- Training fit data through repetition 70.
- Embargo repetitions 71–79.
- Validation from repetition 80 onward.
- LSTM scaler reconstructed using only the documented 6,928 fit attempts.
- Modeling exclusions are handled separately from raw dataset statistics.

**Important:** MLP predicts individual attempts, whereas LSTM and GRU predict ten-attempt windows. Their prediction units are different and should be considered when comparing results.

## 🛠️ Installation and Setup

### Prerequisites
- Python installed.
- Git installed.
- Project repository cloned locally.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd DeepKey
```

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Launch the dashboard

```powershell
.\venv\Scripts\python.exe -m streamlit run practical_tester\app.py
```

### 5. Open in your browser

```text
http://localhost:8501
```

To restrict the server to the local machine:

```powershell
.\venv\Scripts\python.exe -m streamlit run practical_tester\app.py --server.address 127.0.0.1
```

## ⚠️ Important Notes and Limitations

- Model development and evaluation are restricted to Session 1.
- Session 2 is not used for evaluation or model selection.
- The tester uses examples from the training fit partition; therefore, it does not demonstrate unseen-user generalization.
- Live typing functionality is not implemented because verified browser-event-to-feature extraction for all 46 ordered model inputs is pending.
- No raw keyboard capture or fabricated live prediction is used.
- Dataset feature units are undocumented.
- A genuine-user/impostor authentication threshold has not been implemented.
- Historical and single-run results are reported separately from the five-seed comparison.

## 🔮 Future Improvements

- Implement verified live keystroke feature extraction.
- Explore genuine-user versus impostor authentication.
- Develop a suitable authentication threshold.
- Evaluate generalization on independent participants.
- Improve real-time prediction and dashboard usability.
- Extend the evaluation with additional datasets.



Interested in Machine Learning, Deep Learning, Artificial Intelligence, and Software Development.
<img width="1912" height="898" alt="image" src="https://github.com/user-attachments/assets/cec2f3e9-345c-43c5-b2d7-c7d792afbf9a" />
<img width="1907" height="877" alt="image" src="https://github.com/user-attachments/assets/c1176ab8-664a-41ff-9a4e-c995fb3f2a81" />
<img width="1902" height="887" alt="image" src="https://github.com/user-attachments/assets/58747cb7-66b4-4993-b12f-24e5415d8879" />
<img width="1905" height="866" alt="image" src="https://github.com/user-attachments/assets/07c9cd31-d728-4035-b9e5-6899a7e3e430" />
<img width="1917" height="902" alt="image" src="https://github.com/user-attachments/assets/436860f1-664c-4a95-9c36-041fb371372f" />

**


