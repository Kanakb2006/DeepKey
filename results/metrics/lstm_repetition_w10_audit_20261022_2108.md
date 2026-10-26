# Repetition-window LSTM audit

Audit date: 2026-10-22

## Scope and inputs

Read-only audit of `notebooks/04_lstm_repetition_sequences.ipynb`, `data/raw/fixed-text.csv`, `models/deepkey_lstm_repetition_w10.keras`, and `results/metrics/lstm_repetition_w10_validation.csv`. Session 1 only was used to reconstruct data and predictions. No Session 2 score or evaluation was read.

## Partition and window checks

The corrected Session 1 data has 9,876 attempts, 46 ordered timing features, and 99 participants. The chronological partition rules in the notebook produce:

| Partition | Repetitions | Attempt rows |
|---|---:|---:|
| Fit | 1–70 | 6,928 |
| Embargo | 71–79 | 891 |
| Validation | 80 and later | 2,057 |

The nine-repetition embargo equals `window_length - 1` for length-10 windows. Independent reconstruction produced 6,021 fit windows and 1,166 validation windows, with tensor shapes `(6021, 10, 46)` and `(1166, 10, 46)`.

Verified:

- No raw attempt row appears in both fit and validation windows (intersection size 0).
- No window crosses participant boundaries or a missing repetition (0 invalid windows).
- The scaler was fit on 6,928 fit-partition rows; its learned mean matches the fit-row feature means.
- `LabelEncoder` fit participants cover all 99 classes with encoded labels 0–98. The model has 99 softmax output units, in the notebook's `LabelEncoder` class order.

## Reloaded-model metric check

The saved model loaded successfully with input shape `(None, 10, 46)` and output shape `(None, 99)`. Predictions were recomputed on the 1,166 independently rebuilt validation windows. All five metrics exactly matched the saved CSV values at the reported floating-point precision (difference 0.0):

| Metric | Recomputed | CSV |
|---|---:|---:|
| Accuracy | 0.695540309 | 0.695540309 |
| Balanced accuracy | 0.695469850 | 0.695469850 |
| Macro precision | 0.714267975 | 0.714267975 |
| Macro recall | 0.695469850 | 0.695469850 |
| Macro-F1 | 0.666556482 | 0.666556482 |

Accuracy and macro-F1 were computed from the same 1,166 true labels and the same predictions. Their difference is not an inconsistency: accuracy is overall sample accuracy, whereas macro-F1 averages the per-class F1 scores. Balanced accuracy equals macro recall here, as expected. The small difference between accuracy and balanced accuracy reflects unequal validation-window counts per participant.

## Early stopping and provenance limitation

The notebook configures early stopping on `val_loss` with patience 10 and `restore_best_weights=True`, then saves the model after `model.fit` returns. The metrics CSV records best epoch 41. This supports that the training code intended to save the best validation-loss weights.

The saved model archive contains architecture/configuration and weights, but not the training history, a best-epoch checkpoint, or a weight-to-epoch identifier. Therefore, the audit can confirm the callback configuration and reproduce the saved model's metrics, but cannot independently prove from the saved artifact alone that those weights are exactly the epoch-41 checkpoint.

## Conclusion

Partition counts, window counts, boundaries, scaler scope, class coverage/output dimensionality, and all five validation metrics passed independent reconstruction. No existing notebook, model, or metrics file was modified. The only audit output is this report.
