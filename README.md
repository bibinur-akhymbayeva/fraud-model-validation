# Fraud Detection Under Extreme Class Imbalance: A Model Validation Study

Card fraud is rare. In the public dataset used here it is 0.17% of transactions, so a model that flags nothing is 99.8% accurate and completely useless. This project compares four ways of dealing with that imbalance in a LightGBM fraud model, including a custom **Focal Loss** objective, and then validates the selected model the way an independent model risk team would: out-of-time testing, operating-point performance, calibration, stability, drift and segment analysis, ending in a rating.

I work in model validation at a bank, where I review fraud and credit models built by other teams. This repository applies the same checks to a public dataset, with all code written from scratch.

## Results

<!-- RESULTS:START -->
**Overall validation rating: Amber.** Selected model: **Class weights**.

- All four approaches land within 0.025 PR-AUC of each other on the test window, and every one sits inside the selected model's 95% interval, so none is clearly better.
- Focal Loss reached a test PR-AUC of 0.803 against 0.803 for the plain baseline. The difference is within sampling noise, so on this data it does not justify the extra complexity of a custom objective.
- Performance held out of time: test PR-AUC 0.806 vs validation 0.792.
- The input features drift strongly between windows (adversarial AUC 1.00), yet the model score moves much less (PSI 0.12). Drift is the main reason to monitor this model rather than a reason to reject it.
- After Platt scaling, predicted fraud rate is 0.99 times the observed rate on the test window.

| Model | Valid PR-AUC | Test PR-AUC (95% CI) | Test ROC-AUC | Recall at 0.1% FPR | Precision | Alerts | Frauds caught |
|---|---|---|---|---|---|---|---|
| Baseline (log loss) | 0.785 | 0.803 (0.715 to 0.882) | 0.9854 | 81.3% | 53.0% | 115 | 61 |
| **Class weights** | 0.792 | 0.806 (0.718 to 0.883) | 0.9758 | 80.0% | 50.4% | 119 | 60 |
| Random undersampling | 0.780 | 0.781 (0.692 to 0.860) | 0.9840 | 80.0% | 50.4% | 119 | 60 |
| Focal Loss | 0.789 | 0.803 (0.717 to 0.879) | 0.9849 | 80.0% | 49.2% | 122 | 60 |

Out-of-time test window, 75 frauds. Score PSI 0.120; adversarial validation AUC 1.000. Full report: [reports/validation_report.md](reports/validation_report.md)

![Precision-recall curves](reports/figures/pr_curves.png)
<!-- RESULTS:END -->

## What the study does

**Sample design.** A chronological split into train (first 60%), validation (next 20%) and an out-of-time test window (last 20%). Every choice (model selection, early stopping, alert threshold, recalibration) uses the validation window only. The test window is touched once.

**Four candidate approaches, one learner.**

| Approach | Idea |
|---|---|
| Baseline | Standard log loss |
| Class weights | Fraud cases weighted by the negative to positive ratio |
| Random undersampling | All frauds plus 10 legitimate transactions per fraud |
| Focal Loss | `alpha * (1 - p_t)^gamma * log loss`: easy cases count less, so training focuses on hard ones |

Focal Loss is implemented as a custom LightGBM objective with an analytic gradient (`src/fraudval/focal_loss.py`) and checked against finite differences in `tests/`.

**Validation checks.**
- Discrimination: PR-AUC with bootstrap confidence intervals (ROC-AUC is reported but says little at 0.17% prevalence)
- Operating point: recall and precision at a 0.1% false-positive rate, threshold fixed on validation
- Calibration: Brier score, expected calibration error, predicted vs observed fraud rate, before and after Platt scaling
- Stability: Population Stability Index of the score and of each feature
- Drift: adversarial validation (can a model tell the training window from the test window?)
- Segments: performance by transaction amount band
- Rating: Green, Amber or Red per criterion, overall rating set by the worst one

## Run it

```bash
git clone https://github.com/bibinur-akhymbayeva/fraud-model-validation.git
cd fraud-model-validation
pip install -r requirements.txt
# download creditcard.csv into data/ (see data/README.md)
python run_validation.py --data data/creditcard.csv
pytest -q
```

The run takes a few minutes on a laptop and writes `reports/validation_report.md`, the figures and `reports/summary.json`, and updates the Results section above.

## Repository layout

```
run_validation.py          entry point
src/fraudval/
  data.py                  loading, hour-of-day feature, chronological split
  focal_loss.py            Focal Loss objective, gradient, Hessian, initial score
  models.py                the four candidate approaches
  metrics.py               PR-AUC, bootstrap intervals, operating point, calibration
  stability.py             PSI and adversarial validation
  pipeline.py              end-to-end run, figures, report
tests/                     unit tests for the loss and the validation metrics
reports/                   generated validation report and figures
```

## Limitations

Two days of data allow only a short-term stability check. The features are anonymised PCA components, so the economic sense of individual drivers cannot be reviewed. The test window holds fewer than a hundred frauds, so the confidence intervals are wide and small differences between methods should not be over-read.

## References

- Lin, Goyal, Girshick, He and Dollár, *Focal Loss for Dense Object Detection*, ICCV 2017
- Dal Pozzolo, Caelen, Johnson and Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015
- Board of Governors of the Federal Reserve System, *SR 11-7: Guidance on Model Risk Management*, 2011

## Author

Bibinur Akhymbayeva, model validation, Almaty. [LinkedIn](https://www.linkedin.com/in/bibinur-akhymbayeva)

Licensed under the MIT License.
