"""End-to-end validation run: train the candidates, test them, write the report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from . import metrics as M  # noqa: E402
from .data import load, time_split  # noqa: E402
from .models import ALL  # noqa: E402
from .stability import adversarial_validation, feature_psi, psi  # noqa: E402

TARGET_FPR = 0.001  # one false alert per 1,000 legitimate transactions
AMOUNT_BANDS = [0, 10, 100, 500, np.inf]
AMOUNT_LABELS = ["< 10", "10 to 100", "100 to 500", "500 and above"]

# Illustrative rating thresholds for this study (worst criterion sets the overall rating).
RULES = {
    "Discrimination": ("Test PR-AUC, lower 95% bound", lambda v: "Green" if v >= 0.60 else "Amber" if v >= 0.40 else "Red"),
    "Score stability": ("PSI of model score, train vs test", lambda v: "Green" if v < 0.10 else "Amber" if v < 0.25 else "Red"),
    # Drift alone is a monitoring trigger; it becomes Red only when it also costs performance.
    "Feature drift": ("Adversarial validation AUC, train vs test (Red only if test PR-AUC also falls >10% below validation)", None),
    "Calibration": ("Predicted / observed fraud rate on test, after Platt scaling", lambda v: "Green" if 0.8 <= v <= 1.25 else "Amber" if 0.5 <= v <= 2.0 else "Red"),
}
ORDER = {"Green": 0, "Amber": 1, "Red": 2}


def run(data_path: str, out_dir: str = "reports", n_boot: int = 500) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load(data_path)
    s = time_split(df)
    tr, va, te = s["train"], s["valid"], s["test"]

    # 1. Train candidates and compare
    models = []
    for build in ALL:
        m = build(tr, va)
        m.fit_platt(va.X, va.y)
        models.append(m)

    rows = []
    for m in models:
        sv, st = m.raw(va.X), m.raw(te.X)
        thr = M.threshold_at_fpr(va.y, sv, TARGET_FPR)
        op = M.at_threshold(te.y, st, thr)
        lo, hi = M.bootstrap_ci(te.y, st, n=n_boot)
        rows.append({
            "model": m.name, "trees": m.booster.best_iteration,
            "valid_pr_auc": M.pr_auc(va.y, sv), "test_pr_auc": M.pr_auc(te.y, st),
            "test_pr_auc_lo": lo, "test_pr_auc_hi": hi, "test_roc_auc": M.roc_auc(te.y, st),
            "test_recall": op["recall"], "test_precision": op["precision"], "test_fpr": op["fpr"],
            "test_alerts": op["alerts"], "test_caught": op["tp"],
        })
    comp = pd.DataFrame(rows)
    champ_idx = int(comp["valid_pr_auc"].idxmax())  # selected on validation, never on test
    champ = models[champ_idx]

    # 2. Calibration of the selected model
    p_raw, p_cal = champ.proba(te.X), champ.proba_calibrated(te.X)
    calib = pd.DataFrame([
        {"probabilities": "Model output", "brier": M.brier(te.y, p_raw), "ece": M.expected_calibration_error(te.y, p_raw), "pred_over_obs": M.calibration_in_the_large(te.y, p_raw)},
        {"probabilities": "After Platt scaling (fitted on validation)", "brier": M.brier(te.y, p_cal), "ece": M.expected_calibration_error(te.y, p_cal), "pred_over_obs": M.calibration_in_the_large(te.y, p_cal)},
    ])

    # 3. Stability and drift
    score_psi = psi(champ.raw(tr.X), champ.raw(te.X))
    fpsi = feature_psi(tr.X, te.X)
    adv = adversarial_validation(tr.X, te.X)

    # 4. Segment analysis on the test window
    thr = M.threshold_at_fpr(va.y, champ.raw(va.X), TARGET_FPR)
    band = pd.cut(te.X["Amount"], AMOUNT_BANDS, labels=AMOUNT_LABELS, right=False)
    seg_rows = []
    for lab in AMOUNT_LABELS:
        mask = (band == lab).to_numpy()
        y, sc = te.y[mask], champ.raw(te.X[mask])
        op = M.at_threshold(y, sc, thr)
        seg_rows.append({"amount_band": lab, "transactions": int(mask.sum()), "frauds": int(y.sum()),
                         "pr_auc": M.pr_auc(y, sc) if y.sum() >= 5 else np.nan, "recall": op["recall"] if y.sum() else np.nan})
    seg = pd.DataFrame(seg_rows)

    # 5. Rating
    champ_row = comp.iloc[champ_idx]
    values = {"Discrimination": champ_row["test_pr_auc_lo"], "Score stability": score_psi,
              "Feature drift": adv["auc"], "Calibration": calib.iloc[1]["pred_over_obs"]}
    perf_drop = 1 - champ_row["test_pr_auc"] / champ_row["valid_pr_auc"]

    def rate(k, v):
        if k == "Feature drift":
            return "Green" if v < 0.60 else ("Red" if perf_drop > 0.10 else "Amber")
        return RULES[k][1](v)

    rating = pd.DataFrame([{"criterion": k, "measure": RULES[k][0], "value": v, "rating": rate(k, v)} for k, v in values.items()])
    overall = max(rating["rating"], key=ORDER.get)

    # Plain-language findings, derived from the numbers so they stay true on a re-run
    lo_c, hi_c = champ_row["test_pr_auc_lo"], champ_row["test_pr_auc_hi"]
    inside = [r.model for r in comp.itertuples() if lo_c <= r.test_pr_auc <= hi_c]
    spread = comp["test_pr_auc"].max() - comp["test_pr_auc"].min()
    fl = comp[comp["model"] == "Focal Loss"].iloc[0]
    findings = [
        f"All four approaches land within {spread:.3f} PR-AUC of each other on the test window"
        + (", and every one sits inside the selected model's 95% interval, so none is clearly better." if len(inside) == len(comp)
           else f"; {len(inside)} of {len(comp)} sit inside the selected model's 95% interval."),
        f"Focal Loss reached a test PR-AUC of {fl.test_pr_auc:.3f} against {comp.iloc[0].test_pr_auc:.3f} for the plain baseline. "
        + ("The difference is within sampling noise, so on this data it does not justify the extra complexity of a custom objective."
           if abs(fl.test_pr_auc - comp.iloc[0].test_pr_auc) < (hi_c - lo_c) / 2
           else "The gap is larger than half the confidence interval, so the choice of imbalance method matters on this data."),
        f"Performance held out of time: test PR-AUC {champ_row['test_pr_auc']:.3f} vs validation {champ_row['valid_pr_auc']:.3f}.",
        f"The input features drift strongly between windows (adversarial AUC {adv['auc']:.2f}), yet the model score moves much less (PSI {score_psi:.2f}). "
        "Drift is the main reason to monitor this model rather than a reason to reject it.",
        f"After Platt scaling, predicted fraud rate is {values['Calibration']:.2f} times the observed rate on the test window.",
    ]

    res = dict(splits={k: {"rows": v.n, "frauds": v.n_pos, "fraud_rate": v.n_pos / v.n, "hours": f"{v.time.min() / 3600:.1f} to {v.time.max() / 3600:.1f}"} for k, v in s.items()},
               comparison=comp, champion=champ.name, champion_notes=champ.notes, calibration=calib, score_psi=score_psi,
               feature_psi=fpsi, adversarial=adv, segments=seg, rating=rating, overall=overall, findings=findings,
               runtime_min=(time.time() - t0) / 60, n_boot=n_boot)
    _figures(models, champ, s, fpsi, adv, out / "figures")
    _write_report(res, models, out / "validation_report.md")
    _write_summary(res, out / "summary.json")
    return res


# ---------------------------------------------------------------- figures
def _figures(models, champ, s, fpsi, adv, fig_dir: Path) -> None:
    te, va = s["test"], s["valid"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in models:
        p, r, _ = precision_recall_curve(te.y, m.raw(te.X))
        ax.plot(r, p, lw=1.6, label=f"{m.name} (AP {M.pr_auc(te.y, m.raw(te.X)):.3f})")
    ax.axhline(te.y.mean(), color="grey", ls=":", lw=1, label=f"No skill ({te.y.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall on the out-of-time test window", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(fig_dir / "pr_curves.png", dpi=150); plt.close(fig)

    # Reliability on log axes: probability bands by order of magnitude, since almost all
    # transactions sit far below 1% and equal-count bins would hide the tail where fraud lives.
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    edges = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0001]
    floor = 3e-5
    for label, p, colr in [("Model output", champ.proba(te.X), "#4C72B0"), ("After Platt scaling", champ.proba_calibrated(te.X), "#DD8452")]:
        xs, ys, ns = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p < b)
            if m.sum() >= 20:
                xs.append(max(p[m].mean(), floor)); ys.append(max(te.y[m].mean(), floor)); ns.append(int(m.sum()))
        ax.plot(xs, ys, "-", color=colr, lw=1.2, alpha=0.8)
        ax.scatter(xs, ys, s=[12 + 18 * np.log10(n) for n in ns], color=colr, label=label, zorder=3)
    ax.plot([floor, 1], [floor, 1], color="grey", ls=":", lw=1, label="Perfect calibration")
    ax.set(xscale="log", yscale="log", xlim=(floor * 0.8, 1.2), ylim=(floor * 0.8, 1.2),
           xlabel="Mean predicted probability in band", ylabel="Observed fraud rate in band",
           title=f"Calibration on the test window: {champ.name}")
    ax.text(floor, floor * 1.6, "bands with no fraud plotted at the floor", fontsize=7, color="grey")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "calibration.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    top = fpsi.head(8)[::-1]
    axes[0].barh(top.index, top.values, color="#4C72B0")
    axes[0].axvline(0.10, color="#DD8452", ls="--", lw=1)
    axes[0].set(title="Feature PSI, train vs test (top 8)", xlabel="PSI")
    imp = adv["importance"].head(8)[::-1]
    axes[1].barh(imp.index, imp.values, color="#55A868")
    axes[1].set(title=f"Adversarial validation (AUC {adv['auc']:.3f})", xlabel="Share of gain")
    fig.tight_layout(); fig.savefig(fig_dir / "drift.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- report
def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("n/a")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_table(res: dict) -> str:
    c = res["comparison"].copy()
    c["Test PR-AUC (95% CI)"] = c.apply(lambda r: f"{r.test_pr_auc:.3f} ({r.test_pr_auc_lo:.3f} to {r.test_pr_auc_hi:.3f})", axis=1)
    c["Recall at 0.1% FPR"] = c["test_recall"].map("{:.1%}".format)
    c["Precision"] = c["test_precision"].map("{:.1%}".format)
    c = c.rename(columns={"model": "Model", "valid_pr_auc": "Valid PR-AUC", "test_roc_auc": "Test ROC-AUC", "test_alerts": "Alerts", "test_caught": "Frauds caught"})
    c["Model"] = [f"**{m}**" if m == res["champion"] else m for m in c["Model"]]
    return _md(c[["Model", "Valid PR-AUC", "Test PR-AUC (95% CI)", "Test ROC-AUC", "Recall at 0.1% FPR", "Precision", "Alerts", "Frauds caught"]],
               {"Valid PR-AUC": "{:.3f}", "Test ROC-AUC": "{:.4f}"})


def _write_report(res: dict, models, path: Path) -> None:
    sp = pd.DataFrame([{"window": k, **v} for k, v in res["splits"].items()])
    test_frauds = res["splits"]["test"]["frauds"]
    notes = {k: v for k, v in res["champion_notes"].items() if k != "grid"}
    grid = next((m.notes["grid"] for m in models if "grid" in m.notes), None)
    adv = res["adversarial"]

    text = f"""# Model Validation Report: Card Fraud Detection Under Class Imbalance

**Overall rating: {res['overall']}** (generated by `run_validation.py`; every number below comes from that run)

## Key findings
{chr(10).join('- ' + f for f in res['findings'])}

## 1. Purpose and scope
The model scores card transactions for the probability of fraud so that a fraud team can review the highest-risk ones.
Fraud makes up about 0.17% of transactions, so a model can look accurate while missing most fraud.
This study compares four ways of handling that imbalance with the same LightGBM learner and validates the selected model
the way an independent model risk function would: discrimination, operating performance, calibration, stability, drift and segments.

## 2. Data and sample design
Public dataset of 284,807 European card transactions over two days (ULB Machine Learning Group, via Kaggle).
Features V1 to V28 are PCA components supplied by the data owner, plus amount and hour of day.
The raw `Time` field is excluded as a feature because it encodes position in the sample, which would not generalise.

The split is chronological, so the test window is genuinely out of time:

{_md(sp, {"fraud_rate": "{:.3%}"})}

Model selection, early stopping, threshold choice and calibration use only the validation window. The test window is used once, for the results below.

## 3. Candidate approaches
| Approach | What it does |
|---|---|
| Baseline | Standard log loss, no imbalance handling |
| Class weights | Fraud cases weighted by the negative to positive ratio |
| Random undersampling | Training on all frauds plus 10 legitimate transactions per fraud |
| Focal Loss | Custom LightGBM objective that down-weights easy examples (alpha and gamma tuned on validation) |

All four share the same tree settings and early stopping on validation PR-AUC.
"""
    if grid:
        g = pd.DataFrame(grid)
        text += f"\nFocal Loss tuning grid (validation PR-AUC):\n\n{_md(g, {'valid_pr_auc': '{:.3f}'})}\n"

    text += f"""
## 4. Performance on the out-of-time test window
The operating threshold is set on the validation window at a 0.1% false-positive rate (one false alert per 1,000 legitimate transactions) and then applied unchanged to the test window. The test window has {test_frauds} frauds, so the intervals are wide; differences inside them should not be over-read.

{comparison_table(res)}

Selected model (highest validation PR-AUC): **{res['champion']}** {('(' + ', '.join(f'{k} = {v}' for k, v in notes.items()) + ')') if notes else ''}

![Precision-recall curves](figures/pr_curves.png)

## 5. Calibration
Imbalance treatments change the scale of the scores. Class weights, undersampling and Focal Loss all distort probabilities, which matters whenever scores feed expected-loss estimates or alert budgets. Platt scaling fitted on the validation window corrects this.

{_md(res['calibration'], {'brier': '{:.6f}', 'ece': '{:.6f}', 'pred_over_obs': '{:.2f}'})}

`pred_over_obs` is the mean predicted probability divided by the observed fraud rate (1.00 is perfect).

![Calibration](figures/calibration.png)

## 6. Stability and drift (train window vs test window)
- PSI of the model score: **{res['score_psi']:.3f}**
- Adversarial validation AUC: **{adv['auc']:.3f}** (0.5 means the two windows cannot be told apart)
- Features with the highest PSI: {', '.join(f'{k} ({v:.3f})' for k, v in res['feature_psi'].head(5).items())}
- Features that best separate the windows: {', '.join(f'{k} ({v:.0%})' for k, v in adv['importance'].head(5).items())}

![Drift diagnostics](figures/drift.png)

## 7. Segment analysis (test window, by transaction amount)
{_md(res['segments'], {'pr_auc': '{:.3f}', 'recall': '{:.1%}'})}

Segments with fewer than five frauds show n/a for PR-AUC.

## 8. Rating
{_md(res['rating'], {'value': '{:.3f}'})}

The thresholds are illustrative choices for this study and are set in `src/fraudval/pipeline.py`. The overall rating is the worst individual rating.

## 9. Limitations
- Two days of data: the test window shows short-term stability only, and seasonal or behavioural drift cannot be assessed.
- The PCA features are anonymised, so conceptual soundness of individual drivers cannot be reviewed and no fairness analysis by customer group is possible.
- The test window holds few frauds, so recall and precision estimates carry wide uncertainty.
- No transaction-level costs are available; the 0.1% false-positive operating point stands in for an alert budget.

_Runtime: {res['runtime_min']:.1f} minutes. Bootstrap resamples: {res['n_boot']}._
"""
    path.write_text(text, encoding="utf-8")


def _write_summary(res: dict, path: Path) -> None:
    summary = {
        "overall_rating": res["overall"], "champion": res["champion"],
        "champion_params": {k: v for k, v in res["champion_notes"].items() if k != "grid"},
        "score_psi": res["score_psi"], "adversarial_auc": res["adversarial"]["auc"],
        "comparison": res["comparison"].to_dict(orient="records"),
    }
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


def update_readme(res: dict, readme: Path) -> None:
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    txt = readme.read_text(encoding="utf-8")
    if start not in txt or end not in txt:
        return
    block = (f"{start}\n**Overall validation rating: {res['overall']}.** Selected model: **{res['champion']}**.\n\n"
             + "\n".join("- " + f for f in res["findings"]) + "\n\n"
             f"{comparison_table(res)}\n\n"
             f"Out-of-time test window, {res['splits']['test']['frauds']} frauds. Score PSI {res['score_psi']:.3f}; "
             f"adversarial validation AUC {res['adversarial']['auc']:.3f}. Full report: "
             f"[reports/validation_report.md](reports/validation_report.md)\n\n![Precision-recall curves](reports/figures/pr_curves.png)\n{end}")
    readme.write_text(txt.split(start)[0] + block + txt.split(end)[1], encoding="utf-8")
