import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)


def save_test_results(predictions, targets, dataset_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pr, pv = stats.pearsonr(targets, predictions)
    spr, sv = stats.spearmanr(targets, predictions)

    n = len(targets)
    pairs, concordant = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if targets[i] != targets[j]:
                pairs += 1
                if (targets[i] > targets[j]) == \
                        (predictions[i] > predictions[j]):
                    concordant += 1
    ci_val = concordant / pairs if pairs > 0 else 0.5

    pd.DataFrame([{
        "TestSetSize": n, "MSE": mse, "RMSE": np.sqrt(mse),
        "MAE": mae, "Pearson": pr, "p-value": pv,
        "Spearman": spr, "sp-value": sv,
        "R2": r2, "CI": ci_val
    }]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    pd.DataFrame({
        "Actual": targets, "Predicted": predictions,
        "Error": predictions - targets
    }).to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6,
                edgecolors='w', linewidths=0.5)
    lo = min(min(targets), min(predictions))
    hi = max(max(targets), max(predictions))
    plt.plot([lo, hi], [lo, hi], 'r--')
    p = np.poly1d(np.polyfit(targets, predictions, 1))
    plt.plot(targets, p(targets), "b-", alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{dataset_name}  Pearson={pr:.4f}  CI={ci_val:.4f}")
    plt.text(0.05, 0.95,
             f"N={n}\nMSE={mse:.4f}\nMAE={mae:.4f}\n"
             f"Pearson={pr:.4f}\nR2={r2:.4f}\nCI={ci_val:.4f}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "scatter_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(predictions - targets, kde=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {dataset_name}")
    plt.savefig(
        os.path.join(results_dir, "error_distribution.png"), dpi=300)
    plt.close()

    print(f"\n[Results] {dataset_name}: MSE={mse:.4f} "
          f"RMSE={np.sqrt(mse):.4f} MAE={mae:.4f} "
          f"Pearson={pr:.4f} Spearman={spr:.4f} "
          f"R2={r2:.4f} CI={ci_val:.4f}")

    return {"MSE": mse, "Pearson": pr, "R2": r2, "CI": ci_val}
