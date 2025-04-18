import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

torch.set_printoptions(precision=4, sci_mode=False)

mem = [...]
perturb_mem = [...]
nom = [...]
perturb_nom = [...]

mem = torch.tensor(mem)
perturb_mem = torch.tensor(perturb_mem)
nom = torch.tensor(nom)
perturb_nom = torch.tensor(perturb_nom)

tensor_diff_member = mem - perturb_mem
tensor_diff_nomember = nom - perturb_nom

diff1 = tensor_diff_member.tolist()
diff2 = tensor_diff_nomember.tolist()

labels = [1] * len(diff1) + [0] * len(diff2)
all_diff = diff1 + diff2

def optimize_threshold(labels, all_diff):
    best_thr = 0.5
    best_acc = 0.0
    thresholds = np.arange(0, 1, 0.01)
    for thr in thresholds:
        preds = [1 if d > thr else 0 for d in all_diff]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc

best_threshold, best_accuracy = optimize_threshold(labels, all_diff)
print(f"Optimized Threshold: {best_threshold:.2f}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

predictions = [1 if d > best_threshold else 0 for d in all_diff]
y_true = labels
y_scores = all_diff
auc = roc_auc_score(y_true, y_scores)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

red_above = sum(1 for d in diff1 if d > best_threshold)
red_total = len(diff1)
blue_below = sum(1 for d in diff2 if d <= best_threshold)
blue_total = len(diff2)

red_pct = red_above / red_total * 100 if red_total > 0 else 0.0
blue_pct = blue_below / blue_total * 100 if blue_total > 0 else 0.0

print(f"member > threshold: {red_above}/{red_total} = {red_pct:.2f}%")
print(f"nomember <= threshold: {blue_below}/{blue_total} = {blue_pct:.2f}%")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(diff1)), diff1, color="red", label=f"member (> {best_threshold:.2f}): {red_pct:.2f}%")
plt.scatter(range(len(diff2)), diff2, color="blue", label=f"nomember (<= {best_threshold:.2f}): {blue_pct:.2f}%")
plt.ylim(-0.2, 1.2)
plt.axhline(y=best_threshold, color="green", linestyle="--", label=f"Threshold = {best_threshold:.2f}")
plt.title("Score Differences")
plt.xlabel("Index")
plt.ylabel("Difference")
plt.legend()
plt.grid(True)
plt.show()
