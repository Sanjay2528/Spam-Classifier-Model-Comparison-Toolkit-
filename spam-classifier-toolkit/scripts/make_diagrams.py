"""Generates simple architecture/workflow/UML-style diagrams as PNGs
using matplotlib boxes & arrows, for embedding into the PDF report."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT_DIR, exist_ok=True)


def box(ax, xy, w, h, text, color="#e8f0fe", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor="#1a4d8f", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, wrap=True)


def arrow(ax, start, end):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                         color="#1a4d8f", linewidth=1.4)
    ax.add_patch(a)


# ---------------------------------------------------------------------
# 1. System Architecture Diagram
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

box(ax, (0.3, 4.3), 2.2, 1.1, "CLI Layer\n(main.py)")
box(ax, (3.4, 4.3), 2.4, 1.1, "Preprocessing\nModule")
box(ax, (6.6, 4.3), 2.6, 1.1, "Model Training &\nPrediction Module")
box(ax, (3.4, 2.3), 2.4, 1.1, "Evaluation &\nReporting Module")
box(ax, (6.6, 2.3), 2.6, 1.1, "Config / Logger /\nExceptions (cross-cutting)")
box(ax, (0.3, 2.3), 2.2, 1.1, "Data Layer\n(CSV dataset)")
box(ax, (3.4, 0.3), 2.4, 1.1, "Reports (JSON)\n+ Logs")

arrow(ax, (2.5, 4.85), (3.4, 4.85))
arrow(ax, (5.8, 4.85), (6.6, 4.85))
arrow(ax, (2.4, 3.4), (2.4, 4.3))
arrow(ax, (4.6, 4.3), (4.6, 3.4))
arrow(ax, (7.9, 4.3), (4.6, 2.85))
arrow(ax, (4.6, 2.3), (4.6, 1.4))

ax.set_title("System Architecture Diagram", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "architecture.png"), dpi=180)
plt.close()

# ---------------------------------------------------------------------
# 2. Process Flow / Workflow Diagram
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")

steps = ["Load\nDataset", "Clean &\nVectorize Text", "Train Models\n(NB / LR / RF)",
         "Evaluate on\nTest Set", "Print Comparison\n& Save Report"]
x = 0.2
w = 1.7
for i, s in enumerate(steps):
    box(ax, (x, 0.5), w, 1.0, s, color="#eaf7ea", fontsize=9)
    if i < len(steps) - 1:
        arrow(ax, (x + w, 1.0), (x + w + 0.15, 1.0))
    x += w + 0.15

ax.set_title("Process Flow / Workflow Diagram", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "workflow.png"), dpi=180)
plt.close()

# ---------------------------------------------------------------------
# 3. Use Case Diagram (simplified)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(0, 8); ax.set_ylim(0, 6); ax.axis("off")

# actor (stick figure, simplified as circle+lines)
ax.plot(1, 4.6, "o", markersize=18, color="#1a4d8f")
ax.plot([1, 1], [4.3, 3.4], color="#1a4d8f", linewidth=2)
ax.plot([0.6, 1.4], [4.0, 4.0], color="#1a4d8f", linewidth=2)
ax.plot([1, 0.7], [3.4, 2.9], color="#1a4d8f", linewidth=2)
ax.plot([1, 1.3], [3.4, 2.9], color="#1a4d8f", linewidth=2)
ax.text(1, 2.6, "Student /\nUser", ha="center", fontsize=9)

ellipses = [
    ("Load & Preprocess\nDataset", (4.5, 5)),
    ("Train Models", (4.5, 3.8)),
    ("Evaluate & Compare\nModels", (4.5, 2.6)),
    ("Classify Ad-hoc\nMessage", (4.5, 1.4)),
]
for text, (cx, cy) in ellipses:
    e = mpatches.Ellipse((cx, cy), 3.2, 0.9, facecolor="#fff4e0", edgecolor="#b5651d", linewidth=1.4)
    ax.add_patch(e)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5)
    arrow(ax, (1.6, 4.2), (cx - 1.6, cy))

ax.set_title("Use Case Diagram", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "usecase.png"), dpi=180)
plt.close()

# ---------------------------------------------------------------------
# 4. Class / Component Diagram (simplified)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")


def class_box(ax, xy, w, h, title, attrs):
    x, y = xy
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.0",
                                    linewidth=1.4, edgecolor="#1a4d8f", facecolor="#f5f5ff")
    ax.add_patch(rect)
    ax.plot([x, x + w], [y + h - 0.4, y + h - 0.4], color="#1a4d8f", linewidth=1)
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x + 0.1, y + h - 0.55, attrs, ha="left", va="top", fontsize=7.8)


class_box(ax, (0.3, 3.8), 2.6, 1.9, "SpamClassifier",
          "- model_name\n- _model\n- _is_trained\n+ train()\n+ predict()\n+ predict_proba()")
class_box(ax, (3.4, 3.8), 2.6, 1.9, "Preprocessing",
          "+ clean_text()\n+ load_dataset()\n+ preprocess()")
class_box(ax, (6.5, 3.8), 2.9, 1.9, "Evaluation",
          "+ evaluate_models()\n+ print_comparison_table()\n+ save_report()\n+ best_model()")
class_box(ax, (3.4, 1.0), 2.6, 1.9, "Config",
          "- paths\n- hyperparameters\n- model list")
class_box(ax, (6.5, 1.0), 2.9, 1.9, "Exceptions",
          "SpamToolkitError\n- DatasetError\n- ModelNotFoundError\n- ModelNotTrainedError")

arrow(ax, (2.9, 4.7), (3.4, 4.7))
arrow(ax, (6.0, 4.7), (6.5, 4.7))
arrow(ax, (4.7, 3.8), (4.7, 2.9))
arrow(ax, (7.9, 3.8), (7.9, 2.9))

ax.set_title("Class / Component Diagram", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_diagram.png"), dpi=180)
plt.close()

# ---------------------------------------------------------------------
# 5. Sequence Diagram (simplified, textual/lifeline style)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

actors = [("User", 0.8), ("main.py", 2.8), ("preprocessing", 5.0), ("model", 7.0), ("evaluate", 9.0)]
for name, x in actors:
    box(ax, (x - 0.6, 6.2), 1.2, 0.6, name, fontsize=8)
    ax.plot([x, x], [0.3, 6.2], color="#888", linewidth=1, linestyle="--")

msgs = [
    (0.8, 2.8, 5.7, "run CLI command"),
    (2.8, 5.0, 5.1, "load_dataset() / preprocess()"),
    (2.8, 7.0, 4.5, "train_all_models()"),
    (2.8, 9.0, 3.9, "evaluate_models()"),
    (9.0, 2.8, 3.3, "metrics dict"),
    (2.8, 0.8, 2.7, "print comparison + save report"),
]
for x1, x2, y, label in msgs:
    arrow(ax, (x1, y), (x2, y))
    ax.text((x1 + x2) / 2, y + 0.12, label, ha="center", fontsize=7.5)

ax.set_title("Sequence Diagram — Training & Evaluation Flow", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sequence.png"), dpi=180)
plt.close()

print("Diagrams saved to", OUT_DIR)
