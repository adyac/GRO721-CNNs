import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# ── layout constants ──────────────────────────────────────────────────────────
box_x      = 2.2    # left edge of layer boxes
box_w      = 5.0    # box width
box_h      = 0.28   # box height
gap        = 0.10   # vertical gap between boxes
dim_x      = 7.4    # x position of dimension annotations
dim_sym_x  = 7.25   # x of ℝ symbol
arrow_x    = 4.7    # x centre of connector arrows

# ── layers: (label, output_shape_string) ─────────────────────────────────────
layers = [
    # Block 1
    ("Conv2D (C=1, F=16, K=3, S=1, P=1)",  "16×53×53"),
    ("BatchNorm2d (F=16)",                   "16×53×53"),
    ("ReLU",                                 "16×53×53"),
    ("Dropout2d (p=0.2)",                    "16×53×53"),
    ("MaxPool2D (K=2, S=2)",                 "16×26×26"),
    # Block 2
    ("Conv2D (C=16, F=32, K=3, S=1, P=1)", "32×26×26"),
    ("BatchNorm2d (F=32)",                   "32×26×26"),
    ("ReLU",                                 "32×26×26"),
    ("Dropout2d (p=0.2)",                    "32×26×26"),
    ("MaxPool2D (K=2, S=2)",                 "32×13×13"),
    # Block 3
    ("Conv2D (C=32, F=64, K=3, S=1, P=1)", "64×13×13"),
    ("BatchNorm2d (F=64)",                   "64×13×13"),
    ("ReLU",                                 "64×13×13"),
    ("Dropout2d (p=0.2)",                    "64×13×13"),
    ("MaxPool2D (K=2, S=2)",                 "64×6×6"),
    # Block 4
    ("Conv2D (C=64, F=128, K=3, S=1, P=1)","128×6×6"),
    ("BatchNorm2d (F=128)",                  "128×6×6"),
    ("ReLU",                                 "128×6×6"),
    ("Dropout2d (p=0.2)",                    "128×6×6"),
    ("MaxPool2D (K=2, S=2)",                 "128×3×3"),
    # FC
    ("Flatten",                              "1152"),
    ("Linear (Hin=1152, Hout=85)",           "85"),
    ("BatchNorm1d (F=85)",                   "85"),
    ("ReLU",                                 "85"),
    ("Dropout (p=0.2)",                      "85"),
    ("Linear (Hin=85, Hout=3)",              "3"),
    ("Sigmoid",                              "3"),
]

n = len(layers)

# total height used by all boxes + gaps
total_h = n * box_h + (n - 1) * gap
# top y of first box (leave room for input label)
top_y = 9.3

def box_top(i):
    return top_y - i * (box_h + gap)

def box_mid(i):
    return box_top(i) - box_h / 2

# ── input annotation ─────────────────────────────────────────────────────────
ax.annotate("", xy=(arrow_x, box_top(0)), xytext=(arrow_x, box_top(0) + 0.4),
            arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8))
ax.text(arrow_x, box_top(0) + 0.55,
        r"$\mathbf{x} \in \mathbb{R}^{1 \times 53 \times 53}$",
        ha='center', va='bottom', fontsize=8, fontweight='bold')

# ── draw boxes + connectors + dimension labels ────────────────────────────────
for i, (label, dim) in enumerate(layers):
    y_top = box_top(i)
    # box
    rect = FancyBboxPatch((box_x, y_top - box_h), box_w, box_h,
                           boxstyle="square,pad=0", linewidth=0.7,
                           edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    # layer text
    ax.text(box_x + box_w / 2, y_top - box_h / 2, label,
            ha='center', va='center', fontsize=6.5,
            fontfamily='monospace')

    # connector arrow to next box
    if i < n - 1:
        y_bottom = y_top - box_h
        y_next   = box_top(i + 1)
        ax.annotate("", xy=(arrow_x, y_next), xytext=(arrow_x, y_bottom),
                    arrowprops=dict(arrowstyle="-|>", color='black', lw=0.7))

    # dimension on the right
    # horizontal tick line from box edge to dimension
    ax.plot([box_x + box_w, dim_sym_x - 0.05], [y_top - box_h / 2, y_top - box_h / 2],
            color='black', lw=0.5)
    ax.text(dim_sym_x, y_top - box_h / 2,
            r"$\mathbb{R}$",
            ha='right', va='center', fontsize=7)
    ax.text(dim_x, y_top - box_h / 2,
            dim, ha='left', va='center', fontsize=6.5)

# ── output annotation ─────────────────────────────────────────────────────────
last_bottom = box_top(n - 1) - box_h
ax.annotate("", xy=(arrow_x, last_bottom - 0.35), xytext=(arrow_x, last_bottom),
            arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8))
ax.text(arrow_x, last_bottom - 0.55,
        r"$\hat{\mathbf{y}} \in \mathbb{R}^{3}$",
        ha='center', va='top', fontsize=8, fontweight='bold')

# ── section bracket labels (left side) ───────────────────────────────────────
def draw_bracket_label(ax, i_start, i_end, label, x_right=2.15, x_brace=1.85):
    y_top_br    = box_top(i_start)
    y_bot_br    = box_top(i_end) - box_h
    y_mid       = (y_top_br + y_bot_br) / 2
    # vertical line
    ax.plot([x_brace, x_brace], [y_bot_br, y_top_br], color='black', lw=0.8)
    # top & bottom ticks
    ax.plot([x_brace, x_right], [y_top_br, y_top_br], color='black', lw=0.8)
    ax.plot([x_brace, x_right], [y_bot_br, y_bot_br], color='black', lw=0.8)
    # mid tick
    ax.plot([x_brace, x_brace - 0.15], [y_mid, y_mid], color='black', lw=0.8)
    # rotated label
    ax.text(x_brace - 0.25, y_mid, label,
            ha='center', va='center', fontsize=7,
            rotation=90)

# Feature extraction: blocks 1-4 (indices 0-19)
draw_bracket_label(ax, 0, 19, "Extraction des caractéristiques")
# Classification: FC layers (indices 20-26)
draw_bracket_label(ax, 20, 26, "Classification")

# ── save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("classification_architecture.png", dpi=180, bbox_inches='tight',
            facecolor='white')
print("Saved: classification_architecture.png")
