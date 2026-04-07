#!/usr/bin/env python3
"""
Script to generate an architecture diagram of the classification network
showing all layers including BatchNorm and Dropout.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 40)
ax.axis('off')

# Title
ax.text(5, 39, 'Classification Network Architecture', 
        fontsize=18, fontweight='bold', ha='center')
ax.text(5, 38.2, '(Optimized for Shape Detection - ~171k parameters)', 
        fontsize=12, ha='center', style='italic', color='gray')

# Define colors
conv_color = '#FF9999'
bn_color = '#FFCC99'
relu_color = '#CCFFCC'
dropout_color = '#FFE6E6'
pool_color = '#99CCFF'
fc_color = '#FFCCFF'
sigmoid_color = '#FFFFCC'

def draw_layer(y, width, height, color, label, shape_info=''):
    """Draw a layer box and return the center position"""
    box = FancyBboxPatch((5 - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y, label, fontsize=9, ha='center', va='center', fontweight='bold')
    if shape_info:
        ax.text(8.5, y, shape_info, fontsize=8, ha='left', va='center', family='monospace')
    return y

def draw_arrow(y1, y2):
    """Draw arrow between layers"""
    arrow = FancyArrowPatch((5, y1 - 0.5), (5, y2 + 0.5),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# Input
y = 37
draw_layer(y, 3, 0.6, '#E0E0E0', 'Input', 'x ∈ ℝ^(1×48×48)')
draw_arrow(y - 0.3, y - 1)

# Block 1
y = 35.5
block_layers = [
    ('Conv2d(1→16, 3×3)', '#FF9999', 'ℝ^(16×48×48)'),
    ('BatchNorm2d(16)', '#FFCC99', 'ℝ^(16×48×48)'),
    ('ReLU', '#CCFFCC', 'ℝ^(16×48×48)'),
    ('Dropout2d(0.2)', '#FFE6E6', 'ℝ^(16×48×48)'),
    ('MaxPool2d(2,2)', '#99CCFF', 'ℝ^(16×24×24)'),
]

for label, color, shape in block_layers:
    draw_layer(y, 3.5, 0.5, color, label, shape)
    draw_arrow(y - 0.25, y - 0.75)
    y -= 1

# Block 2
y -= 0.5
block_layers = [
    ('Conv2d(16→32, 3×3)', '#FF9999', 'ℝ^(32×24×24)'),
    ('BatchNorm2d(32)', '#FFCC99', 'ℝ^(32×24×24)'),
    ('ReLU', '#CCFFCC', 'ℝ^(32×24×24)'),
    ('Dropout2d(0.2)', '#FFE6E6', 'ℝ^(32×24×24)'),
    ('MaxPool2d(2,2)', '#99CCFF', 'ℝ^(32×12×12)'),
]

for label, color, shape in block_layers:
    draw_layer(y, 3.5, 0.5, color, label, shape)
    draw_arrow(y - 0.25, y - 0.75)
    y -= 1

# Block 3
y -= 0.5
block_layers = [
    ('Conv2d(32→64, 3×3)', '#FF9999', 'ℝ^(64×12×12)'),
    ('BatchNorm2d(64)', '#FFCC99', 'ℝ^(64×12×12)'),
    ('ReLU', '#CCFFCC', 'ℝ^(64×12×12)'),
    ('Dropout2d(0.2)', '#FFE6E6', 'ℝ^(64×12×12)'),
    ('MaxPool2d(2,2)', '#99CCFF', 'ℝ^(64×6×6)'),
]

for label, color, shape in block_layers:
    draw_layer(y, 3.5, 0.5, color, label, shape)
    draw_arrow(y - 0.25, y - 0.75)
    y -= 1

# Block 4
y -= 0.5
block_layers = [
    ('Conv2d(64→128, 3×3)', '#FF9999', 'ℝ^(128×6×6)'),
    ('BatchNorm2d(128)', '#FFCC99', 'ℝ^(128×6×6)'),
    ('ReLU', '#CCFFCC', 'ℝ^(128×6×6)'),
    ('Dropout2d(0.2)', '#FFE6E6', 'ℝ^(128×6×6)'),
    ('MaxPool2d(2,2)', '#99CCFF', 'ℝ^(128×3×3)'),
]

for label, color, shape in block_layers:
    draw_layer(y, 3.5, 0.5, color, label, shape)
    draw_arrow(y - 0.25, y - 0.75)
    y -= 1

# Fully Connected
y -= 0.5
fc_layers = [
    ('Flatten()', '#FFCCCC', 'ℝ^1152'),
    ('Linear(1152→64)', '#FFCCFF', 'ℝ^64'),
    ('BatchNorm1d(64)', '#FFCC99', 'ℝ^64'),
    ('ReLU', '#CCFFCC', 'ℝ^64'),
    ('Dropout(0.5)', '#FFE6E6', 'ℝ^64'),
    ('Linear(64→3)', '#FFCCFF', 'ℝ^3'),
    ('Sigmoid', '#FFFFCC', 'ŷ ∈ [0,1]³'),
]

for label, color, shape in fc_layers:
    draw_layer(y, 3.5, 0.5, color, label, shape)
    draw_arrow(y - 0.25, y - 0.75)
    y -= 1

# Output
y -= 0.25
draw_layer(y, 3, 0.6, '#E0E0E0', 'Output', 'ŷ ∈ ℝ³')

# Add legend
legend_y = 1.5
ax.text(0.5, legend_y + 1, 'Layer Types:', fontsize=11, fontweight='bold')
legend_items = [
    (conv_color, 'Convolutional'),
    (bn_color, 'Batch Normalization'),
    (relu_color, 'ReLU Activation'),
    (dropout_color, 'Dropout'),
    (pool_color, 'MaxPooling'),
    (fc_color, 'Fully Connected'),
    (sigmoid_color, 'Sigmoid Activation'),
]

for i, (color, label) in enumerate(legend_items):
    y_pos = legend_y - (i * 0.4)
    rect = mpatches.Rectangle((0.5, y_pos - 0.15), 0.3, 0.3, 
                              facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(1.0, y_pos, label, fontsize=9, va='center')

# Add statistics box
stats_text = 'Total Parameters: ~171,747\nOptimizers: Adam (lr=1e-3)\nLoss: BCEWithLogitsLoss\nBest Val Accuracy: 83.7%'
ax.text(5, 0.3, stats_text, fontsize=9, ha='center', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        family='monospace')

plt.tight_layout()
output_path = 'figures/classification_architecture_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Architecture diagram saved to: {output_path}")
plt.show()
