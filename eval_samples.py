#!/usr/bin/env python3
"""
Evaluation script to visualize model predictions on sample data.
Run after training to see how the model performs on train and test sets.
"""

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from dataset import ConveyorSimulator
from visualizer import Visualizer
from models.classification_network import build_classification_model
from models.detection_network import build_detection_model
from models.segmentation_network import build_segmentation_model

CLASS_PROBABILITY_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
SEGMENTATION_BACKGROUND_CLASS = 3


def load_model(task, device, weights_path):
    """Load the trained model."""
    if task == 'classification':
        model = build_classification_model()
    elif task == 'detection':
        model = build_detection_model()
    elif task == 'segmentation':
        model = build_segmentation_model()
    else:
        raise ValueError('Not supported task')
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def visualize_classification_samples(images, predictions, class_labels, output_path):

    """Visualize multiple classification samples in a horizontal grid."""
    num_samples = len(images)
    fig = plt.figure(figsize=(4 * num_samples, 4), dpi=100)
    
    for idx in range(num_samples):
        image = images[idx]
        prediction = predictions[idx]
        labels = class_labels[idx]
        
        ax = fig.add_subplot(1, num_samples, idx + 1)
        
        # Get predicted and target shapes
        predicted_shapes = ''
        if prediction[0] >= CLASS_PROBABILITY_THRESHOLD:
            predicted_shapes += 'Circle '
        if prediction[1] >= CLASS_PROBABILITY_THRESHOLD:
            predicted_shapes += 'Triangle '
        if prediction[2] >= CLASS_PROBABILITY_THRESHOLD:
            predicted_shapes += 'Cross '
        
        target_shapes = ''
        if labels[0] >= CLASS_PROBABILITY_THRESHOLD:
            target_shapes += 'Circle '
        if labels[1] >= CLASS_PROBABILITY_THRESHOLD:
            target_shapes += 'Triangle '
        if labels[2] >= CLASS_PROBABILITY_THRESHOLD:
            target_shapes += 'Cross '
        
        ax.set_title(f'Sample {idx+1}\nPred: {predicted_shapes.strip()}\nTarget: {target_shapes.strip()}', fontsize=9)
        ax.imshow(image, cmap='gray', vmax=1)
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def visualize_detection_samples(images, predictions, boxes, output_path):
    """Visualize multiple detection samples in a horizontal grid."""
    num_samples = len(images)
    fig = plt.figure(figsize=(4 * num_samples, 4), dpi=100)
    
    color = ['r', 'g', 'b']
    
    for idx in range(num_samples):
        image = images[idx]
        prediction = predictions[idx]
        target = boxes[idx]
        
        ax = fig.add_subplot(1, num_samples, idx + 1)
        ax.imshow(image, cmap='gray', vmax=1)
        ax.set_title(f'Sample {idx+1}', fontsize=9)
        
        # Draw target boxes (solid lines)
        for i in range(target.shape[0]):
            if target[i, 0] > 0:
                pos_x = target[i, 1] * image.shape[0]
                pos_y = target[i, 2] * image.shape[0]
                size = target[i, 3] * int(0.75 * image.shape[0] / 2)
                class_index = int(target[i, 4])
                rec = patches.RegularPolygon((pos_x, pos_y), 4, orientation=0.78, radius=size, 
                                            linewidth=2, edgecolor=color[class_index], facecolor='none')
                ax.add_patch(rec)
        
        # Draw prediction boxes (dashed lines)
        # Model outputs raw logits — apply sigmoid before thresholding
        for i in range(prediction.shape[0]):
            confidence = 1 / (1 + np.exp(-prediction[i, 0]))
            if confidence > CONFIDENCE_THRESHOLD:
                pos_x = prediction[i, 1] * image.shape[0]
                pos_y = prediction[i, 2] * image.shape[0]
                size = prediction[i, 3] * int(0.75 * image.shape[0] / 2)
                class_index = int(np.argmax(prediction[i, 4:]))
                rec = patches.RegularPolygon((pos_x, pos_y), 4, orientation=0.78, radius=size, 
                                            linewidth=2, edgecolor=color[class_index], 
                                            facecolor='none', linestyle='--')
                ax.add_patch(rec)
        
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def visualize_segmentation_samples(images, predictions, targets, output_path):
    """Visualize multiple segmentation samples in a horizontal grid."""
    num_samples = len(images)
    fig = plt.figure(figsize=(4 * num_samples, 4), dpi=100)
    
    for idx in range(num_samples):
        image = images[idx]
        prediction = predictions[idx]
        target = targets[idx]
        
        n_class = prediction.shape[0]
        prediction_class = np.argmax(prediction, axis=0)
        
        # Concatenate prediction and target for side-by-side view
        combined = np.concatenate((prediction_class, target), axis=1)
        
        ax = fig.add_subplot(1, num_samples, idx + 1)
        ax.imshow(combined, vmax=n_class - 1, vmin=0)
        ax.set_title(f'Sample {idx+1}\nPred | Target', fontsize=9)
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)



def evaluate_samples(task, num_samples=5):
    """Evaluate and visualize samples from train and test sets."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_path = os.path.dirname(__file__)
    train_data_path = os.path.join(dir_path, 'data', 'training')
    test_data_path = os.path.join(dir_path, 'data', 'test')
    weights_path = os.path.join(dir_path, 'weights', 'task_' + task + '_best.pt')
    figures_path = os.path.join(dir_path, 'figures')
    
    # Create figures directory if it doesn't exist
    os.makedirs(figures_path, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(weights_path):
        print(f'Error: Model weights not found at {weights_path}')
        print('Please train the model first.')
        return
    
    print(f'Loading model from {weights_path}...', flush=True)
    model = load_model(task, device, weights_path)
    print(f'Model loaded.', flush=True)

    print(f'Loading train dataset...', flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = ConveyorSimulator(train_data_path, transform)
    print(f'Train dataset: {len(dataset_train)} samples', flush=True)

    print(f'Loading test dataset...', flush=True)
    dataset_test = ConveyorSimulator(test_data_path, transform)
    print(f'Test dataset: {len(dataset_test)} samples', flush=True)

    import random
    train_indices = random.sample(range(len(dataset_train)), min(num_samples, len(dataset_train)))
    test_indices = random.sample(range(len(dataset_test)), min(num_samples, len(dataset_test)))
    print(f'Train indices: {train_indices}', flush=True)
    print(f'Test  indices: {test_indices}', flush=True)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, sampler=train_sampler, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=0)

    # Collect train samples
    print(f'\nCollecting {num_samples} train samples...', flush=True)
    train_images = []
    train_predictions = []
    train_class_labels = []
    train_boxes = []
    train_segmentation = []

    with torch.no_grad():
        for i, (image, segmentation_target, boxes, class_labels) in enumerate(train_loader):
            print(f'  Train sample {i+1}/{num_samples} — loading...', flush=True)
            image = image.to(device)
            segmentation_target = segmentation_target.to(device)
            boxes = boxes.to(device)
            class_labels = class_labels.to(device)
            print(f'  Train sample {i+1}/{num_samples} — running inference...', flush=True)
            prediction = model(image)
            train_images.append(image[0].cpu().detach().numpy()[0])
            train_predictions.append(prediction[0].cpu().detach().numpy())
            train_class_labels.append(class_labels[0].cpu().detach().numpy())
            train_boxes.append(boxes[0].cpu().detach().numpy())
            train_segmentation.append(segmentation_target[0].cpu().detach().numpy())
            print(f'  Train sample {i+1}/{num_samples} — done.', flush=True)

    # Collect test samples
    print(f'\nCollecting {num_samples} test samples...', flush=True)
    test_images = []
    test_predictions = []
    test_class_labels = []
    test_boxes = []
    test_segmentation = []

    with torch.no_grad():
        for i, (image, segmentation_target, boxes, class_labels) in enumerate(test_loader):
            print(f'  Test sample {i+1}/{num_samples} — loading...', flush=True)
            image = image.to(device)
            segmentation_target = segmentation_target.to(device)
            boxes = boxes.to(device)
            class_labels = class_labels.to(device)
            print(f'  Test sample {i+1}/{num_samples} — running inference...', flush=True)
            prediction = model(image)
            test_images.append(image[0].cpu().detach().numpy()[0])
            test_predictions.append(prediction[0].cpu().detach().numpy())
            test_class_labels.append(class_labels[0].cpu().detach().numpy())
            test_boxes.append(boxes[0].cpu().detach().numpy())
            test_segmentation.append(segmentation_target[0].cpu().detach().numpy())
            print(f'  Test sample {i+1}/{num_samples} — done.', flush=True)

    # Visualize
    print(f'\nVisualizing samples...', flush=True)
    
    if task == 'classification':
        train_output = os.path.join(figures_path, 'eval_train_classification_prediction.png')
        test_output = os.path.join(figures_path, 'eval_test_classification_prediction.png')
        visualize_classification_samples(train_images, train_predictions, train_class_labels, train_output)
        visualize_classification_samples(test_images, test_predictions, test_class_labels, test_output)
    
    elif task == 'detection':
        train_output = os.path.join(figures_path, 'eval_train_detection_prediction.png')
        test_output = os.path.join(figures_path, 'eval_test_detection_prediction.png')
        print(f'Saving train figure...', flush=True)
        visualize_detection_samples(train_images, train_predictions, train_boxes, train_output)
        print(f'Saving test figure...', flush=True)
        visualize_detection_samples(test_images, test_predictions, test_boxes, test_output)
    
    elif task == 'segmentation':
        train_output = os.path.join(figures_path, 'eval_train_segmentation_prediction.png')
        test_output = os.path.join(figures_path, 'eval_test_segmentation_prediction.png')
        visualize_segmentation_samples(train_images, train_predictions, train_segmentation, train_output)
        visualize_segmentation_samples(test_images, test_predictions, test_segmentation, test_output)
    
    print(f'\nDone! Check the figures folder for visualizations:')
    print(f'  - eval_train_{task}_prediction.png')
    print(f'  - eval_test_{task}_prediction.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on sample data')
    parser.add_argument('--task', choices=['classification', 'detection', 'segmentation'],
                        help='The CNN task', required=True)
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of samples to evaluate from each set (default: 5)')
    
    args = parser.parse_args()
    evaluate_samples(args.task, args.num_samples)
