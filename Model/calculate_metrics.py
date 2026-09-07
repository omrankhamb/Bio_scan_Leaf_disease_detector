"""
Calculate comprehensive model evaluation metrics:
- Precision, Recall, F1-Score
- Confusion Matrix
- Training/Validation Loss Graphs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from datetime import datetime

# ============================================================================
# 1. SETUP AND CONFIGURATION
# ============================================================================

device = "cpu"
batch_size = 64
epochs = 5

# ============================================================================
# 2. LOAD DATASET
# ============================================================================

transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)

dataset = datasets.ImageFolder("Dataset", transform=transform)
targets_size = len(dataset.class_to_idx)

# Split dataset
indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))
validation = int(np.floor(0.70 * split))

np.random.shuffle(indices)

train_indices = indices[:validation]
validation_indices = indices[validation:split]
test_indices = indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=validation_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)

# ============================================================================
# 3. DEFINE CNN MODEL
# ============================================================================


class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)
        out = self.dense_layers(out)
        return out


# ============================================================================
# 4. INITIALIZE MODEL AND OPTIMIZER
# ============================================================================

model = CNN(targets_size)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# ============================================================================
# 5. TRAINING FUNCTION WITH LOSS TRACKING
# ============================================================================


def train_model(model, criterion, optimizer, train_loader, validation_loader, epochs):
    """Train model and return losses for plotting"""
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for e in range(epochs):
        t0 = datetime.now()
        train_loss = []

        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss_avg = np.mean(train_loss)

        # Validation phase
        validation_loss = []
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            validation_loss.append(loss.item())

        validation_loss_avg = np.mean(validation_loss)

        train_losses[e] = train_loss_avg
        validation_losses[e] = validation_loss_avg

        dt = datetime.now() - t0
        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss_avg:.3f} "
            f"Validation_loss:{validation_loss_avg:.3f} Duration:{dt}"
        )

    return train_losses, validation_losses


# ============================================================================
# 6. PREDICTION FUNCTION
# ============================================================================


def get_predictions(model, loader):
    """Get all predictions and true labels from a data loader"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_predictions), np.array(all_targets)


# ============================================================================
# 7. ACCURACY FUNCTION
# ============================================================================


def calculate_accuracy(loader):
    """Calculate accuracy on a given loader"""
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

    return 100 * (n_correct / n_total)


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PLANT DISEASE DETECTION MODEL - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    # Train the model
    print("\n[1] TRAINING MODEL...")
    print("-" * 80)
    train_losses, validation_losses = train_model(
        model, criterion, optimizer, train_loader, validation_loader, epochs
    )

    # Calculate accuracies
    print("\n[2] CALCULATING ACCURACIES...")
    print("-" * 80)
    train_acc = calculate_accuracy(train_loader)
    validation_acc = calculate_accuracy(validation_loader)
    test_acc = calculate_accuracy(test_loader)

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {validation_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Get predictions for detailed metrics
    print("\n[3] GENERATING PREDICTIONS FOR DETAILED METRICS...")
    print("-" * 80)
    test_predictions, test_targets = get_predictions(model, test_loader)

    # Calculate precision, recall, and F1-score
    print("\n[4] DETAILED CLASSIFICATION METRICS (Test Set)...")
    print("-" * 80)

    # Macro averages (unweighted mean)
    precision_macro = precision_score(test_targets, test_predictions, average="macro")
    recall_macro = recall_score(test_targets, test_predictions, average="macro")
    f1_macro = f1_score(test_targets, test_predictions, average="macro")

    # Weighted averages (weighted by support)
    precision_weighted = precision_score(
        test_targets, test_predictions, average="weighted"
    )
    recall_weighted = recall_score(test_targets, test_predictions, average="weighted")
    f1_weighted = f1_score(test_targets, test_predictions, average="weighted")

    print(f"\n{'Metric':<20} {'Macro':<15} {'Weighted':<15}")
    print("-" * 50)
    print(f"{'Precision':<20} {precision_macro:<15.4f} {precision_weighted:<15.4f}")
    print(f"{'Recall':<20} {recall_macro:<15.4f} {recall_weighted:<15.4f}")
    print(f"{'F1-Score':<20} {f1_macro:<15.4f} {f1_weighted:<15.4f}")

    # Detailed classification report
    print("\n[5] CLASSIFICATION REPORT (Test Set)...")
    print("-" * 80)
    class_labels = list(dataset.class_to_idx.keys())
    print(classification_report(test_targets, test_predictions, target_names=class_labels))

    # Confusion matrix
    print("\n[6] CONFUSION MATRIX...")
    print("-" * 80)
    cm = confusion_matrix(test_targets, test_predictions)
    print(f"Shape: {cm.shape}")
    print("Confusion Matrix saved (see plots below)")

    # ========================================================================
    # 7. GENERATE VISUALIZATION PLOTS
    # ========================================================================
    print("\n[7] GENERATING PLOTS...")
    print("-" * 80)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Training and Validation Loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(range(1, epochs + 1), train_losses, "b-o", label="Train Loss", linewidth=2)
    ax1.plot(
        range(1, epochs + 1), validation_losses, "r-s", label="Validation Loss", linewidth=2
    )
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy Comparison
    ax2 = plt.subplot(2, 2, 2)
    accuracies = [train_acc, validation_acc, test_acc]
    labels = ["Train", "Validation", "Test"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax2.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylim([0, 105])
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Metrics Comparison (Macro)
    ax3 = plt.subplot(2, 2, 3)
    metrics = ["Precision", "Recall", "F1-Score"]
    macro_values = [precision_macro, recall_macro, f1_macro]
    weighted_values = [precision_weighted, recall_weighted, f1_weighted]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width / 2, macro_values, width, label="Macro", color="#9b59b6", edgecolor="black")
    ax3.bar(x + width / 2, weighted_values, width, label="Weighted", color="#f39c12", edgecolor="black")
    ax3.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax3.set_title("Precision, Recall & F1-Score Comparison", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Confusion Matrix (Heatmap)
    ax4 = plt.subplot(2, 2, 4)
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    im = ax4.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    ax4.set_title("Normalized Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax4.set_ylabel("True Label", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax4, label="Normalized Count")

    plt.tight_layout()
    plt.savefig("model_evaluation_summary.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: model_evaluation_summary.png")

    # Save detailed confusion matrix plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Normalized Count"},
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix - All Classes (Normalized)", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_detailed.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: confusion_matrix_detailed.png")

    # ========================================================================
    # 8. SAVE METRICS TO CSV
    # ========================================================================
    print("\n[8] SAVING METRICS TO CSV...")
    print("-" * 80)

    # Summary metrics
    summary_data = {
        "Metric": [
            "Train Accuracy",
            "Validation Accuracy",
            "Test Accuracy",
            "Precision (Macro)",
            "Precision (Weighted)",
            "Recall (Macro)",
            "Recall (Weighted)",
            "F1-Score (Macro)",
            "F1-Score (Weighted)",
        ],
        "Value": [
            f"{train_acc:.4f}",
            f"{validation_acc:.4f}",
            f"{test_acc:.4f}",
            f"{precision_macro:.4f}",
            f"{precision_weighted:.4f}",
            f"{recall_macro:.4f}",
            f"{recall_weighted:.4f}",
            f"{f1_macro:.4f}",
            f"{f1_weighted:.4f}",
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("model_metrics_summary.csv", index=False)
    print("✓ Saved: model_metrics_summary.csv")

    # Loss data
    loss_data = {
        "Epoch": range(1, epochs + 1),
        "Train Loss": train_losses,
        "Validation Loss": validation_losses,
    }
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv("training_validation_loss.csv", index=False)
    print("✓ Saved: training_validation_loss.csv")

    # ========================================================================
    # 9. FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📊 SUMMARY STATISTICS:")
    print(f"   • Training Accuracy: {train_acc:.2f}%")
    print(f"   • Validation Accuracy: {validation_acc:.2f}%")
    print(f"   • Test Accuracy: {test_acc:.2f}%")
    print(f"   • Precision (Weighted): {precision_weighted:.4f}")
    print(f"   • Recall (Weighted): {recall_weighted:.4f}")
    print(f"   • F1-Score (Weighted): {f1_weighted:.4f}")
    print("\n📁 OUTPUT FILES GENERATED:")
    print("   1. model_evaluation_summary.png - Main evaluation plots")
    print("   2. confusion_matrix_detailed.png - Full confusion matrix heatmap")
    print("   3. model_metrics_summary.csv - Summary metrics table")
    print("   4. training_validation_loss.csv - Loss history data")
    print("\n" + "=" * 80)
