########used this
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score, roc_curve

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
train_dir = '/media/adwika/204828a2-e6e4-4a82-8c52-0fd205da4d4e/hruturaj/Documents/adwika/rvf_data/train'
valid_dir = '/media/adwika/204828a2-e6e4-4a82-8c52-0fd205da4d4e/hruturaj/Documents/adwika/rvf_data/valid'

# Hyperparameters
batch_size = 32
num_epochs = 30  # Updated to 30 epochs
learning_rate = 0.001
patience = 5  # Number of epochs to wait for improvement

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Model dictionary with updated weight parameters
model_dict = {
    "resnet": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "xception": timm.create_model('legacy_xception', pretrained=True),
    "efficientnet": timm.create_model('tf_efficientnet_b0', pretrained=True),
    "mobilenet": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
}

# Modify models for binary classification
for model_name, model in model_dict.items():
    if model_name == 'resnet':
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'efficientnet':
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == 'xception':
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'mobilenet':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    model_dict[model_name] = model

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Training and evaluation function with early stopping
def train_and_evaluate(model, train_loader, valid_loader, num_epochs, model_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []

    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()
        with tqdm(total=len(train_loader), desc=f'Training {model_name} Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.update(1)

        train_acc.append(correct / total)
        train_loss.append(running_loss / len(train_loader))
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} training completed in {epoch_time:.2f} seconds.")

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with tqdm(total=len(valid_loader), desc=f'Validating {model_name} Epoch {epoch+1}/{num_epochs}') as pbar:
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    preds = torch.sigmoid(outputs) > 0.5
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    pbar.update(1)

        valid_acc.append(correct / total)
        valid_loss.append(running_loss / len(valid_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f} - "
              f"Valid Loss: {valid_loss[-1]:.4f}, Valid Acc: {valid_acc[-1]:.4f}")

        # Early stopping based on validation loss
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            patience_counter = 0
            best_model_state = model.state_dict()
            best_metrics = {
                'train_loss': train_loss[-1],
                'train_acc': train_acc[-1],
                'valid_loss': valid_loss[-1],
                'valid_acc': valid_acc[-1]
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Load the best model state for testing
    model.load_state_dict(best_model_state)

    return train_acc, valid_acc, train_loss, valid_loss, best_metrics

# ROC-AUC plotting function
def plot_roc_auc(model, valid_loader, model_name, all_labels, all_probs):
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

# Main loop for training and evaluating each model separately
roc_curves = {}
best_metrics_dict = {}

for model_name, model in model_dict.items():
    print(f"Training and evaluating {model_name} model...")
    train_acc, valid_acc, train_loss, valid_loss, best_metrics = train_and_evaluate(model, train_loader, valid_loader, num_epochs, model_name)

    # Save the best metrics
    best_metrics_dict[model_name] = best_metrics

    # Save the best model for each model type
    torch.save(model.state_dict(), f"{model_name}_best_model.pt")
    print(f"Best {model_name} model saved.")

    # Collect ROC curve data for plotting
    all_labels = []
    all_probs = []
    plot_roc_auc(model, valid_loader, model_name, all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    roc_curves[model_name] = (fpr, tpr, auc_score)

    print(f"Final Test Accuracy of {model_name} model: {best_metrics['valid_acc']:.4f}")

    # Clean up
    os.remove(f"{model_name}_best_model.pt")

    print(f"{model_name} model testing completed.\n")                                                                                                                                                                                                                                                                             

# Print best metrics for all models
for model_name, metrics in best_metrics_dict.items():
    print(f"Best metrics for {model_name} model - "
          f"Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_acc']:.4f}, "
          f"Valid Loss: {metrics['valid_loss']:.4f}, Valid Acc: {metrics['valid_acc']:.4f}")

# Plot all ROC curves in one plot
plt.figure()
for model_name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - All Models')
plt.legend(loc="lower right")
plt.savefig('all_models_roc_auc.png')
