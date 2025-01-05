import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt
import io

# Custom dataset class to read data from a list of image paths
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, classes, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.classes = classes # Store the classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
      image_path = self.image_paths[idx]
      image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
      label = self.labels[idx]
      if self.transform:
            image = self.transform(image)
      return image, label

def get_data_loaders(batch_size, train_paths, val_paths, classes):
    # Data augmentation for training dataset (with and without random erasing)
    train_transform_no_erasing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_transform_with_erasing = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.ToTensor(),
      transforms.RandomErasing(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transform for validation dataset
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset instances
    train_dataset_no_erasing = CustomImageDataset(train_paths[0], train_paths[1], classes, transform=train_transform_no_erasing)
    train_dataset_with_erasing = CustomImageDataset(train_paths[0], train_paths[1], classes, transform=train_transform_with_erasing)
    val_dataset = CustomImageDataset(val_paths[0], val_paths[1], classes, transform=val_transform)

    # Data loaders creation
    train_loader_no_erasing = DataLoader(train_dataset_no_erasing, batch_size=batch_size, shuffle=True)
    train_loader_with_erasing = DataLoader(train_dataset_with_erasing, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader_no_erasing, train_loader_with_erasing, val_loader, classes

def build_resnet50_model(num_classes):
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    # Replace the last FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Add Softmax activation (if needed)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    return model

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, model_name):
  # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training {model_name}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation {model_name}'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], {model_name} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return val_accuracies, model

def validate_model(model, val_loader, device, model_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Validating {model_name}'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def plot_accuracy_comparison(accuracy1, accuracy2, model_name1, model_name2):
    labels = [model_name1, model_name2]
    accuracies = [accuracy1, accuracy2]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, accuracies)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    
    plt.savefig('accuracy_comparison.png')
    print("Accuracy comparison plot saved as 'accuracy_comparison.png'")
    plt.close()


def get_image_paths_and_labels(data_dir, classes):
    image_paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        class_path = os.path.join(data_dir, c)
        for image_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, image_name))
            labels.append(class_to_idx[c])
    return image_paths, labels

if __name__ == '__main__':
    # Hyperparameters
    training_dataset_path = r"C:\Users\wang0\Desktop\HW\Processing computer vision\02_Homework\Hw2\Q2_Dataset\dataset\training_dataset"  # Correct training dataset path
    validation_dataset_path = r"C:\Users\wang0\Desktop\HW\Processing computer vision\02_Homework\Hw2\Q2_Dataset\dataset\validation_dataset" # Correct validation dataset path
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001
    num_classes = 2 # cat and dog
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data paths
    classes = ['cat', 'dog']

    train_paths = get_image_paths_and_labels(training_dataset_path, classes)
    val_paths = get_image_paths_and_labels(validation_dataset_path, classes)
    
    # Create Data Loaders
    train_loader_no_erasing, train_loader_with_erasing, val_loader, classes = get_data_loaders(batch_size, train_paths, val_paths, classes)

    # Build Model
    model_no_erasing = build_resnet50_model(num_classes).to(device)
    model_with_erasing = build_resnet50_model(num_classes).to(device)

    # Train model without random erasing
    val_accuracies_no_erasing, model_no_erasing = train_model(model_no_erasing, train_loader_no_erasing, val_loader, device, num_epochs, learning_rate, 'ResNet50_no_erasing')
    torch.save(model_with_erasing.state_dict(), 'resnet50_no_erasing.pth')
    print('Model "resnet50_no_erasing.pth" saved')

    # Train model with random erasing
    val_accuracies_with_erasing, model_with_erasing = train_model(model_with_erasing, train_loader_with_erasing, val_loader, device, num_epochs, learning_rate, 'ResNet50_with_erasing')
    torch.save(model_no_erasing.state_dict(), 'resnet50_with_erasing.pth')
    print('Model "resnet50_with_erasing.pth" saved')


    # Model validation and accuracy comparison
    accuracy1 = validate_model(model_no_erasing, val_loader, device, 'ResNet50_no_erasing')
    accuracy2 = validate_model(model_with_erasing, val_loader, device, 'ResNet50_with_erasing')

    print(f"Validation Accuracy without Random Erasing: {accuracy1:.2f}%")
    print(f"Validation Accuracy with Random Erasing: {accuracy2:.2f}%")
    plot_accuracy_comparison(accuracy1, accuracy2, 'ResNet50_no_erasing', 'ResNet50_with_erasing')