import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchsummary import summary
import matplotlib.pyplot as plt
import io
import sys
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ------- 開始：整合 inference.py 的程式碼 -------
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, classes, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.classes = classes
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

def get_inference_loader(data_root, inference_paths, classes):
    # Transform for validation and inference dataset
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = [class_to_idx[os.path.basename(os.path.dirname(path))] for path in inference_paths]
    inference_dataset = CustomImageDataset(inference_paths, labels, classes, transform=val_transform)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False) # one image at a time
    return inference_loader, inference_dataset.image_paths, inference_dataset.classes

def build_resnet50_model(num_classes):
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    # Replace the last FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Add Softmax activation (if needed)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    return model

def inference(model, image_path, device, classes):
    model.eval()
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()], output.cpu().detach().numpy()[0]

def display_model_structure(model, input_size):
    string_buffer = io.StringIO()
    summary(model, (3, 224, 224))
    model_summary = string_buffer.getvalue()
    print(model_summary)

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
def show_images_with_labels(root, image_paths_with_labels):
    try:
        toplevel = tk.Toplevel(root)
        toplevel.title("Image Display")

        for image_path, label_text in image_paths_with_labels:
            image = Image.open(image_path)
            image = image.resize((200, 200), Image.LANCZOS)  # Resize for the canvas
            photo = ImageTk.PhotoImage(image)
           
            text_label = tk.Label(toplevel, text = label_text, font=("Arial", 12))
            text_label.pack(side = "top", padx=10, pady=5)
            
            canvas = tk.Canvas(toplevel, width=200, height=200, bg="white", highlightthickness=1, highlightbackground="black")
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo  # Keep a reference
            canvas.pack(side="top", padx=10, pady=5)

    except Exception as e:
        messagebox.showerror("Error", f"Could not load images. Error: {e}")
def show_image_q2(root):
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
        image_paths_with_labels = [
           (os.path.join(base_dir,"inference_dataset", "Cat", "8043.jpg"), "Cat"),
           (os.path.join(base_dir,"inference_dataset", "Dog", "12051.jpg"), "Dog")
        ]
        show_images_with_labels(root, image_paths_with_labels)

# ------- 結束：整合 inference.py 的程式碼 -------


def predict_image_vgg(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
        return predicted_class.item()
    except Exception as e:
        messagebox.showerror("Error", f"Could not predict the image. Error: {e}")
        return None

def predict_image_resnet(image_path, model, transform, device, classes):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
        return classes[predicted_class.item()], output.cpu().detach().numpy()[0]
    except Exception as e:
        messagebox.showerror("Error", f"Could not predict the image. Error: {e}")
        return None

def load_and_display_image(image_path, canvas):
    try:
        image = Image.open(image_path)
        image = image.resize((200, 200), Image.LANCZOS)  # Resize for the canvas
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep a reference
        return photo
    except Exception as e:
        messagebox.showerror("Error", f"Could not load the image. Error: {e}")
        return None


def show_model_structure(model_summary_text):
    print("Model Structure:")
    print(model_summary_text)


def get_model_summary_vgg(device):
    # Load a dummy VGG16 model just to get the structure
    model = models.vgg16_bn(pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)

    # Move the model to the correct device
    model = model.to(device)

    # Capture the model summary
    string_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = string_buffer
    summary(model, (1, 32, 32))
    sys.stdout = original_stdout

    # Move the model back to cpu
    model = model.to("cpu")

    model_summary = string_buffer.getvalue()
    return model_summary


def show_loss_accuracy_plot(root):
    try:
        loss_acc_img = Image.open('loss_and_accuracy.png')
        loss_acc_img = loss_acc_img.resize((500, 250))
        loss_acc_photo = ImageTk.PhotoImage(loss_acc_img)
        # 創建一個新的 Toplevel 視窗
        toplevel = tk.Toplevel(root)
        toplevel.title("Loss and Accuracy Plot")
        
        # 創建一個 Label 來顯示圖片
        loss_acc_label = tk.Label(toplevel, image=loss_acc_photo)
        loss_acc_label.image = loss_acc_photo  # keep reference
        loss_acc_label.pack(padx=10, pady=10)  # 圖片放置在彈出視窗中
    except Exception as e:
        messagebox.showerror("Error",
                             f"Could not load the accuracy comparison plot. Make sure 'accuracy_comparison.png' exists. Error: {e}")
def show_comparison(root):
    try:
        loss_acc_img = Image.open('accuracy_comparison.png')
        loss_acc_img = loss_acc_img.resize((500, 250))
        loss_acc_photo = ImageTk.PhotoImage(loss_acc_img)
        # 創建一個新的 Toplevel 視窗
        toplevel = tk.Toplevel(root)
        toplevel.title("Loss and Accuracy Plot")
        
        # 創建一個 Label 來顯示圖片
        loss_acc_label = tk.Label(toplevel, image=loss_acc_photo)
        loss_acc_label.image = loss_acc_photo  # keep reference
        loss_acc_label.pack(padx=10, pady=10)  # 圖片放置在彈出視窗中
    except Exception as e:
        messagebox.showerror("Error",
                             f"Could not load the accuracy comparison plot. Make sure 'accuracy_comparison.png' exists. Error: {e}")


def main():
    root = tk.Tk()
    root.title("MainWindow")

    # 1. 檢查 CUDA 可用性並設定 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Model parameters
    num_classes = 2
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "Q2_Dataset", "dataset")
    inference_dataset_path = os.path.join(data_root, 'inference')  # Inference folder
    classes = ['cat', 'dog']

    # Load the trained VGG16 model
    model_vgg = models.vgg16_bn(pretrained=False)
    model_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    model_vgg.classifier[-1] = nn.Linear(model_vgg.classifier[-1].in_features, 10)
    try:
        model_vgg.load_state_dict(torch.load(os.path.join(base_dir,'model','vgg16_bn_mnist.pth'), map_location=device))
    except Exception as e:
        messagebox.showerror("Error", f"Could not load the trained VGG16 model. Make sure 'vgg16_bn_mnist.pth' exists. Error: {e}")
        return
    model_vgg.to(device)

    # Load the trained ResNet50 models
    model_resnet_no_erasing = build_resnet50_model(num_classes).to(device)
    model_resnet_with_erasing = build_resnet50_model(num_classes).to(device)

    try:
        model_resnet_no_erasing.load_state_dict(torch.load(os.path.join(base_dir,'model','resnet50_no_erasing.pth'), map_location=device))
        model_resnet_with_erasing.load_state_dict(torch.load(os.path.join(base_dir,'model','resnet50_with_erasing.pth'), map_location=device))
    except Exception as e:
        messagebox.showerror("Error", f"Could not load the trained ResNet50 models. Error: {e}")
        return
    
     # Get model summary
    model_summary_text_vgg = get_model_summary_vgg(device)

    # Data preprocessing for prediction
    transform_vgg = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
     # Data preprocessing for ResNet50
    transform_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create inference loader
    inference_loader, inference_paths, classes = get_inference_loader(data_root, [], classes)
    
   # Left frame for buttons and image
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

    # Upper frame for Question 1
    upper_frame = tk.Frame(left_frame)
    upper_frame.pack(side="top", fill="x")

    # Load image button Q1
    load_image_button_q1 = tk.Button(upper_frame, text="Load Image (Q1)", command=lambda: browse_file(canvas_q1))
    load_image_button_q1.pack(side="top", fill="x", pady=5)

    # Load video button Q1
    load_video_button_q1 = tk.Button(upper_frame, text="Load Video (Q1)",
                                    command=lambda: messagebox.showinfo("Video", "Video Loading Not Implemented"))
    load_video_button_q1.pack(side="top", fill="x", pady=5)

    # 1.1 Show structure
    show_structure_button_q1 = tk.Button(upper_frame, text="1.1 Show Structure (Q1)",
                                        command=lambda: show_model_structure(model_summary_text_vgg))
    show_structure_button_q1.pack(side="top", fill="x", pady=5)

    # 1.2 show acc and loss
    show_acc_loss_button_q1 = tk.Button(upper_frame, text="1.2 Show Acc and Loss",
                                        command=lambda: show_loss_accuracy_plot(root))
    show_acc_loss_button_q1.pack(side="top", fill="x", pady=5)

    # 1.3 predict button
    predict_button_q1 = tk.Button(upper_frame, text="1.3 Predict (Q1)",
                                  command=lambda: predict_and_display(canvas_q1, prediction_label_q1, model_vgg, transform_vgg, device))
    predict_button_q1.pack(side="top", fill="x", pady=5)
    
    # Lower frame for Question 2
    lower_frame = tk.Frame(left_frame)
    lower_frame.pack(side="top", fill="x", pady=10)

    # Load image button Q2
    load_image_button_q2 = tk.Button(lower_frame, text="Q2 Load Image", command=lambda: browse_file(canvas_q2))
    load_image_button_q2.pack(side="top", fill="x", pady=5)

    # 2.1 show image Q2
    show_image_button_q2 = tk.Button(lower_frame, text="2.1 Show Image (Q2)", command=lambda: show_image_q2(root))
    show_image_button_q2.pack(side="top", fill="x", pady=5)

    # 2.2 Show model structure Q2
    show_model_button_q2 = tk.Button(lower_frame, text="2.2 Show Model Structure (Q2)",
                                     command=lambda: display_model_structure(model_resnet_with_erasing, (3, 224, 224)))
    show_model_button_q2.pack(side="top", fill="x", pady=5)

    # 2.3 Show Compression Q2
    show_compression_button_q2 = tk.Button(lower_frame, text="2.3 Show Compression (Q2)",
                                           command=lambda: show_comparison(root))
    show_compression_button_q2.pack(side="top", fill="x", pady=5)

    # 2.4 Inference Q2
    inference_button_q2 = tk.Button(lower_frame, text="2.4 Inference (Q2)",
                                    command=lambda: predict_and_display(canvas_q2, text_label_q2, model_resnet_with_erasing, transform_resnet, device, classes))
    inference_button_q2.pack(side="top", fill="x", pady=5)
    
    # Image canvas q1
    canvas_q1 = tk.Canvas(root, width=200, height=200, bg="white", highlightthickness=1, highlightbackground="black")
    canvas_q1.grid(row=0, column=1, padx=10, pady=10)

    # Image canvas q2
    canvas_q2 = tk.Canvas(root, width=200, height=200, bg="white", highlightthickness=1, highlightbackground="black")
    canvas_q2.grid(row=1, column=1, padx=10, pady=10)

    # Text Label q1
    prediction_label_q1 = tk.Label(root, text="predict (Q1)")
    prediction_label_q1.grid(row=0, column=2, sticky=tk.NW, padx=10)

    # Text label Q2
    text_label_q2 = tk.Label(root, text="TextLabel (Q2)", font=("Arial", 12), wraplength=200)
    text_label_q2.grid(row=1, column=2, sticky=tk.NW, padx=10)

    # Keep a reference for the image path (for both questions)
    image_path_q1 = tk.StringVar()
    image_path_q2 = tk.StringVar()

    def browse_file(canvas):
        filename = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if filename:  # Only display if a file was selected
            if canvas == canvas_q1:
                image_path_q1.set(filename)
            elif canvas == canvas_q2:
                image_path_q2.set(filename)
            load_and_display_image(filename, canvas)

    def show_image(canvas):
        if canvas == canvas_q1:
            image_path = image_path_q1.get()
        elif canvas == canvas_q2:
            image_path = image_path_q2.get()
        if not image_path:
            messagebox.showerror("Error", "Please select an image.")
            return
        load_and_display_image(image_path, canvas)

    def predict_and_display(canvas, label, model, transform, device, classes = None):
        if canvas == canvas_q1:
            image_path = image_path_q1.get()
            predicted_class = predict_image_vgg(image_path, model, transform, device)
            if predicted_class is not None:
                label.config(text=f"Prediction: {predicted_class}")

        elif canvas == canvas_q2:
            image_path = image_path_q2.get()
            predicted_class, predicted_vector = predict_image_resnet(image_path, model, transform, device, classes)
            if predicted_class is not None:
                 label.config(text=f"Prediction: {predicted_class}, Vector: {np.round(predicted_vector, 2)}")

    root.mainloop()


if __name__ == '__main__':
    main()