import os
import glob
import re
import math
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TOKEN_VALUES = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
CLASS_TO_IDX = {val: idx for idx, val in enumerate(TOKEN_VALUES)}
IDX_TO_CLASS = {idx: val for idx, val in enumerate(TOKEN_VALUES)}

class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_patch(img_patch):
    """
    Converts RGB patch to 32x32 binary digit mask normalized float32 tensor [0, 1].
    """
    if len(img_patch.shape) == 3:
        gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_patch.copy()
        
    bg_val = np.median(gray)
    _, thresh = cv2.threshold(gray, max(110, bg_val - 25), 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def extract_digit_roi(img_rgb):
    """
    Crops upper/middle 40x40 region of a 56x56 patch.
    """
    h, w = img_rgb.shape[:2]
    roi = img_rgb[4:44, 8:48]
    if roi.size == 0:
        return np.zeros((40, 40, 3), dtype=np.uint8)
    return cv2.resize(roi, (40, 40), interpolation=cv2.INTER_LINEAR)

def generate_synthetic_digit_patch(val):
    """
    Renders clean synthetic digit text onto a 56x56 beige token background patch.
    """
    patch = np.full((56, 56, 3), (218, 222, 228), dtype=np.uint8) # Beige canvas
    text = str(val)
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX])
    scale = 0.85 if val >= 10 else 1.05
    thickness = random.choice([2, 3])
    
    color = (40, 40, 200) if val in (6, 8) else (40, 40, 40) # RGB red or dark
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = (56 - tw) // 2 + random.randint(-2, 2)
    ty = (56 + th) // 2 - 8 + random.randint(-2, 2)
    
    cv2.putText(patch, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)
    return patch

def augment_patch(gray_patch):
    """
    Applies random scale, translation, rotation, brightness/contrast jitter, and noise to a 32x32 grayscale image.
    """
    h, w = gray_patch.shape
    
    # Random rotation (-8 to +8 deg)
    angle = random.uniform(-8, 8)
    scale = random.uniform(0.85, 1.15)
    tx = random.uniform(-3, 3)
    ty = random.uniform(-3, 3)
    
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Border filled with background luma mean
    bg_color = float(np.median(gray_patch[:4, :4]))
    warped = cv2.warpAffine(gray_patch, M, (w, h), borderValue=bg_color)
    
    # Random brightness / contrast
    alpha = random.uniform(0.8, 1.2) # contrast
    beta = random.uniform(-20, 20)  # brightness
    jittered = np.clip(warped.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    
    # Random blur
    if random.random() < 0.3:
        jittered = cv2.GaussianBlur(jittered, (3, 3), 0)
        
    # Random additive Gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(2, 10), (h, w))
        jittered = np.clip(jittered.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return jittered

def parse_board_code(code_str):
    clean = code_str.replace(" ", "")
    if "P" in clean:
        tile_part, _ = clean.split("P", 1)
    else:
        tile_part = clean
    return [f"{r}{num}" for r, num in re.findall(r'([SOGWBD])(\d*)', tile_part)]

def collect_base_samples(base_dir="."):
    """
    Loads base templates and renders synthetic patches for pure, un-corrupted seed samples.
    """
    samples = []
    templates_dir = os.path.join(base_dir, "templates")
    
    for val in TOKEN_VALUES:
        tmpl_path = os.path.join(templates_dir, f"num_{val}.png")
        if os.path.exists(tmpl_path):
            img_bgr = cv2.imread(tmpl_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_56 = cv2.resize(img_rgb, (56, 56))
                digit_roi = extract_digit_roi(img_56)
                gray_32 = preprocess_patch(digit_roi)
                samples.append((gray_32 * 255.0, val))
                
        for _ in range(50):
            synth_patch = generate_synthetic_digit_patch(val)
            digit_roi = extract_digit_roi(synth_patch)
            gray_32 = preprocess_patch(digit_roi)
            samples.append((gray_32 * 255.0, val))
                
    print(f"Collected {len(samples)} base seed samples across all 10 classes.")
    return samples

class SyntheticDigitDataset(Dataset):
    def __init__(self, base_samples, samples_per_class=600):
        self.data = []
        self.labels = []
        
        # Group base samples by class
        by_class = {val: [] for val in TOKEN_VALUES}
        for img, val in base_samples:
            by_class[val].append(img)
            
        for val, img_list in by_class.items():
            class_idx = CLASS_TO_IDX[val]
            if not img_list:
                print(f"Warning: No base samples found for token {val}!")
                continue
                
            for _ in range(samples_per_class):
                base_img = random.choice(img_list)
                aug = augment_patch(base_img.astype(np.uint8))
                norm_tensor = (aug.astype(np.float32) / 255.0)[np.newaxis, :, :] # (1, 32, 32)
                self.data.append(norm_tensor)
                self.labels.append(class_idx)
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train_and_export(base_dir="."):
    base_samples = collect_base_samples(base_dir)
    dataset = SyntheticDigitDataset(base_samples, samples_per_class=800)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Training dataset contains {len(dataset)} augmented samples.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model.train()
    epochs = 25
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        acc = 100.0 * correct / total
        epoch_loss = running_loss / total
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:2d}/{epochs}] Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")
            
    # Export to ONNX format
    model.eval()
    dummy_input = torch.randn(1, 1, 32, 32, device=device)
    
    models_dir = os.path.join(base_dir, "app", "cv", "models")
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = os.path.join(models_dir, "catan_digit_classifier.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    train_and_export()
