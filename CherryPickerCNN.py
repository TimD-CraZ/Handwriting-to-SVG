import os
import cv2
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ==========================================
# 1. THE FAST, CHARACTER-LEVEL BRAIN
# ==========================================
class CharacterJudge(nn.Module):
    def __init__(self):
        super(CharacterJudge, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26) # 26 letters (a-z)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_judge(device, model_path):
    print("No existing Judge brain found. Training on EMNIST... (Takes about 45 seconds)")
    
    # EMNIST is rotated sideways by default, this fixes it so it matches your handwriting
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torchvision.transforms.functional.rotate(img, -90)),
        transforms.Lambda(lambda img: torchvision.transforms.functional.hflip(img)),
        transforms.ToTensor(),
    ])
    
    train_data = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
    train_data.targets = train_data.targets - 1 # Shift labels from 1-26 to 0-25
    
    dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    model = CharacterJudge().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3 
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), model_path)
    print("Brain trained and saved!\n")
    return model

# ==========================================
# 2. THE CHERRY-PICKING ENGINE
# ==========================================
def cherry_pick(input_dir, output_dir, max_per_char=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "custom_handwriting_judge.pth"
    
    # if not os.path.exists(model_path):
    #     model = train_judge(device, model_path)
    # else:
    #     model = CharacterJudge().to(device)
    #     model.load_state_dict(torch.load(model_path, map_location=device))

    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Could not find '{model_path}'!")
        return # Stop the script entirely!
        
    print(f"Loading custom brain '{model_path}' on {device}...")
    model = CharacterJudge().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    rankings = {chr(i + 97): [] for i in range(26)}
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
    print(f"Judging {len(image_files)} characters using Fast CNN...")

    for filename in image_files:
        filepath = os.path.join(input_dir, filename)
        
        # 1. Load image (White/Grey ink on Black background, exactly what EMNIST expects)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # 2. Downscale your 64x64 crop to the 28x28 size the AI expects
        eval_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 3. Convert to tensor
        tensor_img = torch.tensor(eval_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor_img = tensor_img.to(device)
        
        # 4. Predict
        with torch.no_grad():
            logits = model(tensor_img)
            probabilities = F.softmax(logits, dim=1).cpu().squeeze().numpy()
            
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx]
        predicted_letter = chr(best_idx + 97) 
        
        rankings[predicted_letter].append((confidence, filepath, filename))

    # ==========================================
    # 3. EXPORT & REPORTING
    # ==========================================
    total_saved = 0
    missing_chars, short_chars = [], []

    print("\n--- CHERRY PICKING RESULTS ---")
    # Clean out the old output directory so we don't mix TrOCR garbage with the new CNN results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    for letter, items in rankings.items():
        char_count = len(items)
        if char_count == 0:
            missing_chars.append(letter)
            continue
        elif char_count < max_per_char:
            short_chars.append(f"'{letter}' (found {char_count})")
            
        items.sort(key=lambda x: x[0], reverse=True) # Sort by confidence
        top_k = items[:max_per_char]
        
        letter_dir = os.path.join(output_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)
        
        for rank, (score, filepath, filename) in enumerate(top_k):
            safe_score = int(score * 100)
            new_filename = f"rank_{rank+1}_score_{safe_score:03d}_{filename}"
            save_path = os.path.join(letter_dir, new_filename)
            shutil.copy(filepath, save_path)
            total_saved += 1
            
    print("\n==========================================")
    print("         DATASET HEALTH REPORT")
    print("==========================================")
    if missing_chars: print(f"[CRITICAL] Zero images found for: {', '.join(missing_chars)}")
    else: print("[OK] At least one image found for every character!")
        
    if short_chars: print(f"[WARNING] Fewer than {max_per_char} images found for: {', '.join(short_chars)}")
    else: print(f"[OK] All found characters met the target of {max_per_char} images!")
    print("==========================================\n")

if __name__ == "__main__":
    input_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\\All_Anchored_Characters"
    final_output_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Final_Font_Set"
    
    cherry_pick(input_folder, final_output_folder, max_per_char=3)