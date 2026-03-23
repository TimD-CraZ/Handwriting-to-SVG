import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE ARCHITECTURE
# ==========================================
class BBoxRegressor(nn.Module):
    def __init__(self):
        super(BBoxRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def isolate_core_letter(gray_image):
    """
    Uses Connected Component Analysis to keep ONLY the ink blob closest 
    to the AI's anchor point (32, 32), while preserving grayscale gradients!
    """
    # 1. Create a strict binary threshold JUST for the math
    _, binary_math = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    
    # 2. Find all separate blobs of ink using the mathematical binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_math, connectivity=8)
    
    if num_labels <= 1: return gray_image # Blank image, do nothing
    
    # 3. Find the component closest to the exact center (32, 32)
    min_dist = float('inf')
    core_label = 1
    
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        dist = (cx - 32)**2 + (cy - 32)**2 
        if dist < min_dist:
            min_dist = dist
            core_label = i
            
    # 4. Look for 'i' or 'j' dots (small blobs directly above the core blob)
    core_x, core_y, core_w, core_h, core_area = stats[core_label]
    dot_label = -1
    
    for i in range(1, num_labels):
        if i == core_label: continue
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        
        if area < core_area and cy < centroids[core_label][1] and (core_x - 5 <= cx <= core_x + core_w + 5):
            dot_label = i
            break
            
    # 5. THE COOKIE CUTTER FIX
    # Create a mask of the core letter (and dot if found)
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    mask[labels == core_label] = 255
    if dot_label != -1:
        mask[labels == dot_label] = 255
        
    # Use the mask to punch out the original soft grayscale pixels!
    clean_gray_img = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
    return clean_gray_img

# ==========================================
# 2. IMAGE FORMATTING UTILITY
# ==========================================
def format_for_cnn(image, canvas_size=64):
    h, w = image.shape
    # Scale down slightly to ensure we have room to shift it around
    scale = 40.0 / max(h, w) 
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    return canvas

# ==========================================
# 3. INFERENCE & CENTERING LOOP
# ==========================================
def center_handwriting(model_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BBoxRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} letters. Running AI Centering...")
    
    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Place on 64x64 canvas
        canvas = format_for_cnn(raw_img)
        
        tensor_img = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor_img = tensor_img.to(device)
        
        # 2. Ask AI where the core letter is
        with torch.no_grad():
            pred = model(tensor_img).cpu().numpy()[0]
            
        x_min, y_min, x_max, y_max = [int(val * 64) for val in pred]
        
        # 3. Calculate the shift needed
        # Find the center of the AI's predicted box
        ai_center_x = (x_min + x_max) // 2
        ai_center_y = (y_min + y_max) // 2
        
        # Find the distance to the absolute center of the 64x64 canvas (32, 32)
        shift_x = 32 - ai_center_x
        shift_y = 32 - ai_center_y
        
        # 4. Shift the ENTIRE image (tails included!) to perfectly center the core
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered_img = cv2.warpAffine(canvas, translation_matrix, (64, 64))

        centered_img = isolate_core_letter(centered_img)
        
        # 5. Save the perfectly aligned image
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, centered_img)

        # Visualize the first one to prove it works
        if i == 35:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            debug_raw = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_raw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            plt.title("Original (AI Box in Green)")
            plt.imshow(debug_raw)
            
            plt.subplot(1, 2, 2)
            plt.title(f"Shifted (dx:{shift_x}, dy:{shift_y})")
            plt.imshow(centered_img, cmap='gray')
            # Draw a crosshair at dead center (32,32) to prove it's anchored
            plt.axvline(x=32, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=32, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.show()

    print(f"\nSuccess! Check '{output_dir}' for perfectly anchored letters.")

# if __name__ == "__main__":
#     model_weights = "bbox_adjuster_model.pth"
#     input_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Tanm1_crops\isolated_characters"
#     output_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Tanm1_crops\\ai_anchored_characters"
    
#     center_handwriting(model_weights, input_folder, output_folder)

if __name__ == "__main__":
    import glob
    import os
    import torch
    import cv2
    import numpy as np

    # 1. SETUP FOLDERS
    base_craft_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output"
    
    # We will dump all finished characters into ONE massive central folder
    global_output_dir = os.path.join(base_craft_folder, "All_Anchored_Characters")
    os.makedirs(global_output_dir, exist_ok=True)
    
    # 2. LOAD THE CNN BRAIN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CNN brain on {device}...")
    
    model_path = 'bbox_adjuster_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}! Make sure it is in the same folder.")
        exit()
        
    model = BBoxRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("CNN Brain loaded successfully!")
    
    # 3. GATHER ALL RAW CHARACTERS
    # Hunt down every PNG inside every 'isolated_characters' folder
    search_pattern = os.path.join(base_craft_folder, "*_crops", "isolated_characters", "*.png")
    all_char_paths = glob.glob(search_pattern)
    
    if not all_char_paths:
        print("No isolated characters found! Did the Slicer finish running?")
        exit()
        
    print(f"\nFound {len(all_char_paths)} total raw characters across all pages.")
    print(f"Centering them and saving to: {global_output_dir}\n")
    
    # 4. RUN THE BATCH PIPELINE
    for i, image_path in enumerate(all_char_paths):
        filename = os.path.basename(image_path)
        
        # --- COLLISION PREVENTION ---
        # Since Page1 and Page2 might both have a 'crop_20_char_000.png', 
        # we prepend the page folder name to ensure the filename is 100% unique!
        parent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        unique_filename = f"{parent_folder_name}_{filename}"
        
        if i % 500 == 0:
            print(f"Anchoring [{i}/{len(all_char_paths)}]...")
            
        # # --- STANDARD INFERENCE LOGIC ---
        # gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # if gray_img is None: continue
            
        # # The CNN was trained on strict 64x64 binary images
        # _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        # --- STANDARD INFERENCE LOGIC ---
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None: continue
            
        # 1. THE FIX: Place the raw weirdly-shaped crop onto a 64x64 canvas FIRST
        canvas_img = format_for_cnn(gray_img, canvas_size=64)
            
        # 2. Threshold the perfect 64x64 canvas
        _, binary = cv2.threshold(canvas_img, 127, 255, cv2.THRESH_BINARY)
        
        tensor_img = torch.tensor(binary, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor_img = tensor_img.to(device)
        
        with torch.no_grad():
            pred = model(tensor_img).cpu().squeeze().numpy()
            
        dx, dy = int(round(pred[0])), int(round(pred[1]))
        
        # Apply the exact same shift to the beautiful grayscale image
        shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_gray = cv2.warpAffine(gray_img, shift_matrix, (64, 64))
        
        # Vaporize the floating noise using your updated grayscale cookie cutter
        clean_shifted_gray = isolate_core_letter(shifted_gray)
        
        # Save to the central hub
        save_path = os.path.join(global_output_dir, unique_filename)
        cv2.imwrite(save_path, clean_shifted_gray)
        
    print(f"\nSUCCESS! Exported {len(all_char_paths)} perfectly centered characters.")
    print(f"They are ready for Cherry Picking in: {global_output_dir}")