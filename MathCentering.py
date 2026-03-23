import os
import glob
import cv2
import numpy as np

def isolate_and_center_math(gray_image, canvas_size=64):
    """
    1. Uses math to find the biggest ink blob (ignoring noise).
    2. Crops a tight box around it.
    3. Pastes it perfectly in the dead center of a 64x64 canvas.
    """
    # 1. Threshold to find the blobs
    _, binary = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1: 
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8) # Blank
        
    # 2. Find the largest blob (that isn't the background)
    largest_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i
            
    # 3. Check for 'i' or 'j' dots floating above the main blob
    core_x, core_y, core_w, core_h, core_area = stats[largest_label]
    dot_label = -1
    for i in range(1, num_labels):
        if i == largest_label: continue
        cx, cy = centroids[i]
        # If it's smaller, higher up, and horizontally aligned, it's a dot
        if stats[i, cv2.CC_STAT_AREA] < core_area and cy < centroids[largest_label][1] and (core_x - 5 <= cx <= core_x + core_w + 5):
            dot_label = i
            break

    # 4. Create a mask of ONLY the letter and its dot
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    mask[labels == largest_label] = 255
    if dot_label != -1:
        mask[labels == dot_label] = 255
        
    # 5. Extract the clean ink (preserving soft gradients!)
    clean_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    # 6. Find the tight mathematical bounding box of the clean ink
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
    x, y, w, h = cv2.boundingRect(coords)
    tight_ink = clean_gray[y:y+h, x:x+w]
    
    # 7. Paste it onto the exact center of the new canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # If the letter is somehow taller/wider than 64px, scale it down slightly
    if h > canvas_size - 4 or w > canvas_size - 4:
        scale = (canvas_size - 6) / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        tight_ink = cv2.resize(tight_ink, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        
    start_y = (canvas_size - h) // 2
    start_x = (canvas_size - w) // 2
    
    canvas[start_y:start_y+h, start_x:start_x+w] = tight_ink
    return canvas

if __name__ == "__main__":
    # 1. SETUP FOLDERS
    base_craft_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output"
    global_output_dir = os.path.join(base_craft_folder, "All_Anchored_Characters")
    os.makedirs(global_output_dir, exist_ok=True)
    
    # 2. GATHER ALL RAW CHARACTERS (Direct from the Slicer)
    search_pattern = os.path.join(base_craft_folder, "*_crops", "isolated_characters", "*.png")
    all_char_paths = glob.glob(search_pattern)
    
    if not all_char_paths:
        print("No raw characters found! Make sure your paths are correct.")
        exit()
        
    print(f"Found {len(all_char_paths)} raw crops. Executing Mathematical Centering...")
    
    # 3. RUN BATCH
    for i, image_path in enumerate(all_char_paths):
        filename = os.path.basename(image_path)
        parent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        unique_filename = f"{parent_folder_name}_{filename}"
        
        if i % 500 == 0: print(f"Processing [{i}/{len(all_char_paths)}]...")
            
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None: continue
            
        # Execute perfect mathematical centering
        centered_img = isolate_and_center_math(gray_img)
        
        save_path = os.path.join(global_output_dir, unique_filename)
        cv2.imwrite(save_path, centered_img)
        
    print(f"\nSUCCESS! Exported {len(all_char_paths)} flawlessly centered characters to {global_output_dir}")