import os
import cv2
import numpy as np

# ==========================================
# 1. PROPORTIONAL TYPOGRAPHY LOGIC
# ==========================================
def isolate_scale_and_center(gray_image, char, canvas_size=128):
    # 1. Binarize and find the absolute boundaries of the raw ink
    _, binary = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    
    if coords is None:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
    x, y, w, h = cv2.boundingRect(coords)
    tight_ink = gray_image[y:y+h, x:x+w]
    
    # 2. Define our 4-Line Notebook Metrics
    # We will force the letters to scale to these specific pixel heights
    target_x_height = 40  
    
    char = char.lower()
    
    # Group the letters by typographical class to determine their target height
    group_x_height = ['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z']
    group_ascender = ['b', 'd', 'f', 'h', 'k', 'l', 't']
    group_descender = ['g', 'p', 'q', 'y']
    
    if char in group_x_height:
        target_h = target_x_height
    elif char in group_ascender:
        target_h = int(target_x_height * 1.6) # Ascenders are ~60% taller
    elif char in group_descender:
        target_h = int(target_x_height * 1.5) # Descenders are ~50% taller
    elif char == 'i':
        target_h = int(target_x_height * 1.3) # Slightly taller to account for the dot
    elif char == 'j':
        target_h = int(target_x_height * 2.0) # Full span letter
    else:
        target_h = target_x_height # Fallback

    # 3. Proportional Scaling
    # We scale the width by the exact same multiplier so the letter doesn't get squished
    scale_factor = target_h / float(h)
    target_w = int(w * scale_factor)
    
    # Use INTER_AREA for shrinking, INTER_CUBIC for enlarging to preserve smoothness
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    scaled_ink = cv2.resize(tight_ink, (target_w, target_h), interpolation=interpolation)
    
    # Re-threshold slightly to keep the edges crisp after resizing
    _, scaled_ink = cv2.threshold(scaled_ink, 100, 255, cv2.THRESH_BINARY)
    
    # 4. The 4-Line Notebook Anchoring
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    start_x = (canvas_size - target_w) // 2
    
    base_line = int(canvas_size * 0.70)  # Y = 89
    mean_line = base_line - target_x_height # Y = 49
    
    if char in group_descender or char == 'j':
        # Align the TOP of the letter's main body to the mean_line
        start_y = mean_line
    else:
        # Align the BOTTOM of the letter to the base_line
        start_y = base_line - target_h

    # 5. Safety boundary checks
    start_x = max(0, min(canvas_size - target_w, start_x))
    start_y = max(0, min(canvas_size - target_h, start_y))
    
    canvas[start_y:start_y+target_h, start_x:start_x+target_w] = scaled_ink
    
    return canvas

# ==========================================
# 2. THE BATCH PROCESSOR
# ==========================================
if __name__ == "__main__":
    input_dir = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Final_Font_Set"
    output_dir = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Final_Font_Set_Centered"
    
    if not os.path.exists(input_dir):
        print(f"Error: Could not find {input_dir}")
        exit()
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("Applying Proportional Typographical Scaling...")
    
    total_processed = 0
    for letter_folder in os.listdir(input_dir):
        in_letter_path = os.path.join(input_dir, letter_folder)
        
        if not os.path.isdir(in_letter_path): continue
            
        out_letter_path = os.path.join(output_dir, letter_folder)
        os.makedirs(out_letter_path, exist_ok=True)
        
        for filename in os.listdir(in_letter_path):
            if not filename.endswith('.png'): continue
                
            img_path = os.path.join(in_letter_path, filename)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if gray_img is None: continue
                
            # Apply the intelligent scaler
            perfect_img = isolate_scale_and_center(gray_image=gray_img, char=letter_folder)
            
            save_path = os.path.join(out_letter_path, filename)
            cv2.imwrite(save_path, perfect_img)
            total_processed += 1
            
    print(f"\nSuccess! Processed {total_processed} proportional characters.")
    print(f"Check the output in: {output_dir}")