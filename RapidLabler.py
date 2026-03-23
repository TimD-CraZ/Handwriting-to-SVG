import os
import cv2
import shutil

# ==========================================
# 1. MEMORY MANAGEMENT
# ==========================================
def load_history(history_path):
    if not os.path.exists(history_path):
        return set()
    with open(history_path, 'r') as f:
        return set(line.strip() for line in f)

def save_to_history(history_path, identifier):
    with open(history_path, 'a') as f:
        f.write(identifier + '\n')

# ==========================================
# 2. RAPID LABELING ENGINE
# ==========================================
def rapid_labeler(input_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create A-Z folders
    for i in range(26):
        os.makedirs(os.path.join(output_dir, chr(i + 97)), exist_ok=True)
        
    # Set up our memory file
    history_path = os.path.join(output_dir, "labeling_history.txt")
    seen_images = load_history(history_path)

    if not os.path.exists(input_folder):
        print(f"Error: Could not find '{input_folder}'!")
        return

    # Grab all images and filter out the ones we've already processed
    all_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    remaining_files = [f for f in all_files if f not in seen_images]

    if not remaining_files:
        print("No new images to label! You are completely caught up.")
        return

    print("--------------------------------------------------")
    print(f"Found {len(remaining_files)} unlabeled images.")
    print("HOW TO USE:")
    print("- Press [a-z] to label an image.")
    print("- Press SPACEBAR to physically delete a bad image.")
    print("- Press ']' to SKIP the rest of the current page/source.")
    print("- Press ESC to save progress and quit early.")
    print("--------------------------------------------------\n")

    labeled_count = 0
    quit_all = False
    skip_source_prefix = None

    for filename in remaining_files:
        # If we told the script to skip a page, blast through its remaining images silently
        if skip_source_prefix and filename.startswith(skip_source_prefix):
            save_to_history(history_path, filename)
            seen_images.add(filename)
            continue
            
        # Extract the page name from the filename (e.g., "Tanm1_crops" from "Tanm1_crops_crop_20_char_000.png")
        prefix_end_idx = filename.find("_crop")
        current_page_prefix = filename[:prefix_end_idx] if prefix_end_idx != -1 else filename.split('_')[0]

        filepath = os.path.join(input_folder, filename)
        
        img = cv2.imread(filepath)
        if img is None: continue
        
        # Scale up so you can see it clearly
        display_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Show the current page name on the screen so you know where you are!
        # cv2.putText(display_img, f"Page: {current_page_prefix}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Rapid Labeler (Press a-z, Space, ], Esc)", display_img)
        
        while True:
            key = cv2.waitKey(0)
            
            if key == 27: # ESC key
                print("\nExiting early...")
                quit_all = True
                break
                
            elif key == 93: # ']' key
                print(f"\n>>> Skipping the remainder of {current_page_prefix}...")
                skip_source_prefix = current_page_prefix
                save_to_history(history_path, filename)
                seen_images.add(filename)
                break
                
            elif key == 32: # Spacebar
                print(f"Trashed and DELETED: {filename}")
                try:
                    os.remove(filepath)
                except OSError as e:
                    pass # Ignore if Windows file lock prevents deletion
                    
                save_to_history(history_path, filename)
                seen_images.add(filename)
                break
                
            elif 97 <= key <= 122: # lowercase a-z
                letter = chr(key)
                dest_dir = os.path.join(output_dir, letter)
                
                shutil.copy(filepath, os.path.join(dest_dir, filename))
                
                save_to_history(history_path, filename)
                seen_images.add(filename)
                labeled_count += 1
                print(f"[{labeled_count}] Labeled '{letter}' -> {filename}")
                break
                
            else:
                print("Invalid key! Press a-z, Spacebar, ], or Esc.")
        
        if quit_all: 
            break

    cv2.destroyAllWindows()
    print(f"\nAwesome! You successfully labeled {labeled_count} images this session.")

if __name__ == "__main__":
    source_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\All_Anchored_Characters"
    destination_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Hand_Labeled_Data"
    
    rapid_labeler(source_folder, destination_folder)