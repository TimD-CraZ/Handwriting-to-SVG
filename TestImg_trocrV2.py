import os
import glob

# ==========================================
# STAGE 1: THE BATCH CRAFT EXTRACTOR
# ==========================================
def run_batch_craft(source_dir, base_output_dir):
    from craft_text_detector import Craft
    
    print("Loading CRAFT model into memory... (This takes a moment)")
    # We load the model ONCE outside the loop to save massive overhead
    craft = Craft(output_dir=base_output_dir, crop_type="box", cuda=False) 
    
    # Grab all image files from the AllSource folder
    image_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"No images found in {source_dir}!")
        return

    print(f"\nFound {len(image_files)} full page images. Starting extraction...")
    print("-" * 50)
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(source_dir, filename)
        print(f"[{i+1}/{len(image_files)}] Slicing words from: {filename}")
        
        # CRAFT automatically creates a subfolder inside base_output_dir 
        # named after the image file (e.g., 'Page1_crops').
        craft.detect_text(image_path)
        
    # Clean up memory when done
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    
    print("-" * 50)
    print("Stage 1 Complete! All pages have been chopped into word crops.")

if __name__ == "__main__":
    # Your updated paths
    source_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\AllSource"
    output_directory = "K:\Self Coding\Handwriting to SVG\Test\craft_output"
    
    if not os.path.exists(source_folder):
        print(f"Please create the folder '{source_folder}' and put your full page photos in it.")
    else:
        run_batch_craft(source_folder, output_directory)