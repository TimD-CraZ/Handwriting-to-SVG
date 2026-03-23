import os
import cv2
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen

def inject_scaled_handwriting(base_font_path, input_dir, output_path):
    print(f"Cracking open host font: {base_font_path}...")
    font = TTFont(base_font_path)
    
    cmap = font['cmap'].getBestCmap()
    glyf_table = font['glyf']
    hmtx_table = font['hmtx']
    
    units_per_em = font['head'].unitsPerEm
    
    # ==========================================
    # 1. THE SIZE MULTIPLIER
    # 1.0 = Original tiny size.
    # 1.8 = 80% larger (Good starting point for your images).
    # ==========================================
    size_multiplier = 1.5 
    scale = (units_per_em / 128.0) * size_multiplier
    
    # Reduced the margin slightly so the larger letters don't look too far apart
    left_margin = int(units_per_em * 0.05) 

    # ==========================================
    # 2. STROKE THICKNESS
    # *Note: Because we are magnifying the vector, the stroke will also look thicker.
    # If 5 looks too bold at the new size, drop this to 3 or 4.
    # ==========================================
    stroke_thickness = 4 
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_thickness, stroke_thickness))

    print(f"Injecting your handwriting at {size_multiplier}x scale...")

    for i in range(97, 123):
        char = chr(i)
        char_code = ord(char)
        
        if char_code not in cmap: continue
        glyph_name = cmap[char_code]

        char_path = os.path.join(input_dir, char)
        if not os.path.exists(char_path): continue
            
        files = [f for f in os.listdir(char_path) if f.endswith('.png')]
        if not files: continue
        
        # Process the image into a smooth uniform bubble
        img = cv2.imread(os.path.join(char_path, files[0]), cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        skeleton = cv2.ximgproc.thinning(thresh)
        bubble_ink = cv2.dilate(skeleton, brush)
        blurred = cv2.GaussianBlur(bubble_ink, (5, 5), 0)
        _, final_ink = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(final_ink)
        if coords is None: continue
            
        x, y, w, h = cv2.boundingRect(coords)
        min_x = x 
        
        # Extract Vectors
        contours, _ = cv2.findContours(final_ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pen = TTGlyphPen(None)
        
        for cnt in contours:
            if len(cnt) < 3: continue 
                
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            for idx, point in enumerate(approx):
                pt_x, pt_y = point[0]
                
                # Apply the scaled math
                draw_x = int(left_margin + ((pt_x - min_x) * scale))
                draw_y = int((89 - pt_y) * scale) 
                
                if idx == 0:
                    pen.moveTo((draw_x, draw_y))
                else:
                    pen.lineTo((draw_x, draw_y))
            
            pen.closePath() 
            
        glyf_table[glyph_name] = pen.glyph()
        
        # The advance width is also dynamically scaled up
        advance_width = int((w * scale) + (left_margin * 2)) 
        hmtx_table[glyph_name] = (advance_width, left_margin)

    # Change the internal font name
    name_table = font['name']
    for record in name_table.names:
        if record.nameID in (1, 4, 6): 
            record.string = "MyFontHandwriting".encode('utf-16-be')

    font.save(output_path)
    print(f"\nSUCCESS! Hijacked font saved as: {output_path}")

if __name__ == "__main__":
    base_font = "K:\Self Coding\Handwriting to SVG\HostFont.ttf" 
    input_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Final_Font_Set_Centered" 
    output_ttf = "MyFontHandwriting.ttf"
    
    inject_scaled_handwriting(base_font, input_folder, output_ttf)