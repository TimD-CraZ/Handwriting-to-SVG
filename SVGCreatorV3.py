import os
import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

def create_ultimate_monoline_font(input_dir, output_path):
    units_per_em = 1000
    ascent = 800
    descent = -200
    
    fb = FontBuilder(units_per_em, isTTF=True)
    
    glyph_order = [".notdef", "space"] + [chr(i) for i in range(97, 123)]
    fb.setupGlyphOrder(glyph_order)
    
    character_map = {32: "space"}
    for i in range(97, 123):
        character_map[ord(chr(i))] = chr(i)
        character_map[ord(chr(i).upper())] = chr(i)
    
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0)); pen.lineTo((100, 700)); pen.lineTo((500, 700)); pen.lineTo((500, 0)); pen.closePath()
    glyphs = {".notdef": pen.glyph(), "space": TTGlyphPen(None).glyph()}
    metrics = {".notdef": (600, 0), "space": (500, 0)}

    print("Generating perfectly centered, uniform monoline vectors...")

    # ==========================================
    # STROKE THICKNESS CONTROL
    # Increase this number for a bolder monoline font.
    # ==========================================
    stroke_thickness = 6
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_thickness, stroke_thickness))

    for i in range(97, 123):
        char = chr(i)
        char_path = os.path.join(input_dir, char)
        
        if not os.path.exists(char_path): continue
            
        files = [f for f in os.listdir(char_path) if f.endswith('.png')]
        if not files: continue
        
        img = cv2.imread(os.path.join(char_path, files[0]), cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 1. Get the invisible spine
        skeleton = cv2.ximgproc.thinning(thresh)
        
        # 2. Sweep the perfectly circular digital brush
        bubble_ink = cv2.dilate(skeleton, brush)
        
        # 3. Melt away the jagged skeleton artifacts
        blurred = cv2.GaussianBlur(bubble_ink, (5, 5), 0)
        _, final_ink = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Find the absolute boundaries of the new smooth ink
        coords = cv2.findNonZero(final_ink)
        if coords is None:
            metrics[char] = (600, 0)
            continue
            
        x, y, w, h = cv2.boundingRect(coords)
        min_x = x # This is the crucial variable to fix your horizontal offset!
        
        contours, _ = cv2.findContours(final_ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pen = TTGlyphPen(None)
        
        scale = 1000 / 128.0 
        left_margin = 80 # The space before the letter starts
        
        for cnt in contours:
            if len(cnt) < 3: continue 
                
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            for idx, point in enumerate(approx):
                pt_x, pt_y = point[0]
                
                # FIXED MATH: Subtract min_x so the letter always starts exactly at the left_margin
                draw_x = int(left_margin + ((pt_x - min_x) * scale))
                
                # The Y baseline math remains perfect (128px canvas baseline is Y=89)
                draw_y = int((89 - pt_y) * scale) 
                
                draw_x = max(0, min(1500, draw_x))
                draw_y = max(-400, min(1000, draw_y))

                if idx == 0:
                    pen.moveTo((draw_x, draw_y))
                else:
                    pen.lineTo((draw_x, draw_y))
            
            pen.closePath() 
            
        glyphs[char] = pen.glyph()
        
        # The advance width now perfectly wraps the adjusted letter
        advance_width = int((w * scale) + (left_margin * 2)) 
        metrics[char] = (advance_width, 0)

    print("Compiling TrueType tables...")
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupCharacterMap(character_map)
    fb.setupNameTable({"familyName": "MyPerfectMonoline", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=descent, usWinAscent=1000, usWinDescent=400)
    fb.setupPost()
    fb.setupHead(unitsPerEm=units_per_em)
    
    fb.save(output_path)
    print(f"\nSUCCESS! Font saved as: {output_path}")

if __name__ == "__main__":
    input_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Final_Font_Set_Centered" 
    output_ttf = "MyPerfectMonoline.ttf"
    
    create_ultimate_monoline_font(input_folder, output_ttf)