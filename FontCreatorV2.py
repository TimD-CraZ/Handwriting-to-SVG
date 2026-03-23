import os
import cv2
import numpy as np
import random
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

def create_variant(img, angle, y_shift, x_shift):
    """Uses OpenCV Affine Transforms to rotate and bounce the image."""
    rows, cols = img.shape
    
    # 1. Rotation Matrix
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(img, M_rot, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 2. Translation Matrix (Bounce and Wobble)
    M_trans = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated = cv2.warpAffine(rotated, M_trans, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return translated

def inject_opentype_randomness(base_font_path, input_dir, output_path):
    print(f"Cracking open host font for OpenType Injection...")
    font = TTFont(base_font_path)
    
    cmap = font['cmap'].getBestCmap()
    glyf_table = font['glyf']
    hmtx_table = font['hmtx']
    
    units_per_em = font['head'].unitsPerEm
    size_multiplier = 1.8 
    scale = (units_per_em / 128.0) * size_multiplier
    base_left_margin = int(units_per_em * 0.05) 
    
    stroke_thickness = 4 
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_thickness, stroke_thickness))

    new_glyph_names = []
    
    print("Generating 3 unique OpenCV variations per letter...")

    for i in range(97, 123):
        char = chr(i)
        char_code = ord(char)
        
        if char_code not in cmap: continue
        base_glyph_name = cmap[char_code]

        char_path = os.path.join(input_dir, char)
        if not os.path.exists(char_path): continue
            
        files = [f for f in os.listdir(char_path) if f.endswith('.png')]
        if not files: continue
        
        img = cv2.imread(os.path.join(char_path, files[0]), cv2.IMREAD_GRAYSCALE)
        
        variants = [
            {"suffix": "",       "angle": 0,                "y": 0,  "x": 0}, 
            {"suffix": ".alt1",  "angle": random.uniform(2, 6),   "y": -3, "x": 2}, 
            {"suffix": ".alt2",  "angle": random.uniform(-6, -2), "y": 4,  "x": -2} 
        ]

        for var in variants:
            glyph_name = base_glyph_name + var["suffix"]
            if var["suffix"] != "" and glyph_name not in new_glyph_names:
                new_glyph_names.append(glyph_name)
            
            mutated_img = create_variant(img, var["angle"], var["y"], var["x"])
            _, thresh = cv2.threshold(mutated_img, 127, 255, cv2.THRESH_BINARY)
            
            skeleton = cv2.ximgproc.thinning(thresh)
            bubble_ink = cv2.dilate(skeleton, brush)
            blurred = cv2.GaussianBlur(bubble_ink, (5, 5), 0)
            _, final_ink = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            coords = cv2.findNonZero(final_ink)
            if coords is None: continue
                
            x, y, w, h = cv2.boundingRect(coords)
            min_x = x 
            
            contours, _ = cv2.findContours(final_ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            pen = TTGlyphPen(None)
            
            for cnt in contours:
                if len(cnt) < 3: continue 
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                for idx, point in enumerate(approx):
                    pt_x, pt_y = point[0]
                    draw_x = int(base_left_margin + ((pt_x - min_x) * scale))
                    draw_y = int((89 - pt_y) * scale) 
                    
                    if idx == 0: pen.moveTo((draw_x, draw_y))
                    else: pen.lineTo((draw_x, draw_y))
                
                pen.closePath() 
                
            glyf_table[glyph_name] = pen.glyph()
            advance_width = int((w * scale) + (base_left_margin * 2)) 
            hmtx_table[glyph_name] = (advance_width, base_left_margin)
            
            # SAFETY CHECK: If host uses Vertical Metrics, register the glyphs there too
            if 'vmtx' in font:
                font['vmtx'][glyph_name] = (units_per_em, 0)

    # ==========================================
    # THE FIX: STRICT GLYPH REGISTRATION
    # ==========================================
    print("Registering new glyphs to master index...")
    glyph_order = font.getGlyphOrder()
    for name in new_glyph_names:
        if name not in glyph_order:
            glyph_order.append(name)
    font.setGlyphOrder(glyph_order)

    print("Writing OpenType substitution logic...")
    
    fea_script = """
    languagesystem DFLT dflt;
    languagesystem latn dflt;

    @default = [a b c d e f g h i j k l m n o p q r s t u v w x y z];
    @alt1    = [a.alt1 b.alt1 c.alt1 d.alt1 e.alt1 f.alt1 g.alt1 h.alt1 i.alt1 j.alt1 k.alt1 l.alt1 m.alt1 n.alt1 o.alt1 p.alt1 q.alt1 r.alt1 s.alt1 t.alt1 u.alt1 v.alt1 w.alt1 x.alt1 y.alt1 z.alt1];
    @alt2    = [a.alt2 b.alt2 c.alt2 d.alt2 e.alt2 f.alt2 g.alt2 h.alt2 i.alt2 j.alt2 k.alt2 l.alt2 m.alt2 n.alt2 o.alt2 p.alt2 q.alt2 r.alt2 s.alt2 t.alt2 u.alt2 v.alt2 w.alt2 x.alt2 y.alt2 z.alt2];

    feature calt {
        sub @default @default' by @alt1;
        sub @alt1 @default' by @alt2;
    } calt;
    """

    addOpenTypeFeaturesFromString(font, fea_script)

    name_table = font['name']
    for record in name_table.names:
        if record.nameID in (1, 4, 6): 
            record.string = "MyOpenTypeHandwriting".encode('utf-16-be')

    font.save(output_path)
    print(f"\nSUCCESS! Ultimate OpenType font saved as: {output_path}")

    
if __name__ == "__main__":
    base_font = "K:\Self Coding\Handwriting to SVG\HostFont.ttf" 
    input_folder = "K:\Self Coding\Handwriting to SVG\Test\craft_output\Final_Font_Set_Centered" 
    output_ttf = "MyOpenTypeHandwriting.ttf"
    
    inject_opentype_randomness(base_font, input_folder, output_ttf)