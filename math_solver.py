import cv2
import numpy as np
import easyocr
import sympy as sp

class MathAI:
    def __init__(self):
        print("Initializing AI Models...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def solve_canvas(self, canvas_img):
        # 1. Grayscale & Threshold
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 2. NEW: Dilation (Make the handwriting thicker and bolder for the AI)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # 3. Read the text
        results = self.reader.readtext(thresh, detail=0)
        
        if not results:
            return "No math detected. Try drawing larger."
            
        # 4. Clean up the string and fix common AI typos
        equation_str = "".join(results).replace(" ", "").lower()
        equation_str = equation_str.replace('x', '*').replace('=', '')
        equation_str = equation_str.replace('s', '5').replace('z', '2').replace('o', '0')
        
        try:
            expr = sp.sympify(equation_str)
            result = round(float(expr.evalf()), 4)
            # Change the * back to an x for the final display
            display_eq = equation_str.replace('*', 'x')
            return f"{display_eq} = {result}"
        except Exception as e:
            return f"Read: {''.join(results)} (Draw clearer)"