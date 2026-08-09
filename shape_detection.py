import cv2
import numpy as np

class ShapeAI:
    def __init__(self):
        print("Initializing Ultra-Robust Shape AI...")

    def guess_canvas(self, canvas_img):
        # 1. Grayscale & Threshold
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 2. Dilation to connect any gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # 3. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "No shape detected"
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 500:
            return "Draw a larger shape"

        # --- THE MAGIC FIX: SHRINK-WRAP THE SHAPE ---
        # This ignores the brush thickness and hollow centers
        hull = cv2.convexHull(largest_contour)
        
        # 4. Count corners on the smooth shrink-wrapped hull
        peri = cv2.arcLength(hull, True)
        # We also increased the smoothing factor slightly to 0.05 for messy hand-drawings
        approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
        corners = len(approx)
        
        # 5. Identify the Shape
        if corners == 3:
            return "I see: A Triangle!"
        elif corners == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            # Gave even more wiggle room for hand-drawn squares
            if 0.75 <= aspect_ratio <= 1.25:
                return "I see: A Square!"
            else:
                return "I see: A Rectangle!"
        elif corners == 5:
            return "I see: A Pentagon!"
        else:
            return "I see: A Circle!"