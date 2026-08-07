import cv2
import numpy as np

class Canvas:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        # Initialize a black canvas
        self.img_canvas = np.zeros((self.height, self.width, 3), np.uint8)

    def draw_line(self, pt1, pt2, color, thickness):
        """Draws a line on the canvas and returns the new previous point."""
        if pt1 == (0, 0): # If it's the first point, start exactly at pt2
            pt1 = pt2
            
        cv2.line(self.img_canvas, pt1, pt2, color, thickness)
        return pt2 

    def merge_with_frame(self, frame):
        """Overlays the drawings onto the webcam feed."""
        # Convert canvas to grayscale
        img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
        
        # Invert colors: drawings become black, background becomes white
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        
        # Black out the regions on the webcam feed where we drew
        frame = cv2.bitwise_and(frame, img_inv)
        
        # Add the canvas colors to those blacked-out regions
        frame = cv2.bitwise_or(frame, self.img_canvas)
        return frame
        
    def clear(self):
        self.img_canvas = np.zeros((self.height, self.width, 3), np.uint8)