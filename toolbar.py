import cv2

class Toolbar:
    def __init__(self):
        self.colors = [
            (50, 150, (255, 0, 255), "Purple"),
            (200, 300, (255, 0, 0), "Blue"),
            (350, 450, (0, 255, 0), "Green"),
            (500, 600, (0, 165, 255), "Orange")
        ]
        # Two AI Buttons now!
        self.ai_math_btn = (800, 950, (100, 100, 100), "Math")
        self.height = 100
        self.ai_guess_btn = (1000, 1200, (50, 50, 200), "Guess") 

    def draw(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (1280, self.height + 20), (30, 30, 30), cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        # Brushes
        for start_x, end_x, color, name in self.colors:
            cv2.rectangle(img, (start_x, 15), (end_x, self.height), color, cv2.FILLED)
            cv2.rectangle(img, (start_x, 15), (end_x, self.height), (255, 255, 255), 2)
            
        # Math Button
        sx, ex, col, name = self.ai_math_btn
        cv2.rectangle(img, (sx, 15), (ex, self.height), col, cv2.FILLED)
        cv2.putText(img, name, (sx + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Guess Button
        sx, ex, col, name = self.ai_guess_btn
        cv2.rectangle(img, (sx, 15), (ex, self.height), col, cv2.FILLED)
        cv2.rectangle(img, (sx, 15), (ex, self.height), (0, 255, 255), 3)
        cv2.putText(img, name, (sx + 25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
        return img

    def check_interaction(self, x, y):
        if y < self.height + 20: 
            for start_x, end_x, color, name in self.colors:
                if start_x < x < end_x:
                    return 'color', color
                    
            sx, ex, _, _ = self.ai_math_btn
            if sx < x < ex: return 'action', 'Math'
            
            sx, ex, _, _ = self.ai_guess_btn
            if sx < x < ex: return 'action', 'Guess'
            
        return None, None