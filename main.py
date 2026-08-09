import cv2
import numpy as np
import time
import datetime
import winsound 
from camera import Camera
from hand_tracker import HandTracker
from canvas import Canvas
from toolbar import Toolbar
from ai.math_solver import MathAI 
from ai.vision_ai import UltimateAI

def run_canvas(brush_size=25, smoothening=5):
    cam = Camera(width=1280, height=720)
    tracker = HandTracker(max_hands=1, detection_con=0.7)
    canvas = Canvas(width=1280, height=720)
    toolbar = Toolbar()
    
    math_ai = MathAI()
    sketch_ai = UltimateAI() 

    draw_color = (255, 0, 255) 
    
    xp, yp = 0, 0
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    last_save_time = 0 
    
    ai_answer = ""

    print("Phase 4 (Completed). Sketch Classifier loaded!")

    while True:
        success, img = cam.get_frame()
        if not success: break

        img = tracker.find_hands(img)
        lm_list = tracker.find_position(img)
        img = toolbar.draw(img)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1], lm_list[8][2]   
            x2, y2 = lm_list[12][1], lm_list[12][2] 
            
            fingers = tracker.fingers_up()
            thumb_index_dist, _ = tracker.find_distance(4, 8)

            # CLEAR CANVAS
            if thumb_index_dist < 40 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                canvas.clear()
                ai_answer = "" 
                cv2.putText(img, "CLEARED!", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                
            # SAVE IMAGE
            elif fingers == [1, 0, 0, 0, 0]:
                if time.time() - last_save_time > 3: 
                    filename = f"masterpiece_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                    cv2.imwrite(filename, canvas.img_canvas)
                    last_save_time = time.time()
                    flash = np.ones(img.shape, dtype=np.uint8) * 255
                    cv2.imshow("VisionCanvasAI", flash)
                    cv2.waitKey(50) 

            # SELECTION MODE
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                xp, yp = 0, 0 
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 20, draw_color, cv2.FILLED)
                
                interaction_type, value = toolbar.check_interaction(cx, cy)
                
                if interaction_type == 'color':
                    if draw_color != value: winsound.Beep(800, 100) 
                    draw_color = value
                    
                # TRIGGER MATH SOLVER
                elif interaction_type == 'action' and value == 'Math':
                    cv2.putText(img, "SOLVING MATH...", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                    cv2.imshow("VisionCanvasAI", img)
                    cv2.waitKey(1) 
                    ai_answer = math_ai.solve_canvas(canvas.img_canvas)

                # TRIGGER SKETCH GUESSER
                elif interaction_type == 'action' and value == 'Guess':
                    cv2.putText(img, "GUESSING SKETCH...", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                    cv2.imshow("VisionCanvasAI", img)
                    cv2.waitKey(1) 
                    ai_answer = sketch_ai.guess_canvas(canvas.img_canvas)

            # DRAWING MODE
            elif fingers[1] == 1 and fingers[2] == 0: 
                if xp == 0 and yp == 0:
                    curr_x, curr_y = x1, y1
                    prev_x, prev_y = x1, y1
                else:
                    curr_x = prev_x + (x1 - prev_x) / smoothening
                    curr_y = prev_y + (y1 - prev_y) / smoothening
                x_draw, y_draw = int(curr_x), int(curr_y)
                cv2.circle(img, (x_draw, y_draw), brush_size, draw_color, cv2.FILLED)
                xp, yp = canvas.draw_line((xp, yp), (x_draw, y_draw), draw_color, brush_size)
                prev_x, prev_y = curr_x, curr_y
            else:
                xp, yp = 0, 0
                prev_x, prev_y = x1, y1 

        img = canvas.merge_with_frame(img)

        if ai_answer:
            cv2.putText(img, f"{ai_answer}", (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        cv2.imshow("VisionCanvasAI", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
      
if __name__ == "__main__":
    run_canvas()