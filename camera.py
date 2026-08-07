import cv2

class Camera:
    def __init__(self, cam_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(3, width)
        self.cap.set(4, height)

    def get_frame(self):
        success, img = self.cap.read()
        if success:
            # Flip the image horizontally for a selfie-view display
            img = cv2.flip(img, 1) 
        return success, img

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()