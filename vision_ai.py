import os
import cv2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

class UltimateAI:
    def __init__(self):
        print("Connecting to Cloud Vision AI...")
        
        # Load environment variables from the .env file
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            
            # --- NEW: AUTO-DETECT VALID MODEL ---
            print("Searching for the latest active AI model...")
            self.model_name = None
            
            try:
                # 1. Ask Google's servers for every active model
                available_models = list(genai.list_models())
                
                # 2. Filter for models that specifically allow "generateContent" (Vision/Text)
                valid_models = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods]
                
                # 3. Try to pick a fast "flash" model first
                for name in valid_models:
                    if 'flash' in name:
                        self.model_name = name
                        break
                        
                # 4. If no flash model is found, just grab the very first valid one
                if not self.model_name and valid_models:
                    self.model_name = valid_models[0]
                    
                if self.model_name:
                    print(f"Success! Auto-selected: {self.model_name}")
                    self.model = genai.GenerativeModel(self.model_name)
                else:
                    print("Error: No compatible models found on your Google Cloud account.")
                    self.model = None
                    
            except Exception as e:
                print(f"Error fetching model list: {e}")
                self.model = None
        else:
            print("WARNING: GEMINI_API_KEY not found in .env file!")
            self.model = None

    def guess_canvas(self, canvas_img):
        if not self.model:
            return "ERROR: AI connection failed!"
            
        # Convert the OpenCV BGR image format to Standard RGB for Pillow
        rgb_img = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)
        
        try:
            # Send the image to the auto-selected model
            prompt = "You are playing Pictionary. Look at this white sketch on a black background. Guess what the object is in 1 to 3 words. Do not use punctuation."
            response = self.model.generate_content([prompt, pil_image])
            
            guess = response.text.strip().capitalize()
            return f"I see: {guess}"
        except Exception as e:
            print(f"AI Error: {e}")
            return "Connection error..."