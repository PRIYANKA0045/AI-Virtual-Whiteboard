import customtkinter as ctk
from main import run_canvas

# Configure the visual theme
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  

class VisionCanvasUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Vision Canvas AI - Dashboard")
        self.geometry("450x400")
        self.resizable(False, False)
        
        # --- TITLE ---
        self.title_label = ctk.CTkLabel(self, text="Vision Canvas AI", font=ctk.CTkFont(size=28, weight="bold"))
        self.title_label.pack(pady=(30, 10))
        
        self.subtitle = ctk.CTkLabel(self, text="Configure your virtual workspace", text_color="gray")
        self.subtitle.pack(pady=(0, 20))
        
        # --- BRUSH SIZE SLIDER ---
        self.brush_label = ctk.CTkLabel(self, text="Brush Size: 25px", font=ctk.CTkFont(size=14))
        self.brush_label.pack(pady=(10, 0))
        
        self.brush_slider = ctk.CTkSlider(self, from_=5, to=60, command=self.update_brush_text)
        self.brush_slider.set(25)
        self.brush_slider.pack(pady=10, padx=50)
        
        # --- SMOOTHING SLIDER ---
        self.smooth_label = ctk.CTkLabel(self, text="Motion Smoothing (DPI): 5", font=ctk.CTkFont(size=14))
        self.smooth_label.pack(pady=(15, 0))
        
        self.smooth_slider = ctk.CTkSlider(self, from_=1, to=20, command=self.update_smooth_text)
        self.smooth_slider.set(5)
        self.smooth_slider.pack(pady=10, padx=50)
        
        # --- LAUNCH BUTTON ---
        self.launch_btn = ctk.CTkButton(
            self, 
            text="LAUNCH CANVAS", 
            font=ctk.CTkFont(size=16, weight="bold"), 
            height=45,
            command=self.launch_app
        )
        self.launch_btn.pack(pady=30)

    def update_brush_text(self, value):
        self.brush_label.configure(text=f"Brush Size: {int(value)}px")
        
    def update_smooth_text(self, value):
        self.smooth_label.configure(text=f"Motion Smoothing (DPI): {int(value)}")
        
    def launch_app(self):
        # Get the current slider values
        b_size = int(self.brush_slider.get())
        smooth = int(self.smooth_slider.get())
        
        # Hide the settings menu temporarily
        self.withdraw() 
        
        # Boot up the OpenCV Camera with our custom settings!
        print(f"Launching with Brush={b_size}, Smoothing={smooth}...")
        run_canvas(brush_size=b_size, smoothening=smooth)
        
        # When you press 'q' to quit the canvas, the menu comes back
        self.deiconify() 

if __name__ == "__main__":
    app = VisionCanvasUI()
    app.mainloop()