import os
os.environ["YOLO_AUTOINSTALL"] = "false"

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from ultralytics import YOLO
import time
import psutil
from collections import defaultdict, deque
from pathlib import Path
import torch

import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

MODEL_PATH = "./best.onnx"  
VIDEO_PATH = "./Inference1.mp4" 


CUDA_AVAILABLE = torch.cuda.is_available()


def check_onnx_gpu():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return 'CUDAExecutionProvider' in providers
    except ImportError:
        return False

ONNX_GPU_AVAILABLE = check_onnx_gpu()

USE_GPU = CUDA_AVAILABLE and ONNX_GPU_AVAILABLE
DEVICE = 'cuda:0' if USE_GPU else 'cpu'

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class TrafficMonitorLogic:
    def __init__(self):
        self.model = None
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = None
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype="uint8")
        self.device = DEVICE

    def load_model(self, path):
        if not Path(path).exists():
            return False, f"Model file not found at {path}"
        try:
            self.model = YOLO(path, task='detect')
            
            if USE_GPU:
                device_info = f"GPU ({torch.cuda.get_device_name(0)})"
            elif CUDA_AVAILABLE and not ONNX_GPU_AVAILABLE:
                device_info = "CPU (Install onnxruntime-gpu for GPU support)"
            else:
                device_info = "CPU"
            
            return True, f"Model Loaded ✓ | Device: {device_info}"
        except Exception as e:
            return False, str(e)

    def process_frame(self, frame, conf_threshold=0.5):
        current_time = time.time()
        if self.last_frame_time:
            fps = 1.0 / (current_time - self.last_frame_time + 1e-6)
            self.fps_history.append(fps)
        self.last_frame_time = current_time
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0

        results = self.model.track(
            frame, 
            persist=True, 
            conf=conf_threshold, 
            tracker="bytetrack.yaml", 
            verbose=False,
            device=self.device  
        )

        annotated_frame = frame.copy()
        detections = []
        class_counts = defaultdict(int)

        for r in results:
            if r.boxes is None or r.boxes.id is None: 
                continue
                
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(box.id[0].cpu().numpy())
                class_name = self.model.names[cls]
                class_counts[class_name] += 1
                color = tuple(map(int, self.colors[cls % 80]))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 6)
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - text_height - baseline - 10), 
                             (x1 + text_width + 10, y1), 
                             color, -1)
                
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - baseline - 5), 
                            font, font_scale, (255, 255, 255), thickness)

                detections.append({"id": track_id, "class": class_name, "conf": conf})

        self.frame_count += 1
        return annotated_frame, detections, avg_fps, dict(class_counts)

class TrafficApp:
    def __init__(self):
        self.logic = TrafficMonitorLogic()
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.conf_threshold = 0.4
        self.fps_data_x = []
        self.fps_data_y = []
        self.video_aspect_ratio = 16 / 9  
        self.current_tex_width = 0
        self.current_tex_height = 0
        self.latest_frame = None
        self.texture_counter = 0
        self.current_texture_tag = None
        
        dpg.create_context()
        self._setup_gui()
        
        success, msg = self.logic.load_model(MODEL_PATH)
        dpg.set_value("status_text", msg)
        if success:
            dpg.configure_item("status_text", color=(0, 255, 0))

    def _get_system_metrics(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        gpu_load = 0
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus: gpu_load = gpus[0].load * 100
            except: pass
        return cpu, ram.percent, ram.used / (1024**3), gpu_load

    def _toggle_start(self):
        if self.is_running:
            self.is_running = False
            if self.cap: 
                self.cap.release()
            dpg.configure_item("btn_start", label="Start Monitoring")
            dpg.set_value("status_text", "Monitoring Stopped")
            dpg.configure_item("status_text", color=(255, 150, 0))
            self.latest_frame = None
        else:
            self.cap = cv2.VideoCapture(VIDEO_PATH)
            if not self.cap.isOpened():
                dpg.set_value("status_text", f"Error: Cannot open {VIDEO_PATH}")
                dpg.configure_item("status_text", color=(255, 0, 0))
                return
            
            video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if video_height > 0:
                self.video_aspect_ratio = video_width / video_height
            
            self.is_running = True
            dpg.configure_item("btn_start", label="Stop Monitoring")
            dpg.set_value("status_text", "Monitoring Active (ByteTrack)...")
            dpg.configure_item("status_text", color=(0, 255, 0))

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        label = "Resume" if self.is_paused else "Pause"
        status = "Paused" if self.is_paused else "Monitoring Active..."
        color = (255, 255, 0) if self.is_paused else (0, 255, 0)
        
        dpg.configure_item("btn_pause", label=label)
        dpg.set_value("status_text", status)
        dpg.configure_item("status_text", color=color)

    def _calculate_display_size(self, container_width, container_height):
        """Calculate the display size maintaining aspect ratio."""
        if container_width <= 0 or container_height <= 0:
            return 640, 360 
        
        container_aspect = container_width / container_height
        
        if self.video_aspect_ratio > container_aspect:
            display_width = int(container_width)
            display_height = int(container_width / self.video_aspect_ratio)
        else:
            display_height = int(container_height)
            display_width = int(container_height * self.video_aspect_ratio)

        display_width = max(display_width, 16)
        display_height = max(display_height, 16)

        return display_width, display_height

    def _create_or_update_texture(self, width, height):
        """Create or recreate texture if size changed."""
        if width == self.current_tex_width and height == self.current_tex_height:
            return False 
        if self.current_texture_tag and dpg.does_item_exist(self.current_texture_tag):
            dpg.delete_item(self.current_texture_tag)
        
        self.texture_counter += 1
        self.current_texture_tag = f"video_texture_{self.texture_counter}"
        
        texture_data = np.zeros((height, width, 3), dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=width, 
                height=height,
                default_value=texture_data.flatten(),
                format=dpg.mvFormat_Float_rgb,
                tag=self.current_texture_tag
            )
        
        self.current_tex_width = width
        self.current_tex_height = height
        return True  

    def _setup_gui(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 15, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (20, 25, 30))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 80, 180))
        
        dpg.bind_theme(global_theme)
        dpg.create_viewport(title='Live Traffic Monitoring - ByteTrack', width=1920, height=1080)

        with dpg.window(label="Traffic Monitor", tag="main_window", no_close=True, no_collapse=True, no_title_bar=True):
            # Header
            with dpg.child_window(height=80, border=True):
                dpg.add_text("TRAFFIC MONITORING SYSTEM (BYTETRACK)", tag="header_title")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_text("📁 Model:", color=(100, 200, 255)); dpg.add_text(MODEL_PATH, color=(150, 150, 150))
                with dpg.group(horizontal=True):
                    dpg.add_text("🎥 Video:", color=(100, 200, 255)); dpg.add_text(VIDEO_PATH, color=(150, 150, 150))
            
            with dpg.group(horizontal=True):
                # Left Panel
                with dpg.child_window(width=-360, border=True, tag="left_panel"):
                    dpg.add_text("LIVE VIDEO FEED", color=(0, 200, 255))
                    dpg.add_separator()
                    dpg.add_drawlist(tag="video_drawlist", width=800, height=450)
                    
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Start Monitoring", tag="btn_start", callback=self._toggle_start, width=180, height=50)
                        dpg.add_button(label="Pause", tag="btn_pause", callback=self._toggle_pause, width=120, height=50)
                        with dpg.group():
                            dpg.add_text("🎚️ Confidence")
                            dpg.add_slider_float(tag="conf_slider", default_value=0.4, min_value=0.1, max_value=1.0, width=200,
                                                 callback=lambda s, a: setattr(self, 'conf_threshold', a))
                    dpg.add_text("Status:", color=(100, 200, 255), indent=5)
                    dpg.add_text("Ready", tag="status_text", color=(0, 255, 0))

                with dpg.child_window(width=360, border=True):
                    dpg.add_text("ANALYTICS DASHBOARD", color=(0, 200, 255))
                    dpg.add_separator()
                    
                    with dpg.child_window(height=160, border=True):
                        dpg.add_text("System Performance", color=(255, 200, 0))
                        with dpg.group(horizontal=True):
                            dpg.add_text("FPS:"); dpg.add_text("0.0", tag="metric_fps", color=(0, 255, 0))
                        dpg.add_progress_bar(tag="prog_cpu", width=-1, overlay="CPU: 0%")
                        dpg.add_progress_bar(tag="prog_ram", width=-1, overlay="RAM: 0%")
                        dpg.add_progress_bar(tag="prog_gpu", width=-1, overlay="GPU: 0%")
                    
                    with dpg.child_window(height=180, border=True):
                        dpg.add_text("FPS Monitor", color=(255, 200, 0))
                        with dpg.plot(height=140, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis")
                            dpg.add_plot_axis(dpg.mvYAxis, label="FPS", tag="y_axis")
                            dpg.add_line_series([], [], label="FPS", parent="y_axis", tag="fps_series")
                    
                    with dpg.child_window(height=-1, border=True):
                        dpg.add_text("Object Counts", color=(255, 200, 0))
                        dpg.add_separator()
                        dpg.add_text("No detections", tag="txt_counts", color=(150, 150, 150))
        
        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        dpg.maximize_viewport()
        dpg.show_viewport()

    def _update_video_texture(self, frame):
        """Update video display with proper sizing and centering."""
        self.latest_frame = frame

        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        container_width = viewport_width - 360 - 50
        container_height = viewport_height - 80 - 120 - 70
        container_width = max(container_width, 320)
        container_height = max(container_height, 180)

        dpg.configure_item("video_drawlist", width=int(container_width), height=int(container_height))
        display_width, display_height = self._calculate_display_size(container_width, container_height)
        self._create_or_update_texture(display_width, display_height)
        resized = cv2.resize(frame, (display_width, display_height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dpg.set_value(self.current_texture_tag, rgb_frame.flatten())
        offset_x = (container_width - display_width) / 2
        offset_y = (container_height - display_height) / 2
        dpg.delete_item("video_drawlist", children_only=True)
        dpg.draw_image(
            self.current_texture_tag,
            pmin=(offset_x, offset_y),
            pmax=(offset_x + display_width, offset_y + display_height),
            parent="video_drawlist"
        )

    def run(self):
        while dpg.is_dearpygui_running():
            if self.is_running and not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    self._toggle_start()
                else:
                    annotated, detections, fps, counts = self.logic.process_frame(frame, self.conf_threshold)
                    self._update_video_texture(annotated)
                    
                    if self.logic.frame_count % 3 == 0:
                        cpu, ram_per, ram_gb, gpu = self._get_system_metrics()
                        dpg.set_value("metric_fps", f"{fps:.1f}")
                        dpg.set_value("prog_cpu", cpu/100); dpg.configure_item("prog_cpu", overlay=f"CPU: {cpu:.1f}%")
                        dpg.set_value("prog_ram", ram_per/100); dpg.configure_item("prog_ram", overlay=f"RAM: {ram_gb:.1f} GB")
                        dpg.set_value("prog_gpu", gpu/100); dpg.configure_item("prog_gpu", overlay=f"GPU: {gpu:.1f}%")
                        
                        if counts:
                            count_str = "\n".join([f"{k}: {v}" for k, v in counts.items()])
                            dpg.set_value("txt_counts", count_str)
                            dpg.configure_item("txt_counts", color=(255, 255, 0))
                        
                        self.fps_data_x.append(self.logic.frame_count)
                        self.fps_data_y.append(fps)
                        if len(self.fps_data_x) > 100:
                            self.fps_data_x.pop(0); self.fps_data_y.pop(0)
                        dpg.set_value("fps_series", [self.fps_data_x, self.fps_data_y])
                        dpg.fit_axis_data("x_axis"); dpg.fit_axis_data("y_axis")

            dpg.render_dearpygui_frame()
        
        if self.cap: self.cap.release()
        dpg.destroy_context()

if __name__ == "__main__":
    app = TrafficApp()
    app.run()