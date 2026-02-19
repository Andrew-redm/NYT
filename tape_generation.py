import cv2
import numpy as np
import json
import os
import pytesseract
from enum import Enum
from collections import defaultdict
import shutil

# --- CONFIGURATION ---
TESSERACT_PATH = r'C:\Users\andre\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ OCR Engine found at: {TESSERACT_PATH}")
else:
    print(f"⚠️ WARNING: Tesseract not found at {TESSERACT_PATH}")

# Create a debug folder to see exactly what Tesseract sees
DEBUG_FOLDER = "debug_crops"
if os.path.exists(DEBUG_FOLDER):
    shutil.rmtree(DEBUG_FOLDER)
os.makedirs(DEBUG_FOLDER)

# NYT Interface Colors (HSV)
YELLOW_LOWER = np.array([20, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])
BLUE_LOWER = np.array([100, 80, 80])
BLUE_UPPER = np.array([130, 255, 255])

class GameState(Enum):
    WAITING = 0
    CALIBRATING = 1
    RECORDING = 2
    FINISHED = 3

class TapeRecorder:
    def __init__(self):
        self.state = GameState.WAITING
        self.events = []
        self.grid_cells = [] 
        self.grid_state = {} 
        self.last_cursor_pos = None
        self.stability_buffer = defaultdict(lambda: ("?", 0))
        self.grid_bounds = None 

    def find_grid_dynamically(self, frame):
        """
        Robust Grid Finder: Looks for the main game area by thresholding 
        everything that isn't the black background.
        """
        h, w = frame.shape[:2]
        
        # 1. Search Area: Top 60% of screen, with a margin on sides
        # This avoids the status bar (top) and keyboard (bottom)
        margin = int(w * 0.05)
        roi = frame[int(h*0.10):int(h*0.60), margin:w-margin]
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Threshold: The cells are gray/blue/purple (brightness > 40). 
        # The background is black (brightness < 20).
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological Close to fuse the cells into one big block
        kernel = np.ones((7,7), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 5. Find the largest contour (The Grid)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return None
        
        # Pick largest area
        c = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        
        # Sanity Check: Grid should be roughly square
        aspect = cw / float(ch)
        if cw < w * 0.5 or not (0.8 < aspect < 1.2):
            return None

        # Adjust coordinates back to full frame
        final_x = x + margin
        final_y = y + int(h*0.10)
        
        return (final_x, final_y, cw, ch)

    def slice_grid(self, bounds):
        gx, gy, gw, gh = bounds
        cell_w = gw / 5.0
        cell_h = gh / 5.0
        
        cells = []
        for row in range(5):
            for col in range(5):
                cx = int(gx + (col * cell_w))
                cy = int(gy + (row * cell_h))
                cw = int(cell_w)
                ch = int(cell_h)
                cells.append((cx, cy, cw, ch))
        return cells

    def read_letter(self, roi, cell_id):
        if roi.size == 0: return "?"
        try:
            # 1. UPSCALING (Critical for small text)
            # Make it 3x bigger before processing
            roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 2. HIGH CONTRAST THRESHOLDING
            # We want to separate the WHITE text (~200-255) from the GRAY cell (~80-150)
            # A threshold of 160 cuts out the cell background completely.
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            
            # 3. DENSITY CHECK (Is there actually text?)
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = roi.shape[0] * roi.shape[1]
            if (white_pixels / total_pixels) < 0.02: # Less than 2% white = empty
                return "?"

            # 4. INVERT (Black Text on White Background)
            thresh_inv = cv2.bitwise_not(thresh)
            
            # 5. PADDING (Give Tesseract a border)
            padded = cv2.copyMakeBorder(thresh_inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            
            # 6. DILATION (Thicken the letters slightly to close gaps in 'O')
            kernel = np.ones((2,2), np.uint8)
            bolded = cv2.erode(padded, kernel, iterations=1) # Erode because it's black text on white
            
            # DEBUG: Save this image so we can see it!
            filename = f"{DEBUG_FOLDER}/cell_{cell_id[0]}_{cell_id[1]}.png"
            cv2.imwrite(filename, bolded)
            
            config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            text = pytesseract.image_to_string(bolded, config=config)
            clean = text.strip().upper()
            
            if len(clean) == 1 and clean.isalpha():
                return clean
            return "?"
        except Exception as e:
            print(e)
            return "?"

    def detect_cell_state(self, cell_roi):
        if cell_roi.size == 0: return {"active_score": 0}
        hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        mask_b = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        active_score = max(cv2.countNonZero(mask_y), cv2.countNonZero(mask_b))
        return {"active_score": active_score}

    def process_video(self, video_path, output_json="solve_tape.json"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        print(f"📼 Processing {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_time = frame_idx / fps
            frame_idx += 1
            if frame_idx % 5 != 0: continue
            
            debug_frame = frame.copy()

            if self.state == GameState.WAITING:
                bounds = self.find_grid_dynamically(frame)
                if bounds:
                    bx, by, bw, bh = bounds
                    cv2.rectangle(debug_frame, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)
                    print(f"[{current_time:.2f}s] 📐 Grid Found: {bounds}")
                    self.grid_bounds = bounds
                    self.state = GameState.CALIBRATING

            elif self.state == GameState.CALIBRATING:
                self.grid_cells = self.slice_grid(self.grid_bounds)
                for i in range(25):
                    self.grid_state[tuple(list(divmod(i, 5)))] = "?"
                print(f"[{current_time:.2f}s] ✅ Recording started.")
                self.state = GameState.RECORDING

            elif self.state == GameState.RECORDING:
                if self.find_grid_dynamically(frame) is None:
                    # Optional: Add logic to detect end of game
                    pass

                frame_cursor_candidates = []
                frame_events = []

                for i, (x, y, w, h) in enumerate(self.grid_cells):
                    cell_id = list(divmod(i, 5))
                    
                    # Blue Grid for Debug
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

                    # 1. Cursor Check
                    state = self.detect_cell_state(frame[y:y+h, x:x+w])
                    if state['active_score'] > 50:
                        frame_cursor_candidates.append((state['active_score'], cell_id))
                    
                    # 2. OCR Crop (Center 60%)
                    crop_padding = 0.20
                    crop_x = x + int(w * crop_padding)
                    crop_y = y + int(h * crop_padding)
                    crop_w = int(w * (1 - crop_padding*2))
                    crop_h = int(h * (1 - crop_padding*2))
                    ocr_roi = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                    # Red Box for Debug
                    cv2.rectangle(debug_frame, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 1)

                    # 3. Read & Debounce
                    raw_val = self.read_letter(ocr_roi, cell_id)
                    candidate, count = self.stability_buffer[i]
                    
                    confirmed_val = None
                    if raw_val == candidate:
                        count += 1
                        self.stability_buffer[i] = (candidate, count)
                    else:
                        self.stability_buffer[i] = (raw_val, 1)
                        count = 1

                    if count >= 2: confirmed_val = raw_val

                    # 4. Event Logic
                    if confirmed_val is not None:
                        cell_key = tuple(cell_id)
                        prev_val = self.grid_state.get(cell_key, "?")
                        
                        if confirmed_val != "?" and confirmed_val != prev_val:
                            print(f"[{current_time:.2f}s] ✍️ Entry: {confirmed_val} at {cell_id}")
                            frame_events.append({"timestamp": round(current_time, 2), "type": "entry", "cell": cell_id, "value": confirmed_val})
                            self.grid_state[cell_key] = confirmed_val
                        
                        elif confirmed_val == "?" and prev_val != "?":
                            print(f"[{current_time:.2f}s] 🗑️ Deletion at {cell_id}")
                            frame_events.append({"timestamp": round(current_time, 2), "type": "deletion", "cell": cell_id, "value": None})
                            self.grid_state[cell_key] = "?"

                self.events.extend(frame_events)
                
                if frame_cursor_candidates:
                    frame_cursor_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_cell = frame_cursor_candidates[0][1]
                    if self.last_cursor_pos != best_cell:
                        self.events.append({"timestamp": round(current_time, 2), "type": "selection", "cell": best_cell})
                        self.last_cursor_pos = best_cell

                cv2.imshow("Debug: Grid", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
        with open(output_json, "w") as f:
            json.dump({"metadata": {"source": video_path}, "events": self.events}, f, indent=2)
        print(f"💾 Success! {len(self.events)} events saved to {output_json}")

if __name__ == "__main__":
    VIDEO_FILE = "mysolve.mp4" 
    if os.path.exists(VIDEO_FILE):
        TapeRecorder().process_video(VIDEO_FILE)
    else:
        print(f"File '{VIDEO_FILE}' not found.")