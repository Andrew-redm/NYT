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
    print(f"OCR Engine found at: {TESSERACT_PATH}")
else:
    print(f"WARNING: Tesseract not found at {TESSERACT_PATH}")

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

    def build_perfect_grid(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        cell_rects = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect = cw / float(ch)
                
                # Filter for squares that are roughly 1/5th of the screen width
                if 0.85 < aspect < 1.15 and (w * 0.12) < cw < (w * 0.25):
                    if y > h * 0.10 and y < h * 0.65:
                        cell_rects.append((x, y, cw, ch))
                        
        if len(cell_rects) < 10:
            return None 
            
        min_x = min([r[0] for r in cell_rects])
        min_y = min([r[1] for r in cell_rects])
        max_x = max([r[0] + r[2] for r in cell_rects])
        
        # FIX: Force the grid to be a perfect square to avoid vertical stretching
        total_w = max_x - min_x
        total_h = total_w 
        
        step_x = total_w / 5.0
        step_y = total_h / 5.0
        
        cells = []
        for row in range(5):
            for col in range(5):
                cx = int(min_x + (col * step_x))
                cy = int(min_y + (row * step_y))
                cells.append((cx, cy, int(step_x), int(step_y)))
                
        return cells

    def read_letter(self, roi, cell_id):
        if roi.size == 0: return "?"
        try:
            roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = roi.shape[0] * roi.shape[1]
            if (white_pixels / total_pixels) < 0.02: 
                return "?"

            thresh_inv = cv2.bitwise_not(thresh)
            padded = cv2.copyMakeBorder(thresh_inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            
            kernel = np.ones((2,2), np.uint8)
            bolded = cv2.erode(padded, kernel, iterations=1) 
            
            filename = f"{DEBUG_FOLDER}/cell_{cell_id[0]}_{cell_id[1]}.png"
            cv2.imwrite(filename, bolded)
            
            config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            text = pytesseract.image_to_string(bolded, config=config)
            clean = text.strip().upper()
            
            if len(clean) == 1 and clean.isalpha():
                return clean
            return "?"
        except Exception as e:
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
        print(f"Processing {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_time = frame_idx / fps
            frame_idx += 1
            if frame_idx % 5 != 0: continue
            
            debug_frame = frame.copy()

            if self.state == GameState.WAITING:
                cells = self.build_perfect_grid(frame)
                if cells:
                    print(f"[{current_time:.2f}s] Grid Locked.")
                    self.grid_cells = cells
                    self.state = GameState.CALIBRATING

            elif self.state == GameState.CALIBRATING:
                for i in range(25):
                    self.grid_state[tuple(list(divmod(i, 5)))] = "?"
                print(f"[{current_time:.2f}s] Recording started.")
                self.state = GameState.RECORDING

            elif self.state == GameState.RECORDING:
                frame_cursor_candidates = []
                frame_events = []
                last_ocr_crop = None

                for i, (x, y, w, h) in enumerate(self.grid_cells):
                    cell_id = list(divmod(i, 5))
                    
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

                    state = self.detect_cell_state(frame[y:y+h, x:x+w])
                    if state['active_score'] > 50:
                        frame_cursor_candidates.append((state['active_score'], cell_id))
                    
                    crop_padding = 0.20
                    crop_x = x + int(w * crop_padding)
                    crop_y = y + int(h * crop_padding)
                    crop_w = int(w * (1 - crop_padding*2))
                    crop_h = int(h * (1 - crop_padding*2))
                    ocr_roi = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                    cv2.rectangle(debug_frame, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 1)
                    if i == 12: last_ocr_crop = ocr_roi

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

                    if confirmed_val is not None:
                        cell_key = tuple(cell_id)
                        prev_val = self.grid_state.get(cell_key, "?")
                        
                        if confirmed_val != "?" and confirmed_val != prev_val:
                            print(f"[{current_time:.2f}s] Entry: {confirmed_val} at {cell_id}")
                            frame_events.append({"timestamp": round(current_time, 2), "type": "entry", "cell": cell_id, "value": confirmed_val})
                            self.grid_state[cell_key] = confirmed_val
                        
                        elif confirmed_val == "?" and prev_val != "?":
                            print(f"[{current_time:.2f}s] Deletion at {cell_id}")
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
                if last_ocr_crop is not None and last_ocr_crop.size > 0:
                     debug_eye = cv2.resize(last_ocr_crop, (200, 200), interpolation=cv2.INTER_NEAREST)
                     cv2.imshow("Debug: OCR Eye", debug_eye)

                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
        with open(output_json, "w") as f:
            json.dump({"metadata": {"source": video_path}, "events": self.events}, f, indent=2)
        print(f"Success. {len(self.events)} events saved to {output_json}")

if __name__ == "__main__":
    VIDEO_FILE = "mysolve.mp4" 
    if os.path.exists(VIDEO_FILE):
        TapeRecorder().process_video(VIDEO_FILE)
    else:
        print(f"File '{VIDEO_FILE}' not found.")