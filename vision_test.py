import cv2
import os

VIDEO_FILE = "mysolve.mp4"

def test_vision():
    if not os.path.exists(VIDEO_FILE):
        print(f"File {VIDEO_FILE} not found!")
        return

    cap = cv2.VideoCapture(VIDEO_FILE)
    
    print("Press 'q' to quit. Press 'space' to pause/unpause.")
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret: 
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. ADAPTIVE THRESHOLD (The one we are using)
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. SIMPLE THRESHOLD (An alternative)
        # Allows us to see if a simple cutoff works better
        _, thresh_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Show all 3 views
        # Stack them horizontally for easy comparison
        # Resize to fit screen if needed
        h, w = gray.shape
        scale = 0.5
        gray_small = cv2.resize(gray, (0,0), fx=scale, fy=scale)
        thresh1_small = cv2.resize(thresh_adaptive, (0,0), fx=scale, fy=scale)
        thresh2_small = cv2.resize(thresh_simple, (0,0), fx=scale, fy=scale)
        
        cv2.imshow("Original (Gray)", gray_small)
        cv2.imshow("Adaptive (Current)", thresh1_small)
        cv2.imshow("Simple (Alternative)", thresh2_small)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_vision()