import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from collections import deque

class AdvancedHandMouse:
    def __init__(self):
        # Initialize hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Mouse control parameters
        self.smoothing = 5
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.frame_reduction = 100
        
        # Gesture recognition
        self.gesture_history = deque(maxlen=5)
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.3
        
        # States for advanced features
        self.is_dragging = False
        self.last_click_time = 0
        self.click_cooldown = 0.5
        
        # Zoom state
        self.pinch_start_dist = 0
        self.zoom_sensitivity = 20
        self.zoom_active = False
        
        # Swipe detection
        self.swipe_threshold = 150
        
        # Fingers states
        self.finger_states = {
            'thumb': False, 'index': False,
            'middle': False, 'ring': False,
            'pinky': False
        }

    def detect_finger_states(self, landmarks):
        """Detect which fingers are extended"""
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        
        for i, (tip, pip) in enumerate(zip(tips, pips)):
            finger_name = list(self.finger_states.keys())[i]
            self.finger_states[finger_name] = (landmarks[tip].y < landmarks[pip].y)

    def recognize_gesture(self, landmarks):
        """Recognize hand gestures based on finger positions"""
        self.detect_finger_states(landmarks)
        
        thumb = self.finger_states['thumb']
        index = self.finger_states['index']
        middle = self.finger_states['middle']
        ring = self.finger_states['ring']
        pinky = self.finger_states['pinky']
        
        # Check gestures in priority order
        if index and not middle and not ring and not pinky:
            return "POINT"
        elif self.get_distance(landmarks[4], landmarks[8]) < 0.05:
            current_time = time.time()
            if current_time - self.last_click_time < 0.3:
                return "DOUBLE_CLICK"
            self.last_click_time = current_time
            return "LEFT_CLICK"
        elif self.get_distance(landmarks[4], landmarks[12]) < 0.07:
            return "RIGHT_CLICK"
        elif thumb and pinky and not index and not middle and not ring:
            return "SCROLL"
        elif not index and not middle and not ring and not pinky and thumb:
            return "DRAG"
        elif self.get_distance(landmarks[4], landmarks[16]) < 0.07:
            return "CLOSE_TAB"
        elif self.get_distance(landmarks[4], landmarks[8]) < 0.1:
            return "ZOOM"
        elif index and middle and ring and pinky:
            return "SWIPE"
        else:
            return "UNKNOWN"
    
    def get_distance(self, p1, p2):
        """Calculate distance between two landmarks"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def control_mouse(self, landmarks, frame):
        """Move mouse cursor based on index finger position"""
        ix, iy = int(landmarks[8].x * frame.shape[1]), int(landmarks[8].y * frame.shape[0])
        
        # Convert to screen coordinates
        screen_x = np.interp(ix, 
                           (self.frame_reduction, frame.shape[1] - self.frame_reduction),
                           (0, self.screen_w))
        screen_y = np.interp(iy,
                           (self.frame_reduction, frame.shape[0] - self.frame_reduction),
                           (0, self.screen_h))
        
        # Smooth cursor movement
        self.clocX = self.plocX + (screen_x - self.plocX) / self.smoothing
        self.clocY = self.plocY + (screen_y - self.plocY) / self.smoothing
        
        pyautogui.moveTo(self.clocX, self.clocY)
        self.plocX, self.plocY = self.clocX, self.clocY
        
        # Draw cursor marker
        cv2.circle(frame, (ix, iy), 10, (255, 0, 255), cv2.FILLED)

    def execute_gesture_actions(self, gesture, landmarks, frame):
        """Execute actions based on recognized gestures"""
        current_time = time.time()
        
        if gesture == "LEFT_CLICK" and current_time - self.last_click_time > self.click_cooldown:
            pyautogui.click()
            cv2.putText(frame, "LEFT CLICK", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.last_click_time = current_time
        
        elif gesture == "RIGHT_CLICK" and current_time - self.last_click_time > self.click_cooldown:
            pyautogui.rightClick()
            cv2.putText(frame, "RIGHT CLICK", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.last_click_time = current_time
        
        elif gesture == "DOUBLE_CLICK":
            pyautogui.doubleClick()
            cv2.putText(frame, "DOUBLE CLICK", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif gesture == "DRAG":
            if not self.is_dragging:
                pyautogui.mouseDown()
                cv2.putText(frame, "DRAGGING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.is_dragging = True
        elif self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
        
        if gesture == "CLOSE_TAB" and current_time - self.last_click_time > self.click_cooldown:
            pyautogui.hotkey('ctrl', 'w')
            cv2.putText(frame, "CLOSE TAB", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.last_click_time = current_time
        
        elif gesture == "ZOOM":
            current_dist = self.get_distance(landmarks[4], landmarks[8])
            
            if not self.zoom_active:
                self.pinch_start_dist = current_dist
                self.zoom_active = True
            
            zoom_diff = current_dist - self.pinch_start_dist
            
            if abs(zoom_diff) > 0.02:
                pyautogui.keyDown('ctrl')
                pyautogui.scroll(int(zoom_diff * self.zoom_sensitivity))
                pyautogui.keyUp('ctrl')
                
                if zoom_diff > 0:
                    cv2.putText(frame, "ZOOM IN", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "ZOOM OUT", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            self.zoom_active = False
        
        if gesture == "SCROLL":
            middle_y = landmarks[12].y * frame.shape[0]
            
            if middle_y < frame.shape[0] / 3:
                pyautogui.scroll(10)
                cv2.putText(frame, "SCROLL UP", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif middle_y > 2 * frame.shape[0] / 3:
                pyautogui.scroll(-10)
                cv2.putText(frame, "SCROLL DOWN", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif gesture == "SWIPE":
            palm_x = landmarks[0].x * frame.shape[1]
            
            if not hasattr(self, 'swipe_start_x'):
                self.swipe_start_x = palm_x
            
            swipe_diff = palm_x - self.swipe_start_x
            
            if abs(swipe_diff) > self.swipe_threshold:
                if swipe_diff > 0:
                    pyautogui.hotkey('ctrl', 'tab')
                    cv2.putText(frame, "NEXT TAB", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    pyautogui.hotkey('ctrl', 'shift', 'tab')
                    cv2.putText(frame, "PREV TAB", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.swipe_start_x = palm_x
        elif hasattr(self, 'swipe_start_x'):
            del self.swipe_start_x

    def process_frame(self, frame):
        """Process each camera frame for hand gesture recognition"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand detection
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Draw hand landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Recognize gesture
            current_time = time.time()
            if current_time - self.last_gesture_time > self.gesture_cooldown:
                gesture = self.recognize_gesture(landmarks)
                self.gesture_history.append(gesture)
                self.last_gesture_time = current_time
            
            # Get most frequent recent gesture
            if self.gesture_history:
                gesture = max(set(self.gesture_history), 
                             key=self.gesture_history.count)
                
                # Control cursor
                self.control_mouse(landmarks, frame)
                
                # Execute gesture actions
                self.execute_gesture_actions(gesture, landmarks, frame)
        
        # Draw detection area
        cv2.rectangle(frame, (self.frame_reduction, self.frame_reduction),
                      (frame.shape[1] - self.frame_reduction, 
                       frame.shape[0] - self.frame_reduction),
                      (255, 0, 0), 2)
        
        return frame

def main():
    # Add failsafe - move mouse to corner to stop
    pyautogui.FAILSAFE = True
    
    hand_mouse = AdvancedHandMouse()
    cap = cv2.VideoCapture(0)
    
    print("Starting Hand Gesture Mouse...")
    print("Place your hand in the blue rectangle to begin control")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame
        output_frame = hand_mouse.process_frame(frame)
        
        # Display
        cv2.imshow('Hand Gesture Mouse', output_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
