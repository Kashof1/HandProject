import cv2 as cv
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    capture = cv.VideoCapture(0)
    width  = capture.get(3) 
    height = capture.get(4)
    while capture.isOpened():
        val, frame = capture.read()

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for count, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                print(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x)
                x = float(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x)
                y = float(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y)
                xwidth = x * width
                xwidth = round(xwidth)
                ywidth = y * height
                ywidth = round(ywidth)
                print(xwidth, ywidth)
                cv.putText(image, 
                            (f"X: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x}, Y: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y}, Z: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].z}"), 
                            (xwidth, ywidth),
                            cv.FONT_HERSHEY_SIMPLEX, 
                            0.5,
                            (255, 255, 255),
                            2)

        cv.imshow('Window', image)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


capture.release()
cv.destroyAllWindows()
