import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(r"C:\Users\Naing\Desktop\Git_project\AI_2026\Day13 CNN YOLO\runs\detect\train\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(":x: Cannot open camera")
    exit()

cv2.namedWindow("YOLO Webcam", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print(":x: Can't receive frame")
        break

    frame = cv2.flip(frame, 1)

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()