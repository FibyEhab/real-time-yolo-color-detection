from ultralytics import YOLO
import cv2
import random

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)

model = YOLO("yolov8x.pt")

class_colors = {}
class_names = model.names

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, stream = True, verbose=False, conf = 0.5)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls not in class_colors:
                class_colors[cls] = [random.randint(0, 255) for _ in range(3)]

            color = class_colors[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = class_names[cls]
            text = f"{class_names[cls]} {conf:.2f}"

            cv2.putText(frame, text, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Real Time", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
