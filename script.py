import cv2
from ultralytics import YOLO 

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not _:
        break
    
    results = model.predict(source=frame, stream=True, classes=[62, 63, 67])
    if results:
        for result in results:
            boxes = result.boxes.cuda()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 3)
                roi = frame[y1:y2, x1:x2]
                roi = cv2.blur(roi, (15, 15))
                frame[y1:y2, x1:x2] = roi
    else:
        print("no object")
    cv2.imshow("Camera", frame)