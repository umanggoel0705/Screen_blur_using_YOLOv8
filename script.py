import cv2
from ultralytics import YOLO 

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    # _, frame = cap.read()

    # if not _:
    #     break
    
    results = model.predict(source=0, show=True, stream=True, classes=[62, 63, 67])
    if results:
        for result in enumerate(results):
            x1, y1, x2, y2 = result.boxes.xyxy[0]
            print(x1, y1, x2, y2)
    else:
        print("no object")
    # cv2.imshow("Camera", frame)