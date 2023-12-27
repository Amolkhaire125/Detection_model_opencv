
import cv2
from ultralytics import YOLO
import time
import math
import cvzone

class_dict = { 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class_list = list(class_dict.values())

# Load a model
model = YOLO(r"C:\Users\Amol\Downloads\ultralytics-main\ultralytics-main\yolov8n.pt")  # Update the path to your YOLOv8n model

# Open a video capture
cap = cv2.VideoCapture(r"C:\Users\Amol\OneDrive\Desktop\gettyimages-1188462321-640_adpp.mp4")

# Define class labels to keep
desired_classes = ["person", "bag"]
print("model loaded")

confidence_threshold = 0.7

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        break

    # Run inference on the current frame
    print("bs1")
    results = model(img, stream=True)
    print("bs2")
    print("results", results)

    # Process results
    for r in results:
        boxes = r.boxes
        print("boxes", boxes)
        for box in boxes:
            print("box", box)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf >= confidence_threshold:
                if cls == 0 or cls == 24 or cls == 26 or cls == 28 or cls == 63 or cls == 67:
                    print("cls and confidence", cls, conf)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
                    cvzone.putTextRect(
                        img,
                        f"{class_list[cls]} {conf:.2f}",(max(0, x1), max(35, y1 - 13)),thickness=1,colorT=(255, 255, 255),colorR=(0, 0, 0),scale=1,offset=3,)

    # Calculate and print frames per second (FPS)
    fps = 1 / (time.time() - new_frame_time)
    print(f"FPS: {fps:.2f}")

    img = cv2.resize(img, (1280, 640))
    # Display the frame
    cv2.imshow("Detection", img)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()


              



