# vibe coding prompt: coding YOLO inference using ultralytics
# pip install ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["img1.PNG", "img2.PNG"])  # return a list of Results objects

# Process results list
for index, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"result-{index}.jpg")  # save to disk

# vibe coding prompt: coding realtime inference using webcam
# realtime inference using webcam
# Open webcam
import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Display results on the frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Realtime Inference", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()