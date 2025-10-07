import cv2
import numpy as np
import imutils
import argparse
from tensorflow.keras.models import load_model
import pytesseract
import csv
from datetime import datetime
import os

# -----------------------------
# Tesseract OCR path (update if needed)
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# Parse video file path from terminal
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
parser.add_argument("--output", type=str, default="output.avi", help="Path to save output video")
args = parser.parse_args()

video_path = args.video
output_path = args.output

# -----------------------------
# Load YOLOv3 model
# -----------------------------
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -----------------------------
# Load Helmet Classifier
# -----------------------------
model = load_model("helmet-nonhelmet_cnn.h5", compile=False)
print("Helmet model loaded successfully!")

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Error: Cannot open video {video_path}")
    exit()

ret, frame = cap.read()
frame = imutils.resize(frame, height=500)
height, width = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(output_path, fourcc, 5, (width, height))

COLORS = [(0, 255, 0), (0, 0, 255)]

# -----------------------------
# CSV file for violations
# -----------------------------
csv_file = "violations.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Number Plate", "Date", "Time", "Image Path"])

# -----------------------------
# Helmet prediction function
# -----------------------------
def helmet_or_nohelmet(roi):
    try:
        roi = cv2.resize(roi, (224, 224))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi)
        return int(prediction[0][0] > 0.5)  # 0: Helmet, 1: No Helmet
    except Exception as e:
        print("Prediction error:", e)
        return 0

# -----------------------------
# Process video frames
# -----------------------------
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = imutils.resize(img, height=500)
    height, width = img.shape[:2]

    # YOLO blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, classIds = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, center_x - w // 2)
                y = max(0, center_y - h // 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = COLORS[classIds[i] % len(COLORS)]

            # Bike detected
            if classIds[i] == 0:  # Bike class index
                helmet_roi = img[y:y + h // 4, x:x + w]
                if helmet_roi.size != 0:
                    c = helmet_or_nohelmet(helmet_roi)
                    cv2.putText(img, ["Helmet", "No-Helmet"][c], (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                    # No Helmet → save violation
                    if c == 1:
                        # Save cropped bike image
                        violation_img = img[y:y+h, x:x+w]
                        img_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(img_name, violation_img)

                        # Detect number plate (lower half of bike)
                        plate_region = img[y + h//2:y + h, x:x + w]
                        if plate_region.size != 0:
                            plate_text = pytesseract.image_to_string(plate_region, config="--psm 8").strip()
                        else:
                            plate_text = "Unknown"

                        # Save to CSV
                        with open(csv_file, mode="a", newline="") as f:
                            writer_csv = csv.writer(f)
                            writer_csv.writerow([
                                plate_text,
                                datetime.now().strftime("%Y-%m-%d"),
                                datetime.now().strftime("%H:%M:%S"),
                                img_name
                            ])
            else:
                # Other objects
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    writer.write(img)
    cv2.imshow("Helmet Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"✅ Processing finished! Output saved at: {output_path}")
print(f"✅ Violations saved in: {csv_file}")
