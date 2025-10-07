# Helmet-and-Number-plate-detections
The Helmet and Number Plate Detection System is an advanced computer vision project developed using Convolutional Neural Networks (CNN) and the YOLOv3 (You Only Look Once) deep learning architecture. The system automatically detects whether a motorcycle rider is wearing a helmet and simultaneously recognizes the vehicle’s number plate from images or real-time video streams. By training a custom YOLOv3 model with yolov3-custom_7000.weights, the project achieves high accuracy in detecting objects under varying conditions such as different angles, lighting, and traffic density.

This project plays a crucial role in automated traffic surveillance and road safety enforcement, helping authorities identify traffic rule violators efficiently. It eliminates manual monitoring by leveraging AI-driven detection and optical character recognition (OCR) for number plate extraction. The system can be integrated with traffic cameras or smart city infrastructure to automatically generate alerts or reports of violations.

Technically, the model was trained using a custom dataset of helmet and non-helmet rider images, along with vehicle number plates, to ensure real-world robustness. The combination of CNN feature extraction and YOLO’s real-time object detection enables the system to process video frames rapidly and accurately.

Key Highlights:

 Dual Detection: Identifies both helmet usage and vehicle number plates simultaneously.

 Powered by Deep Learning: Built using CNN and YOLOv3 with custom-trained weights.

 Real-Time Monitoring: Works efficiently on live video streams and CCTV footage.

 Smart Integration: Can connect with OCR and alert systems for rule enforcement.

 High Accuracy: Optimized for fast detection and minimal false positives.

Technology Stack:

Frameworks: TensorFlow, OpenCV, Keras

Model: YOLOv3 (Custom-trained on 7000+ iterations)

Languages: Python

Tools: LabelImg, Jupyter Notebook, NumPy, Pandas

Dataset: Custom images for helmet and number plate detection

Objective:
To design an AI-based intelligent system that enhances road safety, supports automated law enforcement, and contributes to smart traffic management through real-time detection and analysis.
