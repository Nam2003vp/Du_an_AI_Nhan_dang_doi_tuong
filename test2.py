import cv2
import torch
from ultralytics import YOLO

# Kiểm tra thiết bị GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🎯 Using device:", device)

# Load mô hình lên GPU
model = YOLO('yolov8n.pt')  # bạn có thể thay bằng 'best.pt' nếu là model của bạn
model.to(device)

# Mở video hoặc webcam
cap = cv2.VideoCapture(0)  # hoặc 0 nếu bạn dùng webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize để tăng tốc nếu cần
    frame = cv2.resize(frame, (640,360 ))

    # Dự đoán và vẽ kết quả
    results = model(frame)
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLOv8 GPU Test", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
