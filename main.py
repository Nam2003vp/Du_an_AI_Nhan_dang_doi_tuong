import cv2
import time
import os 
from yolov8_tracker import YOLOv8ByteTrack
import yolov8_tracker
import supervision as sv


weight_path = "C:\\Users\\Nam\\Documents\\Đồ án tốt nghiệp\\YOLO_ByteTrack_demo\\weight\\train29\\weights\\best.pt"
source = "C:\\Users\\Nam\\Documents\\Đồ án tốt nghiệp\\YOLO_ByteTrack_demo\\videos\\test_2.MOV"  
output_path = "C:\\Users\\Nam\\Documents\\Đồ án tốt nghiệp\\YOLO_ByteTrack_demo\\runs\\result\\result_test_5.mp4"

# **Tạo thư mục và đường dẫn cho video clip 'nohelmet'**
nohelmet_output_dir = "C:\\Users\\Nam\\Documents\\Đồ án tốt nghiệp\\YOLO_ByteTrack_demo\\nohelmet_dir"
os.makedirs(nohelmet_output_dir, exist_ok=True)

# Lấy tên file từ source để tạo tên output tương ứng
source_name = os.path.basename(source)
name_no_ext = os.path.splitext(source_name)[0]
nohelmet_output_path = os.path.join(nohelmet_output_dir, f"{name_no_ext}_nohelmet.mp4")

# Open the video source
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Unable to open video source {source}")
    exit(1)

# Get video properties for writer
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps == 0:
    fps = 30.0  # Default to 30 FPS if fps not available

# Define video writers to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# **VideoWriter cho clip 'nohelmet' 
nohelmet_writer = cv2.VideoWriter(nohelmet_output_path, fourcc, fps, (width, height))

# Initialize the YOLOv8 + ByteTrack tracker
tracker = YOLOv8ByteTrack(weight_path, frame_rate=int(fps))

# Bộ đếm frame có 'nohelmet'
nohelmet_frame_count = 0

# Processing loop
print("Starting video processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video or cannot read frame

    # Process frame: detection and tracking
    start_time = time.time()
    annotated_frame, counts = tracker.process_frame(frame)
    end_time = time.time()

    # Calculate FPS for current frame 
    frame_time = end_time - start_time
    current_fps = 1.0 / frame_time if frame_time > 0 else 0.0

    # Draw FPS on the frame (black outline + white text)
    fps_text = f"FPS: {current_fps:.2f}"
    cv2.putText(annotated_frame, fps_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(annotated_frame, fps_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw object counts for each class on the frame
    y = 50  # starting Y position for the first count (below FPS)
    for class_name, count in counts.items():
        count_text = f"{class_name}: {count}"
        cv2.putText(annotated_frame, count_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, count_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 20  # move to next line for next class

    # lưu frame chứa "nohelmet"
    if 'nohelmet' in counts and counts['nohelmet'] > 0:
        nohelmet_writer.write(annotated_frame)
        nohelmet_frame_count += 1

    # Show the annotated frame in a window (resized for display)
    resized_frame = cv2.resize(annotated_frame, None, fx=0.6, fy=0.6)
    cv2.imshow("YOLOv8 ByteTrack Demo", resized_frame)

    # Write the frame to the full output video file
    writer.write(annotated_frame)

    # Break out of the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
writer.release()
nohelmet_writer.release()
cv2.destroyAllWindows()

# **Sau khi xử lý xong, kiểm tra tổng thời lượng nohelmet**
total_nohelmet_duration = nohelmet_frame_count / fps  # tính bằng giây
if total_nohelmet_duration >= 6:
    print(f"Detected 'nohelmet' for {total_nohelmet_duration:.1f} seconds - clip saved to {nohelmet_output_path}")
else:
    # Nếu không đủ 6 giây, xóa file clip đã ghi 
    try:
        os.remove(nohelmet_output_path)
        print(f"'nohelmet' appeared for only {total_nohelmet_duration:.1f} seconds - clip not saved (file removed).")
    except OSError as e:
        print("No 'nohelmet' clip to remove or error removing file:", e)

print("Processing complete. Full output saved to", output_path)
