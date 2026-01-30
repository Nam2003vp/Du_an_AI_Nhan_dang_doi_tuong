Hệ thống nhận dạng & theo dõi đối tượng – Phát hiện không đội mũ bảo hiểm

YOLOv8 + ByteTrack

📌 Giới thiệu

Đồ án này xây dựng một hệ thống nhận dạng và theo dõi đối tượng trong video dựa trên mô hình YOLOv8 kết hợp với thuật toán ByteTrack, nhằm giải quyết bài toán phát hiện người không đội mũ bảo hiểm trong môi trường thực tế (giao thông, công trường, khu công nghiệp,…).

Hệ thống có khả năng:

- Phát hiện đối tượng theo thời gian thực

- Theo dõi liên tục ID của từng đối tượng: Xe máy, xe đạp, ô tô, xe tải, xe bus

- Nhận diện và cảnh báo các trường hợp không đội mũ bảo hiểm

🧠 Công nghệ sử dụng

- Ngôn ngữ: Python

- Object Detection: YOLOv8 (Ultralytics)

- Multi-Object Tracking: ByteTrack

Thư viện chính:

+ PyTorch

+ OpenCV

+ NumPy

Ultralytics YOLO

📂 Cấu trúc thư mục
YOLO_ByteTrack_demo/
├── ByteTrack/              # Thuật toán theo dõi ByteTrack

├── yolov8/                 # Mã nguồn YOLOv8

├── nohelmet_dir/           # Module nhận diện không đội mũ bảo hiểm

├── nonet_dir/              # Module thử nghiệm / mở rộng

├── README.md               # Mô tả đồ án

├── .gitignore              # Loại trừ dataset, video, weights


📌 Lưu ý: Dataset và video test không được đưa lên GitHub do dung lượng lớn.

⚙️ Cài đặt môi trường
1️⃣ Clone repository
git clone https://github.com/Nam2003vp/Du_an_AI_Nhan_dang_doi_tuong.git
cd YOLO_ByteTrack_demo

2️⃣ Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate    # Linux / MacOS
venv\Scripts\activate       # Windows

3️⃣ Cài đặt thư viện
pip install -r requirements.txt

▶️ Cách sử dụng (Demo)

- Chuẩn bị video đầu vào (không push lên Git)

- Chạy pipeline YOLOv8 + ByteTrack

Quan sát kết quả:

+ Bounding box

+ ID theo dõi

+ Nhãn Helmet / No Helmet

📌 Kết quả được hiển thị trực tiếp trên video đầu ra.

📊 Kết quả đạt được

### Phát hiện đối tượng không đội mũ bảo hiểm
<p align="center">
  <img width="750" src="https://github.com/user-attachments/assets/3050a86b-8d81-45a5-afd0-5003300a2b5f" />
</p>
<p align="center"><em>
Hệ thống YOLOv8 phát hiện người không đội mũ bảo hiểm với bounding box và nhãn phân loại rõ ràng.
</em></p>

### Theo dõi nhiều đối tượng bằng ByteTrack

<p align="center">
  <img width="750" height="404" alt="image" src="https://github.com/user-attachments/assets/b69bf3ff-d4c2-4d01-bea3-2d400480cf41" />
</p>
<p align="center"><em>
Thuật toán ByteTrack duy trì ID ổn định cho từng đối tượng khi di chuyển và xuất hiện che khuất ngắn hạn.
</em></p>

- Phát hiện chính xác người và mũ bảo hiểm

- Theo dõi ổn định nhiều đối tượng cùng lúc

- Giữ nguyên ID khi đối tượng di chuyển hoặc bị che khuất ngắn hạn

Hoạt động tốt trên video thực tế

🚧 Hạn chế

- Hiệu năng phụ thuộc vào chất lượng video đầu vào

- Chưa tối ưu hoàn toàn cho môi trường ánh sáng yếu

- Cần GPU để đạt tốc độ real-time

🔮 Hướng phát triển

- Huấn luyện thêm dữ liệu thực tế

- Tối ưu tốc độ inference

- Tích hợp cảnh báo tự động

Triển khai trên camera giám sát thực tế
