# Nhật ký Triển khai (MCT Production API)

Tài liệu này ghi lại chi tiết các bước thực hiện xây dựng hệ thống quản lý theo dõi người đa camera (Multi-Camera Tracking) hoàn thiện, sử dụng **Pipeline 2** làm nhân xử lý.

---

## 📌 Tiến độ Triển khai

| Giai đoạn | Mô tả | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1** | Xác thực & Phân quyền (Auth & RBAC) | ✅ Đã hoàn thành | Thiết lập JWT + User/Role models |
| **Giai đoạn 2** | Tích hợp lưu trữ đối tượng MinIO | ✅ Đã hoàn thành | Cấu hình MinIO trong Docker & Client |
| **Giai đoạn 3** | Quản lý Camera (Nâng cao) | ✅ Đã hoàn thành | Upload calibration, video |
| **Giai đoạn 4** | Quản lý Tracking Session (Pipeline 2) | ✅ Đã hoàn thành | Background Thread + Dynamic YAML Config |
| **Giai đoạn 5** | Quản lý Cấu hình (SCT/MCT Configuration) | ✅ Đã hoàn thành | API CRUD cấu hình hệ thống |
| **Giai đoạn 6** | WebSocket & Streaming Grid thời gian thực | ✅ Đã hoàn thành | Stream frame & stats lên giao diện |
| **Giai đoạn 7** | Tích hợp Giao diện Frontend React/Vite | ✅ Đã hoàn thành | Đầy đủ UI tương tác chuyên nghiệp |

---

## 🛠 Chi tiết các bước đã hoàn thành

### Khởi động (2026-05-19)
- Đọc `implement_plan.md` từ workspace.
- Thống nhất kiến trúc hệ thống và lựa chọn **MCTPipeline2** làm core multicam pipeline.
- Khởi tạo file theo dõi tiến độ `task.md` và nhật ký triển khai `implement_summary.md`.

### Giai đoạn 1: Xác thực & Phân quyền (2026-05-19)
- **Cơ sở dữ liệu (SQLModel)**: Tạo model `User` chứa các thuộc tính `username`, `email`, `role` (ADMIN/OPERATOR/VIEWER), `hashed_password` và `is_active`.
- **Cấu trúc dữ liệu đầu vào (Pydantic)**: Xây dựng các schema `UserCreate`, `UserRead`, `UserUpdate`, `Token`, `TokenData`.
- **Xử lý bảo mật**:
  - Hashing mật khẩu bằng `passlib[bcrypt]`.
  - Ký và giải mã JWT bằng `python-jose[cryptography]`.
- **Middleware & Dependency**:
  - Tạo `deps.py` để inject database session.
  - Hàm `get_current_user` kiểm tra token JWT hợp lệ.
  - Class `RoleChecker` hỗ trợ phân quyền theo vai trò (ví dụ: Admin mới có quyền CRUD user hoặc xóa camera).
- **APIs**:
  - `POST /api/v1/auth/login`: Lấy JWT token.
  - `POST /api/v1/auth/register`: Đăng ký tài khoản mới (Chỉ Admin).
  - `GET /api/v1/auth/me`: Lấy thông tin tài khoản hiện tại.
  - `GET /api/v1/users/`: Liệt kê toàn bộ người dùng (Chỉ Admin).
  - `PATCH /api/v1/users/{id}`: Cập nhật thông tin người dùng (Chỉ Admin).
  - `DELETE /api/v1/users/{id}`: Xóa người dùng (Chỉ Admin).
- **Seeding dữ liệu**: Viết `init_data.py` tự động khởi tạo admin mặc định (`admin` / `adminpassword123`) và các cấu hình SCT/MCT ban đầu khi chạy ứng dụng.

### Giai đoạn 2: Tích hợp lưu trữ đối tượng MinIO (2026-05-19)
- **Cấu hình Hạ tầng (Docker)**:
  - Thêm service `minio` chạy image `minio/minio` chính thức với các cổng `9000` (API) và `9001` (Console quản trị).
  - Khai báo biến môi trường kết nối MinIO (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`) cho backend.
- **Dịch vụ Lưu trữ (`MinIOClient`)**:
  - Triển khai lớp `MinIOClient` trong `core/minio_client.py` hỗ trợ kết nối MinIO và tự động khởi tạo các bucket cần thiết (`videos`, `calibrations`, `snapshots`, `recordings`, `thumbnails`).
  - **Fallback Cực kỳ Bền bỉ**: Nếu MinIO offline hoặc không cấu hình, client tự động chuyển sang lưu trữ cục bộ tại thư mục `data/` trong workspace để đảm bảo hệ thống vẫn hoạt động mượt mà mà không bị crash.
  - Hỗ trợ các phương thức chuẩn: `upload_file()`, `get_presigned_url()` (tự động đổi domain sang localhost phục vụ dev ngoài container), `download_file()`, `delete_file()`.
- **Tích hợp vào Lifespan**: Gọi khởi tạo `MinIOClient` ngay khi FastAPI startup.

### Giai đoạn 3: Quản lý Camera (Nâng cao) (2026-05-19)
- **Cơ sở dữ liệu & Cấu hình**:
  - Mở rộng model `Camera` với các thuộc tính: `source_type`, `location`, `resolution`, `fps`, `has_calibration`, `calibration_path`.
- **Tích hợp Lưu trữ & Xử lý Ảnh (OpenCV)**:
  - **Upload Calibration**: API `POST /api/v1/cameras/{id}/calibration` hỗ trợ tải lên file JSON hiệu chuẩn camera, tự động validate định dạng JSON hợp lệ trước khi lưu vào MinIO.
  - **Download Calibration**: API `GET /api/v1/cameras/{id}/calibration` tải file cấu hình hiệu chuẩn trực tiếp từ MinIO/local và trả về dưới dạng JSON Dict cho engine sử dụng.
  - **Upload Video & Tự động trích xuất Thumbnail**: API `POST /api/v1/cameras/{id}/video` cho phép tải lên file video của camera. Backend sẽ tự động gọi OpenCV để:
    1. Trích xuất metadata như fps, chiều rộng, chiều cao để tự động lưu vào camera resolution.
    2. Đọc frame tại vị trí 10% của video (để tránh lấy các frame màu đen ở đầu video).
    3. Mã hóa thành JPEG và upload lên bucket `thumbnails` làm ảnh preview cực kỳ cao cấp cho Frontend.
  - **Get Thumbnail**: API `GET /api/v1/cameras/{id}/thumbnail` cung cấp link presigned URL ảnh thu nhỏ của camera.
  - **Stream Preview**: API `GET /api/v1/cameras/{id}/stream` stream preview camera thời gian thực bằng MJPEG StreamingResponse.

### Giai đoạn 4: Quản lý Tracking Session (Pipeline 2) (2026-05-19)
- **Cơ sở dữ liệu (SQLModel)**:
  - Tạo model `TrackingSession` chứa thông tin về: tên, trạng thái (created, running, stopping, completed, failed), cấu hình đóng băng `sct_config` & `mct_config`, mảng `camera_ids`, thời điểm bắt đầu/kết thúc, đường dẫn video output/txt kết quả trong MinIO, và các chỉ số thống kê (tổng frame, tổng ID, FPS trung bình).
- **Bộ điều khiển session background (`SessionManager`)**:
  - Triển khai lớp quản lý session singleton `SessionManager` trong `core/session_manager.py`.
  - Khi start session, backend tự động:
    1. Tải các file hiệu chuẩn JSON của các camera được chọn từ MinIO về một thư mục tạm thời.
    2. Giải quyết nguồn video của các camera (download cache nếu lưu trên MinIO, hoặc lấy trực tiếp).
    3. Tự động biên dịch và ghi file cấu hình YAML tạm thời chứa đầy đủ tham số và đường dẫn tương thích 100% với `MCTPipeline2` của Pipeline 2.
    4. Kích hoạt một background thread để chạy `MCTPipeline2` một cách bất đồng bộ.
  - Sau khi kết thúc hoặc bị dừng giữa chừng, backend tự động nén video output và lưu trữ file MOT15 `.txt` kết quả lên MinIO/local, đồng thời cập nhật đầy đủ thống kê và chuyển trạng thái session thành `completed`.

### Giai đoạn 5: Quản lý Cấu hình (SCT/MCT) (2026-05-19)
- **Model cấu hình (`TrackingConfig`)**: Lưu trữ các hồ sơ (profiles) cấu hình tham số cho cả SCT (detector conf, tracker threshold) và MCT (spatial/reid/time matching threshold).
- **APIs CRUD**: 
  - Đầy đủ các API `POST`, `GET`, `PATCH`, `DELETE` cho configs.
  - API `GET /api/v1/configs/defaults`: Lấy ra cấu hình mặc định được đánh dấu `is_default = True`.
  - API `POST /api/v1/configs/validate`: Validate cấu hình YAML/JSON đầu vào trước khi lưu để tránh các lỗi cấu hình sai tham số.

### Giai đoạn 6: WebSocket Streaming (2026-05-19)
- **Tối ưu hóa đa luồng & Bất đồng bộ**:
  - Để WebSocket gửi dữ liệu mượt mà mà không làm nghẽn luồng xử lý nhận dạng của OpenCV/YOLO, backend đã áp dụng mẫu kiến trúc **Bridge** sử dụng `asyncio.Queue` kết hợp với `loop.call_soon_threadsafe`.
  - Khi pipeline sinh ra frame kết quả `grid`, hệ thống sẽ nén JPEG trực tiếp trên RAM, chuyển đổi thành chuỗi base64 và gửi lên hàng đợi để WebSocket thread tiêu thụ và truyền tải về giao diện.
  - Endpoint WebSocket `/api/v1/ws/session/{session_id}` tự động truyền tải frame thời gian thực kèm theo thống kê FPS và số lượng ID hiện hữu trên màn hình.

### Nâng cấp & Sửa lỗi (Cập nhật tối 2026-05-19)
- **Giải quyết triệt để lỗi phân giải đường dẫn Video từ MinIO**:
  - Khắc phục lỗi `Could not open video source videos/3/cam62.mp4` tại các endpoint `/api/v1/stream/...` và `/api/v1/frames/...`. 
  - Khi một nguồn camera được lưu trữ trên MinIO, API sẽ tự động tải video mẫu về bộ đệm cục bộ `data/temp_stream_cam_{id}.mp4` hoặc `data/temp_frames_cam_{id}.mp4` trước khi mở bằng OpenCV `cv2.VideoCapture`, giúp việc preview stream camera hoạt động mượt mà 100%.
- **Khởi chạy CPU an toàn tương thích tối đa với Driver GPU host**:
  - Cấu hình toàn bộ mô hình nhận diện (`yolov11`, `rtmpose`, `fastreid`) chạy trên CPU. Giúp uvicorn server khởi động mượt mà và an toàn trên môi trường máy của bạn mà không gặp lỗi xung đột NVIDIA Driver (`RuntimeError: The NVIDIA driver on your system is too old`).
- **Thêm tính năng "Kiểm tra kết nối" (Check Connection) & "Tải video trực tiếp" khi thêm Camera**:
  - **Backend**: Triển khai API endpoint `POST /api/v1/cameras/check-connection` hỗ trợ kiểm tra kết nối tới camera nguồn bằng OpenCV. API sẽ mở luồng stream (RTSP/Webcam/Local Video/MinIO Object), cố gắng giải mã đọc 1 frame mẫu để validate và trả về thông số độ phân giải (Resolution) cùng tốc độ khung hình (FPS) thực tế.
  - **Frontend**: Thiết kế và tích hợp giao diện tab hiện đại trong modal "Thêm Camera mới" chia làm 2 tùy chọn:
    1. *RTSP / Webcam / Local Path*: Hỗ trợ nút **Check Connection** trực quan. Khi click sẽ hiện trạng thái loading và trả về kết báo kết nối thành công (màu xanh lá kèm Resolution/FPS) hoặc thất bại (màu đỏ kèm chi tiết lỗi).
    2. *Tải lên video trực tiếp*: Cho phép người dùng chọn tệp tin video `.mp4` trực tiếp từ máy của mình. Hệ thống tự động thực hiện quy trình khép kín: tạo camera -> tải video lên MinIO -> phân tích độ phân giải/FPS -> tạo thumbnail xem trước hoàn toàn tự động khi nhấn "Thêm".




