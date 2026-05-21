# People Tracking System — Workflow & Architecture Diagrams

## 1. Sequence Diagram: Tracking Session Workflow

Biểu đồ này thể hiện trình tự thời gian khi UI (Frontend) gọi Backend, dữ liệu được truyền đi lưu trữ và Engine AI xử lý.

```mermaid
sequenceDiagram
    autonumber
    participant UI as Giao diện (React/Vite)
    participant API as FastAPI Backend
    participant DB as PostgreSQL (pgvector)
    participant S3 as MinIO Storage
    participant Engine as Tracking Engine (SCT/MCT)

    Note over UI,Engine: GIAI ĐOẠN 1: Đăng nhập & Thiết lập Camera
    UI->>API: 1. Đăng nhập (Username/Password)
    API->>DB: Kiểm tra Role & Credentials
    DB-->>API: Trả về thông tin Auth
    API-->>UI: Cấp JWT Token

    UI->>API: 2. Thêm Camera & Upload Calibration (JSON/Video)
    API->>DB: Lưu Metadata Camera
    API->>S3: Lưu file Calibration / Video gốc
    S3-->>API: Trả về URL đường dẫn
    API-->>UI: Xác nhận thêm Camera thành công

    Note over UI,Engine: GIAI ĐOẠN 2: Khởi chạy Tracking Session
    UI->>API: 3. Bấm "Start Tracking" (Gửi Config + Camera IDs)
    API->>DB: Cập nhật trạng thái Session = "running"
    API->>Engine: Kích hoạt Pipeline bằng Background Task
    API-->>UI: Trả về Session ID

    par Quá trình xử lý song song (Background)
        Engine->>S3: Kéo Video/Webcam & Calibration
        Engine->>Engine: Chạy AI (YOLO, Pose, FastReID, Camera Workers)
        Engine->>DB: Lưu lộ trình (Tracks) & Global IDs liên tục
        Engine->>S3: Đẩy kết quả Output (Video MP4, Snapshots)
        
        loop Kết nối Stream Thời gian thực
            Engine-->>API: Emit sự kiện frame, tracks, cảnh báo
            API-->>UI: Đẩy dữ liệu qua WebSocket (Real-time Video)
        end
    end

    Note over UI,Engine: GIAI ĐOẠN 3: Kết thúc & Tra cứu
    UI->>API: 4. Bấm "Stop Session"
    API->>Engine: Gửi lệnh Dừng Pipeline
    Engine-->>API: Xác nhận dừng hoàn tất
    API->>DB: Cập nhật trạng thái = "completed"
    API-->>UI: Dừng thành công

    UI->>API: 5. Search Person / Xem lịch sử Tracking
    API->>DB: Truy vấn dữ liệu Track (Vector Search / SQL)
    API->>S3: Lấy hình ảnh (Thumbnails, Cắt ghép)
    API-->>UI: Hiển thị danh sách kết quả cho User
```

## 2. High-Level Architecture Diagram

Biểu đồ này mô tả lại mức cao (high-level) cách các component nối với nhau (Ai gọi ai, qua giao thức nào).

```mermaid
graph TD
    %% Định nghĩa các node
    User((Người dùng\nAdmin/Operator))
    FE["Frontend (React / Vite)"]
    
    subgraph API_Layer ["API Layer (FastAPI Backend)"]
        AUTH("Auth & RBAC")
        CAM_API("Camera API")
        SESS_API("Session API")
        WS_API("WebSocket Stream")
    end

    subgraph Core_Engine ["Core Tracking Engine"]
        WORKERS["Camera Workers (SCT)"]
        MCT["MCT Pipeline v3"]
    end

    subgraph Storage ["Dữ liệu (Storage)"]
        DB[("PostgreSQL\n(User, Camera, Config,\nTracks, Session)")]
        MINIO[("MinIO Object Storage\n(Videos, Cals, Outputs)")]
    end

    %% Các luồng dữ liệu
    User -->|Tương tác trên Web| FE
    FE -->|HTTP/REST (Kèm JWT)| AUTH
    FE -.->|WebSocket (Real-time)| WS_API
    
    AUTH --> CAM_API
    AUTH --> SESS_API
    
    %% API tương tác Storage
    CAM_API -->|Đọc/Ghi Metadata| DB
    CAM_API -->|Upload Files| MINIO
    SESS_API -->|Truy vấn cấu hình/Track| DB
    
    %% Gọi Engine
    SESS_API -->|Trigger Start/Stop| MCT
    MCT -->|Phân việc| WORKERS
    
    %% Engine tương tác Storage
    WORKERS -->|Đọc Video/Calib| MINIO
    WORKERS -->|Lưu Track Local| DB
    MCT -->|Lưu Global ID & Metrics| DB
    MCT -->|Lưu Video Output| MINIO
    
    %% Stream feedback
    MCT -.->|Bắn sự kiện/Frames\nqua Queue/Memory| WS_API

    %% Styling 
    classDef frontend fill:#3178C6,stroke:#fff,stroke-width:2px,color:#fff;
    classDef api fill:#009688,stroke:#fff,stroke-width:2px,color:#fff;
    classDef engine fill:#FF9800,stroke:#fff,stroke-width:2px,color:#fff;
    classDef storage fill:#607D8B,stroke:#fff,stroke-width:2px,color:#fff;

    class FE frontend
    class AUTH,CAM_API,SESS_API,WS_API api
    class WORKERS,MCT engine
    class DB,MINIO storage
```
