import cv2

def save_video_segment(input_video_path, output_video_path, start_frame, end_frame):
    # Mở video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError("Không thể mở video")

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Kiểm tra frame hợp lệ
    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        raise ValueError("start_frame hoặc end_frame không hợp lệ")

    # Codec và writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Nhảy đến frame bắt đầu
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)
        current_frame += 1

    # Giải phóng tài nguyên
    cap.release()
    out.release()

    print(f"Đã lưu video từ frame {start_frame} đến {end_frame} vào: {output_video_path}")

save_video_segment('/home/dungnt/People-Tracking-System/cam24.mp4', 'swap_1.mp4', 129, 174)
save_video_segment('/home/dungnt/People-Tracking-System/cam24.mp4', 'swap_2.mp4', 339, 370)
save_video_segment('/home/dungnt/People-Tracking-System/cam24.mp4', 'swap_3.mp4', 560, 583)
