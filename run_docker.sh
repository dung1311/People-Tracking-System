#!/bin/bash

# Màu sắc hiển thị
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}    HỆ THỐNG TRUY VẾT MCT & TÌM KIẾM ĐỐI TƯỢNG - INFRA RUNNER${NC}"
echo -e "${BLUE}================================================================${NC}"

check_port() {
    local port=$1
    local service_name=$2
    if timeout 0.5 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$port" &>/dev/null; then
        return 1
    fi
    return 0
}

echo -e "\n${BLUE}[1/2] Kiểm tra xung đột cổng dịch vụ local...${NC}"
conflict=0

check_port 5432 "PostgreSQL Local"
if [ $? -eq 1 ]; then
    echo -e "${GREEN}[THÔNG TIN] Phát hiện cổng 5432 (PostgreSQL local) đang hoạt động.${NC}"
    echo -e "   -> ${GREEN}Không lo bị xung đột! Docker đã tự động đổi cổng host sang 5435 để chạy.${NC}"
fi

check_port 5435 "PostgreSQL Docker"
if [ $? -eq 1 ]; then
    conflict=1
    echo -e "${RED}[LỖI] Cổng 5435 (cổng dự phòng cho Postgres Docker) cũng đã bị chiếm!${NC}"
fi

check_port 9000 "MinIO API"
if [ $? -eq 1 ]; then
    conflict=1
    echo -e "${YELLOW}[CẢNH BÁO] Phát hiện cổng 9000 (MinIO API) đang được sử dụng chạy local trên Host.${NC}"
fi

check_port 9001 "MinIO Console"
if [ $? -eq 1 ]; then
    conflict=1
    echo -e "${YELLOW}[CẢNH BÁO] Phát hiện cổng 9001 (MinIO Console) đang được sử dụng chạy local trên Host.${NC}"
fi

if [ $conflict -eq 1 ]; then
    echo -e "${RED}[LƯU Ý] Nếu Docker báo lỗi 'address already in use', vui lòng dừng các dịch vụ trên host trước hoặc đổi cấu hình cổng trong docker-compose.yml.${NC}"
else
    echo -e "${GREEN}   -> Không phát hiện xung đột cổng. Sẵn sàng khởi chạy hạ tầng!${NC}"
fi

# Dọn dẹp và khởi động docker compose
echo -e "\n${BLUE}[2/2] Đang khởi dựng các Container Hạ tầng (PostgreSQL pgvector, MinIO)...${NC}"
echo -e "Chạy lệnh: ${GREEN}docker compose up -d --remove-orphans${NC}"
echo -e "Vui lòng đợi trong giây lát..."

docker compose up -d --remove-orphans

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================================================${NC}"
    echo -e "${GREEN}        KHỞI ĐỘNG HẠ TẦNG DOCKER THÀNH CÔNG!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo -e "• ${BLUE}PostgreSQL pgvector (mct-db):${NC} localhost:5435 (User/Pass/DB: user / password / db)"
    echo -e "• ${BLUE}MinIO API (mct-minio):${NC} localhost:9000"
    echo -e "• ${BLUE}MinIO Console:${NC} http://localhost:9001 (User/Pass: minioadmin / minioadminpassword)"
    echo -e "\n${YELLOW}HƯỚNG DẪN CHẠY BACKEND VÀ FRONTEND TỰ TAY:${NC}"
    echo -e "----------------------------------------------------------------"
    echo -e "${GREEN}1. Chạy Backend (FastAPI):${NC}"
    echo -e "   Bước A: Mở terminal mới, kích hoạt môi trường và cài đặt dependencies (nếu cần):"
    echo -e "      ${YELLOW}conda activate DATN${NC}"
    echo -e "      ${YELLOW}cd be && pip install -r requirements.txt${NC}"
    echo -e "   Bước B: Khởi chạy Backend:"
    echo -e "      ${YELLOW}python src/main.py${NC}"
    echo -e "      *(Đã cấu hình tự động kết nối sang Postgres ở localhost:5435)*"
    echo -e ""
    echo -e "${GREEN}2. Chạy Frontend (Vite/React):${NC}"
    echo -e "   Bước A: Mở terminal mới và di chuyển vào thư mục fe:"
    echo -e "      ${YELLOW}cd fe${NC}"
    echo -e "   Bước B: Khởi chạy dev server:"
    echo -e "      ${YELLOW}npm run dev${NC}"
    echo -e "      *(Frontend sẽ chạy trên http://localhost:5173 và gọi API tới http://localhost:8000)*"
    echo -e "----------------------------------------------------------------"
    echo -e "\nĐể xem log hạ tầng, hãy chạy: ${YELLOW}docker compose logs -f${NC}"
    echo -e "Để dừng hạ tầng, hãy chạy: ${YELLOW}docker compose down${NC}"
    echo -e "${GREEN}================================================================${NC}"
else
    echo -e "\n${RED}[LỖI] Khởi động các container hạ tầng thất bại. Vui lòng kiểm tra lỗi hiển thị ở trên.${NC}"
fi
