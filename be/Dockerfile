FROM ultralytics/ultralytics:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y ffmpeg libsm6 libxext6 curl python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python opencv-contrib-python-headless
RUN pip install --no-cache-dir opencv-python

# Copy source code
COPY . .

# Environment variable to ensure src imports work if running from root
ENV PYTHONPATH=/app/src

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
