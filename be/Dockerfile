FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variable to ensure src imports work if running from root
# But usually we install as package or set PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Command to run the application
# We use src.main:app because src is a package now
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
