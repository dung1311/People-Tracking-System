import os
import logging
from io import BytesIO
from datetime import timedelta
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

class MinIOClient:
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadminpassword")
        self.secure = os.getenv("MINIO_SECURE", "False").lower() in ("true", "1", "yes")
        
        self.client = None
        self.fallback_mode = False
        self.buckets = ["videos", "calibrations", "snapshots", "recordings", "thumbnails"]
        
        try:
            logger.info(f"Connecting to MinIO at {self.endpoint}...")
            self.client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            # Verify connection by listing buckets or making a simple call
            # This triggers fallback if connection fails
            self.init_buckets()
            logger.info("MinIO storage client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to MinIO: {e}. Falling back to local filesystem storage.")
            self.fallback_mode = True
            self.init_fallback_dirs()

    def init_buckets(self):
        for bucket in self.buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created MinIO bucket: '{bucket}'")
            except Exception as e:
                # Re-raise to trigger fallback if initial ping fails
                raise e

    def init_fallback_dirs(self):
        for bucket in self.buckets:
            os.makedirs(f"data/{bucket}", exist_ok=True)
        logger.info("Fallback local directories initialized under 'data/'")

    def upload_file(self, bucket_name: str, object_name: str, file_data, length: int = -1, content_type: str = "application/octet-stream") -> str:
        """
        Uploads a file. file_data can be a path string, bytes, or a file-like object.
        Returns the path/URI of the saved object.
        """
        if self.fallback_mode:
            local_dir = f"data/{bucket_name}"
            # Ensure nested folders work
            local_path = os.path.join(local_dir, object_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            if isinstance(file_data, str): # Path to file
                import shutil
                shutil.copy2(file_data, local_path)
            elif isinstance(file_data, bytes):
                with open(local_path, "wb") as f:
                    f.write(file_data)
            else: # file-like object
                with open(local_path, "wb") as f:
                    f.write(file_data.read())
            logger.info(f"[Fallback Storage] Saved file to {local_path}")
            return local_path
        
        try:
            if isinstance(file_data, str):
                self.client.fput_object(bucket_name, object_name, file_data, content_type=content_type)
            elif isinstance(file_data, bytes):
                self.client.put_object(
                    bucket_name, 
                    object_name, 
                    BytesIO(file_data), 
                    length=len(file_data), 
                    content_type=content_type
                )
            else:
                self.client.put_object(
                    bucket_name, 
                    object_name, 
                    file_data, 
                    length=length if length > 0 else -1, 
                    content_type=content_type
                )
            logger.info(f"[MinIO] Uploaded '{object_name}' to bucket '{bucket_name}'")
            return f"{bucket_name}/{object_name}"
        except S3Error as e:
            logger.error(f"MinIO upload error: {e}")
            raise e

    def get_presigned_url(self, bucket_name: str, object_name: str, expires_hours: int = 24) -> str:
        """
        Generates a URL that can be accessed by the frontend.
        """
        if self.fallback_mode:
            # Under fallback, serve static files via FastAPI's static mount `/static`
            # E.g. data/videos/cam1.mp4 is served at /static/videos/cam1.mp4
            # We assume the base domain is handled by frontend or use absolute paths relative to api root
            return f"/static/{bucket_name}/{object_name}"
            
        try:
            url = self.client.presigned_get_object(
                bucket_name, 
                object_name, 
                expires=timedelta(hours=expires_hours)
            )
            # Convert Docker container name inside endpoint to localhost if request is from outside docker
            # Highly useful when developing on host machine
            if "minio:9000" in url:
                url = url.replace("minio:9000", "localhost:9000")
            return url
        except Exception as e:
            logger.error(f"MinIO presigned url error: {e}")
            # Safe fallback URL
            return f"/static/{bucket_name}/{object_name}"

    def download_file(self, bucket_name: str, object_name: str, file_path: str):
        if self.fallback_mode:
            import shutil
            local_path = f"data/{bucket_name}/{object_name}"
            shutil.copy2(local_path, file_path)
            return file_path
            
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            return file_path
        except Exception as e:
            logger.error(f"MinIO download error: {e}")
            raise e

    def delete_file(self, bucket_name: str, object_name: str):
        if self.fallback_mode:
            local_path = f"data/{bucket_name}/{object_name}"
            if os.path.exists(local_path):
                os.remove(local_path)
            return
            
        try:
            self.client.remove_object(bucket_name, object_name)
        except Exception as e:
            logger.error(f"MinIO delete error: {e}")
            raise e

# Singleton Client instance
minio_client = None

def get_minio_client() -> MinIOClient:
    global minio_client
    if minio_client is None:
        minio_client = MinIOClient()
    return minio_client
