"""MinIO object storage client."""

from __future__ import annotations

import io
import logging
from datetime import timedelta
from typing import BinaryIO, Optional

from minio import Minio
from minio.error import S3Error

from core.config import settings

logger = logging.getLogger(__name__)


class MinIOService:
    _instance: Optional["MinIOService"] = None

    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created MinIO bucket: %s", self.bucket)
        except S3Error as e:
            logger.error("MinIO bucket init error: %s", e)

    # ── Upload ──
    def upload_file(
        self,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file-like object. Returns the object path."""
        self.client.put_object(
            self.bucket, object_name, data, length, content_type=content_type,
        )
        logger.info("Uploaded %s (%d bytes)", object_name, length)
        return object_name

    def upload_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        return self.upload_file(
            object_name, io.BytesIO(data), len(data), content_type,
        )

    # ── Download ──
    def get_file(self, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_presigned_url(
        self, object_name: str, expires: timedelta = timedelta(hours=1),
    ) -> str:
        return self.client.presigned_get_object(self.bucket, object_name, expires=expires)

    # ── Delete ──
    def delete_file(self, object_name: str):
        self.client.remove_object(self.bucket, object_name)
        logger.info("Deleted %s", object_name)

    # ── List ──
    def list_files(self, prefix: str = "") -> list[str]:
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def file_exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False

    # ── Singleton ──
    @classmethod
    def get_instance(cls) -> "MinIOService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_minio() -> MinIOService:
    return MinIOService.get_instance()
