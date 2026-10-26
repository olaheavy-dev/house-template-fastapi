"""Object storage over the S3 API: Cloudflare R2 in production, RustFS locally (compose.yaml).

One boto3 client for the process, built in the lifespan (clients are thread-safe; sessions are
not). boto3 is synchronous, so calls that reach the network run in a worker thread; presigned
URLs are signed locally and need none. Browsers upload and download directly with presigned
URLs, so file bodies never pass through the API:

    @router.post("/uploads")
    async def start_upload(body: UploadIn, storage: StorageDep) -> UploadOut:
        key = f"uploads/{uuid.uuid4()}"
        url = storage.presigned_put(key, content_type=body.content_type, size=body.size)
        return UploadOut(key=key, url=url)

Storage is optional: with STORAGE_BUCKET unset there is no client, no readiness check, and
StorageDep answers 503. The bucket is created outside the app (`make storage-up` locally; the
provider's console or infrastructure code in production), so the app needs no admin rights.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated
from urllib.parse import quote

import anyio
import boto3
from botocore.config import Config
from fastapi import Depends, Request

from app.problems import Problem
from app.settings import Settings

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

# Keys are paths inside the bucket: no leading slash, no "." or ".." segments, no control
# characters, and S3's 1,024-byte limit.
_KEY = re.compile(r"^(?!/)(?!.*(?:^|/)\.{1,2}(?:/|$))[^\x00-\x1f\x7f]{1,1024}$")


class InvalidStorageKey(Problem):
    def __init__(self, key: str) -> None:
        # The key may come from the caller: it goes to the logs' exception, never the response.
        super().__init__(400, detail="That isn't a valid file key.")
        self.key = key


class ObjectNotFound(Problem):
    def __init__(self) -> None:
        super().__init__(404, detail="No such file.")


class Storage:
    def __init__(self, settings: Settings) -> None:
        if settings.storage_bucket is None:
            raise ValueError("STORAGE_BUCKET is not set")
        self.bucket = settings.storage_bucket
        self.url_seconds = settings.storage_url_seconds
        self.max_upload_bytes = settings.storage_max_upload_bytes
        self.upload_types = settings.storage_upload_types
        self.client = _client(settings, settings.storage_endpoint_url)
        # Presigned URLs carry their host in the signature, so they are signed for the address
        # browsers use. Inside Compose that differs from the API's (storage:9000 vs localhost:9000);
        # signing makes no network call, so the second client never connects.
        public = settings.storage_public_endpoint_url
        self.signer = _client(settings, public) if public else self.client

    def presigned_get(
        self, key: str, *, seconds: int | None = None, filename: str | None = None
    ) -> str:
        """A URL anyone holding it can download the object from until it expires. It downloads
        as an attachment (optionally named), so a stored file never renders as a page on the
        storage domain."""
        disposition = "attachment"
        if filename:
            disposition += f"; filename*=UTF-8''{quote(filename, safe='')}"
        return self.signer.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": check_key(key),
                "ResponseContentDisposition": disposition,
            },
            ExpiresIn=seconds or self.url_seconds,
        )

    def presigned_put(
        self, key: str, *, content_type: str, size: int, seconds: int | None = None
    ) -> str:
        """A URL to upload one object of exactly `size` bytes with exactly this Content-Type:
        both are signed, so a different size or type is refused. The type must be one of
        STORAGE_UPLOAD_TYPES and the size at most STORAGE_MAX_UPLOAD_BYTES (else 415 or 413).
        Browser uploads also need a CORS rule on the bucket."""
        if content_type not in self.upload_types:
            raise Problem(415, detail="That type of file can't be uploaded.")
        if not 0 < size <= self.max_upload_bytes:
            raise Problem(413, detail=f"Files can be at most {self.max_upload_bytes} bytes.")
        return self.signer.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": check_key(key),
                "ContentType": content_type,
                "ContentLength": size,
            },
            ExpiresIn=seconds or self.url_seconds,
        )

    async def put(self, key: str, body: bytes, *, content_type: str) -> None:
        await anyio.to_thread.run_sync(
            lambda: self.client.put_object(
                Bucket=self.bucket, Key=check_key(key), Body=body, ContentType=content_type
            )
        )

    async def get(self, key: str) -> bytes:
        def read() -> bytes:
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=check_key(key))
            except self.client.exceptions.NoSuchKey:
                raise ObjectNotFound from None
            return response["Body"].read()

        return await anyio.to_thread.run_sync(read)

    async def delete(self, key: str) -> None:
        await anyio.to_thread.run_sync(
            lambda: self.client.delete_object(Bucket=self.bucket, Key=check_key(key))
        )

    async def ping(self, seconds: float = 3.0) -> None:
        # Bounded like the database's: an unreachable endpoint must not hang readiness. The
        # abandoned thread finishes on botocore's own timeouts.
        with anyio.fail_after(seconds):
            await anyio.to_thread.run_sync(
                lambda: self.client.head_bucket(Bucket=self.bucket), abandon_on_cancel=True
            )

    def close(self) -> None:
        self.client.close()
        if self.signer is not self.client:
            self.signer.close()


def _client(settings: Settings, endpoint_url: str | None) -> S3Client:
    key_id, secret = settings.storage_access_key_id, settings.storage_secret_access_key
    return boto3.session.Session().client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=settings.storage_region,
        aws_access_key_id=key_id.get_secret_value() if key_id else None,
        aws_secret_access_key=secret.get_secret_value() if secret else None,
        config=Config(
            signature_version="s3v4",
            # Path-style: RustFS serves virtual-host buckets only with RUSTFS_SERVER_DOMAINS set;
            # R2 accepts both.
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "max_attempts": 3},
        ),
    )


def check_key(key: str) -> str:
    if not _KEY.match(key) or len(key.encode()) > 1024:
        raise InvalidStorageKey(key)
    return key


def get_storage(request: Request) -> Storage:
    storage: Storage | None = request.app.state.storage
    if storage is None:
        raise Problem(503, detail="File storage isn't configured.")
    return storage


StorageDep = Annotated[Storage, Depends(get_storage)]


def create_bucket(settings: Settings) -> None:
    """Create the bucket if it's missing: for local storage and tests, never production."""
    storage = Storage(settings)
    existing = {b["Name"] for b in storage.client.list_buckets().get("Buckets", [])}
    if storage.bucket not in existing:
        storage.client.create_bucket(Bucket=storage.bucket)
    storage.close()


if __name__ == "__main__":  # `make storage-up` runs this after starting the local storage
    from app.settings import get_settings

    if get_settings().storage_bucket is None:
        raise SystemExit("Set STORAGE_BUCKET (backend/.env) to use storage.")
    create_bucket(get_settings())
