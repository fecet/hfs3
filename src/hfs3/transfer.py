"""Core streaming transfer logic using pure asyncio."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import ssl
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Literal

import aiohttp
from aiobotocore.session import get_session
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi, RepoFile, hf_hub_url
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .progress import ProgressTracker

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

RepoType = Literal["model", "dataset", "space"]

MIN_PART_SIZE = 5 * 1024 * 1024  # 5 MB, S3 multipart minimum
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100 MB, use multipart above this


@dataclass
class FileInfo:
    repo_id: str
    path_in_repo: str
    s3_key: str
    size: int
    revision: str
    repo_type: RepoType


@dataclass
class TransferConfig:
    """Transfer configuration."""

    concurrency: int = 4
    chunk_size: int = DEFAULT_CHUNK_SIZE
    endpoint_url: str | None = None
    hf_token: str | None = None


@dataclass
class MultipartUpload:
    """Manages S3 multipart upload lifecycle."""

    client: "S3Client"
    bucket: str
    key: str
    upload_id: str | None = None
    parts: list[dict] = field(default_factory=list)

    async def start(self) -> str:
        """Initiate multipart upload."""
        response = await self.client.create_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
        )
        self.upload_id = response["UploadId"]
        return self.upload_id

    async def upload_part(self, part_number: int, data: bytes) -> dict:
        """Upload a single part."""
        response = await self.client.upload_part(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            PartNumber=part_number,
            Body=data,
        )
        part_info = {"PartNumber": part_number, "ETag": response["ETag"]}
        self.parts.append(part_info)
        return part_info

    async def complete(self) -> dict:
        """Complete multipart upload."""
        sorted_parts = sorted(self.parts, key=lambda p: p["PartNumber"])
        return await self.client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            MultipartUpload={"Parts": sorted_parts},
        )

    async def abort(self) -> None:
        """Abort multipart upload (cleanup on failure)."""
        if self.upload_id:
            await self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self.upload_id,
            )


def matches_patterns(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def list_repo_files(
    repo_id: str,
    revision: str,
    repo_type: RepoType,
    s3_dest: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[FileInfo]:
    """List files in a HuggingFace repo using HfApi."""
    api = HfApi()
    files: list[FileInfo] = []

    for item in api.list_repo_tree(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        recursive=True,
    ):
        if not isinstance(item, RepoFile):
            continue

        path_in_repo = item.path
        filename = path_in_repo.split("/")[-1]

        if not matches_patterns(filename, include_patterns):
            continue
        if exclude_patterns and matches_patterns(filename, exclude_patterns):
            continue

        s3_key = f"{s3_dest.rstrip('/')}/{path_in_repo}"
        files.append(
            FileInfo(
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                s3_key=s3_key,
                size=item.size or 0,
                revision=revision,
                repo_type=repo_type,
            )
        )

    return files


@asynccontextmanager
async def create_s3_client(endpoint_url: str | None = None, max_connections: int = 100):
    """Create async S3 client context."""
    session = get_session()
    config = BotoConfig(max_pool_connections=max_connections)
    async with session.create_client(
        "s3", endpoint_url=endpoint_url, config=config
    ) as client:
        yield client


async def stream_chunks_from_hf(
    url: str,
    session: aiohttp.ClientSession,
    chunk_size: int,
    hf_token: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream file content from HuggingFace URL."""
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with session.get(url, headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.content.iter_chunked(chunk_size):
            yield chunk


def _parse_s3_path(s3_key: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    path = s3_key
    if path.startswith("s3://"):
        path = path[5:]
    bucket, key = path.split("/", 1)
    return bucket, key


async def _multipart_transfer(
    file_info: FileInfo,
    download_url: str,
    bucket: str,
    key: str,
    s3_client: "S3Client",
    http_session: aiohttp.ClientSession,
    config: TransferConfig,
    progress: ProgressTracker | None,
) -> None:
    """Multipart upload for large files."""
    mpu = MultipartUpload(client=s3_client, bucket=bucket, key=key)

    try:
        await mpu.start()

        part_number = 1
        buf = bytearray()

        async for chunk in stream_chunks_from_hf(
            download_url, http_session, config.chunk_size, config.hf_token
        ):
            buf.extend(chunk)

            while len(buf) >= config.chunk_size:
                part_data = bytes(buf[: config.chunk_size])
                del buf[: config.chunk_size]

                await mpu.upload_part(part_number, part_data)
                part_number += 1

                if progress:
                    progress.update(file_info.path_in_repo, len(part_data))

        if buf:
            await mpu.upload_part(part_number, bytes(buf))
            if progress:
                progress.update(file_info.path_in_repo, len(buf))

        await mpu.complete()

    except Exception:
        await mpu.abort()
        raise

    if progress:
        progress.complete_file(file_info.path_in_repo)


async def _simple_transfer(
    file_info: FileInfo,
    download_url: str,
    bucket: str,
    key: str,
    s3_client: "S3Client",
    http_session: aiohttp.ClientSession,
    config: TransferConfig,
    progress: ProgressTracker | None,
) -> None:
    """Simple PutObject for small files (< 100MB)."""
    data = bytearray()

    async for chunk in stream_chunks_from_hf(
        download_url, http_session, config.chunk_size, config.hf_token
    ):
        data.extend(chunk)
        if progress:
            progress.update(file_info.path_in_repo, len(chunk))

    await s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=bytes(data),
    )

    if progress:
        progress.complete_file(file_info.path_in_repo)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, ConnectionError, OSError, ssl.SSLError)),
    reraise=True,
)
async def transfer_single_file(
    file_info: FileInfo,
    s3_client: "S3Client",
    http_session: aiohttp.ClientSession,
    config: TransferConfig,
    progress: ProgressTracker | None = None,
) -> None:
    """Stream file from HF to S3 using multipart upload."""
    download_url = hf_hub_url(
        repo_id=file_info.repo_id,
        filename=file_info.path_in_repo,
        revision=file_info.revision,
        repo_type=file_info.repo_type,
    )

    bucket, key = _parse_s3_path(file_info.s3_key)

    if file_info.size >= MULTIPART_THRESHOLD:
        await _multipart_transfer(
            file_info,
            download_url,
            bucket,
            key,
            s3_client,
            http_session,
            config,
            progress,
        )
    else:
        await _simple_transfer(
            file_info,
            download_url,
            bucket,
            key,
            s3_client,
            http_session,
            config,
            progress,
        )


async def stream_repo_to_s3(
    repo_id: str,
    s3_dest: str,
    revision: str = "main",
    repo_type: RepoType = "model",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    concurrency: int = 4,
    chunk_size_mb: int = 8,
    dry_run: bool = False,
    endpoint_url: str | None = None,
    hf_token: str | None = None,
) -> None:
    """Stream HF repo to S3 using pure asyncio."""
    include_patterns = include_patterns or ["*"]
    exclude_patterns = exclude_patterns or []
    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")
    token = hf_token or os.environ.get("HF_TOKEN")

    chunk_size = max(chunk_size_mb * 1024 * 1024, MIN_PART_SIZE)

    config = TransferConfig(
        concurrency=concurrency,
        chunk_size=chunk_size,
        endpoint_url=endpoint,
        hf_token=token,
    )

    files = list_repo_files(
        repo_id, revision, repo_type, s3_dest, include_patterns, exclude_patterns
    )

    if not files:
        print("No files found matching the specified patterns.")
        return

    total_size = sum(f.size for f in files)
    print(f"Found {len(files)} files, total size: {total_size / (1024**3):.2f} GB")

    if dry_run:
        for f in files:
            print(f"  {f.path_in_repo} -> {f.s3_key} ({f.size / (1024**2):.1f} MB)")
        return

    progress = ProgressTracker(total_size=total_size)
    for f in files:
        progress.add_file(f.path_in_repo, f.size)

    semaphore = asyncio.Semaphore(concurrency)

    async def transfer_with_semaphore(
        file_info: FileInfo,
        s3_client: "S3Client",
        http_session: aiohttp.ClientSession,
    ) -> FileInfo | BaseException:
        async with semaphore:
            try:
                await transfer_single_file(
                    file_info, s3_client, http_session, config, progress
                )
                return file_info
            except Exception as e:
                progress.complete_file(file_info.path_in_repo)
                return e

    async with create_s3_client(endpoint, max_connections=concurrency * 2) as s3_client:
        connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=60)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as http_session:
            async with progress:
                tasks = [
                    transfer_with_semaphore(f, s3_client, http_session) for f in files
                ]
                results = await asyncio.gather(*tasks)

    succeeded = [r for r in results if isinstance(r, FileInfo)]
    failed = [(f, r) for f, r in zip(files, results) if isinstance(r, BaseException)]

    if failed:
        print(f"\nFailed {len(failed)}/{len(files)} files:")
        for file_info, err in failed:
            print(f"  {file_info.path_in_repo}: {err}")

    if succeeded:
        print(
            f"Successfully transferred {len(succeeded)}/{len(files)} files to {s3_dest}"
        )
