"""Core streaming transfer logic."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from dataclasses import dataclass
from typing import Literal

from huggingface_hub import HfFileSystem
import s3fs

from .progress import ProgressTracker

RepoType = Literal["model", "dataset", "space"]

_REPO_PREFIXES: dict[RepoType, str] = {"dataset": "datasets", "space": "spaces", "model": ""}


@dataclass
class FileInfo:
    hf_path: str
    s3_key: str
    size: int


def get_hf_prefix(repo_id: str, repo_type: RepoType, revision: str) -> str:
    prefix = _REPO_PREFIXES[repo_type]
    return f"{prefix}/{repo_id}@{revision}" if prefix else f"{repo_id}@{revision}"


def matches_patterns(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def list_repo_files(
    hf: HfFileSystem,
    repo_id: str,
    revision: str,
    repo_type: RepoType,
    s3_dest: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[FileInfo]:
    prefix = get_hf_prefix(repo_id, repo_type, revision)
    files: list[FileInfo] = []

    for full_path, info in hf.find(prefix, detail=True).items():
        relative_path = full_path.removeprefix(f"{prefix}/")
        filename = relative_path.split("/")[-1]

        if not matches_patterns(filename, include_patterns):
            continue
        if exclude_patterns and matches_patterns(filename, exclude_patterns):
            continue

        s3_key = f"{s3_dest.rstrip('/')}/{relative_path}"
        files.append(
            FileInfo(
                hf_path=full_path,
                s3_key=s3_key,
                size=info.get("size", 0),
            )
        )

    return files


def stream_single_file(
    hf: HfFileSystem,
    s3: s3fs.S3FileSystem,
    file_info: FileInfo,
    chunk_size: int,
    progress: ProgressTracker | None,
) -> None:
    with hf.open(file_info.hf_path, "rb") as src:
        with s3.open(file_info.s3_key, "wb") as dst:
            while chunk := src.read(chunk_size):
                dst.write(chunk)
                if progress:
                    progress.update(file_info.hf_path, len(chunk))

    if progress:
        progress.complete_file(file_info.hf_path)


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
) -> None:
    include_patterns = include_patterns or ["*"]
    exclude_patterns = exclude_patterns or []

    hf = HfFileSystem()
    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")
    s3 = s3fs.S3FileSystem(endpoint_url=endpoint)

    files = list_repo_files(
        hf, repo_id, revision, repo_type, s3_dest, include_patterns, exclude_patterns
    )

    if not files:
        print("No files found matching the specified patterns.")
        return

    total_size = sum(f.size for f in files)
    print(f"Found {len(files)} files, total size: {total_size / (1024**3):.2f} GB")

    if dry_run:
        for f in files:
            print(f"  {f.hf_path} -> {f.s3_key} ({f.size / (1024**2):.1f} MB)")
        return

    semaphore = asyncio.Semaphore(concurrency)
    chunk_size = chunk_size_mb * 1024 * 1024

    async def transfer_with_limit(
        file_info: FileInfo, progress: ProgressTracker
    ) -> None:
        async with semaphore:
            await asyncio.to_thread(
                stream_single_file, hf, s3, file_info, chunk_size, progress
            )

    progress = ProgressTracker(total_size=total_size)
    for f in files:
        progress.add_file(f.hf_path, f.size)

    with progress:
        await asyncio.gather(*[transfer_with_limit(f, progress) for f in files])

    print(f"Successfully transferred {len(files)} files to {s3_dest}")
