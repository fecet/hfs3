"""Core streaming transfer logic using hf_xet for optimized downloads."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download, RepoFile
import s3fs
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .progress import ProgressTracker

RepoType = Literal["model", "dataset", "space"]


@dataclass
class FileInfo:
    repo_id: str
    path_in_repo: str
    s3_key: str
    size: int
    revision: str
    repo_type: RepoType


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((OSError, ConnectionError)),
    reraise=True,
)
def transfer_single_file(
    file_info: FileInfo,
    chunk_size: int,
    endpoint_url: str | None,
    progress: ProgressTracker | None,
) -> None:
    """Download file using hf_xet, then upload to S3."""
    # Download using hf_hub_download (leverages hf_xet for optimized transfers)
    local_path = hf_hub_download(
        repo_id=file_info.repo_id,
        filename=file_info.path_in_repo,
        revision=file_info.revision,
        repo_type=file_info.repo_type,
    )

    # Upload to S3
    s3 = s3fs.S3FileSystem(endpoint_url=endpoint_url)
    with open(local_path, "rb") as src:
        with s3.open(file_info.s3_key, "wb") as dst:
            while chunk := src.read(chunk_size):
                dst.write(chunk)
                if progress:
                    progress.update(file_info.path_in_repo, len(chunk))

    if progress:
        progress.complete_file(file_info.path_in_repo)


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
    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")

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

    chunk_size = chunk_size_mb * 1024 * 1024
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=concurrency)

    async def transfer_file(file_info: FileInfo) -> FileInfo | BaseException:
        try:
            await loop.run_in_executor(
                executor, transfer_single_file, file_info, chunk_size, endpoint, progress
            )
            return file_info
        except Exception as e:
            progress.complete_file(file_info.path_in_repo)
            return e

    progress = ProgressTracker(total_size=total_size)
    for f in files:
        progress.add_file(f.path_in_repo, f.size)

    with progress:
        results = await asyncio.gather(*[transfer_file(f) for f in files])

    executor.shutdown(wait=False)

    succeeded = [r for r in results if isinstance(r, FileInfo)]
    failed = [(f, r) for f, r in zip(files, results) if isinstance(r, BaseException)]

    if failed:
        print(f"\nFailed {len(failed)}/{len(files)} files:")
        for file_info, err in failed:
            print(f"  {file_info.path_in_repo}: {err}")

    if succeeded:
        print(f"Successfully transferred {len(succeeded)}/{len(files)} files to {s3_dest}")
