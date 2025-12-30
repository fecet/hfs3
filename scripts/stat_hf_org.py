#!/usr/bin/env python3
"""统计 HuggingFace 组织的所有数据集和大小"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_dataset_size(api: HfApi, repo_id: str) -> tuple[str, int, int]:
    """Get dataset size using repo_info with files_metadata."""
    try:
        info = api.dataset_info(repo_id, files_metadata=True)
        size = sum(
            sibling.size or 0
            for sibling in (info.siblings or [])
        )
        downloads = info.downloads or 0
        return (repo_id, size, downloads)
    except Exception as e:
        print(f"  Warning: {repo_id}: {e}")
        return (repo_id, 0, 0)


def main():
    org = "RoboCOIN"
    api = HfApi()

    print(f"Fetching datasets from {org}...")
    datasets = list(api.list_datasets(author=org))
    print(f"Found {len(datasets)} datasets\n")
    print("Fetching file sizes (this may take a while)...\n")

    results = []
    total_size = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(get_dataset_size, api, ds.id): ds.id
            for ds in datasets
        }
        for i, future in enumerate(as_completed(futures), 1):
            repo_id, size, downloads = future.result()
            results.append((repo_id, size, downloads))
            total_size += size
            print(f"\r[{i}/{len(datasets)}] Processed", end="", flush=True)

    print("\n\n" + "=" * 75)
    print(f"{'Dataset':<45} {'Size':>14} {'Downloads':>12}")
    print("=" * 75)

    for repo_id, size, downloads in sorted(results, key=lambda x: -x[1]):
        name = repo_id.replace(f"{org}/", "")
        print(f"{name:<45} {format_size(size):>14} {downloads:>12,}")

    print("=" * 75)
    print(f"{'Total':<45} {format_size(total_size):>14}")
    print(f"{'Dataset count':<45} {len(datasets):>14}")


if __name__ == "__main__":
    main()
