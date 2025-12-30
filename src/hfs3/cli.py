"""CLI entry point for hfs3."""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

import typer

from .transfer import stream_repo_to_s3

app = typer.Typer(
    name="hfs3",
    help="Stream HuggingFace repositories directly to S3.",
    add_completion=False,
)


@app.command()
def main(
    repo_id: Annotated[
        str,
        typer.Argument(help="HuggingFace repo ID (e.g., meta-llama/Llama-2-7b)"),
    ],
    s3_dest: Annotated[
        str,
        typer.Argument(help="S3 destination (e.g., s3://bucket/prefix)"),
    ],
    revision: Annotated[
        str,
        typer.Option("--revision", "-r", help="Git revision (branch/tag/commit)"),
    ] = "main",
    repo_type: Annotated[
        str,
        typer.Option("--repo-type", "-t", help="Repository type: model/dataset/space"),
    ] = "model",
    include: Annotated[
        Optional[list[str]],
        typer.Option("--include", "-i", help="Include glob patterns (can repeat)"),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option("--exclude", "-e", help="Exclude glob patterns (can repeat)"),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Number of concurrent transfers"),
    ] = 4,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Chunk size in MB"),
    ] = 8,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="List files without transferring"),
    ] = False,
    endpoint_url: Annotated[
        Optional[str],
        typer.Option("--endpoint-url", help="S3 endpoint URL (or set AWS_ENDPOINT_URL)"),
    ] = None,
) -> None:
    """Stream a HuggingFace repository to S3."""
    if not s3_dest.startswith("s3://"):
        raise typer.BadParameter("S3 destination must start with 's3://'")

    if repo_type not in ("model", "dataset", "space"):
        raise typer.BadParameter("repo-type must be one of: model, dataset, space")

    asyncio.run(
        stream_repo_to_s3(
            repo_id=repo_id,
            s3_dest=s3_dest,
            revision=revision,
            repo_type=repo_type,
            include_patterns=include,
            exclude_patterns=exclude,
            concurrency=concurrency,
            chunk_size_mb=chunk_size,
            dry_run=dry_run,
            endpoint_url=endpoint_url,
        )
    )


if __name__ == "__main__":
    app()
