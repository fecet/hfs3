"""Progress tracking and display using Rich."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


@dataclass
class ProgressTracker:
    """Async-compatible progress tracker.

    In pure asyncio (single-threaded), locks are not needed.
    Rich's Progress is thread-safe internally.
    """

    total_size: int
    progress: Progress = field(init=False)
    overall_task: TaskID = field(init=False)
    file_tasks: dict[str, TaskID] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        self.overall_task = self.progress.add_task(
            "[cyan]Overall", total=self.total_size
        )

    async def __aenter__(self) -> Self:
        self.progress.__enter__()
        return self

    async def __aexit__(self, *args) -> None:
        self.progress.__exit__(*args)

    def add_file(self, file_path: str, size: int) -> None:
        """Add a file to track. Called before transfer starts."""
        short_name = file_path.split("/")[-1]
        if len(short_name) > 30:
            short_name = short_name[:27] + "..."
        task_id = self.progress.add_task(short_name, total=size)
        self.file_tasks[file_path] = task_id

    def update(self, file_path: str, bytes_transferred: int) -> None:
        """Update progress for a file."""
        if file_path in self.file_tasks:
            self.progress.update(
                self.file_tasks[file_path], advance=bytes_transferred
            )
        self.progress.update(self.overall_task, advance=bytes_transferred)

    def complete_file(self, file_path: str) -> None:
        """Mark file as complete and remove from display."""
        if file_path in self.file_tasks:
            self.progress.remove_task(self.file_tasks[file_path])
            del self.file_tasks[file_path]
