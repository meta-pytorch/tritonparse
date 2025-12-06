# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Rich TUI interface for bisect operations.

This module provides a split-screen terminal UI for displaying bisect progress
and real-time command output. It gracefully falls back to plain text when
Rich is not available or when running in non-TTY environments.
"""

import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# Graceful fallback if rich not installed
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class BisectProgress:
    """
    Bisect progress state for UI display.

    Attributes:
        phase: Current phase name (e.g., "Triton Bisect").
        phase_number: Current phase number (1-4).
        total_phases: Total number of phases.
        current_commit: Currently testing commit hash.
        commits_tested: Number of commits tested so far.
        total_commits: Total commits to test (estimated).
        elapsed_seconds: Time elapsed since start.
        status_message: Additional status message.
        is_building: Whether currently building.
        is_testing: Whether currently running test.
    """

    phase: str = "Initializing"
    phase_number: int = 1
    total_phases: int = 4
    current_commit: Optional[str] = None
    commits_tested: int = 0
    total_commits: int = 0
    elapsed_seconds: float = 0.0
    status_message: Optional[str] = None
    is_building: bool = False
    is_testing: bool = False


class BisectUI:
    """
    Rich-based TUI for bisect operations.

    Provides a split-screen interface with:
    - Top panel: Progress information (phase, commit, progress, elapsed time)
    - Bottom panel: Scrolling command output (build/test logs)

    Automatically falls back to plain text output when:
    - Rich library is not installed
    - Running in non-TTY environment (CI, pipes)
    - Explicitly disabled via enabled=False

    Example:
        >>> ui = BisectUI()
        >>> with ui:
        ...     ui.update_progress(phase="Triton Bisect", phase_number=1)
        ...     ui.append_output("Building...")
        ...     # ... do work ...
        ...     ui.update_progress(commits_tested=5, total_commits=12)
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialize the TUI.

        Args:
            enabled: Whether to enable Rich TUI. If False or Rich unavailable,
                    falls back to plain text output.
        """
        # Check if we should use Rich TUI
        self._rich_enabled = (
            enabled
            and RICH_AVAILABLE
            and sys.stdout.isatty()
            and sys.stderr.isatty()
        )

        self.progress = BisectProgress()
        self.output_lines: List[str] = []
        self.max_output_lines = 100
        self.start_time: Optional[float] = None

        if self._rich_enabled:
            self._console = Console()
            self._layout = self._create_layout()
            self._live: Optional[Live] = None
        else:
            self._console = None
            self._layout = None
            self._live = None

    @property
    def is_tui_enabled(self) -> bool:
        """Check if Rich TUI is enabled."""
        return self._rich_enabled

    def _create_layout(self) -> "Layout":
        """Create the split-screen layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=10),
            Layout(name="output"),
        )
        return layout

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _render_progress_panel(self) -> "Panel":
        """Render the progress information panel."""
        p = self.progress

        text = Text()

        # Phase info
        text.append("Phase: ", style="bold")
        phase_style = "green" if p.phase_number < p.total_phases else "blue"
        text.append(f"{p.phase} ", style=phase_style)
        text.append(f"({p.phase_number}/{p.total_phases})\n", style="dim")

        # Current commit
        text.append("Commit: ", style="bold")
        if p.current_commit:
            commit_display = p.current_commit[:12] if len(p.current_commit) > 12 else p.current_commit
            text.append(f"{commit_display}\n", style="cyan")
        else:
            text.append("N/A\n", style="dim")

        # Progress
        text.append("Progress: ", style="bold")
        if p.total_commits > 0:
            pct = p.commits_tested / p.total_commits * 100
            text.append(f"{p.commits_tested}/{p.total_commits} ", style="yellow")
            text.append(f"({pct:.0f}%)\n", style="dim")
        else:
            text.append(f"{p.commits_tested} commits tested\n", style="yellow")

        # Elapsed time
        text.append("Elapsed: ", style="bold")
        text.append(f"{self._format_elapsed(p.elapsed_seconds)}\n", style="magenta")

        # Status indicators
        if p.is_building:
            text.append("\n⚙️  Building...", style="yellow")
        elif p.is_testing:
            text.append("\n🧪 Testing...", style="cyan")

        # Status message
        if p.status_message:
            text.append(f"\n{p.status_message}", style="dim italic")

        return Panel(
            text,
            title="[bold white]Bisect Progress[/bold white]",
            border_style="green",
        )

    def _render_output_panel(self) -> "Panel":
        """Render the scrolling output panel."""
        # Get last N lines that fit
        display_lines = self.output_lines[-50:]  # Show last 50 lines

        text = Text()
        for line in display_lines:
            # Truncate very long lines
            if len(line) > 200:
                line = line[:197] + "..."
            text.append(line + "\n")

        return Panel(
            text,
            title="[bold white]Output[/bold white]",
            border_style="blue",
        )

    def _update_layout(self) -> None:
        """Update the layout with current state."""
        if not self._rich_enabled or not self._layout:
            return

        self._layout["progress"].update(self._render_progress_panel())
        self._layout["output"].update(self._render_output_panel())

    def start(self) -> None:
        """Start the live display."""
        self.start_time = time.time()

        if not self._rich_enabled:
            return

        self._update_layout()
        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_progress(self, **kwargs) -> None:
        """
        Update progress information.

        Args:
            **kwargs: Fields to update on BisectProgress.
                     Valid fields: phase, phase_number, total_phases,
                     current_commit, commits_tested, total_commits,
                     elapsed_seconds, status_message, is_building, is_testing.
        """
        # Update elapsed time automatically
        if self.start_time and "elapsed_seconds" not in kwargs:
            kwargs["elapsed_seconds"] = time.time() - self.start_time

        # Update progress fields
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)

        # Update display
        if self._rich_enabled and self._live:
            self._update_layout()

    def append_output(self, line: str) -> None:
        """
        Append a line to the output panel.

        Args:
            line: Output line to append.
        """
        # Strip trailing newline if present
        line = line.rstrip("\n")

        self.output_lines.append(line)

        # Limit stored lines
        if len(self.output_lines) > self.max_output_lines:
            self.output_lines = self.output_lines[-self.max_output_lines:]

        if not self._rich_enabled:
            # Plain text fallback
            print(line)
        elif self._live:
            self._update_layout()

    def print_summary(self) -> None:
        """Print a final summary after TUI stops."""
        p = self.progress
        elapsed = self._format_elapsed(p.elapsed_seconds)

        print()
        print("=" * 60)
        print(f"Phase: {p.phase} ({p.phase_number}/{p.total_phases})")
        if p.current_commit:
            print(f"Last commit: {p.current_commit[:12]}")
        print(f"Commits tested: {p.commits_tested}")
        print(f"Total time: {elapsed}")
        print("=" * 60)

    def create_output_callback(self) -> Callable[[str], None]:
        """
        Create a callback function for streaming output.

        Returns:
            Callback that appends lines to the output panel.
        """
        def callback(line: str) -> None:
            self.append_output(line)

        return callback

    def __enter__(self) -> "BisectUI":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.stop()
        return False


def is_rich_available() -> bool:
    """Check if Rich library is available."""
    return RICH_AVAILABLE
