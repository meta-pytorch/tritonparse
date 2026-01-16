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
from typing import List, Optional

# Graceful fallback if rich not installed
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


@dataclass
class BisectProgress:
    """
    Bisect progress state for UI display.

    Attributes:
        phase: Current phase name (e.g., "Triton Bisect").
        phase_number: Current phase number.
        total_phases: Total number of phases (depends on mode).
        current_commit: Currently testing commit hash.
        commits_tested: Number of commits tested so far.
        steps_remaining: Estimated steps remaining (from git bisect output).
        elapsed_seconds: Time elapsed since start.
        status_message: Additional status message.
        is_building: Whether currently building.
        is_testing: Whether currently running test.
        log_dir: Directory containing log files.
        log_file: Main log file name.
        command_log: Command log file name.
        range_start_index: Pair test specific - track the starting index
            when range filter is applied.
    """

    phase: str = "Initializing"
    phase_number: int = 1
    total_phases: int = 1  # Default to 1, CLI will set correct value based on mode
    current_commit: Optional[str] = None
    commits_tested: int = 0
    steps_remaining: Optional[int] = None
    elapsed_seconds: float = 0.0
    status_message: Optional[str] = None
    is_building: bool = False
    is_testing: bool = False
    log_dir: Optional[str] = None
    log_file: Optional[str] = None
    command_log: Optional[str] = None
    # Pair test specific: track the starting index when range filter is applied
    range_start_index: Optional[int] = None


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
        # Track why TUI was disabled for status message
        self._disabled_reason: Optional[str] = None

        # Check if we should use Rich TUI
        if not enabled:
            self._rich_enabled = False
            self._disabled_reason = "disabled by --no-tui flag"
        elif not RICH_AVAILABLE:
            self._rich_enabled = False
            self._disabled_reason = "rich library not installed (pip install rich)"
        elif not sys.stdout.isatty() or not sys.stderr.isatty():
            self._rich_enabled = False
            self._disabled_reason = "not running in a TTY (e.g., piped output or CI)"
        else:
            self._rich_enabled = True

        # Initialize progress state
        self.progress = BisectProgress()
        self.output_lines: List[str] = []
        self.max_output_lines = 100
        self.start_time: Optional[float] = None

        # Initialize Rich components (only if enabled)
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

    @property
    def disabled_reason(self) -> Optional[str]:
        """Get the reason TUI was disabled, or None if enabled."""
        return self._disabled_reason

    def get_tui_status_message(self) -> str:
        """Get a human-readable message about TUI status."""
        if self._rich_enabled:
            return "Rich TUI enabled"
        else:
            return f"Rich TUI disabled: {self._disabled_reason}"

    def _create_layout(self) -> "Layout":
        """
        Create the split-screen layout.

        Returns:
            Layout with progress panel (top) and output panel (bottom).
        """
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=6),
            Layout(name="output"),
        )
        return layout

    def _render_progress_panel(self) -> "Panel":
        """Render the progress information panel."""
        p = self.progress

        # Auto-update elapsed time on each render
        if self.start_time:
            p.elapsed_seconds = time.time() - self.start_time

        text = Text()

        # Line 1: Core status (Phase, Commit, Progress, Elapsed) - all in one line
        text.append("Phase: ", style="bold")
        text.append(f"{p.phase} ", style="green bold")
        text.append(f"({p.phase_number}/{p.total_phases})  ", style="green")

        text.append("Commit: ", style="bold")
        if p.current_commit:
            commit_display = (
                p.current_commit[:9] if len(p.current_commit) > 9 else p.current_commit
            )
            text.append(f"{commit_display}  ", style="cyan")
        else:
            text.append("N/A  ", style="dim")

        text.append("Progress: ", style="bold")
        progress_text = f"{p.commits_tested} tested"
        if p.steps_remaining is not None:
            progress_text += f", ~{p.steps_remaining} steps left"
        text.append(progress_text + "  ", style="yellow")

        text.append("Elapsed: ", style="bold")
        text.append(f"{_format_elapsed(p.elapsed_seconds)}\n", style="magenta")

        # Line 2-3: Log files
        if p.log_dir:
            text.append("📁 Logs: ", style="bold")
            text.append(f"{p.log_dir}\n", style="dim")
            log_files = []
            if p.log_file:
                log_files.append(p.log_file)
            if p.command_log:
                log_files.append(p.command_log)
            if log_files:
                text.append("   └─ ", style="dim")
                text.append(", ".join(log_files) + "\n", style="bright_black")

        # Line 4: Status indicator + last result (combined)
        status_parts = []
        if p.is_building:
            status_parts.append(("⚙️  Building...", "yellow"))
        elif p.is_testing:
            status_parts.append(("🧪 Testing...", "cyan"))

        if p.status_message:
            status_parts.append((p.status_message, "bright_black italic"))

        if status_parts:
            for i, (msg, style) in enumerate(status_parts):
                if i > 0:
                    text.append("  ", style="default")
                text.append(msg, style=style)

        return Panel(
            text,
            title="[bold bright_green]Bisect Progress[/bold bright_green]",
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
            title="[bold bright_cyan]Output[/bold bright_cyan]",
            border_style="blue",
        )

    def _update_layout(self) -> None:
        """Update the layout with current state."""
        if not self._rich_enabled or not self._layout:
            return

        self._layout["progress"].update(self._render_progress_panel())
        self._layout["output"].update(self._render_output_panel())
