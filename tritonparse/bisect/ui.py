# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Rich TUI interface for bisect operations.

This module provides a split-screen terminal UI for displaying bisect progress
and real-time command output. It gracefully falls back to plain text when
Rich is not available or when running in non-TTY environments.
"""

import re
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

# GitHub commit URL mapping - for generating clickable links
GITHUB_COMMIT_URLS = {
    "triton": "https://github.com/triton-lang/triton/commit/",
    "llvm": "https://github.com/llvm/llvm-project/commit/",
}

# Display order - from upper layer to lower layer (call stack order)
CULPRIT_DISPLAY_ORDER = ["triton", "llvm"]

# Display names for each component
CULPRIT_DISPLAY_NAMES = {
    "triton": "Triton",
    "llvm": "LLVM",
}


class _LiveContent:
    """
    Wrapper that regenerates layout on each render.

    This class implements the Rich renderable protocol (__rich__) to ensure
    that the layout is regenerated on each Live refresh cycle. This enables
    automatic updating of dynamic content like elapsed time without requiring
    explicit calls to update the display.
    """

    def __init__(self, ui: "BisectUI") -> None:
        self._ui = ui

    def __rich__(self) -> "Layout":
        """
        Called by Rich Live on each refresh cycle.

        Returns:
            The updated layout with fresh elapsed time.
        """
        # Update elapsed time on each render
        if self._ui.start_time:
            self._ui.progress.elapsed_seconds = time.time() - self._ui.start_time

        # Rebuild layout with fresh data
        self._ui._update_layout()
        return self._ui._layout


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
        """Create the split-screen layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=6),
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
        text.append(f"{self._format_elapsed(p.elapsed_seconds)}\n", style="magenta")

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

    def start(self) -> None:
        """Start the live display."""
        self.start_time = time.time()

        if not self._rich_enabled:
            return

        self._update_layout()
        # Use _LiveContent wrapper to enable auto-refresh of elapsed time
        # The __rich__() method is called on each Live refresh cycle
        # refresh_per_second=1 is sufficient since elapsed time displays in seconds
        self._live = Live(
            _LiveContent(self),
            console=self._console,
            refresh_per_second=2,
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
            self.output_lines = self.output_lines[-self.max_output_lines :]

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
            Callback that appends lines to the output panel and
            parses progress information from the output.
        """

        def callback(line: str) -> None:
            self.append_output(line)
            self._parse_and_update_progress(line)

        return callback

    def _parse_and_update_progress(self, line: str) -> None:
        """
        Parse output line and update progress state.

        This method analyzes the output from bisect scripts to extract
        progress information and update the UI accordingly.

        Note: phase_number and total_phases are NOT set here - they should
        be set by the CLI based on the current mode (triton only, llvm only,
        or full workflow).

        Args:
            line: Output line to parse.
        """
        # Detect phase from script header (only update phase name, not phase_number)
        if "=== Triton Bisect Run ===" in line:
            self.update_progress(
                phase="Triton Bisect",
                is_building=False,
                is_testing=False,
            )
        elif "=== LLVM Bisect Run ===" in line:
            self.update_progress(
                phase="LLVM Bisect",
                is_building=False,
                is_testing=False,
            )

        # Detect current commit (from script output)
        # Matches: "Commit: abc123", "LLVM Commit: abc123", "Triton Commit: abc123"
        match = re.search(r"^(?:LLVM |Triton )?Commit: ([a-fA-F0-9]{7,40})", line)
        if match:
            self.update_progress(current_commit=match.group(1))

        # Detect short commit (from script output)
        # Matches: "Short: abc123def", "LLVM Short: abc123def", "Triton Short: abc123def"
        match = re.search(r"^(?:LLVM |Triton )?Short: ([a-fA-F0-9]{7,12})", line)
        if match:
            self.update_progress(current_commit=match.group(1))

        # Detect building status
        if "Building Triton" in line or "Building..." in line:
            self.update_progress(
                is_building=True,
                is_testing=False,
            )

        # Detect testing status
        if "Running test" in line:
            self.update_progress(
                is_building=False,
                is_testing=True,
            )

        # Detect test result (commit completed)
        if "✅ Passed" in line:
            self.update_progress(
                is_building=False,
                is_testing=False,
                commits_tested=self.progress.commits_tested + 1,
                status_message="Last: Passed ✅",
            )
        elif "❌ Failed" in line:
            self.update_progress(
                is_building=False,
                is_testing=False,
                commits_tested=self.progress.commits_tested + 1,
                status_message="Last: Failed ❌",
            )

        # Detect git bisect progress from "roughly N steps"
        # Format: "Bisecting: 284 revisions left to test after this (roughly 8 steps)"
        match = re.search(r"\(roughly (\d+) steps?\)", line)
        if match:
            steps_remaining = int(match.group(1))
            self.update_progress(steps_remaining=steps_remaining)

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


def _generate_title(culprits: Optional[dict]) -> str:
    """
    Generate panel title based on culprits dictionary.

    Args:
        culprits: Dictionary mapping component name to culprit commit hash.

    Returns:
        Title string for the summary panel.
    """
    if not culprits:
        return "Bisect Failed"

    if len(culprits) == 1:
        key = list(culprits.keys())[0]
        return f"{CULPRIT_DISPLAY_NAMES.get(key, key)} Bisect Result"
    else:
        return "Bisect Result (Full Workflow)"


def print_final_summary(
    culprits: Optional[dict] = None,
    llvm_bump_info: Optional[object] = None,
    error_msg: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    command_log: Optional[str] = None,
) -> None:
    """
    Print final bisect summary with Rich formatting (or plain text fallback).

    Args:
        culprits: Dictionary mapping component name to culprit commit hash.
                  Keys can be: 'triton', 'llvm' (extensible).
                  Example: {'triton': 'abc123', 'llvm': 'def456'}
        llvm_bump_info: LLVMBumpInfo object if Triton culprit is an LLVM bump.
        error_msg: Error message if bisect failed.
        log_dir: Directory containing log files.
        log_file: Main log file path (shown on error).
        command_log: Command log file path (shown on error).
    """
    success = culprits is not None and len(culprits) > 0

    # Generate title based on culprits
    title = _generate_title(culprits)

    # Check if Triton culprit is an LLVM bump
    is_llvm_bump = (
        llvm_bump_info is not None
        and hasattr(llvm_bump_info, "is_llvm_bump")
        and llvm_bump_info.is_llvm_bump
    )

    if RICH_AVAILABLE:
        _print_final_summary_rich(
            culprits,
            is_llvm_bump,
            llvm_bump_info,
            error_msg,
            log_dir,
            log_file,
            command_log,
            title,
            success,
        )
    else:
        _print_final_summary_plain(
            culprits,
            is_llvm_bump,
            llvm_bump_info,
            error_msg,
            log_dir,
            log_file,
            command_log,
            title,
            success,
        )


def _print_final_summary_rich(
    culprits: Optional[dict],
    is_llvm_bump: bool,
    llvm_bump_info: Optional[object],
    error_msg: Optional[str],
    log_dir: Optional[str],
    log_file: Optional[str],
    command_log: Optional[str],
    title: str,
    success: bool,
) -> None:
    """Print final summary with Rich formatting."""
    console = Console()
    text = Text()

    if success:
        text.append("✅ Bisect Completed\n\n", style="bold green")

        # Print all culprits in order
        for key in CULPRIT_DISPLAY_ORDER:
            if culprits and key in culprits:
                name = CULPRIT_DISPLAY_NAMES.get(key, key)
                commit = culprits[key]
                url = GITHUB_COMMIT_URLS.get(key, "") + commit

                text.append(f"🔍 {name} culprit: ", style="bold")
                text.append(f"{commit}\n", style="cyan bold")
                if url:
                    text.append("   🔗 Link:\n", style="bold")
                    text.append(f"      {url}\n", style="blue underline")

                # Show LLVM bump info after Triton culprit (if applicable)
                if key == "triton":
                    if is_llvm_bump:
                        text.append(
                            "\n⚠️  This Triton commit is an LLVM bump!\n",
                            style="yellow bold",
                        )
                        text.append("   LLVM: ", style="bold")
                        text.append(f"{llvm_bump_info.old_hash}", style="dim")
                        text.append(" → ", style="bold")
                        text.append(f"{llvm_bump_info.new_hash}\n", style="yellow")
                    elif culprits and len(culprits) == 1:
                        # Only show "not an LLVM bump" for Triton-only mode
                        text.append(
                            "\nℹ️  This is a regular Triton commit (not an LLVM bump)\n",
                            style="dim italic",
                        )

                text.append("\n")

        if log_dir:
            text.append("📁 Log directory: ", style="bold")
            text.append(f"{log_dir}", style="dim")
    else:
        text.append("❌ Bisect Failed\n\n", style="bold red")
        if error_msg:
            text.append(f"{error_msg}", style="red")
        # Show specific log files for easier debugging
        if command_log:
            text.append("\n\n📄 Check command log for details:\n", style="bold")
            text.append(f"   {command_log}", style="yellow")
        if log_file:
            text.append("\n📄 Module log: ", style="bold")
            text.append(f"{log_file}", style="dim")
        if log_dir and not command_log and not log_file:
            text.append("\n\n📁 Log directory: ", style="bold")
            text.append(f"{log_dir}", style="dim")

    panel = Panel(
        text,
        title=f"[bold]{title}[/bold]",
        border_style="green" if success else "red",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def _print_final_summary_plain(
    culprits: Optional[dict],
    is_llvm_bump: bool,
    llvm_bump_info: Optional[object],
    error_msg: Optional[str],
    log_dir: Optional[str],
    log_file: Optional[str],
    command_log: Optional[str],
    title: str,
    success: bool,
) -> None:
    """Print final summary with plain text (fallback when Rich not available)."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    if success:
        print("✅ Bisect Completed")

        # Print all culprits in order
        for key in CULPRIT_DISPLAY_ORDER:
            if culprits and key in culprits:
                name = CULPRIT_DISPLAY_NAMES.get(key, key)
                commit = culprits[key]
                url = GITHUB_COMMIT_URLS.get(key, "") + commit

                print(f"🔍 {name} culprit: {commit}")
                if url:
                    print(f"   🔗 {url}")

                # Show LLVM bump info after Triton culprit (if applicable)
                if key == "triton":
                    if is_llvm_bump:
                        print(f"⚠️  This Triton commit is an LLVM bump!")
                        print(
                            f"   LLVM: {llvm_bump_info.old_hash} → {llvm_bump_info.new_hash}"
                        )
                    elif culprits and len(culprits) == 1:
                        # Only show "not an LLVM bump" for Triton-only mode
                        print("ℹ️  This is a regular Triton commit (not an LLVM bump)")

        if log_dir:
            print(f"📁 Log directory: {log_dir}")
    else:
        print(f"❌ Bisect Failed: {error_msg}")
        # Show specific log files for easier debugging
        if command_log:
            print("📄 Check command log for details:")
            print(f"   {command_log}")
        if log_file:
            print(f"📄 Module log: {log_file}")
        if log_dir and not command_log and not log_file:
            print(f"📁 Log directory: {log_dir}")

    print("=" * 60)
