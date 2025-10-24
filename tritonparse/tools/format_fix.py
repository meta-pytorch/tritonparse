#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Format fix script for tritonparse project.

This script runs all linter tools to format and fix code issues:
- usort: Import sorting
- ruff: Linting only
- black: Code formatting

Usage:
    python -m tritonparse.tools.format_fix [options]

Options:
    --check-only    Only check for issues, don't fix them
    --verbose       Verbose output
    --help          Show this help message
"""

import argparse
import subprocess
import sys


def run_command(cmd: list[str], verbose: bool = False) -> bool:
    """Run a command and return success status."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if verbose:
                print(f"Command failed with return code {result.returncode}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            return False

        if verbose and result.stdout:
            print(result.stdout)

        return True
    except Exception as e:
        if verbose:
            print(f"Error running command: {e}")
        return False


def run_usort(check_only: bool = False, verbose: bool = False) -> bool:
    """Run usort for import sorting."""
    cmd = ["usort"]

    if check_only:
        cmd.extend(["check", "."])
    else:
        cmd.extend(["format", "."])

    return run_command(cmd, verbose)


def run_ruff_check(check_only: bool = False, verbose: bool = False) -> bool:
    """Run ruff for linting only."""
    cmd = ["ruff", "check", "."]

    if check_only:
        cmd.append("--diff")
    else:
        cmd.append("--fix")

    return run_command(cmd, verbose)


def run_black(check_only: bool = False, verbose: bool = False) -> bool:
    """Run black for code formatting."""
    cmd = ["black"]

    if check_only:
        cmd.extend(["--check", "--diff", "."])
    else:
        cmd.append(".")

    return run_command(cmd, verbose)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Format fix script for tritonparse project",
        epilog="""
Examples:
    # Fix all formatting issues
    python -m tritonparse.tools.format_fix
    
    # Check for issues without fixing
    python -m tritonparse.tools.format_fix --check-only
    
    # Verbose output
    python -m tritonparse.tools.format_fix --verbose
        """,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for issues, don't fix them",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run formatters on the entire project
    success = True

    # 1. Run usort for import sorting
    print("Running usort for import sorting...")
    if not run_usort(args.check_only, args.verbose):
        print("❌ usort failed")
        success = False
    else:
        print("✅ usort completed")

    # 2. Run ruff for linting only
    print("Running ruff for linting...")
    if not run_ruff_check(args.check_only, args.verbose):
        print("❌ ruff linting failed")
        success = False
    else:
        print("✅ ruff linting completed")

    # 3. Run black for code formatting
    print("Running black for code formatting...")
    if not run_black(args.check_only, args.verbose):
        print("❌ black failed")
        success = False
    else:
        print("✅ black completed")

    if success:
        print("\n🎉 All formatting tools completed successfully!")
        return 0
    else:
        print("\n❌ Some formatting tools failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
