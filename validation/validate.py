#!/usr/bin/env python3
"""Main Entry Point for SpectralReader End-to-End Automated Validation Framework."""

import os
import sys
import argparse

# Ensure UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure validation package directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config_loader import load_config
from validator import Validator
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="SpectralReader End-to-End Automated Backend Validation Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "validation_cases.yaml"),
        help="Path to validation cases YAML file (default: configs/validation_cases.yaml)"
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help="Override backend service base URL (e.g. http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override HTTP request timeout in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override report output directory (default: reports)"
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(SCRIPT_DIR, config_path))

    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Error loading configuration file:[/bold red] {e}")
        sys.exit(1)

    # CLI Overrides
    if args.backend_url:
        config.settings.backend_url = args.backend_url.rstrip("/")
    if args.timeout:
        config.settings.timeout_seconds = args.timeout
    if args.output_dir:
        config.settings.output_dir = args.output_dir

    # Resolve directories relative to validation folder
    pdfs_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "pdfs"))
    reports_dir = os.path.abspath(os.path.join(SCRIPT_DIR, config.settings.output_dir))

    validator = Validator(config=config, pdfs_dir=pdfs_dir, reports_dir=reports_dir)
    exit_code = validator.run_all()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
