"""
AIR Command Line Interface

Provides the main entry point for the AIR command-line tool.
Supports various subcommands for inference, benchmarking, and configuration.

Usage:
    air --version
    air run --config config.yaml
    air benchmark --model llama-7b
"""

import argparse
import sys
from typing import Optional

from air import __version__
from air.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the AIR CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="air",
        description="Adaptive Inference Runtime - Dynamic model routing and optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  air --version              Show version information
  air run -c config.yaml     Run inference with configuration
  air benchmark -m llama-7b  Benchmark a model

For more information, visit: https://github.com/danielsilva010/Adaptive-Inference-Runtime
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output (WARNING level logging)",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available commands",
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run inference with specified configuration",
        description="Execute inference using AIR's adaptive runtime",
    )
    run_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML)",
    )
    run_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model identifier (overrides config)",
    )
    run_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input text or path to input file",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file",
    )

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks",
        description="Benchmark AIR performance with various configurations",
    )
    bench_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model identifier to benchmark",
    )
    bench_parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to benchmark (default: 100)",
    )
    bench_parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    bench_parser.add_argument(
        "--output-format",
        choices=["json", "csv", "table"],
        default="table",
        help="Output format for results (default: table)",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="Manage AIR configuration files",
    )
    config_parser.add_argument(
        "action",
        choices=["show", "validate", "generate"],
        help="Configuration action to perform",
    )
    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file",
    )

    return parser


def handle_run(args: argparse.Namespace) -> int:
    """
    Handle the 'run' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    logger.info(f"Loading configuration from: {args.config}")

    # TODO: Implement inference runtime
    logger.warning("Inference runtime not yet implemented")
    print(f"Would run inference with config: {args.config}")

    if args.model:
        print(f"Model override: {args.model}")
    if args.input:
        print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")

    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """
    Handle the 'benchmark' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    logger.info(f"Running benchmark for model: {args.model}")

    # TODO: Implement benchmarking
    logger.warning("Benchmarking not yet implemented")
    print(f"Would benchmark model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Output format: {args.output_format}")

    return 0


def handle_config(args: argparse.Namespace) -> int:
    """
    Handle the 'config' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    action = args.action

    if action == "show":
        if args.config:
            logger.info(f"Showing configuration: {args.config}")
            # TODO: Load and display config
            print(f"Would display config from: {args.config}")
        else:
            print("No configuration file specified")
            return 1

    elif action == "validate":
        if args.config:
            logger.info(f"Validating configuration: {args.config}")
            # TODO: Validate config
            print(f"Would validate config: {args.config}")
        else:
            print("No configuration file specified")
            return 1

    elif action == "generate":
        logger.info("Generating default configuration")
        # TODO: Generate default config
        print("Would generate default configuration")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entry point for the AIR CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging based on verbosity flags
    if args.verbose:
        setup_logging(level="DEBUG", colored=True)
    elif args.quiet:
        setup_logging(level="WARNING", colored=True)
    else:
        setup_logging(level="INFO", colored=True)

    # Handle commands
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return handle_run(args)
    elif args.command == "benchmark":
        return handle_benchmark(args)
    elif args.command == "config":
        return handle_config(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
