### run_edi.py
#!/usr/bin/env python3
"""
EDI Claims Processing System Entry Point
"""
import sys
import argparse
import logging
from pathlib import Path
import cProfile
import pstats
from contextlib import contextmanager

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import EDIProcessingSystem
from config.config_manager import ConfigurationManager
from utils.logging_config import setup_logging


@contextmanager
def profiling_context(enable_profiling=False, output_file="profile_output.prof"):
    """Context manager for optional profiling."""
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            profiler.dump_stats(output_file)
            
            # Print top functions
            stats = pstats.Stats(output_file)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
    else:
        yield None


def main():
    """Main entry point for the EDI Claims Processing System."""
    parser = argparse.ArgumentParser(description="EDI Claims Processing System")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--profile", 
        action="store_true",
        help="Enable performance profiling"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging(level=args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting EDI Claims Processing System")
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Profiling enabled: {args.profile}")
        
        # Load configuration
        config_manager = ConfigurationManager(args.config)
        config = config_manager.get_config()
        
        # Initialize and run system with optional profiling
        with profiling_context(args.profile):
            system = EDIProcessingSystem(config)
            system.run()
            
        logger.info("EDI Claims Processing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())