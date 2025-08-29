import cProfile
import functools
import io
import logging
import os
import pstats

logger = logging.getLogger(__name__)


def profile(func=None, *, filter_module=None):
    """Decorator to profile a function and save results for gprof2dot visualization.

    Saves to: /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/finbarrt/${GIT_COMMIT}_<func_name>.pstats

    Args:
        func: The function to profile (when used without arguments)
        filter_module: Optional module name to filter profiling results (e.g., 'open_instruct.vllm_utils3')

    Usage:
        @profile
        def my_function():
            # Your code here
            pass

        # Or with module filtering:
        @profile(filter_module='open_instruct.vllm_utils3')
        def my_function():
            # Your code here
            pass

        Then visualize with:
        gprof2dot -f pstats <output_file>.pstats | dot -Tpng -o output.png
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            git_commit = os.environ.get("GIT_COMMIT", "unknown")
            # Keep the suffix in filename to distinguish different profiling runs
            suffix = f"_filter_{filter_module.replace('.', '_')}" if filter_module else ""
            # Use /tmp for local testing, /weka path on the cluster
            base_path = "/tmp" if os.path.exists("/tmp") and not os.path.exists("/weka") else "/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/finbarrt"
            profile_path = f"{base_path}/{git_commit}_{func.__name__}{suffix}.pstats"

            profiler = cProfile.Profile()
            profiler.enable()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()

                # Ensure directory exists
                os.makedirs(os.path.dirname(profile_path), exist_ok=True)

                # Always save full stats for proper analysis with gprof2dot
                profiler.dump_stats(profile_path)

                if filter_module:
                    # Log filtered summary showing only functions from the target module
                    # but keeping cumulative times intact
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                    # Print only functions from the target module, sorted by cumulative time
                    ps.print_stats(f".*{filter_module.replace('.', '/')}.*")
                    logger.info(f"\nProfile for {func.__name__} (filtered to {filter_module}, sorted by cumulative time):")
                    logger.info(s.getvalue())
                else:
                    # Log summary to console (optional)
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                    ps.print_stats(10)  # Print top 10 functions
                    logger.info(f"\nProfile for {func.__name__}:")
                    logger.info(s.getvalue())

                logger.info(f"Profile saved to {profile_path}")

        return wrapper

    # Handle both @profile and @profile(filter_module='...')
    if func is None:
        return decorator
    else:
        return decorator(func)


# Example usage
if __name__ == "__main__":

    @profile("example.pstats")
    def example_function():
        import time

        total = 0
        for i in range(1000000):
            total += i**2
        time.sleep(0.1)
        return total

    result = example_function()
    print(f"Result: {result}")
