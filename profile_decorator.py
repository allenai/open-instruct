import cProfile
import pstats
import functools
import io
import logging
import os

logger = logging.getLogger(__name__)


def profile(func):
    """Decorator to profile a function and save results for gprof2dot visualization.
    
    Saves to: /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/finbarrt/${GIT_COMMIT}_<func_name>.pstats
    
    Usage:
        @profile
        def my_function():
            # Your code here
            pass
            
        Then visualize with:
        gprof2dot -f pstats <output_file>.pstats | dot -Tpng -o output.png
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        git_commit = os.environ.get("GIT_COMMIT", "unknown")
        profile_path = f"/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/finbarrt/{git_commit}_{func.__name__}.pstats"
        
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            
            # Save to file for gprof2dot
            profiler.dump_stats(profile_path)
            
            # Log summary to console (optional)
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(10)  # Print top 10 functions
            logger.info(f"\nProfile for {func.__name__}:")
            logger.info(s.getvalue())
            logger.info(f"Profile saved to {profile_path}")
            
    return wrapper


# Example usage
if __name__ == "__main__":
    @profile("example.pstats")
    def example_function():
        import time
        total = 0
        for i in range(1000000):
            total += i ** 2
        time.sleep(0.1)
        return total
    
    result = example_function()
    print(f"Result: {result}")