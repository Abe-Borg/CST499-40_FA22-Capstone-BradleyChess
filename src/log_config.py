import logging
import functools
import time

logging.basicConfig(filename = 'app.log', filemode = 'w', format = '%(message)s', level = logging.DEBUG)
logger = logging.getLogger(__name__)


def log_execution_time(func):
    """A decorator that logs the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: A new function that logs the execution time of the decorated function.

    Example:
        @log_execution_time
        def my_function():
            # do something
            pass

    This decorator logs the execution time of a function using the Python logging module. It takes a function `func` as an argument and returns a new function `wrapper` that wraps `func` with additional code to log the execution time.

    The `functools.wraps` decorator is used to preserve the metadata of the original function `func`, such as its name, docstring, and signature. This is important because it allows the decorated function to be used in place of the original function without losing any of its original metadata.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f'{func.__name__}: {elapsed_time: .4f} seconds.')
        return result
    return wrapper


def log_execution_time_every_N(n = 10):
    """A decorator that logs the execution time of a function every N calls.

    Args:
        n (int, optional): The number of calls after which to log the execution time. Defaults to 1000.

    Returns:
        function: The actual decorator that takes a function as an argument and returns a new function that logs the execution time every N calls.

    Example:
        @log_execution_time_every_N(n=1000)
        def my_function():
            # do something
            pass

    This decorator takes an argument `n` that specifies the number of calls after which to log the execution time. It returns a decorator function `actual_decorator` that takes a function as an argument and returns a new function that logs the execution time every N calls.

    The `counter` variable is defined as a list to allow it to be modified within the `wrapper` function. The `wrapper` function increments the counter every time it is called, and if the counter is a multiple of `n`, it logs the execution time of the decorated function using the Python logging module.

    The `functools.wraps` decorator is used to preserve the metadata of the original function `func`, such as its name, docstring, and signature. This is important because it allows the decorated function to be used in place of the original function without losing any of its original metadata.
    """
    def actual_decorator(func):
        counter = [0]  # Using a list as a mutable object to hold the counter

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            counter[0] += 1  # Increment the counter

            if counter[0] % n == 0:
                elapsed_time = end_time - start_time
                logger.info(f"{func.__name__}: {elapsed_time:.4f} seconds (Call #{counter[0]})")

            return result

        return wrapper
    return actual_decorator
