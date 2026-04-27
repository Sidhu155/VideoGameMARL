import time
import os
import psutil


def assert_agent_set_up(func):
    """
    A decorator to be used for agent objects to ensure that they are set-up with action and observation
    spaces before calling certain methods
    """

    def decorator(obj, *args, **kwargs):
        if obj.set_up_bool:
            return func(obj, *args, **kwargs)
        else:
            raise Exception("Agent has not been set up!")
    return decorator

def time_func(log_name: str):
    """
    A decorator used to time functions and append this information to the objects logger.
    """
    
    def timed_func(func):
        def decorator(obj, *args, **kwargs):
            if obj.logger.logging and obj.logger.hasKeyInLogs(log_name):
                start = time.perf_counter()
                return_val = func(obj, *args, **kwargs)
                end = time.perf_counter()
                obj.logger.updateLogs(log_name, end - start)
            else:
                return_val = func(obj, *args, **kwargs)
            return return_val
        return decorator
    return timed_func

def log_memory_func(log_name: str):

    def memory_logger_func(func):
        def decorator(obj, *args, **kwargs):
            if obj.logger.logging and obj.logger.hasKeyInLogs(log_name):
                process = psutil.Process(os.getpid())
                
                before = process.memory_info().rss
                return_val = func(obj, *args, **kwargs)
                after = process.memory_info().rss

                obj.logger.updateLogs(log_name, ((after + before)/2)/(1024 ** 2))
            else:
                return_val = func(obj, *args, **kwargs)
            return return_val
        return decorator
    return memory_logger_func