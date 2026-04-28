import time
import os
import psutil


def assert_agent_set_up(func):
    """
    A decorator to be used for agent objects to ensure that the set-up method has
    been called on them. This prevents methods like get_action and update from being called
    without defined action or observation spaces.
    """

    def decorator(obj, *args, **kwargs):
        if obj.set_up_bool:
            return func(obj, *args, **kwargs)
        else:
            raise Exception("Agent has not been set up!")
    return decorator

def time_func(log_name: str):
    """
    Args:
        log_name: The key under which to log the time taken by the function.

    A decorator used to time functions and append this information to the objects logger.
    """
    
    def timed_func(func):
        def decorator(obj, *args, **kwargs):
            #Check if object logging is turned on and whether the key is allowed by logger.
            #Otherwise only complete the function
            if obj.logger.logging and obj.logger.keyAllowed(log_name):
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
    """
    Args:
        log_name: The key under which to log the memory usage of the function
        
    A decorator used to log the average memory usage of a particular function.
    Instead of looking at the increase in memory from before and after, this instead logs the
    midpoint between the memory used before and memory used after. This is to better demonstrate
    persistent memory usage by tabular-based agents.
    """

    def memory_logger_func(func):
        def decorator(obj, *args, **kwargs):
            #Check if object logging is turned on and whether the key is allowed by logger.
            #Otherwise only complete the function
            if obj.logger.logging and obj.logger.keyAllowed(log_name):
                process = psutil.Process(os.getpid())
                
                before = process.memory_info().rss
                return_val = func(obj, *args, **kwargs)
                after = process.memory_info().rss
                
                #Divide the average memory by 1024**2 to convert into MB
                obj.logger.updateLogs(log_name, ((after + before)/2)/(1024 ** 2))
            else:
                return_val = func(obj, *args, **kwargs)
            return return_val
        return decorator
    return memory_logger_func