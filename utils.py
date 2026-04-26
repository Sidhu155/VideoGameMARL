import time


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

def time_func(func_name: str):
    """
    A decorator used to time functions and append this information to the objects logger.
    """
    
    def timed_func(func):
        def decorator(obj, *args, **kwargs):
            start = time.perf_counter_ns()
            return_val = func(obj, *args, **kwargs)
            end = time.perf_counter_ns()
            obj.logger.updateLogs(func_name, end - start)
            return return_val
        return decorator
    return timed_func