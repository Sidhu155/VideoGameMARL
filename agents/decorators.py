import time

def assert_agent_set_up(func):
    def decorator(obj, *args, **kwargs):
        if obj.set_up_bool:
            return func(obj, *args, **kwargs)
        else:
            raise Exception("Agent has not been set up!")
    return decorator

def time_func(func_name):
    def timed_func(func):
        def decorator(obj, *args, **kwargs):
            start = time.perf_counter()
            return_val = func(obj, *args, **kwargs)
            end = time.perf_counter()
            obj.logger[func_name].append(end - start)
            return return_val
        return decorator
    return timed_func