class Logger:
    """
    A class used by agent and environment objects in order to record information
    about themselves. This information can be used by the Evaluator class in order
    to produce plots and data and compare between different objects.
    """

    allowed_keys = set(
        ('get_action', 'update', 'record', 'training_error', 'history_actions',      #agents
        'num_iterations', 'num_states', 'run', 'mem_run')                            #environments
    )
    
    def __init__(self):
        """
        Initialises a dictionary of strings to lists and enables logging.
        The strings are the keys to the logs and the lists are the logs themselves.
        """
        self.dict_lists = {}
        self.logging = True

    def updateLogs(self, key: str, val: int | float) -> None:
        """
        Args:
            key: The key under which the logs are held
            val: The value to append to the end of the logs
            
        Updates the logs corresponding to the given key by appending a value
        """

        #Check if logging is enabled and if the key is in allowed_keys.
        if self.logging and self.keyAllowed(key):
            if self.hasKeyInLogs(key):
                self.dict_lists[key].append(val)
            else:
                self.dict_lists[key] = [val]

    def getLogs(self, key: str) -> list:
        """
        Args:
            key: The key under which the logs are held

        Get the logs corresponding to the provided key
        """

        if self.hasKeyInLogs(key):
            return self.dict_lists[key]
        else:
            return []
    
    def hasKeyInLogs(self, key: str) -> bool:
        """
        Args:
            key: A string against which the existence of a log is checked

        Checks if there is a log attributed to the key
        """

        return key in self.dict_lists
    
    def keyAllowed(self, key:str) -> bool:
        """
        Args:
            key: A string representing the key to a potential log
            
        Check whether the key provided can hold logs or not
        """
        
        return key in self.allowed_keys
    
    def enableLogging(self) -> None:
        self.logging = True

    def disableLogging(self) -> None:
        self.logging = False

    def resetLogger(self) -> None:
        self.dict_lists = {}

    @staticmethod
    def get_default_keys() -> set[str]:
        return set(
            ('get_action', 'update', 'record', 'training_error', 'history_actions',      #agents
            'num_iterations', 'num_states', 'run', 'mem_run')                            #environments
        )

    @classmethod
    def set_allowed_keys(cls, new_keys: list[str]):
        cls.allowed_keys = new_keys

    @classmethod
    def reset_allowed_keys(cls):
        cls.allowed_keys = cls.get_default_keys()

    

    