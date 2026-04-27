class Logger:

    def __init__(self):
        self.allowed_keys = [
            'get_action', 'update', 'record', 'training_error',
            'num_iterations', 'run', 'mem_run'
        ]
        self.dict_lists = {}
        self.logging = True

    def updateLogs(self, key: str, val: int | float) -> None:
        if self.logging and key in self.allowed_keys:
            if self.hasKeyInLogs(key):
                self.dict_lists[key].append(val)
            else:
                self.dict_lists[key] = [val]

    def getLogs(self, key: str) -> list:
        if self.hasKeyInLogs(key):
            return self.dict_lists[key]
        else:
            return []
    
    def hasKeyInLogs(self, key: str) -> bool:
        return key in self.dict_lists
    
    def keyAllowed(self, key:str) -> bool:
        return key in self.allowed_keys
    
    def enableLogging(self) -> None:
        self.logging = True

    def disableLogging(self) -> None:
        self.logging = False

    def resetLogger(self) -> None:
        self.dict_lists = {}

    

    