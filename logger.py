from collections import defaultdict

class Logger:

    def __init__(self):
        self.dict_lists = defaultdict(list)
        self.logging = True

    def updateLogs(self, key: str, val: int | float) -> None:
        if self.logging:
            self.dict_lists[key].append(val)

    def getLogs(self, key: str) -> list:
        return self.dict_lists[key]
    
    def hasKeyInLogs(self, key: str) -> bool:
        return key in self.dict_lists
    
    def enableLogging(self) -> None:
        self.logging = True

    def disableLogging(self) -> None:
        self.logging = False

    def resetLogger(self) -> None:
        self.dict_lists = defaultdict(list)

    

    