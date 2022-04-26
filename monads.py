from types import FunctionType


class Option:

    def __init__(self, func:FunctionType, working_condition:FunctionType=None) -> None:

        self.working_condition = lambda _ : True if working_condition is None else working_condition 
        self.logs = {}
        self.item = self.__operate(func)

    def perform(self, func:FunctionType):
        return self.__operate(lambda: func(self.item))

    def unwrap(self):
        return self.item

    def __bool__(self):
        return self.item is None

    def apply(self, func:FunctionType):

        self.item = self.__operate(lambda: func(self.item)) 
        if not self.item is None:
            
            if not self.working_condition(self.item):
                self.logs['conditions_not_met'] = self.unwrap()
                self.item = None

        return self

    def __operate(self, func:FunctionType):    

        try:
            return func()

        except Exception as e:
            self.logs[type(e)] = e.with_traceback(None)
            return None

    def __repr__(self) -> str:
        return f"object of type [{type(self.item)}]"         
