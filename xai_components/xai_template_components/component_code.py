from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

@xai_component(color="red")
class HelloNewLibrary(Component):

    greet_str: InArg[str]


    def __init__(self):

        self.done = False
        self.greet_str = InArg.empty()

    def execute(self, ctx) -> None:

        import os 

        greet_str = self.greet_str.value if self.greet_str.value else "Xircuits Project!"
        print("Hello, " + greet_str)

        self.done = True