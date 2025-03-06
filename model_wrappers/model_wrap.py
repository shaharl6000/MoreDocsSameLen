

class Model:
    def __init__(self, model):
        self.model = model

    def get_max_window(self):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def batch(self, prompts, num_truncation_tokens):
        raise NotImplementedError("This method needs to be implemented by subclasses")


