import llm

@llm.hookimpl
def register_models(register):
    register(Dust())

class Dust(llm.Model):
    model_id = "dust"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]