import llm

@llm.hookimpl
def register_models(register):
    register(Markov())

class Markov(llm.Model):
    model_id = "dust"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]