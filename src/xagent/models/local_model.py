from transformers import pipeline

class LocalModel:
    def __init__(self,model_id:str):
        self.model_id = model_id
        self.model = pipeline(model="meta-llama/Llama-2-7b-hf",device="cuda:1")
         
    def reply(self,prompt:str):
        return self.model(prompt)

    