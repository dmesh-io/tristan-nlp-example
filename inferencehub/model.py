from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class ModelWrapper(GPT2LMHeadModel):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")

    def generate(self, x: dict, input_parameters: dict):
        return self.model.generate(x, max_length=input_parameters['max_length'], temperature=input_parameters['temperature'])


# We don't have the model in the repo or on inferencehub, so we need to download it
def get_model(weights_path: str = None) -> Dict:
    model = ModelWrapper()
    tokenizer = AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")
    return {'model': model, 'tokenizer': tokenizer}
