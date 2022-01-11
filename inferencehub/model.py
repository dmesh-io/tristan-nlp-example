
from transformers import AutoTokenizer, AutoModelForCausalLM
class ModelWrapper(nn.Module):

    def __init__(self, weights_path:str, parameters: Dict, device: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")
        self.add_module('real_module', self.model)
        self.tokenizer = AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")

    def forward(self, x: any, runtime_parameters: dict) -> torch.tensor:
        input_ids = self.tokenizer.encode(x, return_tensors="pt")
        generated_ids = self.model.generate(input_ids, max_length=500)
        generated_sequence = self.tokenizer.decode(generated_ids[0])
        return generated_sequence

#We don't have the model in the repo, so we need to download it
def get_model(weights_path: str = None, map_location="cpu",
              model_initialization_parameters: Dict = None) -> torch.nn.Module:
    model = ModelWrapper(weights_path, model_initialization_parameters, map_location)
    return model
    


        
