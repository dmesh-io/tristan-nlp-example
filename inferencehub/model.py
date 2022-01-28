from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM


# We don't have the model in the repo or on inferencehub, so we need to download it
def get_model(weights_path: str = None) -> Dict:
    model = AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")
    tokenizer = AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")
    return {'model': model, 'tokenizer': tokenizer}
