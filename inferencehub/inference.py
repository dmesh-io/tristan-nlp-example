import json


def preprocess_function(input_payload, model):
    input_decoded = json.loads(input_payload)
    return model.get('tokenizer').encode(input_decoded, return_tensors="pt")


def postprocess_function(output, model):
    generated_sequence = model.get('tokenizer').decode(output[0])
    return generated_sequence
