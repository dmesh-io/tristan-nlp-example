import json


def preprocess_function(input_payload, model):
    input_decoded = input_payload.read().decode("utf-8")
    return model.get('tokenizer').encode(input_decoded, return_tensors="pt")


def postprocess_function(output, model):
    generated_sequence = model.get('tokenizer').decode(output[0])
    return generated_sequence
