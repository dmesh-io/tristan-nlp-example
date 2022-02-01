def preprocess_function(input_payload, model):
    return model.get('tokenizer').encode(input_payload, return_tensors="pt")


def postprocess_function(output, model):
    generated_sequence = model.get('tokenizer').decode(output[0])
    return generated_sequence
