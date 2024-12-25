import torch

def infer_with_no_grad(model, group1, group2, discrete):
    with torch.no_grad():  # Use the context manager correctly
        output = model(group1, group2, discrete)
    return output
