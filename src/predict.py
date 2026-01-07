import torch
from src.model import StarClassifier

def load_model(model_path):
    model = StarClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_star(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return pred.item()
