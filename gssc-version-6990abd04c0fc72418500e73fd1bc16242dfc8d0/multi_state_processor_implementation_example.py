import torch

def main(sleep_data_path, mat_path):
    # Load model directly from checkpoint
    model = torch.load('gssc_weights.ckpt', map_location='cpu')
    model.eval() 