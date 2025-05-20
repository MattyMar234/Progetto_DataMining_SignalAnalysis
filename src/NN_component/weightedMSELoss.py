import torch
import torch.nn as nn
import numpy as np

class WeightedMSELoss(nn.Module):
    def __init__(self, bpm_bins, weights):
        super(WeightedMSELoss, self).__init__()
        self.bpm_bins = torch.tensor(bpm_bins.detach().clone(), dtype=torch.float32) # Punti di separazione dei bucket
        self.weights = torch.tensor(weights.detach().clone(), dtype=torch.float32)   # Pesi corrispondenti ai bucket


    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape, f"I due tensori hanno shape differente. predictions: {predictions.shape}, targets: {targets.shape}"
        
        # Determina a quale bucket appartiene ogni target BPM
        # bucket_indices conterrà l'indice del bucket per ogni target
        bucket_indices = torch.bucketize(targets, self.bpm_bins, right=True) # right=True per intervalli aperti a destra (es. [0, 50), [50, 100))

        # Assicurati che gli indici non superino i limiti dei pesi
        bucket_indices = torch.clamp(bucket_indices, 0, len(self.weights) - 1)


        # Calcola l'MSE standard
        mse = (predictions - targets) ** 2
        
        # Applica i pesi all'MSE
        weighted_mse = mse * self.weights[bucket_indices]
        
        #print(f"mse: {mse} bucket_indices: {bucket_indices} weighted_mse: {weighted_mse}")

        # Ritorna la media della loss ponderata per il batch
        return torch.mean(weighted_mse)