from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_length=280, num_channels=2):
        super(CNNFeatureExtractor, self).__init__()
        self.input_length = input_length
        self.num_channels = num_channels

        # Le dimensioni per il reshape da 280 punti, come da Figura 2 del paper originale.
        # Questo assume che i 280 punti di un canale possano essere interpretati come 10 "sottocanali"
        # ciascuno lungo 28 punti per la Conv1D.
        self.input_reshape_channels_per_channel = 10
        self.input_reshape_length_per_channel = 28
        self.feature_output_length = 3 # Per allinearsi alla "3x1" del paper

        # Livelli CNN per un singolo canale
        self.conv1_c = nn.Conv1d(in_channels=self.input_reshape_channels_per_channel, out_channels=32, kernel_size=2, stride=1)
        self.relu1_c = nn.ReLU()
        self.maxpool1_c = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv2_c = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        self.relu2_c = nn.ReLU()
        self.maxpool2_c = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv3_c = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1)
        self.relu3_c = nn.ReLU()

        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.feature_output_length)

        # Calcola la dimensione delle feature estratte per un singolo canale
        self.features_per_channel = 128 * self.feature_output_length
        # Calcola la dimensione totale delle feature per un singolo battito (combinando tutti i canali)
        self.output_features_per_beat = self.features_per_channel * self.num_channels

    def _process_single_channel(self, x_single_channel):
        # x_single_channel: (batch_size, original_beat_length)
        # Reshape per l'elaborazione CNN (divide la sequenza lunga in "sottocanali" e "sottosequenze")
        x_reshaped = x_single_channel.view(-1, self.input_reshape_channels_per_channel, self.input_reshape_length_per_channel)

        x = self.relu1_c(self.conv1_c(x_reshaped))
        x = self.maxpool1_c(x)
        x = self.relu2_c(self.conv2_c(x))
        x = self.maxpool2_c(x)
        x = self.relu3_c(self.conv3_c(x))

        x = self.adaptive_pool(x)
        c = x.view(x.size(0), -1) # Appiattisce l'output in un vettore di feature per canale
        return c

    def forward(self, x):
        # x: (batch_size, num_channels, original_beat_length)
        # Questa funzione ora processa un singolo battito (con i suoi canali) per ogni elemento del batch.
        batch_size, num_channels, original_beat_length = x.shape # Unpacking corretto

        # Processa ogni canale separatamente
        channel_features = []
        for i in range(num_channels):
            features = self._process_single_channel(x[:, i, :]) # Seleziona il canale i-esimo
            channel_features.append(features)

        # Concatena le feature da tutti i canali
        combined_features = torch.cat(channel_features, dim=1) # (batch_size, self.output_features_per_beat)

        # Aggiungi una dimensione di lunghezza 1 per la "sequenza" per renderlo compatibile con ViT
        # che si aspetta (batch_size, sequence_length, features)
        combined_features = combined_features.unsqueeze(1) # (batch_size, 1, output_features_per_beat)

        return combined_features


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class PatchEmbedding(nn.Module):
    """
    Simula il patch embedding per i feature vector della CNN.
    """
    def __init__(self, in_features, embed_dim):
        super().__init__()
        # Se in_features != embed_dim, usiamo un linear layer per mappare.
        # Altrimenti, l'identità o un'operazione più complessa.
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)
        # Output: (batch_size, sequence_length, embed_dim)
        return self.projection(x)
    
class ViT1D_2(nn.Module):
    def __init__(
        self,
        signal_length: int,
        in_channels: int = 1,
        emb_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 256,
        classes_output_dim: int = 1,
        categories_output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        
        self.cnn_extractor = CNNFeatureExtractor(
            input_length=signal_length,
            num_channels=in_channels
        )
        
        self.cnn_output_features = self.cnn_extractor.output_features_per_beat
        self.seq_length = signal_length
        self.embed_dim = emb_dim

        
        self.patch_embedding = PatchEmbedding(in_features=self.cnn_output_features, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Using nn.Embedding for positional encoding for better flexibility
        self.pos_embed = nn.Embedding(self.seq_length + 1, self.embed_dim)
        self.pos_ids = torch.arange(self.seq_length + 1).unsqueeze(0)


        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_dim)
        self.head1 = nn.Linear(emb_dim, classes_output_dim)
        self.head2 = nn.Linear(emb_dim, categories_output_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = self.cnn_extractor(x)
        print(x.shape)
        B, N, _ = x.shape

        # 1. Patch Embedding
        x = self.patch_embedding(x) # (B, N, embed_dim)

        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 2, embed_dim)

        # 3. Positional Embedding
        pos_ids_batch = self.pos_ids.to(x.device).expand(B, -1) # (B, N+1)
        x = x + self.pos_embed(pos_ids_batch)
        
        x = self.dropout(x)
        
        # 4. Transformer Blocks
        x = self.encoder(x) 
        
        # 5. Normalization
        x = self.norm(x)
        
        
        cls_token = x[:, 0]
        x1 = self.head1(cls_token) 
        x2 = self.head2(cls_token) 
        return x1, x2