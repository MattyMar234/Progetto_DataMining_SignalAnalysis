from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_length=280, num_channels=2):
        super(CNNFeatureExtractor, self).__init__()
        assert input_length % 10 == 0, "Input length must be divisible by 10"
        
        self.input_length = input_length
        self.num_channels = num_channels
        self.patchSize = 10

        self.input_reshape_channels = int(self.input_length/self.patchSize) * self.num_channels

        self.convLayes = nn.Sequential(
            [
                nn.Conv1d(in_channels= self.input_reshape_channels, out_channels=32, kernel_size=2, stride=1, padding=1),
                 nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=1),
                
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),  
            ]
        )

    def forward(self, x):
        
        batch_size, _, _ = x.shape
        
        # Reshape to (batch_size, 10*num_channels, 28)
        x = x.view(batch_size, self.input_reshape_channels, self.patchSize)
        return self.convLayers(x)






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