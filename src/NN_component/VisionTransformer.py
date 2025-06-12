import torch
import torch.nn as nn
import math


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_length=280, num_channels=2,embed_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        assert input_length % 10 == 0, "Input length must be divisible by 10"
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4"
        
        self.embed_dim = embed_dim
        self.embed_dim_2 = embed_dim // 2
        self.embed_dim_4 = embed_dim // 4
        self.input_length = input_length
        self.num_channels = num_channels
        self.patchSize = 10

        self.input_reshape_channels = int(self.input_length/self.patchSize) * self.num_channels

        self.convLayers = nn.Sequential(
            
            nn.Conv1d(in_channels= self.input_reshape_channels, out_channels=self.embed_dim_4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.embed_dim_4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output length: 10 -> 5
            
            nn.Conv1d(in_channels=self.embed_dim_4, out_channels=self.embed_dim_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.embed_dim_2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),# Output length: 5 -> 3
            
            nn.Conv1d(in_channels=self.embed_dim_2, out_channels=self.embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),  
        )

    def forward(self, x):
        
        batch_size, _, _ = x.shape
        
        # Reshape to (batch_size, 10*num_channels, 28)
        x = x.view(batch_size, self.input_reshape_channels, self.patchSize)
        return self.convLayers(x)


class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, L]
        x = self.proj(x)  # [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim pari
        pe[:, 1::2] = torch.cos(position * div_term)  # dim dispari
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        return x + self.pe[:, :x.size(1)]


class ViT1D(nn.Module):
    def __init__(
        self,
        signal_length: int,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert signal_length % patch_size == 0, "Signal length must be divisible by patch size."
        
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, embed_dim)
        num_patches = signal_length // patch_size
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_patches + 1)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x) -> torch.Tensor:
        # x: [B, C, L]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        B, N, E = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, embed_dim]
        x = self.pos_encoding(x)  # [B, N+1, embed_dim]

        x = x.permute(1, 0, 2)  # Transformer expects [N+1, B, embed_dim]
        x = self.transformer_encoder(x)  # [N+1, B, embed_dim]
        x = x[0]  # cls token output: [B, embed_dim]

        return self.output_head(x)  # [B, num_classes]
    
    
class ViT1D_2V(nn.Module):
    def __init__(
        self,
        signal_length: int,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        #[B, embed_dim, 3]
        self.feature_extractor = CNNFeatureExtractor(input_length=signal_length, num_channels=in_channels, embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(embed_dim, max_len= 4 + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x) -> torch.Tensor:
        # x: [B, C, L] (e.g., [B, 2, 280])

        # Estrai le feature usando la CNN
        # output_cnn_features: [B, cnn_embed_dim, num_patches] (e.g., [B, 128, 3])
        cnn_features = self.feature_extractor(x)

        # Prepara le feature per il Transformer
        # Transformer si aspetta [B, num_patches, embed_dim]
        x = cnn_features.permute(0, 2, 1) # [B, num_patches, cnn_embed_dim]
        B, N, _ = x.shape
        assert N == 3, f"Expected 3 patches, got {x.shape}"

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, cnn_embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, cnn_embed_dim]
        x = self.pos_encoding(x)  # [B, N+1, cnn_embed_dim]

        x = x.permute(1, 0, 2)  # Transformer expects [N+1, B, cnn_embed_dim]
        x = self.transformer_encoder(x)  # [N+1, B, cnn_embed_dim]
        x = x[0]  # cls token output: [B, cnn_embed_dim]

        return self.output_head(x)  # [B, num_classes]
    
    
class ViT1D_2V_CLASSES(ViT1D_2V):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class ViT1D_2V_CATEGORIES(ViT1D_2V):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)