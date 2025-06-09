from typing import Tuple
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implementa il Positional Encoding standard basato su seno e coseno.
    Aggiunge informazioni sulla posizione del token agli embedding.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crea la matrice di encoding posizionale
        # Shape: (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Registra la matrice come buffer (non sarà addestrata)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor input con shape (sequence_length, batch_size, d_model)
               o (batch_size, sequence_length, d_model) - dipendente dall'input del TransformerEncoder
               PyTorch standard usa (sequence_length, batch_size, feature_dim) per PE.
               Useremo (batch_size, sequence_length, feature_dim) e trasporremo se necessario.
        Returns:
            Tensor con shape uguale a x, con l'aggiunta del positional encoding.
        """
        # Assumiamo input shape (batch_size, sequence_length, d_model)
        batch_size, seq_len, d_model = x.shape

        x = x.permute(1, 0, 2) # -> (sequence_length, batch_size, d_model)
        x = x + self.pe[:seq_len]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x

class Transformer_BPM_Regressor(nn.Module):
    def __init__(
            self, 
            input_samples_num: int,
            conv_kernel_size: int, # Dimensione del kernel convoluzionale
            conv_stride: int,      # Passo della convoluzione
            in_channels: int,
            d_model: int, 
            head_num: int, 
            num_encoder_layers: int, 
            dim_feedforward: int, 
            dropout: float, 
        ):
        super(Transformer_BPM_Regressor, self).__init__()
        assert input_samples_num % conv_kernel_size == 0
        
        self.token_number = math.floor((input_samples_num  - conv_kernel_size) / conv_stride) + 1
        self.in_channels = in_channels
        
        
        self.conv1d_projection = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model, # L'output channels della conv diventa la dimensione del modello
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=0,
            # Non usiamo bias=False a meno di usare Batch Norm subito dopo
        )
        
        self.conv_activation = nn.ReLU()
        
        # CLS token: a learnable vector
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=self.token_number + 11)

        # 3. Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=head_num, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        # 4. Regression Head
        # Dopo l'encoder, abbiamo (batch_size, sequence_length, model_dim).
        # Aggreghiamo lungo la dimensione della sequenza e mappiamo a 1.
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1) # Output scalare per regressione
        )
        
   
    def forward(self, x):

        # 1. Proiezione dell'input
        x = self.conv1d_projection(x)
        x = self.conv_activation(x)
  
        # 2. Trasponi per il Positional Encoding e Transformer (batch_first=True)
        # Da (batch_size, d_model, new_sequence_length) a (batch_size, new_sequence_length, d_model)
        x = x.permute(0, 2, 1) # -> (batch_size, new_sequence_length, d_model)

        cls_token = self.cls_token.expand(x.size(0),-1,-1)
        x = torch.cat((cls_token, x), dim=1)
         
        # 2. Aggiunta del Positional Encoding
        x = self.positional_encoding(x) # -> (batch_size, sequence_length, model_dim)

        # 3. Passaggio attraverso il Transformer Encoder
        transformer_output = self.transformer_encoder(x) # -> (batch_size, sequence_length, model_dim)

        # # 4. Aggregazione e Regression Head
        # aggregated_output = torch.mean(transformer_output, dim=1) # -> (batch_size, model_dim)

        # Passa l'output aggregato attraverso lo strato di regressione
        cls_output = transformer_output[:, 0] # Only the [CLS] token output
        regression_output = self.regression_head(cls_output) # -> (batch_size, 1)

        return regression_output

class SimpleECGRegressor(nn.Module):
    def __init__(self, in_channels, input_length):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
    
    



class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, signal_length):
        super().__init__()
        assert signal_length % patch_size == 0, "La lunghezza del segnale deve essere divisibile per il patch size."
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

    def forward(self, x):
        # x: [B, C, L]
        x = self.proj(x)                      # [B, D, N]
        x = x.transpose(1, 2)                 # [B, N, D]
        B, N, D = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_token, x), dim=1)          # [B, N+1, D]
        x = x + self.pos_embed                       # Positional embedding
        return x

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


class ViT1D(nn.Module):
    def __init__(
        self,
        signal_length: int,
        patch_size: int = 16,
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
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, emb_dim, signal_length)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_dim)
        self.head1 = nn.Linear(emb_dim, classes_output_dim)
        self.head2 = nn.Linear(emb_dim, categories_output_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)    # [B, N+1, D]
        x = self.encoder(x)        # [B, N+1, D]
        x = self.norm(x)           # [B, N+1, D]
        cls_token = x[:, 0]        # [B, D]
        x1 = self.head1(cls_token) 
        x2 = self.head2(cls_token) 
        return x1, x2