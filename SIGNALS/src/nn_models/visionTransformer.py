import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

class PatchEmbedding1D(nn.Module):
    def __init__(self, signal_length, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = signal_length // patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x
   

class CNNFeatureExtractor2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=128):
        super(CNNFeatureExtractor2D, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4"

        self.embed_dim = embed_dim
        self.embed_dim_2 = embed_dim // 2
        self.embed_dim_4 = embed_dim // 4
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Patch Embedding (tipo ViT)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.embed_dim_4,  # primo livello ridotto
            kernel_size=patch_size,
            stride=patch_size
        )

        # Numero di patch
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        # CNN layers su feature map ridotte
        self.convLayers = nn.Sequential(
            nn.Conv2d(self.embed_dim_4, self.embed_dim_4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim_4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(self.embed_dim_4, self.embed_dim_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(self.embed_dim_2, self.embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        B = x.size(0)

        # Patch embedding iniziale
        x = self.proj(x)  # (B, E//4, H/P, W/P)

        # Passaggio attraverso CNN
        x = self.convLayers(x)  # (B, E, H', W')

        # Flatten per concatenare con cls_token
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, E)

        # Aggiunta posizioni
        x = x + self.pos_embed[:, :x.size(1), :]  # adattato al numero di token

        return x



class VisionTransformer1(nn.Module):
    """
    Vision Transformer (ViT) con:
      - Patch embedding apprendibile via Conv2d (kernel/stride configurabili)
      - Token [CLS] + positional embeddings apprendibili
      - TransformerEncoder di PyTorch
      - Predizione basata sul token [CLS]

    Parametri principali:
      img_size:         lato dell'immagine (assunta quadrata), es. 256
      in_channels:      canali in input (spettrogrammi spesso 1, ma configurabile)
      num_classes:      numero di classi da predire
      embed_dim:        dimensione dei token/embedding
      depth:            numero di layer (blocchi) del Transformer
      n_heads:          numero di teste di attenzione
      mlp_ratio:        rapporto tra dim. feedforward e embed_dim (dim_ff = embed_dim * mlp_ratio)
      patch_kernel:     kernel della Conv2d per l'encoder dei patch (può essere int o tuple)
      patch_stride:     stride della Conv2d per l'encoder dei patch (int o tuple)
      patch_padding:    padding della Conv2d per l'encoder dei patch
      dropout:          dropout globale (attn+ff useranno quello del layer)
      attn_dropout:     (non usato direttamente: il TransformerEncoderLayer usa 'dropout' unico)
      norm_first:       se True usa pre-norm nei layer Transformer (PyTorch >= 1.12/2.x)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 5,
        embed_dim: int = 384,
        depth: int = 8,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # --- Transformer Encoder di PyTorch ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, S, E)
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

class VisionTransformer1_1D(nn.Module):
    """
    Vision Transformer per segnali 1D con:
      - Patch embedding apprendibile via Conv1d
      - Token [CLS] + positional embeddings apprendibili
      - TransformerEncoder di PyTorch
      - Predizione basata sul token [CLS]

    Parametri principali:
      signal_length:    lunghezza del segnale 1D, es. 3600
      patch_size:       dimensione del patch
      in_channels:      canali in input (1 per segnali ECG)
      num_classes:      numero di classi da predire
      embed_dim:        dimensione dei token/embedding
      depth:            numero di layer (blocchi) del Transformer
      n_heads:          numero di teste di attenzione
      mlp_ratio:        rapporto tra dim. feedforward e embed_dim
      norm_first:       se True usa pre-norm nei layer Transformer
      dropout:          dropout globale
    """
    def __init__(
        self,
        signal_length: int = 3600,
        patch_size: int = 36,
        in_channels: int = 1,
        num_classes: int = 5,
        embed_dim: int = 256,
        depth: int = 6,
        n_heads: int = 4,
        mlp_ratio: float = 3.0,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        assert signal_length % patch_size == 0, "Signal length must be divisible by patch size"
        
        self.patch_embed = PatchEmbedding1D(signal_length, patch_size, in_channels, embed_dim)
        
        # --- Transformer Encoder di PyTorch ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, S, E)
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assicurati che l'input abbia la forma corretta [batch_size, 1, signal_length]
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, signal_length]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, signal_length]
            
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)


class VisionTransformer2(nn.Module):
    """
    Vision Transformer (ViT) con:
      - Patch embedding apprendibile via Conv2d (kernel/stride configurabili)
      - Token [CLS] + positional embeddings apprendibili
      - TransformerEncoder di PyTorch
      - Predizione basata sul token [CLS]

    Parametri principali:
      img_size:         lato dell'immagine (assunta quadrata), es. 256
      in_channels:      canali in input (spettrogrammi spesso 1, ma configurabile)
      num_classes:      numero di classi da predire
      embed_dim:        dimensione dei token/embedding
      depth:            numero di layer (blocchi) del Transformer
      n_heads:          numero di teste di attenzione
      mlp_ratio:        rapporto tra dim. feedforward e embed_dim (dim_ff = embed_dim * mlp_ratio)
      patch_kernel:     kernel della Conv2d per l'encoder dei patch (può essere int o tuple)
      patch_stride:     stride della Conv2d per l'encoder dei patch (int o tuple)
      patch_padding:    padding della Conv2d per l'encoder dei patch
      dropout:          dropout globale (attn+ff useranno quello del layer)
      attn_dropout:     (non usato direttamente: il TransformerEncoderLayer usa 'dropout' unico)
      norm_first:       se True usa pre-norm nei layer Transformer (PyTorch >= 1.12/2.x)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 5,
        embed_dim: int = 384,
        depth: int = 8,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = CNNFeatureExtractor2D(img_size, patch_size, in_channels)
        
        # --- Transformer Encoder di PyTorch ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, S, E)
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)
    
    
    
class ViT1(VisionTransformer1):
    def __init__(self, 
            img_size: int = 256, 
            patch_size: int = 16, 
            in_channels: int = 1, 
            num_classes: int = 5, 
        ):
        super().__init__(
            img_size, 
            patch_size, 
            in_channels, 
            num_classes, 
            embed_dim=256, 
            depth=6, 
            n_heads=4, 
            mlp_ratio=3, 
            norm_first = True, 
            dropout = 0.1
        )
        
class ViT1_1D(VisionTransformer1_1D):
    def __init__(self, 
            signal_length: int = 3600, 
            patch_size: int = 72, 
            in_channels: int = 1, 
            num_classes: int = 5, 
        ):
        super().__init__(
            signal_length, 
            patch_size, 
            in_channels, 
            num_classes, 
            embed_dim=256, 
            depth=6, 
            n_heads=4, 
            mlp_ratio=3, 
            norm_first = True, 
            dropout = 0.1
        )
              
class ViT2(VisionTransformer1):
    def __init__(self, 
            img_size: int = 256, 
            patch_size: int = 16, 
            in_channels: int = 1, 
            num_classes: int = 5, 
        ):
        super().__init__(
            img_size, 
            patch_size, 
            in_channels, 
            num_classes, 
            embed_dim=256, 
            depth=6, 
            n_heads=16, 
            mlp_ratio=3, 
            norm_first = True, 
            dropout = 0.1
        )
        
        
