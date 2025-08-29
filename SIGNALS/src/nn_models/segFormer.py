import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SegFormer(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=19, 
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 dropout=0.1,
                 decoder_dim=256):
        super().__init__()
        
        # Encoder gerarchico
        self.encoder = HierarchicalTransformerEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            dropout=dropout
        )
        
        # Decoder MLP leggero
        self.decoder = MLPSegmentationDecoder(
            embed_dims=embed_dims,
            num_classes=num_classes,
            decoder_dim=decoder_dim
        )
    
    def forward(self, x):
        # Encoder: estrae feature a diverse scale
        features = self.encoder(x)
        
        # Decoder: fonde le feature e produce la segmentazione
        seg_map = self.decoder(features)
        
        return seg_map

class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 dropout=0.1):
        super().__init__()
        
        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims
        
        # Blocchi per ogni stage
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            # Primo stage: patch embedding con stride 4
            if i == 0:
                patch_embed = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=embed_dims[i],
                    kernel_size=7,
                    stride=4,
                    padding=3
                )
                norm = nn.LayerNorm(embed_dims[i])
            else:
                # Stage successivi: riduzione risoluzione con stride 2
                patch_embed = nn.Sequential(
                    nn.Conv2d(embed_dims[i-1], embed_dims[i], 
                             kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dims[i])
                )
                norm = nn.LayerNorm(embed_dims[i])
            
            # Transformer blocks
            transformer_blocks = nn.ModuleList()
            for _ in range(depths[i]):
                transformer_blocks.append(
                    TransformerEncoderLayer(
                        d_model=embed_dims[i],
                        nhead=num_heads[i],
                        dim_feedforward=embed_dims[i] * mlp_ratios[i],
                        dropout=dropout,
                        activation='gelu',
                        batch_first=True
                    )
                )
            
            stage = Stage(
                patch_embed=patch_embed,
                norm=norm,
                transformer_blocks=transformer_blocks,
                downsample=(i != 0)  # Downsampling per tutti tranne il primo stage
            )
            self.stages.append(stage)
    
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class Stage(nn.Module):
    def __init__(self, patch_embed, norm, transformer_blocks, downsample=False):
        super().__init__()
        self.patch_embed = patch_embed
        self.norm = norm
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.downsample = downsample
    
    def forward(self, x):
        # Patch embedding
        if self.downsample:
            # Stage successivi: input 4D (B, H, W, C)
            B, H_prev, W_prev, C_prev = x.shape
            
            # Permuta per convoluzione: (B, C_prev, H_prev, W_prev)
            x = x.permute(0, 3, 1, 2)
            
            # Applica convoluzione per riduzione risoluzione
            x = self.patch_embed(x)  # (B, embed_dim, H_prev//2, W_prev//2)
            
            # Calcola nuove dimensioni spaziali
            H_new = x.shape[2]
            W_new = x.shape[3]
            
            # Permuta per Transformer: (B, H_new, W_new, embed_dim)
            x = x.permute(0, 2, 3, 1)
            
            # Appiattisci per Transformer: (B, H_new*W_new, embed_dim)
            x = x.flatten(1, 2)
        else:
            # Primo stage: input 4D (B, C, H, W)
            B, C, H, W = x.shape
            
            # Applica convoluzione per patch embedding
            x = self.patch_embed(x)  # (B, embed_dim, H//4, W//4)
            
            # Calcola nuove dimensioni spaziali
            H_new = x.shape[2]
            W_new = x.shape[3]
            
            # Appiattisci per Transformer: (B, H_new*W_new, embed_dim)
            x = x.flatten(2).transpose(1, 2)
        
        # Applica normalizzazione
        x = self.norm(x)
        
        # Applica transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Riporta a formato 4D (B, H_new, W_new, embed_dim)
        x = x.reshape(B, H_new, W_new, -1)
        
        return x

class MLPSegmentationDecoder(nn.Module):
    def __init__(self, embed_dims, num_classes, decoder_dim=256):
        super().__init__()
        
        # Layer per fondere le feature multi-scale
        self.linear_fuse = nn.ModuleList()
        for dim in embed_dims:
            self.linear_fuse.append(
                nn.Conv2d(dim, decoder_dim, kernel_size=1)
            )
        
        # Layer finale per la segmentazione
        self.linear_pred = nn.Conv2d(
            decoder_dim, num_classes, kernel_size=1
        )
        
        # Upsampling per allineare le risoluzioni
        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True
        )
    
    def forward(self, features):
        # Features da tutti gli stage
        f1, f2, f3, f4 = features
        
        # Converti in formato (B, C, H, W)
        f1 = f1.permute(0, 3, 1, 2)
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)
        
        # Applica convoluzioni 1x1 per ridurre la dimensione
        f1 = self.linear_fuse[0](f1)
        f2 = self.linear_fuse[1](f2)
        f3 = self.linear_fuse[2](f3)
        f4 = self.linear_fuse[3](f4)
        
        # Upsample tutte le feature alla risoluzione più alta (f1)
        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)
        
        # Somma le feature
        fused = f1 + f2 + f3 + f4
        
        # Predizione finale
        seg_map = self.linear_pred(fused)
        
        # Upsample alla risoluzione originale (256x256)
        seg_map = self.upsample(seg_map)
        
        return seg_map
