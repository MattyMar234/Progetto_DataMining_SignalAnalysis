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


# Example usage
if __name__ == "__main__":
    max_token = 500
    input_dim = 10
    d_model = 64
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 128
    dropout = 0.1
    output_dim = 1

    model = Transformer_BPM_Regressor(max_token, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim)
    sample_input = torch.randn(32, 50, input_dim)  # Batch of 32, sequence length 50, input_dim 10
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (32, output_dim)