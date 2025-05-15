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
            max_token: int,
            in_channels: int,
            d_model: int, 
            head_num: int, 
            num_encoder_layers: int, 
            dim_feedforward: int, 
            dropout: float, 
        ):
        super(Transformer_BPM_Regressor, self).__init__()
        self.max_token = max_token
        self.in_channels = in_channels
        
        # 1. Proiezione dell'input raw nella dimensione del modello
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_token + 10)

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
        self.regression_head = nn.Linear(d_model, 1) # Output scalare per regressione


   
    def forward(self, x):
  
        if x.shape[-1] != self.max_token:
            raise ValueError(f"Input sequence length must be {self.max_token}, but got {x.shape[1]}. Shape: {x.shape}")

        # Trasponi le dimensioni: [batch, channels, elements] -> [batch, elements, channels]
        x = x.permute(0, 2, 1) # -> (batch_size, sequence_length, in_channels)

        
        # 1. Proiezione dell'input
        x = self.input_projection(x) # -> (batch_size, sequence_length, model_dim)

        # 2. Aggiunta del Positional Encoding
        x = self.positional_encoding(x) # -> (batch_size, sequence_length, model_dim)

        # 3. Passaggio attraverso il Transformer Encoder
        transformer_output = self.transformer_encoder(x) # -> (batch_size, sequence_length, model_dim)

        # 4. Aggregazione e Regression Head
        aggregated_output = torch.mean(transformer_output, dim=1) # -> (batch_size, model_dim)

        # Passa l'output aggregato attraverso lo strato di regressione
        regression_output = self.regression_head(aggregated_output) # -> (batch_size, 1)

        return regression_output

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