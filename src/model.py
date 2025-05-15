import torch
import torch.nn as nn
import math

class TransformerRegressor(nn.Module):
    def __init__(
            self, 
            max_token: int,
            input_dim: int, 
            d_model: int, 
            head_num: int, 
            num_encoder_layers: int, 
            dim_feedforward: int, 
            dropout: float, 
            output_dim: int
        ):
        super(TransformerRegressor, self).__init__()
        
        self.positional_encoding = self._generate_positional_encoding(max_token, d_model)
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=head_num, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.regressor = nn.Linear(d_model, output_dim)

    def _generate_positional_encoding(self, max_token: int, d_model: int):
        
        pe = torch.zeros(max_token + 1, d_model) #max_token + 1 perchè tengo in considerazione il token <CLS>
        
        position = torch.arange(0, max_token, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Ensure seq_len does not exceed max_token
        if seq_len > self.positional_encoding.size(1) - 1:
            x = x[:, :self.positional_encoding.size(1) - 1, :]
            seq_len = self.positional_encoding.size(1) - 1
        
        # Add <CLS> token (initialized as zeros)
        cls_token = torch.zeros(batch_size, 1, x.size(2), device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # New shape: (batch_size, seq_len + 1, input_dim)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len + 1, :]
        
        # Transformer expects (seq_len, batch_size, d_model)
        #x = x.permute(1, 0, 2)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the <CLS> token representation for regression
        cls_representation = x[0, :, :]  # Shape: (batch_size, d_model)
        
        # Pass through regressor
        output = self.regressor(cls_representation)
        return output

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

    model = TransformerRegressor(max_token, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim)
    sample_input = torch.randn(32, 50, input_dim)  # Batch of 32, sequence length 50, input_dim 10
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (32, output_dim)