import torch.nn as nn

class TabularAttentionClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dims):
        super(TabularAttentionClassifier, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attention_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Create attention, linear, and normalization layers based on hidden_dims
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.attention_layers.append(nn.MultiheadAttention(embed_dim, num_heads))
            self.fc_layers.append(nn.Linear(current_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim  # Update current_dim for the next layer
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # x = self.embedding(x)  # (batch_size, input_dim) -> (batch_size, embed_dim)
        x = x.unsqueeze(0)  # Add sequence length dimension: (1, batch_size, embed_dim)

        for attn_layer, fc_layer, layer_norm in zip(self.attention_layers, self.fc_layers, self.layer_norms):
            attn_output, _ = attn_layer(x, x, x)  # Multi-head attention
            x = attn_output + x  # Residual connection
            x = x.permute(1, 0, 2)  # (1, batch_size, dim) -> (batch_size, 1, dim)
            x = fc_layer(x)  # Apply the linear layer
            x = x.permute(1, 0, 2)  # (batch_size, 1, dim) -> (1, batch_size, dim)
            x = layer_norm(x)  # Layer normalization
            x = nn.ReLU()(x)  # ReLU activation

        x = x.squeeze(0)  # Remove sequence length dimension: (batch_size, dim)
        x = self.output_layer(x)  # Final output layer

        return x