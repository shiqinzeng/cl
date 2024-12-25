import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class EnhancedGroupedMLP(nn.Module):
    def __init__(self, group_dims, discrete_dim, combined_dim, dropout_prob=0.5, num_heads=4):
        super(EnhancedGroupedMLP, self).__init__()

        # Group-specific MLPs with residual connections
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(128, combined_dim)
            )
            for dim in group_dims
        ])

        # Projection layers for residual connection (align input and output dims)
        self.group_projections = nn.ModuleList([
            nn.Linear(dim, combined_dim) for dim in group_dims
        ])

        # Discrete feature MLP with deeper layers and residual connection
        self.discrete_proj = nn.Linear(discrete_dim, combined_dim)  # Projection layer for discrete features
        self.discrete_mlp = nn.Sequential(
            nn.Linear(discrete_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, combined_dim),
            nn.LayerNorm(combined_dim),
            nn.GELU()
        )

        # Multi-Head Attention Mechanism with post-attention transformation
        self.multihead_attention = nn.MultiheadAttention(embed_dim=combined_dim, num_heads=num_heads, batch_first=True)
        self.attn_post_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, combined_dim)
        )

        # Final MLP with more layers for classification
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2)  # Final output layer
        )

    def forward(self, group_features, discrete_features):
        # Process each group of features with residual connections
        group_outputs = []
        for mlp, proj, features in zip(self.group_mlps, self.group_projections, group_features):
            residual = proj(features)  # Project input to match combined_dim
            processed = mlp(features)
            group_outputs.append(processed + residual)  # Add residual

        # Process discrete features with residual connection
        projected_discrete = self.discrete_proj(discrete_features)  # Project discrete features
        discrete_output = self.discrete_mlp(discrete_features) + projected_discrete

        # Stack group outputs and apply multi-head attention
        group_outputs_tensor = torch.stack(group_outputs, dim=1)  # Shape: (batch, num_groups, combined_dim)
        attn_output, _ = self.multihead_attention(discrete_output.unsqueeze(1), group_outputs_tensor, group_outputs_tensor)

        # Post-process attention output
        attended_features = self.attn_post_proj(attn_output.squeeze(1))  # Shape: (batch, combined_dim)

        # Combine attended features and discrete output
        combined = torch.cat([attended_features, discrete_output], dim=-1)  # Shape: (batch, combined_dim * 2)

        # Final classification
        return self.final_mlp(combined)


# Model Integration
class BoostedMLP(nn.Module):
    def __init__(self, base_model, group_dims, discrete_dim, combined_dim, num_boosting_steps=4, dropout_prob=0.3, num_heads=4):
        super(BoostedMLP, self).__init__()
        self.num_boosting_steps = num_boosting_steps

        # Initialize multiple models
        self.models = nn.ModuleList([
            base_model(group_dims, discrete_dim, combined_dim, dropout_prob, num_heads)
            for _ in range(num_boosting_steps)
        ])

        # Model Weights
        self.model_weights = nn.Parameter(torch.ones(num_boosting_steps) / num_boosting_steps)

    def forward(self, group_features1, group_features2, discrete_features):
        group_features = [group_features1, group_features2]
        predictions = 0
        for i, model in enumerate(self.models):
            logits = model(group_features, discrete_features)
            weighted_logits = self.model_weights[i] * logits
            predictions += weighted_logits
        return predictions
