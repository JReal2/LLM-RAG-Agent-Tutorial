# https://medium.com/@wangdk93/moe-mixture-of-experts-for-llm-5d593e625e0b
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2):
        """
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            num_experts (int): Number of experts in the model.
            top_k (int): Number of experts to activate for each input.
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Experts: Each expert is a simple linear layer.
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        
        # Gating network: Maps input to the logits for experts.
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass through the MoE model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Compute gating scores (logits).
        gate_logits = self.gate(x)  # Shape: (batch_size, num_experts)
        
        # Convert logits to probabilities using softmax.
        gate_probs = F.softmax(gate_logits, dim=-1)  # Shape: (batch_size, num_experts)
        
        # Select the top-k experts for each input.
        top_k_values, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # Shape: (batch_size, top_k)
        
        # Normalize the top-k probabilities (optional, keeps weights sum to 1).
        top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)  # Shape: (batch_size, top_k)
        
        # Compute the output by combining the top-k experts' outputs.
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)  # Shape: (batch_size, num_experts, output_dim)
        
        # Select outputs of top-k experts and weight them by their probabilities.
        batch_size = x.size(0)
        output = torch.zeros(batch_size, expert_outputs.size(-1)).to(x.device)
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k]
                weight = top_k_values[b, k]
                output[b] += weight * expert_outputs[b, expert_idx]
        
        return output

# Example Usage
if __name__ == "__main__":
    batch_size = 8
    input_dim = 16
    output_dim = 8
    num_experts = 4
    top_k = 2

    model = MoE(input_dim, output_dim, num_experts, top_k)
    inputs = torch.randn(batch_size, input_dim)  # Random input data.
    outputs = model(inputs)

    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)