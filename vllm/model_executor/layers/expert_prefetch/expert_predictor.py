import os
from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertPredictorMLP(nn.Module):
    """
    ExpertPredictorMLP: Linear -> SiLU -> Linear
    
    Args:
        input_dim (int): Input dimension
        num_experts (int): Number of expert outputs
        hidden_dim (int): Hidden dimension (default: 2048)
    """
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 2048):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_experts)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class ExpertPredictorModel:
    """
    Wrapper class for easy inference with the trained expert prediction model (ExpertPredictorMLP).
    
    Supports:
    - Single sample and batch inference
    - Top-K expert extraction
    - GPU and CPU execution
    - Probability output
    - Direct weight loading from checkpoint
    """
    
    def __init__(self, weight_path: str, input_dim: int = 2880, num_experts: int= 128, 
                 hidden_dim: int = 2048, device: str = 'cuda', verbose: bool = False):
        """
        Initialize the predictor by loading weights.
        
        Args:
            weight_path (str): Path to the checkpoint file (.ckpt or .pt)
            input_dim (int): Input dimension
            num_experts (int): Number of experts
            hidden_dim (int): Hidden dimension (default: 2048)
            device (str): Device to use ('cuda' or 'cpu')
            verbose (bool): Whether to print loading information
        """
        self.weight_path = weight_path
        self.device = device
        self.verbose = verbose
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Validate checkpoint exists
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        # Initialize model
        if self.verbose:
            print(f"🔄 Initializing ExpertPredictorMLP model...")
            print(f"  - Input Dim: {input_dim}")
            print(f"  - Hidden Dim: {hidden_dim}")
            print(f"  - Num Experts: {num_experts}")
        
        self.model = ExpertPredictorMLP(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
        )
        self.model = self.model.to(device)
        
        # Load weights
        if self.verbose:
            print(f"🔄 Loading weights from: {weight_path}")
        
        self._load_weights(weight_path)
        self.model.eval()  # Set to evaluation mode
        
        if self.verbose:
            print(f"✓ Model loaded successfully")
            print(f"  - Device: {self.device}")
            print(f"  - Total Parameters: {self._count_parameters():,}")
    
    def _load_weights(self, weight_path: str):
        """Load weights from checkpoint (handles both PyTorch Lightning .ckpt and .pt formats)"""
        try:
            # Try loading as PyTorch Lightning checkpoint first
            if weight_path.endswith('.ckpt'):
                checkpoint = torch.load(weight_path, map_location=self.device)
                
                # Extract state dict from Lightning checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Remove 'model.' prefix if it exists (from Lightning wrapper)
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        # Handle keys like 'model.linear1.weight' -> 'linear1.weight'
                        if key.startswith('model.'):
                            new_key = key.replace('model.', '', 1)
                        else:
                            new_key = key
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                
                self.model.load_state_dict(state_dict)
            else:
                # Regular .pt file
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
            if self.verbose:
                print(f"✓ Weights loaded successfully")
        except RuntimeError as e:
            # Handle size mismatch errors with helpful message
            if "size mismatch" in str(e):
                if "linear1.weight" in str(e):
                    # Extract actual input dimension from checkpoint
                    print(f"\n❌ Error loading weights: {e}\n")
                    print(f"📌 This usually means your --input-dim parameter is wrong!")
                    print(f"\nTip: The checkpoint was trained with a different input dimension.")
                    print(f"Try using one of these common values:")
                    print(f"  - 2880 (GPT hidden_dim)")
                    print(f"  - 4096 (Standard LLM hidden_dim)")
                    print(f"\nExample:")
                    print(f"  python inference.py --weight-path {self.weight_path} --input-dim 2880")
            raise
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            raise
    
    def _count_parameters(self) -> int:
        """Count total number of parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def predict(self, x: torch.Tensor, return_probabilities: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict expert selection for input samples.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            return_probabilities (bool): If True, return both logits and probabilities
        
        Returns:
            torch.Tensor: Predicted logits [batch_size, num_experts]
            torch.Tensor (optional): Probabilities if return_probabilities=True
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor [batch_size, input_dim], got shape {x.shape}")
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {x.shape[1]}")
        
        # Move input to device
        x = x.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(x)
        
        if return_probabilities:
            probs = F.softmax(logits, dim=-1)
            return logits, probs
        
        return logits

    def predict_top_k(self, x: torch.Tensor, top_k: int = 4) -> Dict[str, torch.Tensor]:
        """
        Predict top-K experts for given input.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            top_k (int): Number of top experts to return
        
        Returns:
            dict: Contains 'indices', 'scores', and 'probabilities'
        """
        logits = self.predict(x)
        
        # Get top-K values and indices
        topk_scores, topk_indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        topk_probs = torch.gather(probs, -1, topk_indices)
        
        return {
            'indices': topk_indices,
            'scores': topk_scores,
            'probabilities': topk_probs,
            'logits': logits
        }

    def predict_batch(self, x: torch.Tensor, top_k: int = 4) -> Dict[str, torch.Tensor]:
        """
        Simple batch inference - just get top-K experts and scores.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            top_k (int): Number of top experts to return
        
        Returns:
            dict: {
                'indices': [batch_size, top_k],
                'scores': [batch_size, top_k]
            }
        """
        logits = self.predict(x)
        
        # Get top-K values and indices
        topk_scores, topk_indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        
        return {
            'indices': topk_indices,
            'scores': topk_scores
        }

    def measure_inference_time(self, x: torch.Tensor, num_runs: int = 10, return_stats: bool = True) -> Union[float, Dict]:
        """
        Measure average inference time.
        
        Args:
            x (torch.Tensor): Input tensor
            num_runs (int): Number of runs for averaging
            return_stats (bool): If True, return dict with statistics
        
        Returns:
            float or dict: Average time in milliseconds (or dict with detailed stats)
        """
        x = x.to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = self.model(x)
        
        # Measure
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        if return_stats:
            return {
                'avg_ms': np.mean(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'std_ms': np.std(times),
                'num_runs': num_runs,
                'batch_size': x.shape[0]
            }
        else:
            return np.mean(times)

