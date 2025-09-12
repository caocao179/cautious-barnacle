"""
Neural Network Models for Industrial Electrical Equipment Carbon Footprint Analysis

This module implements GRU-based models with multi-head self-attention for appliance 
identification and state classification. Python 3.13 compatible implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import warnings
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for sequence modeling.
    
    This implementation is optimized for Python 3.13 and PyTorch compatibility.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Dimension of the model (embedding size)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, 
                                   query: torch.Tensor, 
                                   key: torch.Tensor, 
                                   value: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, d_k]
            key: Key tensor [batch_size, num_heads, seq_len, d_k]
            value: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # Store residual connection
        residual = x
        
        # Linear projections
        query = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        # Add residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


class GRUWithAttention(nn.Module):
    """
    GRU-based model with multi-head self-attention for appliance identification.
    
    This model follows the framework: Data Embedding → GRU → Multi-head Self-Attention → Output States
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 num_classes: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        """
        Initialize GRU with attention model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size for GRU
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            num_classes: Number of output classes (appliance states)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate attention input size based on bidirectional setting
        attention_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            d_model=attention_input_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output layers for state identification
        self.classifier = nn.Sequential(
            nn.Linear(attention_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Global average pooling for sequence-level prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Dictionary containing:
                - 'logits': Classification logits [batch_size, num_classes]
                - 'embeddings': GRU embeddings [batch_size, seq_len, hidden_size * directions]
                - 'attention_output': Attention output [batch_size, seq_len, hidden_size * directions]
        """
        batch_size, seq_len, _ = x.size()
        
        # Step 1: Data Embedding
        embedded = self.input_embedding(x)  # [batch_size, seq_len, hidden_size]
        embedded = F.relu(embedded)
        
        # Step 2: GRU Layers
        gru_output, hidden_state = self.gru(embedded)  # [batch_size, seq_len, hidden_size * directions]
        
        # Step 3: Multi-head Self-Attention
        attention_output = self.attention(gru_output)  # [batch_size, seq_len, hidden_size * directions]
        
        # Step 4: Output States for State Identification
        # Global pooling to get sequence-level representation
        pooled = self.global_pool(attention_output.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_size * directions]
        
        # Classification
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        
        return {
            'logits': logits,
            'embeddings': gru_output,
            'attention_output': attention_output,
            'pooled_features': pooled
        }


class ApplianceDataset(Dataset):
    """
    PyTorch Dataset for appliance power consumption data.
    """
    
    def __init__(self, 
                 sequences: np.ndarray, 
                 targets: Optional[np.ndarray] = None,
                 transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences [num_samples, seq_len, num_features]
            targets: Target labels [num_samples] (optional)
            transform: Optional transform function
        """
        # Ensure proper data types
        self.sequences = torch.FloatTensor(sequences.astype(np.float32))
        self.targets = torch.LongTensor(targets.astype(np.int64)) if targets is not None else None
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sequence = self.sequences[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.targets is not None:
            return sequence, self.targets[idx]
        else:
            return sequence, None


class CarbonFootprintPredictor:
    """
    High-level interface for carbon footprint prediction using the GRU-attention model.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 num_classes: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize the carbon footprint predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size for GRU
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            num_classes: Number of output classes
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = GRUWithAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            if targets is None:
                continue
                
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            dataloader: Validation/test data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, targets in dataloader:
                if targets is None:
                    continue
                    
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                logits = outputs['logits']
                
                # Compute loss
                loss = self.criterion(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def fit(self, 
            train_sequences: np.ndarray,
            train_targets: np.ndarray,
            val_sequences: Optional[np.ndarray] = None,
            val_targets: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_sequences: Training sequences
            train_targets: Training targets
            val_sequences: Validation sequences (optional)
            val_targets: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Create datasets and dataloaders
        train_dataset = ApplianceDataset(train_sequences, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_sequences is not None and val_targets is not None:
            val_dataset = ApplianceDataset(val_sequences, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        return self.training_history
    
    def predict(self, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            sequences: Input sequences
            batch_size: Batch size for prediction
            
        Returns:
            Predicted class labels
        """
        dataset = ApplianceDataset(sequences)
        
        # Custom collate function to handle None targets
        def collate_fn(batch):
            sequences_batch = torch.stack([item[0] for item in batch])
            return sequences_batch, None
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for sequences_batch, _ in dataloader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                logits = outputs['logits']
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def get_embeddings(self, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from the model.
        
        Args:
            sequences: Input sequences
            batch_size: Batch size
            
        Returns:
            Feature embeddings
        """
        dataset = ApplianceDataset(sequences)
        
        # Custom collate function to handle None targets
        def collate_fn(batch):
            sequences_batch = torch.stack([item[0] for item in batch])
            return sequences_batch, None
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for sequences_batch, _ in dataloader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                batch_embeddings = outputs['pooled_features'].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.concatenate(embeddings, axis=0)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing GRU with Multi-head Self-Attention model...")
    
    # Generate sample data
    batch_size = 32
    seq_len = 100
    input_size = 10
    num_classes = 4
    
    # Create sample data
    sample_sequences = np.random.randn(batch_size * 5, seq_len, input_size).astype(np.float32)
    sample_targets = np.random.randint(0, num_classes, batch_size * 5)
    
    # Split into train/val
    train_size = int(0.8 * len(sample_sequences))
    train_sequences = sample_sequences[:train_size]
    train_targets = sample_targets[:train_size]
    val_sequences = sample_sequences[train_size:]
    val_targets = sample_targets[train_size:]
    
    # Initialize predictor
    predictor = CarbonFootprintPredictor(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_classes=num_classes,
        dropout=0.1,
        learning_rate=0.001
    )
    
    print(f"Model initialized on device: {predictor.device}")
    print(f"Training data shape: {train_sequences.shape}")
    print(f"Validation data shape: {val_sequences.shape}")
    
    # Train model
    history = predictor.fit(
        train_sequences=train_sequences,
        train_targets=train_targets,
        val_sequences=val_sequences,
        val_targets=val_targets,
        epochs=20,
        batch_size=16,
        verbose=True
    )
    
    # Make predictions
    predictions = predictor.predict(val_sequences)
    accuracy = np.mean(predictions == val_targets)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Extract embeddings
    embeddings = predictor.get_embeddings(val_sequences[:10])
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("\nModel testing completed successfully!")