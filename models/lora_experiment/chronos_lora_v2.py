from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

class ChronosLoRARankerV2(nn.Module):
    """
    Triple-Expert Hybrid System: Foundation Expert (Chronos-T5 + LoRA).
    Fine-tunes the Chronos-T5-small backbone to predict Sector Alpha from 10 RAMT features.
    """
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        input_dim: int = 10,
        lora_r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Load the base Chronos-T5 model
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 2. Freeze base parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 3. Apply LoRA to the encoder attention projections
        # We focus on the encoder because we are using Chronos as a feature extractor/ranker
        peft_config = LoraConfig(
            task_type=None,  # Set to None since we are using a sub-module (encoder) directly
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            target_modules=["q", "k", "v", "o"]
        )
        self.encoder = get_peft_model(self.base_model.get_encoder(), peft_config)
        
        # 4. Input projection: Map 10 RAMT features to T5 d_model (512 for small)
        self.d_model = self.config.d_model
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # 5. Ranking head: Linear layer to output a single scalar alpha prediction
        self.ranking_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch,) scalar alpha prediction
        """
        # Project inputs: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Pass through LoRA-adapted T5 encoder
        # The encoder expects (batch, seq_len, d_model) if we bypass embedding, 
        # but Chronos usually takes raw values. Here we provide the projected embeddings.
        encoder_outputs = self.encoder(inputs_embeds=x)
        
        # Use the representation of the last token for ranking
        # encoder_outputs.last_hidden_state: (batch, seq_len, d_model)
        last_hidden = encoder_outputs.last_hidden_state[:, -1, :]
        
        # Final ranking head: (batch, d_model) -> (batch, 1)
        out = self.ranking_head(last_hidden)
        return out.squeeze(-1)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
