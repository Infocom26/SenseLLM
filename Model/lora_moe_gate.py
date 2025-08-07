import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

class LoraMoEGate(nn.Module):
    def __init__(self, gpt2_model, expert_cfgs, gate_input_dim, gate_hidden_dim=128):
        super().__init__()
        self.num_experts = len(expert_cfgs)
        self.experts = nn.ModuleList()
        for cfg in expert_cfgs:
            lora_config = LoraConfig(
                r=cfg.get('r', 8),
                lora_alpha=cfg.get('lora_alpha', 16),
                target_modules=cfg.get('target_modules', ["c_attn"]),
                lora_dropout=cfg.get('lora_dropout', 0.05),
                bias=cfg.get('bias', "none"),
                task_type=cfg.get('task_type', TaskType.FEATURE_EXTRACTION)
            )
            expert = get_peft_model(gpt2_model, lora_config)
            for n, p in expert.named_parameters():
                if 'lora' not in n:
                    p.requires_grad = False
            self.experts.append(expert)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_experts)
        )

    def forward(self, x, gate_feature, task_id=None, force_expert=None):
        max_seq_len = getattr(self.experts[0].base_model.config, 'n_positions', 1024)
        B, L, D = x.shape
        
        if force_expert is not None:
            gate_weights = torch.zeros(B, self.num_experts, device=x.device)
            gate_weights[:, force_expert] = 1.0
        else:
            gate_logits = self.gate(gate_feature)
            if task_id is not None:
                bias = torch.zeros_like(gate_logits)
                bias[:, task_id] = 10.0
                gate_logits = gate_logits + bias
            gate_weights = torch.softmax(gate_logits, dim=-1)
        
        expert_outputs = []
        for expert in self.experts:
            outputs = []
            for start in range(0, L, max_seq_len):
                end = min(start + max_seq_len, L)
                chunk = x[:, start:end, :]
                out = expert(inputs_embeds=chunk).last_hidden_state
                outputs.append(out)
            out_full = torch.cat(outputs, dim=1)
            expert_outputs.append(out_full)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)
        fused = (expert_outputs * gate_weights).sum(dim=1)
        return fused, gate_weights.squeeze(-1).squeeze(-1) 