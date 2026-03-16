import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class UnifiedLLaDA(nn.Module):
    """
    Unified hierarchical diffusion model with a single shared LLaDA backbone.

    Two forward passes per denoising step:
      Pass 1 (hierarchy head): token_embeds + hier_context_embeds → backbone
                               → hierarchy_head → hier_probs
      Pass 2 (LM head):        token_embeds + scale * hier_embeds → backbone
                               → lm_head → lm_logits

    The hierarchy embedding matrix W is shared between both passes.
    Gradients from both L_lm and L_hier flow through the shared backbone.
    """

    def __init__(
        self,
        model_path,
        num_levels=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05
    ):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(base_model, lora_config)

        hidden = self.model.config.hidden_size
        self.num_levels = num_levels

        # Shared hierarchy embedding matrix W (num_levels, hidden)
        # Used in both passes: as context input (Pass 1) and soft conditioning (Pass 2)
        self.hierarchy_embedding = nn.Embedding(num_levels, hidden)

        # Hierarchy prediction head (Pass 1 output)
        self.hierarchy_head = nn.Linear(hidden, num_levels)

        # Scale for hierarchy signal added to token embeddings
        self.scale = 0.1


    def build_hier_context_embeds(self, hierarchy_labels, mask):
        """
        Build hier_context_embeds for Pass 1 during training.

        Unmasked positions: W[gold_label]
        Masked positions:   mean(W)  — uniform average over all levels

        Args:
            hierarchy_labels: (B, L) long — gold hierarchy labels
            mask:             (B, L) bool — True for masked (unknown) positions

        Returns:
            (B, L, hidden) tensor
        """
        ctx = self.hierarchy_embedding(hierarchy_labels)  # (B, L, H)
        uniform = self.hierarchy_embedding.weight.mean(0)  # (H,)
        ctx = ctx.clone()
        ctx[mask] = uniform
        return ctx

    def forward_hierarchy(self, input_ids, hier_context_embeds, attention_mask=None):
        """
        Pass 1: backbone conditioned on hierarchy context → hierarchy head.

        Args:
            input_ids:           (B, L) — possibly with mask tokens
            hier_context_embeds: (B, L, H) — known hierarchy context per position
            attention_mask:      (B, L) optional

        Returns:
            hier_logits: (B, L, num_levels)
            hier_probs:  (B, L, num_levels)
        """
        token_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = token_embeds + self.scale * hier_context_embeds.to(token_embeds.dtype)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        last_hidden = outputs.hidden_states[-1]          # (B, L, H)
        hier_logits = self.hierarchy_head(last_hidden.float())  # (B, L, num_levels)
        hier_probs = F.softmax(hier_logits, dim=-1)

        return hier_logits, hier_probs

    def forward_lm(self, input_ids, hier_embeds, attention_mask=None):
        """
        Pass 2: backbone conditioned on hierarchy embeddings → LM head.

        Args:
            input_ids:      (B, L) — possibly with mask tokens
            hier_embeds:    (B, L, H) — weighted hierarchy embeddings per position
            attention_mask: (B, L) optional

        Returns:
            lm_logits: (B, L, vocab_size)
        """
        token_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = token_embeds + self.scale * hier_embeds.to(token_embeds.dtype)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        return outputs.logits

    def forward(self, input_ids, hier_context_embeds, attention_mask=None):
        """
        Combined forward for training: runs Pass 1 then Pass 2.

        Pass 2 uses hier_probs @ W for all positions (training setting).
        During inference, the sampler calls forward_hierarchy and forward_lm
        separately to apply frozen embeddings for unmasked positions.

        Returns:
            lm_logits:   (B, L, vocab_size)
            hier_logits: (B, L, num_levels)
            hier_probs:  (B, L, num_levels)
        """
        # Pass 1
        hier_logits, hier_probs = self.forward_hierarchy(
            input_ids, hier_context_embeds, attention_mask
        )

        # Compute hier_embeds from Pass 1 output
        hier_embeds = torch.matmul(
            hier_probs.to(self.hierarchy_embedding.weight.dtype),
            self.hierarchy_embedding.weight
        )

        # Pass 2
        lm_logits = self.forward_lm(input_ids, hier_embeds, attention_mask)

        return lm_logits, hier_logits, hier_probs
