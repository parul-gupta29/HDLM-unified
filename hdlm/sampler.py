import torch
import torch.nn.functional as F
from .models.noise_schedule import HierarchicalNoiseSchedule


class HierarchicalSampler:
    """
    Inference sampler for the unified UnifiedLLaDA model.

    At each denoising step t:
      Pass 1 (hierarchy head):
        - Unmasked positions: frozen_hier_embeds (from when token was unmasked)
        - Masked positions:   prev_hier_probs @ W  (soft prediction from step t-1)
        → produces fresh hier_probs used for unmasking probability π_i(t)

      Pass 2 (LM head):
        - Unmasked positions: frozen_hier_embeds  (same as Pass 1)
        - Masked positions:   hier_probs @ W       (fresh from current Pass 1)
        → produces token_probs for sampling

    Hierarchy embeddings for a position are frozen the moment its token is
    unmasked and never updated again.
    """

    STRATEGIES = ("stochastic", "topk_pi")

    def __init__(
        self,
        model,
        tokenizer,
        mask_token_id,
        num_levels=2,
        num_steps=128,
        device="cuda",
        strategy="stochastic",
    ):
        assert strategy in self.STRATEGIES, \
            f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}"
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.num_levels = num_levels
        self.num_steps = num_steps
        self.device = device
        self.strategy = strategy

        self.noise_schedule = HierarchicalNoiseSchedule(
            num_levels=num_levels,
            num_timesteps=num_steps
        ).to(device)

    def initialize_sequence(self, prompt, max_length=512):
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"][0]

        seq = torch.full(
            (max_length,), self.mask_token_id, dtype=torch.long
        )
        seq[:len(input_ids)] = input_ids
        seq = seq.unsqueeze(0).to(self.device)

        prompt_mask = torch.zeros_like(seq).bool()
        prompt_mask[:, :len(input_ids)] = True

        return seq, prompt_mask, len(input_ids)

    @torch.no_grad()
    def sample(self, prompt, max_length=512):
        x, prompt_mask, prompt_len = self.initialize_sequence(prompt, max_length)
        attention_mask = torch.ones_like(x)

        B, L = x.shape
        W = self.model.hierarchy_embedding.weight  # (K, H)
        H = W.shape[1]

        # frozen_hier_embeds: stores the hierarchy embedding for each position
        # once it is unmasked; prompt positions initialised to W[0] (level 0 = text)
        frozen_hier_embeds = torch.zeros(B, L, H, device=self.device, dtype=W.dtype)
        frozen_hier_embeds[:, :prompt_len] = W[0]

        # is_unmasked: True for positions whose token is already revealed
        is_unmasked = prompt_mask.clone()

        # prev_hier_probs: soft hierarchy predictions from the previous step
        # initialised to uniform — used as Pass 1 context for masked positions
        prev_hier_probs = torch.full(
            (B, L, self.num_levels), 1.0 / self.num_levels,
            device=self.device, dtype=W.dtype
        )

        for t in reversed(range(1, self.num_steps)):

            # --- Build hier_context_embeds for Pass 1 ---
            hier_context = frozen_hier_embeds.clone()
            if (~is_unmasked).any():
                hier_context[~is_unmasked] = torch.matmul(
                    prev_hier_probs[~is_unmasked], W
                )

            # Pass 1: get fresh hierarchy predictions
            hier_logits, hier_probs = self.model.forward_hierarchy(
                x, hier_context, attention_mask
            )

            # --- Build hier_embeds for Pass 2 ---
            # All positions: fresh hier_probs @ W (matches training)
            hier_embeds = torch.matmul(hier_probs.to(W.dtype), W)

            # Pass 2: get token predictions
            lm_logits = self.model.forward_lm(x, hier_embeds, attention_mask)
            token_probs = F.softmax(lm_logits.float(), dim=-1)

            # --- Compute unmasking probabilities using fresh hier_probs ---
            # π_i(t) = Σ_k p(h_i=k | x_t) · λ_k(t)
            lambda_k = self.noise_schedule.get_hazard_rate(t).to(W.dtype).view(1, 1, -1)
            pi = (hier_probs * lambda_k).sum(-1)  # (B, L)

            masked = (x == self.mask_token_id)
            if self.strategy == "stochastic":
                to_unmask = (torch.rand_like(pi) < pi) & masked & (~prompt_mask)
            else:  # topk_pi
                n_gen = int((~prompt_mask).sum(-1).min().item())
                n_unmask = max(1, n_gen // self.num_steps)
                pi_score = pi.clone()
                pi_score[~masked | prompt_mask] = -float('inf')
                topk_indices = pi_score.topk(n_unmask, dim=-1).indices
                to_unmask = torch.zeros_like(masked)
                to_unmask.scatter_(-1, topk_indices, True)
                to_unmask = to_unmask & masked & (~prompt_mask)

            # Unmask selected positions
            sampled_tokens = token_probs.argmax(-1)
            x[to_unmask] = sampled_tokens[to_unmask]

            # Freeze hierarchy embeddings for newly unmasked positions
            frozen_hier_embeds[to_unmask] = torch.matmul(
                hier_probs[to_unmask].to(W.dtype), W
            )
            is_unmasked[to_unmask] = True

            # Update prev_hier_probs for next step's masked-position context
            prev_hier_probs = hier_probs.to(W.dtype)

        # --- Fill any remaining masks ---
        remaining = (x == self.mask_token_id)
        if remaining.any():
            hier_context = frozen_hier_embeds.clone()
            if (~is_unmasked).any():
                hier_context[~is_unmasked] = torch.matmul(
                    prev_hier_probs[~is_unmasked], W
                )

            _, hier_probs = self.model.forward_hierarchy(x, hier_context, attention_mask)

            hier_embeds = torch.matmul(hier_probs.to(W.dtype), W)

            lm_logits = self.model.forward_lm(x, hier_embeds, attention_mask)
            x[remaining] = lm_logits.argmax(-1)[remaining]

        return x
