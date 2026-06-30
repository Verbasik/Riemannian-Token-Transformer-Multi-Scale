# file: model.py
# -*- coding: utf-8 -*-
"""
Base RTTMultiScale model with optional subject embeddings and support for
returning attention statistics.

The model uses multi-scale signal decomposition, Riemannian features
(SPD matrices), and a Transformer for EEG signal classification.
Supports subject embeddings to improve generalization.
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import math
from typing import Any, Dict, Optional, Tuple, Union

# =============================================================================
# Third-Party Libraries
# =============================================================================
import torch
import torch.nn as nn

# =============================================================================
# Local Imports
# =============================================================================
from riemannian_utils import (
    cov_shrinkage_oas,
    spd_correlation_from_cov,
    spd_logm,
    spd_vectorize,
    window_signal,
)


# =============================================================================
# Positional encoding
# =============================================================================

class SinusoidalPE(nn.Module):
    """
    Description:
    ---------------
        Sinusoidal positional encoding for adding time-step order
        information to the input sequence. Uses fixed sine and cosine
        frequencies.

    Args:
    ---------------
        d_model: int - Model dimensionality (embedding depth).
        dropout: float - Element dropout probability (default: 0.1).
        max_len: int - Maximum sequence length (default: 1024).

    Returns:
    ---------------
        Tensor with added positional encoding.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> pe = SinusoidalPE(d_model=64)
        >>> x = torch.randn(2, 10, 64)  # [batch, seq_len, d_model]
        >>> out = pe(x)
        >>> out.shape
        torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create the positional encoding matrix.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute logarithmic frequency steps.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Fill even indices with sine and odd indices with cosine.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (not updated by gradients).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Adds positional encoding to the input tensor and applies dropout.

        Args:
        ---------------
            x: torch.Tensor [B, L, d_model] - Input sequence.

        Returns:
        ---------------
            torch.Tensor [B, L, d_model] - Tensor with positional encoding.
        """
        L = x.size(1)
        return self.dropout(x + self.pe[:L, :])


# =============================================================================
# Main model (RTTMultiScale)
# =============================================================================

class RTTMultiScale(nn.Module):
    """
    Description:
    ---------------
        Multi-scale Transformer-based model for EEG classification.
        Uses Riemannian geometry (SPD matrices) to extract features from
        sliding windows of different sizes.

        Architecture:
        1. Channel projection.
        2. Windowing (small and large windows).
        3. Covariance matrix computation and logarithmic mapping.
        4. Vectorization and projection into Transformer space.
        5. Scale encoding and positional encoding.
        6. Transformer encoder with attention.
        7. Attention pooling and classification.
        8. Optional subject embeddings.

    Args:
    ---------------
        n_channels: int - Number of EEG channels.
        n_classes: int - Number of classification classes.
        proj_channels: int - Dimensionality after channel projection.
        window_size_small: int - Small window size (high resolution).
        stride_small: int - Small-window stride.
        window_size_large: int - Large window size (low resolution).
        stride_large: int - Large-window stride.
        d_model: int - Transformer-space dimensionality.
        n_heads: int - Number of attention heads.
        ff_dim: int - Feed-forward network dimensionality inside the Transformer.
        n_layers: int - Number of Transformer layers.
        dropout: float - Dropout rate.
        eps: float - Small value for numerical stability.
        attn_heads: int - Number of heads for attention pooling.
        cov_type: str - Covariance normalization type ('corr' or 'trace').
        oas_min_alpha: float - Minimum OAS shrinkage coefficient.
        use_subject_embed: bool - Whether to use subject embeddings.
        n_subjects: int - Total number of subjects (for the Embedding layer).
        subject_embed_dim: int - Subject embedding dimensionality.
        subject_embed_dropout: float - Dropout for subject embedding.
        unknown_subject_policy: str - Policy for subject_id=-1:
            'error', 'zero', or 'mean'.

    Returns:
    ---------------
        logits: torch.Tensor - Class logits.
        attn_stats: Dict (optional) - Attention statistics.

    Raises:
    ---------------
        ValueError: If use_subject_embed=True but subject_ids are not passed.

    Examples:
    ---------------
        >>> model = RTTMultiScale(
        ...     n_channels=22, n_classes=4, proj_channels=16,
        ...     window_size_small=50, stride_small=25,
        ...     window_size_large=200, stride_large=100,
        ...     d_model=64, n_heads=4, ff_dim=128, n_layers=2,
        ...     dropout=0.1, eps=1e-6, attn_heads=4, cov_type='corr'
        ... )
        >>> x = torch.randn(2, 22, 1000)  # [batch, channels, time]
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        proj_channels: int,
        window_size_small: int,
        stride_small: int,
        window_size_large: int,
        stride_large: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        dropout: float,
        eps: float,
        attn_heads: int,
        cov_type: str,
        oas_min_alpha: float = 0.1,
        use_subject_embed: bool = False,
        n_subjects: int = 5,
        subject_embed_dim: int = 16,
        subject_embed_dropout: float = 0.0,
        unknown_subject_policy: str = 'error',
    ):
        super().__init__()

        # Store window and normalization parameters.
        self.ws_s = window_size_small
        self.st_s = stride_small
        self.ws_l = window_size_large
        self.st_l = stride_large
        self.eps = eps
        self.cov_type = cov_type
        self.oas_min_alpha = oas_min_alpha
        self.use_subject_embed = use_subject_embed
        self.unknown_subject_policy = unknown_subject_policy

        allowed_unknown_policies = {'error', 'zero', 'mean'}
        if self.unknown_subject_policy not in allowed_unknown_policies:
            raise ValueError(
                "unknown_subject_policy must be one of "
                f"{sorted(allowed_unknown_policies)}, got "
                f"'{self.unknown_subject_policy}'."
            )

        # Initialize subject embeddings (optional).
        if self.use_subject_embed:
            self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
            # Use Identity if dropout <= 0 to avoid unnecessary computation.
            if subject_embed_dropout and subject_embed_dropout > 0:
                self.subject_embed_drop = nn.Dropout(subject_embed_dropout)
            else:
                self.subject_embed_drop = nn.Identity()
            classifier_input_dim = d_model * 2 + subject_embed_dim
        else:
            classifier_input_dim = d_model * 2

        # Channel projection (linear transform without bias).
        self.channel_proj = nn.Linear(
            n_channels, proj_channels, bias=False
        )

        # SPD matrix vector dimensionality: N*(N+1)/2.
        spd_vec_dim = proj_channels * (proj_channels + 1) // 2
        self.feature_proj = nn.Linear(spd_vec_dim, d_model)

        # Trainable embeddings for distinguishing scales (small/large).
        self.scale_emb = nn.Parameter(torch.zeros(2, d_model))

        # Classification token [CLS].
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding.
        self.pos_enc = SinusoidalPE(d_model, dropout)

        # Transformer layers.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN architecture for stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Attention pooling mechanism.
        self.attn_pool_W = nn.Linear(d_model, attn_heads)
        self.head_weights = nn.Parameter(torch.zeros(attn_heads))

        # Classifier (MLP head).
        self.head = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

        # Weight initialization.
        self._reset_parameters()

    def _lookup_subject_embeddings(
        self,
        subject_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Returns subject embeddings with explicit unknown ID handling.
            Index -1 means a subject missing from the train mapping.

        Args:
        ---------------
            subject_ids: torch.Tensor [B] - Subject indices.

        Returns:
        ---------------
            torch.Tensor [B, subject_embed_dim].

        Raises:
        ---------------
            ValueError: If unseen subjects are forbidden by policy.
        """
        n_subjects = self.subject_embed.num_embeddings
        subject_ids = subject_ids.to(dtype=torch.long)

        too_large = subject_ids >= n_subjects
        if bool(too_large.any()):
            bad_ids = torch.unique(subject_ids[too_large]).detach().cpu()
            raise ValueError(
                f"subject_ids contain indices outside embedding table: "
                f"{bad_ids.tolist()} >= {n_subjects}."
            )

        unknown_mask = subject_ids < 0
        if not bool(unknown_mask.any()):
            return self.subject_embed(subject_ids)

        if self.unknown_subject_policy == 'error':
            raise ValueError(
                "Validation batch contains subject_id=-1 (unseen subject), "
                "but unknown_subject_policy='error'. For subject-held-out "
                "evaluation set model.use_subject_embed=False or choose "
                "unknown_subject_policy='zero'/'mean'."
            )

        safe_ids = subject_ids.clamp(min=0)
        subject_emb = self.subject_embed(safe_ids)

        if self.unknown_subject_policy == 'zero':
            fallback = torch.zeros_like(subject_emb)
        else:
            mean_emb = self.subject_embed.weight.mean(dim=0)
            fallback = mean_emb.view(1, -1).expand_as(subject_emb)

        return torch.where(
            unknown_mask.unsqueeze(-1),
            fallback,
            subject_emb
        )

    def _reset_parameters(self) -> None:
        """
        Description:
        ---------------
            Initializes model weights.
            - Linear layers: Xavier Uniform initialization.
            - Biases: zeros.
            - Learnable parameters (cls_token, scale_emb): Normal(0, 0.02).

        Returns:
        ---------------
            None
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Use Xavier for weight matrices.
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Zero biases.
                nn.init.zeros_(param)

        # Special initialization for tokens and scale embeddings.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.scale_emb, std=0.02)

    def _tokens_for_scale(
        self,
        x: torch.Tensor,
        w: int,
        s: int,
        scale_id: int
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Generates feature tokens for the specified window scale.
            Process:
            1. Channel projection.
            2. Split into overlapping windows.
            3. Compute an SPD covariance matrix for each window.
            4. Normalize (correlation or trace).
            5. Logarithmic mapping (Log-Euclidean metric).
            6. Vectorize the upper triangular part.
            7. Project into d_model and add the scale embedding.

        Args:
        ---------------
            x: torch.Tensor [B, C, T] - Input signal.
            w: int - Window size.
            s: int - Window stride.
            scale_id: int - Scale index (0 for small, 1 for large).

        Returns:
        ---------------
            torch.Tensor [B, L, d_model] - Token sequence.
        """
        # Channel projection: [B, C, T] -> [B, Proj, T].
        x_pc = self.channel_proj(x.transpose(1, 2)).transpose(1, 2)

        # Windowing: [B, Proj, T] -> [B, L, Proj, W].
        x_win = window_signal(x_pc, w, s)

        B, L, c, _ = x_win.shape

        # Flatten for covariance computation: [B*L, Proj, W].
        x_flat = x_win.reshape(B * L, c, -1)

        # Compute covariance with OAS shrinkage.
        cov = cov_shrinkage_oas(
            x_flat,
            eps=self.eps,
            min_alpha=self.oas_min_alpha
        )

        # Normalize the covariance matrix.
        if self.cov_type == 'corr':
            # Convert to a correlation matrix.
            cov = spd_correlation_from_cov(cov, eps=self.eps)
        else:
            # Trace normalization.
            tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            cov = cov / tr.clamp(min=self.eps).unsqueeze(-1)

        # Log-Euclidean mapping: Log(Cov)
        log_cov = spd_logm(cov, eps=self.eps)

        # Vectorize the upper triangular part.
        vec = spd_vectorize(log_cov)

        # Project into Transformer space.
        tok = self.feature_proj(vec).view(B, L, -1)

        # Add the scale embedding.
        return tok + self.scale_emb[scale_id]

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Description:
        ---------------
            Model forward pass.
            1. Extract features for two scales.
            2. Concatenate tokens and add [CLS].
            3. Encode position and run through the Transformer.
            4. Apply token attention pooling.
            5. Combine the [CLS] and pooled vectors.
            6. Add subject embedding (if enabled).
            7. Classify.

        Args:
        ---------------
            x: torch.Tensor [B, C, T] - Input EEG signal.
            subject_ids: Optional[torch.Tensor] [B] - Subject IDs.
                Required if use_subject_embed=True.
            return_attn: bool - If True, also returns attention statistics.

        Returns:
        ---------------
            logits: torch.Tensor [B, n_classes] - Model predictions.
            attn_stats: Dict (only if return_attn=True):
                - 'weights_tok_mean': Mean attention weights over tokens.
                - 'head_weights': Attention head weights.
                - 'scale_lengths': Scale sequence lengths.

        Raises:
        ---------------
            ValueError: If use_subject_embed=True and subject_ids=None.
        """
        # Extract tokens for the small scale.
        t_s = self._tokens_for_scale(x, self.ws_s, self.st_s, 0)

        # Extract tokens for the large scale.
        t_l = self._tokens_for_scale(x, self.ws_l, self.st_l, 1)

        # Concatenate scale sequences.
        tokens = torch.cat([t_s, t_l], dim=1)

        # Prepare the [CLS] token for the batch.
        cls = self.cls_token.expand(x.size(0), -1, -1)

        # Add position and run through the encoder.
        seq = self.pos_enc(torch.cat([cls, tokens], dim=1))
        h = self.encoder(seq)

        # Split [CLS] and sequence tokens.
        h_cls = h[:, 0]       # [B, d_model]
        toks = h[:, 1:]       # [B, L, d_model]

        # Attention pooling: compute weights for each token.
        scores = self.attn_pool_W(toks)  # [B, L, H]
        weights_tok = torch.softmax(scores, dim=1)

        # Weighted token sum for each attention head.
        # einsum: (batch, len, heads) x (batch, len, dim) -> (batch, heads, dim)
        h_heads = torch.einsum('blh,bld->bhd', weights_tok, toks)

        # Global weights for combining heads.
        head_alpha = torch.softmax(self.head_weights, dim=0)

        # Final head aggregation: (heads) x (batch, heads, dim) -> (batch, dim).
        h_attn = torch.einsum('h,bhd->bd', head_alpha, h_heads)

        # Combine the [CLS] representation and attention-pooled vector.
        combined = torch.cat([h_cls, h_attn], dim=-1)

        # Add subject embedding if needed.
        if self.use_subject_embed:
            if subject_ids is None:
                raise ValueError(
                    "subject_ids required when use_subject_embed=True"
                )
            subject_emb = self._lookup_subject_embeddings(subject_ids)
            subject_emb = self.subject_embed_drop(subject_emb)
            combined = torch.cat([combined, subject_emb], dim=-1)

        # Classification.
        logits = self.head(combined)

        # Return attention statistics if requested.
        if return_attn:
            attn_stats = {
                'weights_tok_mean': weights_tok.mean(dim=0),  # [L, H]
                'head_weights': head_alpha,
                'scale_lengths': (t_s.size(1), t_l.size(1)),
            }
            return logits, attn_stats

        return logits
