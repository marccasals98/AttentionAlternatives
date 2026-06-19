from torch import nn
import torch
from torch.nn import functional as F
import logging
import copy
import math
import ipdb
import os
import warnings
def _warn_once(msg: str):
    # avoid log spam under torchrun (one warning per rank)
    if os.environ.get("RANK", "0") == "0":
        warnings.warn(msg, RuntimeWarning)

# --- Optional FLA backend ---
try:
    from fla.layers import (
        MultiScaleRetention,
        GatedSlotAttention,
        ForgettingAttention,
        KimiDeltaAttention,
        LightNetAttention,
        ReBasedLinearAttention,
        Mamba2 as Mamba2FLA,
    )
    from fla.ops import log_linear_attn
except Exception as e:
    MultiScaleRetention = None
    GatedSlotAttention = None
    ForgettingAttention = None
    KimiDeltaAttention = None
    LightNetAttention = None
    ReBasedLinearAttention = None
    Mamba2FLA = None
    log_linear_attn = None
    _warn_once(f"'fla' is not available ({e}). FLA-based methods will be disabled in this environment.")

try:
    from mamba_ssm import Mamba2 as Mamba2Official
except Exception as e:
    Mamba2Official = None
    _warn_once(f"'Mamba2Official' is not available ({e}). The original Mamba methods will be disabled in this environment.")
# Based on https://peterbloem.nl/blog/transformers
# TODO make dim asserts in every new class

# ---------------------------------------------------------------------
#region Logging

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion
# ---------------------------------------------------------------------

#region 1 - Sequence to sequence components 
# (sequence to sequence blocks, the input dimension is the same than the output dimension)

class NoneSeqToSeq(torch.nn.Module):

    def __init__(self):
        super().__init__()
    

    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        return input_tensors


class SelfAttention(nn.Module):

    """
    Sequence to sequence component, the input dimension is the same than the output dimension.
    Sequence length is not fixed.
    Self-attention without trainable parameters.
    """

    def __init__(self):

        super().__init__()


    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)

        raw_weights = torch.bmm(input_tensors, input_tensors.transpose(1, 2))

        weights = F.softmax(raw_weights, dim = 2)

        output = torch.bmm(weights, input_tensors)

        return output
    

class MultiHeadAttention(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        emb_in is the dimension of every input vector (embedding).
        heads is the number of heads to use in the Multi-Head Attention.
    """

    def __init__(self, emb_in, heads, skip_connections):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads
        self.skip_connections = skip_connections

        self.init_matrix_transformations()
    

    def init_matrix_transformations(self):

        # Matrix transformations to stack every head keys, queries and values matrices
        self.to_keys = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_queries = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_values = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)

        # Linear projection. For each input vector we get self.heads heads, we project them into only one.
        self.unify_heads = nn.Linear(self.heads * self.emb_out, self.emb_out)
    
    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        keys = self.to_keys(input_tensors).view(b, t, self.heads, self.emb_out)
        queries = self.to_queries(input_tensors).view(b, t, self.heads, self.emb_out)
        values = self.to_values(input_tensors).view(b, t, self.heads, self.emb_out)

        # 1 - Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)

        # - Instead of dividing the dot products by sqrt(e), we scale the queries and keys.
        #   This should be more memory efficient
        queries = queries / (self.emb_out ** (1/4))
        keys    = keys / (self.emb_out ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * self.heads, t, t), f'Matrix has size {dot.size()}, expected {(b * self.heads, t, t)}.'

        dot = F.softmax(dot, dim = 2) # dot now has row-wise self-attention probabilities

        # 2 - Apply the self attention to the values
        output = torch.bmm(dot, values).view(b, self.heads, t, self.emb_out)

        # swap h, t back
        output = output.transpose(1, 2).contiguous().view(b, t, self.heads * self.emb_out)

        # unify heads
        output = self.unify_heads(output)

        if self.skip_connections:
            output = output + torch.cat((speech, text), dim=1)
        else:
            output = output

        return output


class MultiHeadStandardVersion(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        emb_in is the dimension of every input vector (embedding).
        
        Heads is the number of heads to use in the Multi-Head Attention.
        In the standard version, the input is splitted in heads, each head has emb_in / heads dimension.

    """

    def __init__(self, emb_in, heads, skip_connections):

        super().__init__()
        assert emb_in % heads == 0, "Input dimension (emb_in) must be divisible by heads"

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads
        self.head_dim = emb_in // heads
        self.skip_connections = skip_connections

        self.init_matrix_transformations()
    

    def init_matrix_transformations(self):

        # Matrix transformations to stack every head keys, queries and values matrices
        self.to_keys = nn.Linear(self.emb_in , self.head_dim * self.heads, bias=False)
        self.to_queries = nn.Linear(self.emb_in, self.head_dim * self.heads, bias=False)
        self.to_values = nn.Linear(self.emb_in, self.head_dim * self.heads, bias=False)

        # Linear projection. For each input vector we get self.heads heads, we project them into only one.
        self.unify_heads = nn.Linear(self.heads * self.head_dim, self.emb_out)
    
    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1) # (b,t, emb)
        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # (b, t, heads, head_dim)
        keys = self.to_keys(input_tensors).view(b, t, self.heads, self.head_dim)
        queries = self.to_queries(input_tensors).view(b, t, self.heads, self.head_dim)
        values = self.to_values(input_tensors).view(b, t, self.heads, self.head_dim)

        # 1 - Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.head_dim)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.head_dim)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.head_dim)

        # - Instead of dividing the dot products by sqrt(e), we scale the queries and keys.
        #   This should be more memory efficient
        queries = queries / (self.head_dim ** (1/4))
        keys    = keys / (self.head_dim ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * self.heads, t, t), f'Matrix has size {dot.size()}, expected {(b * self.heads, t, t)}.'

        dot = F.softmax(dot, dim = 2) # dot now has row-wise self-attention probabilities

        # 2 - Apply the self attention to the values
        output = torch.bmm(dot, values).view(b, self.heads, t, self.head_dim)

        # swap h, t back
        output = output.transpose(1, 2).contiguous().view(b, t, self.heads * self.head_dim)

        # unify heads
        output = self.unify_heads(output)

        if self.skip_connections:
            output = output + torch.cat((speech, text), dim=1)
        else:
            output = output

        return output

class RetNet(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        emb_in is the dimension of every input vector (embedding).
        heads is the number of heads to use in the Multi-Head Attention.
    """

    def __init__(self, emb_in, heads):

        super().__init__()
        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads  
        self.multi_scale_retention = MultiScaleRetention(hidden_size=emb_in, num_heads=heads)
    

    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()
        output, *_ = self.multi_scale_retention(input_tensors)        
        # unify heads
        # output = self.unify_heads(output)


        return output

class GSA(nn.Module):
    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads  
        self.gated_slot_attention = GatedSlotAttention(hidden_size=emb_in, num_heads=heads)
    

    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()

        # Create attention mask: 1 for valid tokens, 0 for padding
        # Since we're concatenating speech and text, all tokens are valid
        attention_mask = torch.ones(b, t, device=input_tensors.device, dtype=torch.long)
        
        output, *_ = self.gated_slot_attention(input_tensors, attention_mask =attention_mask)

        return output
    
class FoX(nn.Module):
    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads  
        self.forgetting_attention = ForgettingAttention(hidden_size=emb_in, num_kv_heads=self.heads)
    

    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()

        # Create attention mask: 1 for valid tokens, 0 for padding
        # Since we're concatenating speech and text, all tokens are valid
        # attention_mask = torch.ones(b, t, device=input_tensors.device, dtype=torch.long)
        
        output, *_ = self.forgetting_attention(input_tensors)

        return output
    

class LogLinearAttention(nn.Module):
    """
    Attention mechanism that balances linear attentions efficiency and the expressiveness of softmax attention.
    Log-linear attention replaces the fixed-size hidden state with a logarithmically growing set of hidden states.

    Lets denote
    `B` as the batch size,
    `T` as the sequence length,
    `H` as the number of heads,
    `K` as the dimension of keys and queries,
    `V` as the dimension of values,
    `L` as the number of levels.
    
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]`.
        level_scales (torch.Tensor):
            Scales for each level of shape `[B, T, H, L]`.
        initial_state (Optional[LogLinearAttentionState]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of type `LogLinearAttentionState` if `output_final_state=True` else `None`.

    """
    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads
        self.num_levels = 2 
        # self.log_linear_attention = log_linear_attn(hidden_size=emb_in)
        self.init_matrix_transformations()
    

    def init_matrix_transformations(self):

        # Matrix transformations to stack every head keys, queries and values matrices
        self.to_keys = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_queries = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_values = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)

        # Linear projection. For each input vector we get self.heads heads, we project them into only one.
        self.unify_heads = nn.Linear(self.heads * self.emb_out, self.emb_out)
        
        self.to_gates = nn.Linear(self.emb_in, self.heads, bias=True)
        self.level_scales = nn.Parameter(torch.linspace(0.9, 0.999, self.heads * self.num_levels).view(self.heads, self.num_levels))
    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()

        # Shape [B, T, H, K]
        keys = self.to_keys(input_tensors).view(b, t, 1, self.emb_out * self.heads).contiguous()
        queries = self.to_queries(input_tensors).view(b, t, 1, self.emb_out * self.heads).contiguous()
        values = self.to_values(input_tensors).view(b, t, self.heads, self.emb_out).contiguous()

        # Forget gate g in (0,1), shape [B, T, H]
        g = torch.sigmoid(self.to_gates(input_tensors))  # [B, T, H]

        # Level Scales of shape [B, T, H, L]
        level_scales_expanded = self.level_scales.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)


        # Create attention mask: 1 for valid tokens, 0 for padding
        # Since we're concatenating speech and text, all tokens are valid
        # attention_mask = torch.ones(b, t, device=input_tensors.device, dtype=torch.long)
        print(f"Before log linear attention: input_tensors.size(): {input_tensors.size()}", flush=True)
        output, final_state = log_linear_attn.chunk_log_linear_attn(q=queries,
                                                                    k=keys,
                                                                    v=values,
                                                                    g=g, 
                                                                    level_scales=level_scales_expanded)
        output = output.view(b, t, self.heads * self.emb_out)

        print(f"After log linear attention: output.size(): {output.size()}", flush=True)

        return output
    
class KDA(nn.Module):
    """
    Kimi Delta Attention (KDA) layer implementation.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Kimi Delta Attention kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads  
        self.kimi_delta_attention = KimiDeltaAttention(hidden_size=emb_in, num_heads=heads)
    

    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()

        # Create attention mask: 1 for valid tokens, 0 for padding
        # Since we're concatenating speech and text, all tokens are valid
        # attention_mask = torch.ones(b, t, device=input_tensors.device, dtype=torch.long)
        
        output, *_ = self.kimi_delta_attention(input_tensors)

        return output
    
class LightNet(nn.Module):

    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads  
        self.lightnet_attention = LightNetAttention(hidden_size=emb_in, num_heads=heads)
    

    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()

        # Create attention mask: 1 for valid tokens, 0 for padding
        # Since we're concatenating speech and text, all tokens are valid
        # attention_mask = torch.ones(b, t, device=input_tensors.device, dtype=torch.long)
        
        output, *_ = self.lightnet_attention(input_tensors)

        return output

class ReBased(nn.Module):
    def __init__(self, emb_in, heads):
        super().__init__()
        assert emb_in % heads == 0
        self.emb_in = emb_in
        self.heads = 16 # HACK I harcoded bc they hardcoded.
        self.rebased_linear_attention = ReBasedLinearAttention(hidden_size=emb_in, num_heads=self.heads)
        fm = self.rebased_linear_attention.feature_map
        head_dim = emb_in // heads
        if hasattr(fm, "gamma") and fm.gamma.shape[0] != head_dim:
            fm.gamma = nn.Parameter(torch.ones(head_dim, dtype=fm.gamma.dtype, device=fm.gamma.device))
        if hasattr(fm, "beta") and fm.beta is not None and fm.beta.shape[0] != head_dim:
            fm.beta = nn.Parameter(torch.zeros(head_dim, dtype=fm.beta.dtype, device=fm.beta.device))
    

    
    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)
        b, t, e = input_tensors.size()

        print(f"rebased_linear_attention.heads: {self.rebased_linear_attention.num_heads}", flush=True)
        print(f"input_tensors.size(1) / self.heads: {input_tensors.size(1) / self.heads}", flush=True)
        # I will only consider cases where sequence length is divisible by number of heads

        """
        cropped_length = (input_tensors.size(1) // self.heads) * self.heads
        input_tensors = input_tensors[:, :cropped_length, :]
        print(f"cropped input_tensors.size(): {input_tensors.size()}", flush=True)
        """
        # Create attention mask: 1 for valid tokens, 0 for padding
        # Since we're concatenating speech and text, all tokens are valid
        # attention_mask = torch.ones(b, t, device=input_tensors.device, dtype=torch.long)
        output, *_ = self.rebased_linear_attention(input_tensors)

        return output

class Mamba2Block(nn.Module):

    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads
        head_dim = emb_in // heads
        self.mamba2 = Mamba2FLA(
            hidden_size=emb_in,
            num_heads=heads,
            head_dim=head_dim,
            expand=1,
        )

    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim=1)
        output = self.mamba2(input_tensors)
        return output
  
class Mamba2Official(nn.Module):
    def __init__(self, emb_in, heads):
        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads
        head_dim = emb_in // heads
        self.mamba2 = Mamba2Official(
            hidden_size=emb_in,
            num_heads=heads,
            head_dim=head_dim,
            expand=1,
        )

    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim=1)
        output = self.mamba2(input_tensors)
        return output

class TransformerBlock(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        One Transformer block.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, expansion_coef, drop_out_p, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we want the same dimension
        self.expansion_coef = expansion_coef
        self.drop_out_p = drop_out_p
        self.heads = heads
        

        self.init_attention_layer()
        self.init_norm_layers()
        self.init_feed_forward_layer()
        self.drop_out = nn.Dropout(drop_out_p)


    def init_attention_layer(self):

        self.attention_layer = MultiHeadAttention(self.emb_in, self.heads)


    def init_norm_layers(self):

        self.norm1 = nn.LayerNorm(self.emb_out)
        self.norm2 = nn.LayerNorm(self.emb_out)


    def init_feed_forward_layer(self):

        self.feed_forward_layer = nn.Sequential(
            nn.Linear(self.emb_out, self.expansion_coef * self.emb_out),
            nn.ReLU(),
            nn.Linear(self.expansion_coef * self.emb_out, self.emb_out),
            )


    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Pass through the attention component
        attention_layer_output = self.attention_layer(input_tensors)

        # Make the skip connection
        skip_connection_1 = attention_layer_output + input_tensors

        # Normalization layer
        normalized_1 = self.norm1(skip_connection_1)

        # Feed forward component
        feed_forward = self.feed_forward_layer(self.drop_out(normalized_1))
        
        # Make the skip connection
        skip_connection_2 = feed_forward + normalized_1

        # Normalization layer
        norm_attended_2 = self.norm2(skip_connection_2)

        # Output
        output = norm_attended_2

        return output


class TransformerStacked(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Stack of n_blocks Transformer blocks.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, n_blocks, expansion_coef, drop_out_p, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we force the same input and output dimension
        self.n_blocks = n_blocks
        self.expansion_coef = expansion_coef
        self.drop_out_p = drop_out_p
        self.heads = heads

        self.init_transformer_blocks()


    def init_transformer_block(self, emb_in, expansion_coef, drop_out_p, heads):

        # Init one transformer block

        transformer_block = TransformerBlock(emb_in, expansion_coef, drop_out_p, heads)

        return transformer_block


    def init_transformer_blocks(self):

        self.transformer_blocks = nn.Sequential()

        for num_block in range(self.n_blocks):

            transformer_block_name = f"transformer_block_{num_block}"
            transformer_block = self.init_transformer_block(self.emb_in, self.expansion_coef, self.drop_out_p, self.heads)
                
            self.transformer_blocks.add_module(transformer_block_name, transformer_block)


    def forward(self, speech, text):
        input_tensors = torch.cat((speech, text), dim = 1)

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        transformer_output = self.transformer_blocks(input_tensors)

        output = transformer_output

        return output


# We call "Reduced Multi-Head Attention" to the implementation of the paper: https://arxiv.org/abs/2007.13199

def new_parameter(*size):

    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)

    return out


def innerKeyValueAttention(query, key, value):

    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / math.sqrt(d_k), dim1=-2, dim2=-1).view(value.size(0),value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim = -2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class ReducedMultiHeadAttention(nn.Module):
    
    def __init__(self, encoder_size, heads_number):
        super().__init__()

        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0 # d_model
        self.head_size = self.encoder_size // heads_number 
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None

        
    def getAlignments(self,ht):

        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)

        return self.alignment 
    

    def getHeadsContextVectors(self,ht):    

        batch_size = ht.size(0)
        logger.debug(f"ht.size(): {ht.size()}")
        logger.debug(f"batch_size: {batch_size}")
        logger.debug(f"self.head_size: {self.head_size}")
        logger.debug(f"self.heads_number: {self.heads_number}")
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return headsContextVectors


    def forward(self, speech, text):
        ht = torch.cat((speech, text), dim = 1)

        logger.debug(f"ht.size(): {ht.size()}")

        headsContextVectors = self.getHeadsContextVectors(ht)
        logger.debug(f"headsContextVectors.size(): {headsContextVectors.size()}")

        # original line
        #return headsContextVectors.view(headsContextVectors.size(0),-1), copy.copy(self.alignment)
        
        return headsContextVectors
#endregion
class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism between different modalities.

    Args:
        nn (Module): The base nn.Module class
    """
    def __init__(self, emb_in, heads, skip_connections):
        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in
        self.heads = heads
        self.skip_connections = skip_connections
        self.init_matrix_transformations()
    

    def init_matrix_transformations(self):

        # Matrix transformations to stack every head keys, queries and values matrices
        self.to_keys = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_queries = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_values = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)

        # Linear projection. For each input vector we get self.heads heads, we project them into only one.
        self.unify_heads = nn.Linear(self.heads * self.emb_out, self.emb_out)

    def cross_attention_forward(self, modality1, modality2):
        """Cross-attention mechanism between different modalities.

        We want to extract query for speech and compare with text keys, but then do the vice versa.

        Thats what this code is about 

        Args:
            modality1 (_type_): The query one
            modality2 (_type_): The key/value one
        """
        #ipdb.set_trace()
        b, t1, e = modality1.size()
        b, t2, e = modality2.size()
        
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # 0 - Define which modality will represent the queries and which the keys and values.
        queries = self.to_queries(modality1).view(b, t1, self.heads, self.emb_out)
        keys = self.to_keys(modality2).view(b, t2, self.heads, self.emb_out)
        values = self.to_values(modality2).view(b, t2, self.heads, self.emb_out)

        # 1 - Compute scaled dot-product attention

        # - fold heads into the batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t1, self.emb_out)
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t2, self.emb_out)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t2, self.emb_out)

        # - Instead of dividing the dot products by sqrt(e), we scale the queries and keys.
        #   This should be more memory efficient
        queries = queries / (self.emb_out ** (1/4))
        keys    = keys / (self.emb_out ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * self.heads, t1, t2), f'Matrix has size {dot.size()}, expected {(b * self.heads, t1, t2)}.'

        dot = F.softmax(dot, dim = 2) # dot now has row-wise self-attention probabilities

        # 2 - Apply the cross-attention to the values
        output = torch.bmm(dot, values).view(b, self.heads, t1, self.emb_out)

        # Swap h, t back
        output = output.transpose(1, 2).contiguous().view(b, t1, self.heads * self.emb_out)

        # unify heads
        output = self.unify_heads(output)

        return output

    def forward(self, speech, text):
        output1 = self.cross_attention_forward(speech, text)
        output2 = self.cross_attention_forward(text, speech)
        
        output = torch.cat((output1, output2), dim = 1)
        if self.skip_connections:
            output = output + torch.cat((speech, text), dim=1)
        else:
            output = output
        return output

class CrossAttentionReduced(CrossAttention):
    """
    CrossAttention but considering speech queries and text keys/values only.

    Args:
        CrossAttention (nn.Module): The base CrossAttention class
    """
    def forward(self, speech, text):

        if self.skip_connections:
            return self.cross_attention_forward(speech, text) + speech
        else:
            return self.cross_attention_forward(speech, text)

# ---------------------------------------------------------------------
#region 2 - Pooling components (sequence to one components, the input dimension is the same than the output dimension)

class StatisticalPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their average as output.
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()
        
        self.emb_in = emb_in 


    def forward(self, input_tensors):

        logger.debug(f"input_tensors.size(): {input_tensors.size()}")

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Get the average of the input vectors (dim = 0 is the batch dimension)
        output = input_tensors.mean(dim = 1)

        return output


class AttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their weighted average as output. These weights comes from an attention mechanism.
        It can be seen as a One Head Self-Attention, where a unique query is used and input vectors are the values and keys.   
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.init_query()

        
    def init_query(self):

        # Init the unique trainable query.
        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)


    def forward(self, input_tensors):

        #logger.debug(f"input_tensors.size(): {input_tensors.size()}")

        #logger.debug(f"self.query[0]: {self.query[0]}")

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        attention_scores = torch.matmul(input_tensors, self.query)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        #logger.debug(f"self.query.size(): {self.query.size()}")
        attention_scores = attention_scores.squeeze(dim = -1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        attention_scores = F.softmax(attention_scores, dim = 1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        attention_scores = attention_scores.unsqueeze(dim = -1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")

        output = torch.bmm(attention_scores.transpose(1, 2), input_tensors)
        #logger.debug(f"output.size(): {output.size()}")
        output = output.view(output.size()[0], output.size()[1] * output.size()[2])
        #logger.debug(f"output.size(): {output.size()}")
        
        return output
    
#endregion
