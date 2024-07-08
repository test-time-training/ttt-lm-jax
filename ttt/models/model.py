import os
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax
import flax.linen as nn
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.linen import partitioning as nn_partitioning

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import ModelOutput

from ml_collections import ConfigDict
from mlxu import function_args_to_config, load_pickle, open_file

from ttt.models.bpt import blockwise_ffn, blockwise_attn
from ttt.infra.jax_utils import with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy
from ttt.models.ttt_layer import TTTLinear, TTTMLP, TTTLinearBase, TTTMLPBase, precompute_freqs_cis, apply_rotary_emb


@flax.struct.dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    ttt_stats: Optional[Tuple[jnp.ndarray]] = None
    logits: jnp.ndarray = None


CausalLMOutput = BaseModelOutput

remat = nn_partitioning.remat

CONFIGS = {
    "125m": {
        "vocab_size": 32000,
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "self_attention",
        "use_rotary_emb": "sequence",
        "rope_theta": 10000.0,
        "pre_conv": False,
    },
    "125m-TTT": {
        "vocab_size": 32000,
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "ttt_linear",
        "ttt_base_lr": 1.0,
        "ttt_base_lr_init": -1.0,
        "ttt_base_lr_warmup": -1,
        "mini_batch_size": 16,
        "remat_mini_batch_group_size": 4,
        "rope_theta": 10000.0,
        "pre_conv": True,
        "conv_width": 4,
    },
    "350m": {
        "vocab_size": 32000,
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2736,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "self_attention",
        "use_rotary_emb": "sequence",
        "rope_theta": 10000.0,
        "pre_conv": False,
    },
    "350m-TTT": {
        "vocab_size": 32000,
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2736,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "ttt_linear",
        "ttt_base_lr": 1.0,
        "ttt_base_lr_init": -1.0,
        "ttt_base_lr_warmup": -1,
        "mini_batch_size": 16,
        "remat_mini_batch_group_size": 4,
        "rope_theta": 10000.0,
        "pre_conv": True,
        "conv_width": 4,
    },
    "760m": {
        "vocab_size": 32000,
        "num_hidden_layers": 24,
        "hidden_size": 1536,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "self_attention",
        "use_rotary_emb": "sequence",
        "rope_theta": 10000.0,
        "pre_conv": False,
    },
    "760m-TTT": {
        "vocab_size": 32000,
        "num_hidden_layers": 24,
        "hidden_size": 1536,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "ttt_linear",
        "ttt_base_lr": 1.0,
        "ttt_base_lr_init": -1.0,
        "ttt_base_lr_warmup": -1,
        "mini_batch_size": 16,
        "remat_mini_batch_group_size": 4,
        "rope_theta": 10000.0,
        "pre_conv": True,
        "conv_width": 4,
    },
    "1b": {
        "vocab_size": 32000,
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "intermediate_size": 5504,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "self_attention",
        "use_rotary_emb": "sequence",
        "rope_theta": 10000.0,
        "pre_conv": False,
    },
    "1b-TTT": {
        "vocab_size": 32000,
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "intermediate_size": 5504,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "seq_modeling_block": "ttt_linear",
        "ttt_base_lr": 1.0,
        "ttt_base_lr_init": -1.0,
        "ttt_base_lr_warmup": -1,
        "mini_batch_size": 16,
        "remat_mini_batch_group_size": 4,
        "rope_theta": 10000.0,
        "pre_conv": True,
        "conv_width": 4,
    },
}


class ModelConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        remat_block="",
        remat_attention="",
        remat_mlp="",
        remat_conv="",
        scan_attention=False,
        scan_mlp=False,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        fcm_min_ratio=0.0,
        fcm_max_ratio=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.remat_conv = remat_conv
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ("dp", "fsdp", "mp"))

    @staticmethod
    def get_partition_rules():
        """Partition rules. Note that these rules are orderd, so that
        the beginning rules match first. It is important to use
        PartitionSpec() instead of None here because JAX does not treat
        None as a pytree leaf.
        """
        return (
            # Embeddings
            ("model/wte/embedding", PS("mp", "fsdp")),
            # Attention/TTT
            ("seq_modeling_block/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("seq_modeling_block/wo/kernel", PS("mp", "fsdp")),
            # TTT
            ("seq_modeling_block/ttt_norm/scale", PS(None)),
            ("seq_modeling_block/ttt_norm/bias", PS(None)),
            ("seq_modeling_block/post_norm/scale", PS(None)),
            ("seq_modeling_block/post_norm/bias", PS(None)),
            ("seq_modeling_block/learnable_ttt_lr/kernel", PS(None)),
            ("seq_modeling_block/learnable_ttt_lr/bias", PS(None)),
            ("seq_modeling_block/ttt_dense_0", PS(None)),
            ("seq_modeling_block/ttt_dense_1", PS(None)),
            ("seq_modeling_block/ttt_bias_0", PS(None)),
            ("seq_modeling_block/ttt_bias_1", PS(None)),
            # SwiGLU MLP
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            # RMS Norm
            ("seq_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # Output Head
            ("model/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            (".*", PS(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ("params", "dropout", "fcm")

    @classmethod
    def load_config(cls, path):
        if path in CONFIGS:
            return cls.from_dict(CONFIGS[path])
        load_type, load_path = path.split("::", 1)
        if load_type == "pickle":
            return cls.from_dict(load_pickle(load_path)["config"])
        elif load_type == "json":
            with open_file(load_path, "r") as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f"Unsupported load config type: {load_type}")


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param("kernel", nn.initializers.ones, (self.dim,), self.param_dtype)

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class SwiGLUMLP(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config
        self.w1 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w3 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x


class ConvModule(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config

        if self.config.remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(self.config.remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv

        self.conv1 = conv_module(
            config.hidden_size,
            (config.conv_width,),
            padding="CAUSAL",
            feature_group_count=config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype
        )

    def __call__(self, hidden_states):
        x = hidden_states
        x = self.conv_norm(x)
        x = self.conv1(x)
        return x


class Attention(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, config.max_sequence_length * 2, theta=config.rope_theta, dtype=self.dtype
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
    ):
        xq, xk, xv = (self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states))

        xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.config.scan_attention and not (self.has_variable("cache", "cached_key") or init_cache):
            # doesn't need blockwise attention if we are doing autoregressive decoding since no quadratic memory

            # attention mask without nxn materlization, blockwise_attn will handle the rest
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            # transform boolean mask into float mask
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = None
            attn_output = blockwise_attn(
                xq,
                xk,
                xv,
                bias=attention_bias,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
                attn_pdrop=self.config.attn_pdrop,
                causal=True,
                query_chunk_size=self.config.scan_query_chunk_size,
                key_chunk_size=self.config.scan_key_chunk_size,
                dtype=self.dtype,
                policy=get_gradient_checkpoint_policy("nothing_saveable"),
                precision=self.precision,
                float32_logits=True,
                prevent_cse=True,
            )
            attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp", None))
        else:
            query_length, key_length = xq.shape[1], xk.shape[1]

            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]

            batch_size = hidden_states.shape[0]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            # transform boolean mask into float mask
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            attn_weights = dot_product_attention_weights(
                xq,
                xk,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attn_pdrop,
                deterministic=deterministic,
                dtype=jnp.promote_types(self.dtype, jnp.float32),
                precision=self.precision,
            )
            attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class Block(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        if self.config.seq_modeling_block == "self_attention":
            seq_modeling_block = Attention

        elif self.config.seq_modeling_block == "ttt_linear":
            seq_modeling_block = TTTLinear

        elif self.config.seq_modeling_block == "ttt_mlp":
            seq_modeling_block = TTTMLP

        elif self.config.seq_modeling_block == "ttt_linear_base":
            seq_modeling_block = TTTLinearBase

        elif self.config.seq_modeling_block == "ttt_mlp_base":
            seq_modeling_block = TTTMLPBase

        else:
            raise NotImplementedError("Sequence Modeling Layer %s Not Implemented." % (self.config.seq_modeling_block))

        mlp_module = SwiGLUMLP

        if self.config.remat_attention != "":
            if self.config.seq_modeling_block == "self_attention":
                static_argnums_tuple = (3, 4, 5)
            else:
                static_argnums_tuple = (3, 4)
            seq_modeling_block = remat(
                seq_modeling_block,
                static_argnums=static_argnums_tuple,
                policy=get_gradient_checkpoint_policy(self.config.remat_attention),
                prevent_cse=True,
            )
        if self.config.remat_mlp != "":
            mlp_module = remat(
                SwiGLUMLP,
                static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp),
                prevent_cse=True,
            )

        self.seq_modeling_block = seq_modeling_block(
            self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.feed_forward = mlp_module(
            self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.seq_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype
        )
        if self.config.pre_conv:
            self.conv = ConvModule(
                self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
            )

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        ttt_lr_mult=1.0,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_ttt_stats: bool = False,
        fcm_mask: Optional[jnp.ndarray] = None,
    ):
        if self.config.pre_conv:
            conv_outputs = self.conv(hidden_states)
            hidden_states = hidden_states + conv_outputs

        hidden_states_pre_normed = self.seq_norm(hidden_states)

        if self.config.seq_modeling_block == "self_attention":
            seq_modeling_outputs = self.seq_modeling_block(
                hidden_states_pre_normed,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                fcm_mask,
            )
        else:
            seq_modeling_outputs = self.seq_modeling_block(
                hidden_states_pre_normed, input_ids, position_ids, deterministic, output_ttt_stats, ttt_lr_mult
            )

        seq_modeling_output = seq_modeling_outputs[0]
        hidden_states = hidden_states + seq_modeling_output

        feed_forward_input = self.ffn_norm(hidden_states)
        if self.config.scan_mlp:
            feed_forward_hidden_states = blockwise_ffn(
                self.feed_forward, feed_forward_input, self.config.scan_mlp_chunk_size, deterministic
            )
        else:
            feed_forward_hidden_states = self.feed_forward(feed_forward_input, deterministic)
        feed_forward_hidden_states = with_sharding_constraint(
            feed_forward_hidden_states, PS(("dp", "fsdp"), None, "mp")
        )
        hidden_states = hidden_states + feed_forward_hidden_states

        if len(seq_modeling_outputs) > 1:
            if isinstance(seq_modeling_outputs[1], tuple):
                return (hidden_states,) + (seq_modeling_outputs[1],)
            else:
                return (hidden_states,) + ((seq_modeling_outputs[1],),)
        else:
            return (hidden_states,)


class BlockCollection(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        block = Block
        if self.config.remat_block != "":
            block = remat(
                Block, static_argnums=(5, 6, 7, 8), policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        self.blocks = [
            block(self.config, name=str(i), dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        ttt_lr_mult=1.0,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_ttt_stats: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_ttt_stats = () if output_ttt_stats else None

        if not deterministic and self.config.fcm_max_ratio > 0:
            batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
            fcm_ratio = jax.random.uniform(
                self.make_rng("fcm"),
                shape=(batch_size, 1, 1, 1),
                minval=self.config.fcm_min_ratio,
                maxval=self.config.fcm_max_ratio,
            )
            fcm_mask = jax.random.uniform(self.make_rng("fcm"), shape=(batch_size, 1, 1, seq_length)) > fcm_ratio
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype("bool")
        else:
            fcm_mask = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                input_ids,
                attention_mask,
                position_ids,
                ttt_lr_mult,
                deterministic,
                init_cache,
                output_attentions,
                output_ttt_stats,
                fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

            if output_ttt_stats:
                all_ttt_stats += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions, all_ttt_stats)
        return outputs


class Model(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = BlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.ln_f = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        ttt_lr_mult=1.0,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_ttt_stats: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))
        hidden_states = self.dropout(input_embeds, deterministic=deterministic)
        outputs = self.h(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ttt_lr_mult=ttt_lr_mult,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_ttt_stats=output_ttt_stats,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=outputs[1], attentions=outputs[2], ttt_stats=outputs[3]
        )


class CausalLM(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.model = Model(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        ttt_lr_mult=1.0,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_ttt_stats: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0), (batch_size, seq_length)
            )
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            ttt_lr_mult,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_ttt_stats=output_ttt_stats,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ttt_stats=outputs.ttt_stats,
        )
