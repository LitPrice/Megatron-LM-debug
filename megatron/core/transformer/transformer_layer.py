# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel
import torch.nn.functional as F



@dataclass
class TransformerLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, moe=False,
                 enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__(config=config)
        # args = get_args()
        self.layer_fusion = True # args.mlp_layer_fusion
        self.add_bias = config.add_bias_linear
        ffn_hidden_size = config.ffn_hidden_size
        self.gated_linear_unit = config.gated_linear_unit
        if config.gated_linear_unit or self.layer_fusion:
            ffn_hidden_size *= 2
        # ensure_valid(sum([args.add_gate, config.gated_linear_unit]) <= 1,
        #              f"only can use one method in [add_gate :"
        #              f"{args.add_gate},gated_linear_unit :{config.gated_linear_unit}],")
        self.add_gate = True # args.add_gate

        if self.add_gate:
            if self.layer_fusion:
                self.proj = tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    ffn_hidden_size,
                    config=config,
                    init_method=config.init_method,
                    bias=self.add_bias,
                    gather_output=False,
                    skip_bias_add=True,
                    # moe=moe,
                    # enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
                )
            else:
                self.gate_proj = tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    ffn_hidden_size,
                    config=config,
                    init_method=config.init_method,
                    bias=self.add_bias,
                    gather_output=False,
                    skip_bias_add=True,
                    moe=moe,
                    enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
                )

        if not self.layer_fusion:
            # Project to 4h. If using swiglu double the output width
            self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                # moe=moe,
                # enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = False # args.swiglu

        # if args.openai_gelu:
        #     self.activation_func = openai_gelu
        # elif args.onnx_safe:
        #     self.activation_func = erf_gelu
        # elif args.swiglu:
        #     def swiglu(x):
        #         x = torch.chunk(x, 2, dim=-1)
        #         return F.silu(x[0]) * x[1]

        #     self.activation_func = swiglu
        # elif args.squared_relu:
        #     def squared_relu(x):
        #         return torch.pow(F.relu(x), 2)

        #     self.activation_func = squared_relu
        # else:
        self.bias_gelu_fusion = False # args.bias_gelu_fusion
        self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            input_is_parallel=True,
            skip_bias_add=False
            # moe=moe,
            # enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

    def forward(self, hidden_states):

        if self.add_gate:
            if self.layer_fusion:
                gate_and_up_proj = self.proj(hidden_states)[0]
                (gate, up_proj) = tensor_parallel.utils.split_tensor_along_last_dim(
                    gate_and_up_proj, 2, contiguous_split_chunks=True)
                intermediate_parallel = F.silu(gate) * up_proj
            else:
                intermediate_parallel = F.silu(
                    self.gate_proj(hidden_states)[0]) * self.dense_h_to_4h(hidden_states)[0]

        else:
            # [s, b, 4hp]
            intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
            if self.bias_gelu_fusion:
                ensure_valid(self.add_bias is True)
                # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
                ensure_valid(hasattr(self, "__flops__") or self.activation_func == F.gelu)
                intermediate_parallel = \
                    torch_npu.fast_gelu(intermediate_parallel + bias_parallel)
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class BaseTransformerLayer(ABC):
    """ A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)
        self.submodules_config = submodules

        self.layer_number = layer_number + self._get_layer_offset()
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        # self.input_layernorm = build_module(
        #     submodules.input_layernorm,
        #     config=self.config,
        #     hidden_size=self.config.hidden_size,
        #     eps=self.config.layernorm_epsilon,
        # )
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            dim=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention, config=self.config, layer_number=layer_number,
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention, config=self.config, layer_number=layer_number,
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        # self.pre_mlp_layernorm = build_module(
        #     submodules.pre_mlp_layernorm,
        #     config=self.config,
        #     hidden_size=self.config.hidden_size,
        #     eps=self.config.layernorm_epsilon,
        # )
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            dim=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)
        # self.mlp = ParallelMLP(self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
