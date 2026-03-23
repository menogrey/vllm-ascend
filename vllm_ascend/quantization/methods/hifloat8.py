#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
import numpy as np
from en_dtypes import hifloat8
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.quantization.utils import replace_parameter


from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_hifloat8_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts

from .base import AscendLinearScheme, AscendMoEScheme, QuantType
from .registry import register_scheme


@register_scheme("W8A8_HIFLOAT8", "linear")
class AscendW8A8Hifloat8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_HIFLOAT8 quantization.

    This scheme uses HiFloat8 quantization with per-group scales.
    """

    model_dtype = None

    def __init__(self):
        ensure_hifloat8_available("W8A8_HIFLOAT8 linear quantization")
        vllm_config = get_current_vllm_config()

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        # 
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        # params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.uint8)
        # pertoken_scale = dynamic_scale
        # output_dtype = x.dtype

        # output = torch_npu.npu_quant_matmul(
        #     quantized_x,
        #     layer.weight,
        #     layer.weight_scale,
        #     scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        #     pertoken_scale=pertoken_scale,
        #     pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        #     bias=bias,
        #     output_dtype=output_dtype,
        #     group_sizes=[1, 1, self.group_size],
        # )


        output = torch.matmul(
            x,
            layer.weight.data
        )

        return output

    def process_weights_after_loading(self, layer):
        # n_dim, k_dim = layer.weight_scale.data.shape
        # layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        # layer.weight.data = layer.weight.data.transpose(0, 1)
        # layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

        # transform weight to HiFloat8 format
        weight = layer.weight.data

        array = weight.cpu().to(torch.float32).numpy().astype(hifloat8)

        new_tensor = torch.from_numpy(array.astype(np.float16)).to(weight.device)

        replace_parameter(layer, "weight", new_tensor)

        layer.weight.data = layer.weight.data.transpose(0, 1)

