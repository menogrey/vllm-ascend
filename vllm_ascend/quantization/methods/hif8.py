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

from typing import Any

import torch
import torch_npu

from .base import AscendLinearScheme
from .registry import register_scheme


@register_scheme("HIF8", "linear")
class AscendHiF8LinearMethod(AscendLinearScheme):
    """Linear method for Ascend HIF8 quantization.

    This scheme uses static per-tensor quantization for activations
    and per-channel quantization for weights.
    """

    def __init__(self) -> None:
        # NOTE: This is a switch
        self.act_dynamic_quant = False
        self.act_pertensor = True
        pass

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        params_dict = {"quantized_weight": torch.empty(input_size, output_size, dtype=torch.uint8)}

        if not self.act_dynamic_quant and self.act_pertensor:
            # NOTE: This is a mock pertensor scale, we are using the dynamic quant weight, not having act quant scale.
            params_dict["pertensor_scale"] = torch.ones(
                [
                    1,
                ],
                dtype=torch.float32,
            )
        return params_dict

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        params_dict = {}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["scale_w"] = torch.empty(output_size, dtype=torch.float32)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if self.act_dynamic_quant:
            quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch_npu.hifloat8, dst_type_max=15)

            output = torch_npu.npu_quant_matmul(
                quant_x,
                layer.quantized_weight,
                layer.scale_w,  # weight_scale_i64?
                pertoken_scale=pertoken_scale,
                output_dtype=layer.params_dtype,
                x1_dtype=torch_npu.hifloat8,
                x2_dtype=torch_npu.hifloat8,
            )
            return output

        else:
            if self.act_pertensor:
                quant_x = torch_npu.npu_quantize(x, layer.pertensor_scale, None, dtype=torch_npu.hifloat8)
                output = torch_npu.npu_quant_matmul(
                    quant_x,
                    layer.quantized_weight,
                    layer.weight_scale_i64,
                    pertoken_scale=None,
                    output_dtype=layer.params_dtype,
                    x1_dtype=torch_npu.hifloat8,
                    x2_dtype=torch_npu.hifloat8,
                )
                return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quantized_weight = layer.quantized_weight.data.contiguous()
        layer.scale_w.data = layer.scale_w.data.contiguous()
        layer.weight_scale_i64 = torch_npu.npu_trans_quant_param(layer.scale_w.data)
