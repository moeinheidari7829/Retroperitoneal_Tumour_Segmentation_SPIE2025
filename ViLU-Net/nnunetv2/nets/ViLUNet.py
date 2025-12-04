import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import List, Tuple, Type, Union

from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    convert_dim_to_conv_op,
    get_matching_instancenorm,
    get_matching_pool_op,
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch.cuda.amp import autocast

from .vision_lstm import SequenceTraversal, ViLBlock


class UpsampleLayer(nn.Module):
    """Nearest-neighbor upsample followed by 1x1 convolution to mix channels."""

    def __init__(self, conv_op, input_channels, output_channels, pool_op_kernel_size, mode: str = "nearest"):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        return self.conv(x)


class ViLLayer(nn.Module):
    """
    Wraps ViLBlock to run either over spatial tokens (flattened HxW/xL) or channel tokens when spatial size is tiny.
    """

    def __init__(self, dim: int, channel_token: bool = False):
        super().__init__()
        self.dim = dim
        self.channel_token = channel_token
        self.vil = ViLBlock(dim=self.dim, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()

        if self.channel_token:
            # treat channels as tokens, spatial dims as embedding
            B, n_tokens = x.shape[:2]
            d_model = x.shape[2:].numel()
            assert d_model == self.dim, f"Expected embedding dim {self.dim}, got {d_model}"
            img_dims = x.shape[2:]
            x_flat = x.flatten(2)  # (B, n_tokens, d_model)
            x_vil = self.vil(x_flat)
            out = x_vil.reshape(B, n_tokens, *img_dims)
        else:
            # treat spatial positions as tokens, channels as embedding
            B, d_model = x.shape[:2]
            assert d_model == self.dim, f"Expected embedding dim {self.dim}, got {d_model}"
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]
            x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  # (B, tokens, d_model)
            x_vil = self.vil(x_flat)
            out = x_vil.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out


class BasicResBlock(nn.Module):
    """2-layer residual conv block with optional 1x1 projection on the skip."""

    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        norm_op,
        norm_op_kwargs,
        kernel_size=3,
        padding=1,
        stride=1,
        use_1x1conv=False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    ):
        super().__init__()
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class ViLUNetEncoder(nn.Module):
    """Encoder: stem convs + per-stage residual conv stack + ViL blocks."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        vil_repeats: int = 2,
        pool_type: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(strides) == n_stages

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != "conv" else None
        self.pool_op = pool_op

        self.conv_pad_sizes = [[k // 2 for k in krnl] for krnl in kernel_sizes]

        # feature map sizes and channel-token heuristic
        feature_map_sizes = []
        feature_map_size = input_size
        do_channel_token = [False] * n_stages
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        self.feature_map_sizes = feature_map_sizes
        self.do_channel_token = do_channel_token

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ],
        )

        input_channels = stem_channels
        stages = []
        vil_layers = []
        for s in range(n_stages):
            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=conv_op,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        input_channels=input_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        padding=self.conv_pad_sizes[s],
                        stride=strides[s],
                        use_1x1conv=True,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=conv_op,
                            input_channels=features_per_stage[s],
                            output_channels=features_per_stage[s],
                            kernel_size=kernel_sizes[s],
                            stride=1,
                            conv_bias=conv_bias,
                            norm_op=norm_op,
                            norm_op_kwargs=norm_op_kwargs,
                            nonlin=nonlin,
                            nonlin_kwargs=nonlin_kwargs,
                        )
                        for _ in range(n_blocks_per_stage[s] - 1)
                    ],
                )
            )

            vil_dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s]
            vil_layers.append(
                nn.Sequential(
                    *[
                        ViLLayer(dim=int(vil_dim), channel_token=do_channel_token[s])
                        for _ in range(vil_repeats)
                    ]
                )
            )
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.vil_layers = nn.ModuleList(vil_layers)
        self.output_channels = features_per_stage

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage, vil_layer in zip(self.stages, self.vil_layers):
            x = stage(x)
            x = vil_layer(x)
            skips.append(x)
        return skips


class ViLUNetDecoder(nn.Module):
    """Decoder: upsample + concat + residual convs + ViL blocks per stage."""

    def __init__(
        self,
        encoder: ViLUNetEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        vil_repeats: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []
        vil_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode="nearest",
                )
            )

            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=encoder.conv_op,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        input_channels=2 * input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        padding=encoder.conv_pad_sizes[-(s + 1)],
                        stride=1,
                        use_1x1conv=True,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=encoder.conv_op,
                            input_channels=input_features_skip,
                            output_channels=input_features_skip,
                            kernel_size=encoder.kernel_sizes[-(s + 1)],
                            stride=1,
                            conv_bias=encoder.conv_bias,
                            norm_op=encoder.norm_op,
                            norm_op_kwargs=encoder.norm_op_kwargs,
                            nonlin=encoder.nonlin,
                            nonlin_kwargs=encoder.nonlin_kwargs,
                        )
                        for _ in range(n_conv_per_stage[s - 1] - 1)
                    ],
                )
            )

            # Decoder ViL uses the spatial size of the corresponding skip
            skip_map_size = encoder.feature_map_sizes[-(s + 1)]
            channel_token = np.prod(skip_map_size) <= input_features_skip
            vil_dim = np.prod(skip_map_size) if channel_token else input_features_skip
            vil_layers.append(
                nn.Sequential(
                    *[
                        ViLLayer(dim=int(vil_dim), channel_token=channel_token)
                        for _ in range(vil_repeats)
                    ]
                )
            )

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.vil_layers = nn.ModuleList(vil_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            x = self.vil_layers[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs[0] if not self.deep_supervision else seg_outputs


class ViLUNet(nn.Module):
    """Full ViL U-Net with ViL blocks in both encoder and decoder."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        vil_repeats: int = 2,
    ):
        super().__init__()
        self.encoder = ViLUNetEncoder(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            vil_repeats=vil_repeats,
        )

        self.decoder = ViLUNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            vil_repeats=vil_repeats,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        output = np.int64(0)
        feature_map_size = input_size

        for s in range(len(self.encoder.strides)):
            if s == 0:
                output += self.encoder.stem.compute_conv_feature_map_size(feature_map_size)
            output += self.encoder.stages[s].compute_conv_feature_map_size(feature_map_size)
            feature_map_size = [i // j for i, j in zip(feature_map_size, self.encoder.strides[s])]
            skip_sizes.append(feature_map_size)

        # decoder feature map sizes mirror encoder except last
        for s in range(len(self.decoder.stages)):
            output += self.decoder.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.decoder.deep_supervision or (s == (len(self.decoder.stages) - 1)):
                output += np.prod([self.decoder.seg_layers[s].out_channels, *skip_sizes[-(s + 1)]], dtype=np.int64)

        return output


def get_vilunet_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
):
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)
    features = [
        min(configuration_manager.UNet_base_num_features * 2 ** i, configuration_manager.unet_max_num_features)
        for i in range(num_stages)
    ]

    model = ViLUNet(
        input_size=configuration_manager.patch_size,
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=features,
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
        num_classes=label_manager.num_segmentation_heads,
        n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=get_matching_instancenorm(conv_op),
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
        vil_repeats=2,
    )
    model.apply(InitWeights_He(1e-2))
    return model


def get_vilunet_2d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
):
    assert len(configuration_manager.patch_size) == 2
    return get_vilunet_from_plans(
        plans_manager=plans_manager,
        dataset_json=dataset_json,
        configuration_manager=configuration_manager,
        num_input_channels=num_input_channels,
        deep_supervision=deep_supervision,
    )


def get_vilunet_3d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
):
    assert len(configuration_manager.patch_size) == 3
    return get_vilunet_from_plans(
        plans_manager=plans_manager,
        dataset_json=dataset_json,
        configuration_manager=configuration_manager,
        num_input_channels=num_input_channels,
        deep_supervision=deep_supervision,
    )
