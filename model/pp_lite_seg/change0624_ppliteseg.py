import torch
import torch.nn as nn
import torch.nn.functional as F
# from typing import Union, List, Tuple, Optional
import math
from typing import Mapping, Optional, Tuple, Union, List, Dict, Any, Iterable
# from super_gradients.training.utils.module_utils import ConvBNReLU, make_upsample_module
# from super_gradients.training.utils import WrappedModel
# from module_utils import ConvBNReLU, make_upsample_module
# from super_gradients.common import UpsampleMode
# from super_gradients.training.models.segmentation_models.stdc import SegmentationHead, AbstractSTDCBackbone,\
#     STDC1Backbone, STDC2Backbone
# from super_gradients.training.models.segmentation_models.segmentation_module import SegmentationModule
# from super_gradients.training.utils import HpmStruct, get_param

from enum import Enum
from abc import ABC, abstractmethod
import re


def fuzzy_keys(params: Mapping) -> List[str]:
    """
    Returns params.key() removing leading and trailing white space, lower-casing and dropping symbols.
    :param params: Mapping, the mapping containing the keys to be returned.
    :return: List[str], list of keys as discussed above.
    """
    return [fuzzy_str(s) for s in params.keys()]


def fuzzy_str(s: str):
    """
    Returns s removing leading and trailing white space, lower-casing and drops
    :param s: str, string to apply the manipulation discussed above.
    :return: str, s after the manipulation discussed above.
    """
    return re.sub(r"[^\w]", "", s).replace("_", "").lower()


def _get_fuzzy_attr_map(params):
    return {fuzzy_str(a): a for a in params.__dir__()}


def _has_fuzzy_attr(params, name):
    return fuzzy_str(name) in _get_fuzzy_attr_map(params)


def get_fuzzy_mapping_param(name: str, params: Mapping):
    """
    Returns parameter value, with key=name with no sensitivity to lowercase, uppercase and symbols.
    :param name: str, the key in params which is fuzzy-matched and retruned.
    :param params: Mapping, the mapping containing param.
    :return:
    """
    fuzzy_params = {fuzzy_str(key): params[key] for key in params.keys()}
    return fuzzy_params[fuzzy_str(name)]


def get_fuzzy_attr(params: Any, name: str):
    """
    Returns attribute (same functionality as getattr), but non sensitive to symbols, uppercase and lowercase.
    :param params: Any, any object which wed looking for the attribute name in.
    :param name: str, the attribute of param to be returned.
    :return: Any, the attribute value or None when not fuzzy matching of the attribute is found
    """
    return getattr(params, _get_fuzzy_attr_map(params)[fuzzy_str(name)])


def fuzzy_idx_in_list(name: str, lst: List[str]) -> int:
    """
    Returns the index of name in lst, with non sensitivity to symbols, uppercase and lowercase.
    :param name: str, the name to be searched in lst.
    :param lst: List[str], the list as described above.
    :return: int, index of name in lst in the matter discussed above.
    """
    return [fuzzy_str(x) for x in lst].index(fuzzy_str(name))


def get_param(params, name, default_val=None):
    """
    Retrieves a param from a parameter object/dict . If the parameter does not exist, will return default_val.
    In case the default_val is of type dictionary, and a value is found in the params - the function
    will return the default value dictionary with internal values overridden by the found value
    IMPORTANT: Not sensitive to lowercase, uppercase and symbols.

    i.e.
    default_opt_params = {'lr':0.1, 'momentum':0.99, 'alpha':0.001}
    training_params = {'optimizer_params': {'lr':0.0001}, 'batch': 32 .... }
    get_param(training_params, name='OptimizerParams', default_val=default_opt_params)
    will return {'lr':0.0001, 'momentum':0.99, 'alpha':0.001}


    :param params:      an object (typically HpmStruct) or a dict holding the params
    :param name:        name of the searched parameter
    :param default_val: assumed to be the same type as the value searched in the params
    :return:            the found value, or default if not found
    """
    if isinstance(params, Mapping):
        if name in params:
            param_val = params[name]

        elif fuzzy_str(name) in fuzzy_keys(params):
            param_val = get_fuzzy_mapping_param(name, params)

        else:
            param_val = default_val
    elif hasattr(params, name):
        param_val = getattr(params, name)
    elif _has_fuzzy_attr(params, name):
        param_val = get_fuzzy_attr(params, name)
    else:
        param_val = default_val

    if isinstance(default_val, Mapping):
        return {**default_val, **param_val}
    else:
        return param_val


def recursive_override(base: dict, extension: dict):
    for k, v in extension.items():
        if k in base:
            if isinstance(v, Mapping) and isinstance(base[k], Mapping):
                recursive_override(base[k], extension[k])
            else:
                base[k] = extension[k]
        else:
            base[k] = extension[k]




class HpmStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.schema = None

    def set_schema(self, schema: dict):
        self.schema = schema

    def override(self, **entries):
        recursive_override(self.__dict__, entries)

    def to_dict(self, include_schema=True) -> dict:
        """Convert this HpmStruct instance into a dict.
        :param include_schema: If True, also return the field "schema"
        :return: Dict representation of this HpmStruct instance.
        """
        out_dict = self.__dict__.copy()
        if not include_schema:
            out_dict.pop("schema")
        return out_dict

    # def validate(self):
    #     """
    #     Validate the current dict values according to the provided schema
    #     :raises
    #         `AttributeError` if schema was not set
    #         `jsonschema.exceptions.ValidationError` if the instance is invalid
    #         `jsonschema.exceptions.SchemaError` if the schema itselfis invalid
    #     """
    #     if self.schema is None:
    #         raise AttributeError("schema was not set")
    #     else:
    #         validate(self.__dict__, self.schema)


class SgModule(nn.Module):
    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """
        return [{"named_params": self.named_parameters()}]

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        """

        :param param_groups: list of dictionaries containing the params
        :return: list of dictionaries containing the params
        """
        for param_group in param_groups:
            param_group["lr"] = lr
        return param_groups

    def get_include_attributes(self) -> list:
        """
        This function is used by the EMA. When updating the EMA model, some attributes of the main model (used in training)
        are updated to the EMA model along with the model weights.
        By default, all attributes are updated except for private attributes (starting with '_')
        You can either set include_attributes or exclude_attributes. By returning a non empty list from this function,
        you override the default behaviour and only attributes named in this list will be updated.
        Note: This will also override the get_exclude_attributes list.
            :return: list of attributes to update from main model to EMA model
        """
        return []

    def get_exclude_attributes(self) -> list:
        """
        This function is used by the EMA. When updating the EMA model, some attributes of the main model (used in training)
        are updated to the EMA model along with the model weights.
        By default, all attributes are updated except for private attributes (starting with '_')
        You can either set include_attributes or exclude_attributes. By returning a non empty list from this function,
        you override the default behaviour and attributes named in this list will also be excluded from update.
        Note: if get_include_attributes is not empty, it will override this list.
            :return: list of attributes to not update from main model to EMA mode
        """
        return []

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """

    def replace_head(self, **kwargs):
        """
        Replace final layer for pretrained models. Since this varies between architectures, we leave it to the inheriting
        class to implement.
        """

        raise NotImplementedError


class SegmentationModule(SgModule, ABC):
    """
    Base SegmentationModule class
    """

    def __init__(self, use_aux_heads: bool):
        super().__init__()
        self._use_aux_heads = use_aux_heads

    @property
    def use_aux_heads(self):
        return self._use_aux_heads

    @use_aux_heads.setter
    def use_aux_heads(self, use_aux: bool):
        """
        public setter for self._use_aux_heads, called every time an assignment to self.use_aux_heads is applied.
        if use_aux is False, `_remove_auxiliary_heads` is called to delete auxiliary and detail heads.
        if use_aux is True, and self._use_aux_heads was already set to False a ValueError is raised, recreating
            aux and detail heads outside init method is not allowed, and the module should be recreated.
        """
        if use_aux is True and self._use_aux_heads is False:
            raise ValueError(
                "Cant turn use_aux_heads from False to True. Try initiating the module again with"
                " `use_aux_heads=True` or initiating the auxiliary heads modules manually."
            )
        if not use_aux:
            self._remove_auxiliary_heads()
        self._use_aux_heads = use_aux

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        # set to false and delete auxiliary and detail heads modules.
        self.use_aux_heads = False

    @abstractmethod
    def _remove_auxiliary_heads(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """
        For SgTrainer load_backbone compatibility.
        """
        raise NotImplementedError()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Residual(nn.Module):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a residual skip connection within a single block.
    """

    def forward(self, x):
        return x


class UpsampleMode(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    SNPE_BILINEAR = "snpe_bilinear"


class ConvBNReLU(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Relu layer. Default behaviour is Conv-BN-Relu. To exclude Batchnorm module use
        `use_normalization=False`, to exclude Relu activation use `use_activation=False`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    For relu arguments documentation see `nn.Relu`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 use_normalization: bool = True,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None,
                 use_activation: bool = True,
                 inplace: bool = False):

        super(ConvBNReLU, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv", nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              padding_mode=padding_mode))

        if use_normalization:
            self.seq.add_module("bn", nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                                     track_running_stats=track_running_stats, device=device,
                                                     dtype=dtype))
        if use_activation:
            self.seq.add_module("relu", nn.ReLU(inplace=inplace))

    def forward(self, x):
        return self.seq(x)


def make_upsample_module(scale_factor: int,
                         upsample_mode: Union[str, UpsampleMode],
                         align_corners: Optional[bool] = None):
    """
    Factory method for creating upsampling modules.
    :param scale_factor: upsample scale factor
    :param upsample_mode: see UpsampleMode for supported options.
    :return: nn.Module
    """
    upsample_mode = upsample_mode.value if isinstance(upsample_mode, UpsampleMode) else upsample_mode
    if upsample_mode == UpsampleMode.NEAREST.value:
        # Prevent ValueError when passing align_corners with nearest mode.
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)
    elif upsample_mode in [UpsampleMode.BILINEAR.value, UpsampleMode.BICUBIC.value]:
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode, align_corners=align_corners)
    else:
        raise NotImplementedError(f"Upsample mode: `{upsample_mode}` is not supported.")
    return module


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, dropout: float):
        super(SegmentationHead, self).__init__()
        self.seg_head = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.seg_head(x)

    def replace_num_classes(self, num_classes: int):
        """
        This method replace the last Conv Classification layer to output a different number of classes.
        Note that the weights of the new layers are random initiated.
        """
        old_cls_conv = self.seg_head[-1]
        self.seg_head[-1] = nn.Conv2d(old_cls_conv.in_channels, num_classes, kernel_size=1, bias=False)


class STDCBlock(nn.Module):
    """
    STDC building block, known as Short Term Dense Concatenate module.
    In STDC module, the kernel size of first block is 1, and the rest of them are simply set as 3.
    """

    def __init__(self, in_channels: int, out_channels: int, steps: int, stdc_downsample_mode: str, stride: int):
        """
        :param steps: The total number of convs in this module, 1 conv 1x1 and (steps - 1) conv3x3.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        """
        super().__init__()
        if steps not in [2, 3, 4]:
            raise ValueError(f"only 2, 3, 4 steps number are supported, found: {steps}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.steps = steps
        self.stdc_downsample_mode = stdc_downsample_mode
        self.stride = stride
        self.conv_list = nn.ModuleList()
        # build first step conv 1x1.
        self.conv_list.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, bias=False))
        # build skip connection after first convolution.
        if stride == 1:
            self.skip_step1 = Residual()
        elif stdc_downsample_mode == "avg_pool":
            self.skip_step1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif stdc_downsample_mode == "dw_conv":
            self.skip_step1 = ConvBNReLU(
                out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False, groups=out_channels // 2, use_activation=False
            )
        else:
            raise ValueError(f"stdc_downsample mode is not supported: found {stdc_downsample_mode}," f" must be in [avg_pool, dw_conv]")

        in_channels = out_channels // 2
        mid_channels = in_channels
        # build rest conv3x3 layers.
        for idx in range(1, steps):
            if idx < steps - 1:
                mid_channels //= 2
            conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_list.append(conv)
            in_channels = mid_channels

        # add dw conv before second step for down sample if stride = 2.
        if stride == 2:
            self.conv_list[1] = nn.Sequential(
                ConvBNReLU(
                    out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, groups=out_channels // 2, use_activation=False, bias=False
                ),
                self.conv_list[1],
            )

    def forward(self, x):
        out_list = []
        # run first conv
        x = self.conv_list[0](x)
        out_list.append(self.skip_step1(x))

        for conv in self.conv_list[1:]:
            x = conv(x)
            out_list.append(x)

        out = torch.cat(out_list, dim=1)
        return out



class AbstractSTDCBackbone(nn.Module, ABC):
    """
    All backbones for STDC segmentation models must implement this class.
    """

    def validate_backbone(self):
        if len(self.get_backbone_output_number_of_channels()) != 3:
            raise ValueError(f"Backbone for STDC segmentation must output 3 feature maps," f" found: {len(self.get_backbone_output_number_of_channels())}.")

    @abstractmethod
    def get_backbone_output_number_of_channels(self) -> List[int]:
        """
        :return: list on stages num channels.
        """
        raise NotImplementedError()



class STDCBackbone(AbstractSTDCBackbone):
    def __init__(
        self,
        block_types: list,
        ch_widths: list,
        num_blocks: list,
        stdc_steps: int = 4,
        stdc_downsample_mode: str = "avg_pool",
        in_channels: int = 3,
        out_down_ratios: Union[tuple, list] = (32,),
    ):
        """
        :param block_types: list of block type for each stage, supported `conv` for ConvBNRelu with 3x3 kernel.
        :param ch_widths: list of output num of channels for each stage.
        :param num_blocks: list of the number of repeating blocks in each stage.
        :param stdc_steps: num of convs steps in each block.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :param in_channels: num channels of the input image.
        :param out_down_ratios: down ratio of output feature maps required from the backbone,
            default (32,) for classification.
        """
        super(STDCBackbone, self).__init__()
        if not (len(block_types) == len(ch_widths) == len(num_blocks)):
            raise ValueError(
                f"STDC architecture configuration, block_types, ch_widths, num_blocks, must be defined for the same number"
                f" of stages, found: {len(block_types)} for block_type, {len(ch_widths)} for ch_widths, "
                f"{len(num_blocks)} for num_blocks"
            )

        self.out_widths = []
        self.stages = nn.ModuleDict()
        self.out_stage_keys = []
        down_ratio = 2
        for block_type, width, blocks in zip(block_types, ch_widths, num_blocks):
            block_name = f"block_s{down_ratio}"
            self.stages[block_name] = self._make_stage(
                in_channels=in_channels,
                out_channels=width,
                block_type=block_type,
                num_blocks=blocks,
                stdc_steps=stdc_steps,
                stdc_downsample_mode=stdc_downsample_mode,
            )
            if down_ratio in out_down_ratios:
                self.out_stage_keys.append(block_name)
                self.out_widths.append(width)
            in_channels = width
            down_ratio *= 2

    def _make_stage(self, in_channels: int, out_channels: int, block_type: str, num_blocks: int, stdc_downsample_mode: str, stdc_steps: int = 4):
        """
        :param in_channels: input channels of stage.
        :param out_channels: output channels of stage.
        :param block_type: stage building block, supported `conv` for 3x3 ConvBNRelu, or `stdc` for STDCBlock.
        :param num_blocks: num of blocks in each stage.
        :param stdc_steps: number of conv3x3 steps in each STDC block, referred as `num blocks` in paper.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :return: nn.Module
        """
        if block_type == "conv":
            block = ConvBNReLU
            kwargs = {"kernel_size": 3, "padding": 1, "bias": False}
        elif block_type == "stdc":
            block = STDCBlock
            kwargs = {"steps": stdc_steps, "stdc_downsample_mode": stdc_downsample_mode}
        else:
            raise ValueError(f"Block type not supported: {block_type}, excepted: `conv` or `stdc`")

        # first block to apply stride 2.
        blocks = nn.ModuleList([block(in_channels, out_channels, stride=2, **kwargs)])
        # build rest of blocks
        for i in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels, stride=1, **kwargs))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        outputs.append(x)#jim 0616
        for stage_name, stage in self.stages.items():
            x = stage(x)
            if stage_name in self.out_stage_keys:
                outputs.append(x)
        return tuple(outputs)

    def get_backbone_output_number_of_channels(self) -> List[int]:
        return self.out_widths


class STDC1Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,)):
        super().__init__(
            block_types=["conv", "conv", "stdc", "stdc", "stdc"],
            ch_widths=[32, 64, 256, 512, 1024],
            num_blocks=[1, 1, 2, 2, 2],
            stdc_steps=4,
            in_channels=in_channels,
            out_down_ratios=out_down_ratios,
        )


class STDC2Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,)):
        super().__init__(
            block_types=["conv", "conv", "stdc", "stdc", "stdc"],
            ch_widths=[32, 64, 256, 512, 1024],
            num_blocks=[1, 1, 4, 5, 3],
            stdc_steps=4,
            in_channels=in_channels,
            out_down_ratios=out_down_ratios,
        )



class GaussianRelativePE(nn.Module):
    def __init__(self, side, sigma=None):
        super().__init__()
        if sigma is None:
            sigma = (side / 5)
        self.sigma_square = sigma ** 2
        self.alpha = 1 / (2 * math.pi * self.sigma_square)
        self.side = side
        coord_r = torch.stack([torch.arange(side) for _ in range(side)])
        coord_c = coord_r.T
        self.register_buffer('coord_r', coord_r)
        self.register_buffer('coord_c', coord_c)

    def forward(self, x, center):
        #print("ff",x.device,center.device)
        pe = self.alpha * torch.exp(- ((self.coord_r.view(1, self.side, self.side) - center[:, 0:1].unsqueeze(1)) ** 2 + \
            (self.coord_c.view(1, self.side, self.side) - center[:, 1:2].unsqueeze(1)) ** 2) / (2 * self.sigma_square))
        pe /= pe.amax(dim=(-1, -2)).view(-1, 1, 1)
        return x + pe.unsqueeze(1)



class SPPM(nn.Module):
    """
    Simple Pyramid Pooling context Module.
    """
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 pool_sizes: List[Union[int, Tuple[int, int]]],
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
                 align_corners: bool = False):
        """
        :param inter_channels: num channels in each pooling branch.
        :param out_channels: The number of output channels after pyramid pooling module.
        :param pool_sizes: spatial output sizes of the pooled feature maps.
        """
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBNReLU(in_channels, inter_channels, kernel_size=1, bias=False),
            ) for pool_size in pool_sizes
        ])
        self.conv_out = ConvBNReLU(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.pool_sizes = pool_sizes

    def forward(self, x):
        out = None
        input_shape = x.shape[2:]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=input_shape, mode=self.upsample_mode, align_corners=self.align_corners)
            out = y if out is None else out + y
        out = self.conv_out(out)
        return out

    def prep_model_for_conversion(self, input_size: Union[tuple, list], stride_ratio: int = 32, **kwargs):
        """
        Replace Global average pooling with fixed kernels Average pooling, since dynamic kernel sizes are not supported
        when compiling to ONNX: `Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not accessible.`
        """
        input_size = [x / stride_ratio for x in input_size[-2:]]
        for branch in self.branches:
            global_pool: nn.AdaptiveAvgPool2d = branch[0]
            out_size = global_pool.output_size
            out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)
            kernel_size = [int(i / o) for i, o in zip(input_size, out_size)]
            branch[0] = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)


class UAFM(nn.Module):
    """
    Unified Attention Fusion Module, which uses mean and max values across the spatial dimensions.
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 up_factor: int,
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.BILINEAR,
                 align_corners: bool = False):
        """
        :params in_channels: num_channels of input feature map.
        :param skip_channels: num_channels of skip connection feature map.
        :param out_channels: num out channels after features fusion.
        :param up_factor: upsample scale factor of the input feature map.
        :param upsample_mode: see UpsampleMode for valid options.
        """
        super().__init__()
        self.conv_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False),
            ConvBNReLU(2, 1, kernel_size=3, padding=1, bias=False, use_activation=False)
        )

        self.proj_skip = nn.Identity() if skip_channels == in_channels else \
            ConvBNReLU(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.up_x = nn.Identity() if up_factor == 1 else \
            make_upsample_module(scale_factor=up_factor, upsample_mode=upsample_mode, align_corners=align_corners)
        self.conv_out = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skip):
        """
        :param x: input feature map to upsample before fusion.
        :param skip: skip connection feature map.
        """
        x = self.up_x(x)
        skip = self.proj_skip(skip)

        atten = torch.cat([
            *self._avg_max_spatial_reduce(x, use_concat=False),
            *self._avg_max_spatial_reduce(skip, use_concat=False)
        ], dim=1)
        atten = self.conv_atten(atten)
        atten = torch.sigmoid(atten)
        test=x*atten
        out = x * atten + skip * (1 - atten)
        out = self.conv_out(out)
        return out

    @staticmethod
    def _avg_max_spatial_reduce(x, use_concat: bool = False):
        reduced = [
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ]
        if use_concat:
            reduced = torch.cat(reduced, dim=1)
        return reduced


class PPLiteSegEncoder(nn.Module):
    """
    Encoder for PPLiteSeg, include backbone followed by a context module.
    """
    def __init__(self,
                 backbone: AbstractSTDCBackbone,
                 projection_channels_list: List[int],
                 context_module: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.context_module = context_module
        feats_channels = backbone.get_backbone_output_number_of_channels()
        self.proj_convs = nn.ModuleList([
            ConvBNReLU(feat_ch, proj_ch, kernel_size=3, padding=1, bias=False)
            for feat_ch, proj_ch in zip(feats_channels, projection_channels_list)
        ])
        self.projection_channels_list = projection_channels_list

    def get_output_number_of_channels(self) -> List[int]:
        channels_list = self.projection_channels_list
        if hasattr(self.context_module, "out_channels"):
            channels_list.append(self.context_module.out_channels)
        return channels_list

    def forward(self, x):
        feats = self.backbone(x)
        feats_reverse = feats[::-1]
        feats_reverse_list=list(feats_reverse)
        orin_feat = feats_reverse_list.pop()
        orin_first_channel=orin_feat[:, :1, :, :]
        feats_reverse_tuple=tuple(feats_reverse_list)
        feats=feats_reverse_tuple[::-1]
      

        y = self.context_module(feats[-1])
        feats = [conv(f) for conv, f in zip(self.proj_convs, feats)]
        return feats + [y],orin_first_channel


class PPLiteSegDecoder(nn.Module):
    """
    PPLiteSegDecoder using UAFM blocks to fuse feature maps.
    """
    def __init__(self,
                 encoder_channels: List[int],
                 up_factors: List[int],
                 out_channels: List[int],
                 upsample_mode,
                 align_corners: bool):
        super().__init__()
        # Make a copy of channels list, to prevent out of scope changes.
        encoder_channels = encoder_channels.copy()
        encoder_channels.reverse()
        in_channels = encoder_channels.pop(0)

        # TODO - assert argument length
        self.up_stages = nn.ModuleList()
        for skip_ch, up_factor, out_ch in zip(encoder_channels, up_factors, out_channels):
            self.up_stages.append(UAFM(
                in_channels=in_channels,
                skip_channels=skip_ch,
                out_channels=out_ch,
                up_factor=up_factor,
                upsample_mode=upsample_mode,
                align_corners=align_corners
            ))
            in_channels = out_ch

    def forward(self, feats: List[torch.Tensor],orin_map_channel:torch.Tensor):
        feats.reverse()
        x = feats.pop(0)

        for up_stage, skip in zip(self.up_stages, feats):
            x = up_stage(x, skip)
        sizex=x.size()
        out_dim=sizex[-1]
        orin_map_channel = F.interpolate(orin_map_channel, size=(out_dim,out_dim), mode='bilinear', align_corners=False)
        mask=1-orin_map_channel
        x = x*mask
        # up_stage_0 = self.up_stages[0]
        # up_stage_1 = self.up_stages[1]
        # up_stage_2 = self.up_stages[2]

        # skip_0 = feats[0]
        # skip_1 = feats[1]
        # skip_2 = feats[2]
        # x_1 = up_stage_0(x, skip_0)
        # x_2 = up_stage_1(x_1, skip_1)
        # x_3 = up_stage_2(x_2, skip_2)


        return x


class PPLiteSegBase(SegmentationModule):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.
    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".
    """
    def __init__(self,
                 num_classes,
                 backbone: AbstractSTDCBackbone,
                 projection_channels_list: List[int],
                 sppm_inter_channels: int,
                 sppm_out_channels: int,
                 sppm_pool_sizes: List[int],
                 sppm_upsample_mode: Union[UpsampleMode, str],
                 align_corners: bool,
                 decoder_up_factors: List[int],
                 decoder_channels: List[int],
                 decoder_upsample_mode: Union[UpsampleMode, str],
                 head_scale_factor: int,
                 head_upsample_mode: Union[UpsampleMode, str],
                 head_mid_channels: int,
                 dropout: float,
                 use_aux_heads: bool,
                 aux_hidden_channels: List[int],
                 aux_scale_factors: List[int]
                 ):
        """
        :param backbone: Backbone nn.Module should implement the abstract class `AbstractSTDCBackbone`.
        :param projection_channels_list: channels list to project encoder features before fusing with the decoder
            stream.
        :param sppm_inter_channels: num channels in each sppm pooling branch.
        :param sppm_out_channels: The number of output channels after sppm module.
        :param sppm_pool_sizes: spatial output sizes of the pooled feature maps.
        :param sppm_upsample_mode: Upsample mode to original size after pooling.
        :param decoder_up_factors: list upsample factor per decoder stage.
        :param decoder_channels: list of num_channels per decoder stage.
        :param decoder_upsample_mode: upsample mode in decoder stages, see UpsampleMode for valid options.
        :param head_scale_factor: scale factor for final the segmentation head logits.
        :param head_upsample_mode: upsample mode to final prediction sizes, see UpsampleMode for valid options.
        :param head_mid_channels: num of hidden channels in segmentation head.
        :param use_aux_heads: set True when training, output extra Auxiliary feature maps from the encoder module.
        :param aux_hidden_channels: List of hidden channels in auxiliary segmentation heads.
        :param aux_scale_factors: list of uppsample factors for final auxiliary heads logits.
        """
        super().__init__(use_aux_heads=use_aux_heads)

        # jim 230424
        self.pe = GaussianRelativePE(256)

        # Init Encoder
        backbone_out_channels = backbone.get_backbone_output_number_of_channels()
        assert len(backbone_out_channels) == len(projection_channels_list), \
            f"The length of backbone outputs ({backbone_out_channels}) should match the length of projection channels" \
            f"({len(projection_channels_list)})."
        context = SPPM(in_channels=backbone_out_channels[-1],
                       inter_channels=sppm_inter_channels,
                       out_channels=sppm_out_channels,
                       pool_sizes=sppm_pool_sizes,
                       upsample_mode=sppm_upsample_mode,
                       align_corners=align_corners)
        self.encoder = PPLiteSegEncoder(backbone=backbone,
                                        context_module=context,
                                        projection_channels_list=projection_channels_list)
        encoder_channels = self.encoder.get_output_number_of_channels()

        # Init Decoder
        self.decoder = PPLiteSegDecoder(encoder_channels=encoder_channels,
                                        up_factors=decoder_up_factors,
                                        out_channels=decoder_channels,
                                        upsample_mode=decoder_upsample_mode,
                                        align_corners=align_corners)

        # Init Segmentation classification heads
        self.seg_head = nn.Sequential(
            SegmentationHead(in_channels=decoder_channels[-1],
                             mid_channels=head_mid_channels,
                             num_classes=num_classes,
                             dropout=dropout),
            make_upsample_module(scale_factor=head_scale_factor, upsample_mode=head_upsample_mode,
                                 align_corners=align_corners)
        )
        # Auxiliary heads
        if self.use_aux_heads:
            encoder_out_channels = projection_channels_list
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    SegmentationHead(backbone_ch, hidden_ch, num_classes, dropout=dropout),
                    make_upsample_module(scale_factor=scale_factor, upsample_mode=head_upsample_mode,
                                         align_corners=align_corners)
                ) for backbone_ch, hidden_ch, scale_factor in zip(encoder_out_channels, aux_hidden_channels,
                                                                  aux_scale_factors)
            ])
        self.init_params()

    # jim 230424
    def pe_forward(self, x, start, goal):
        zeros = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        #print("dd",zeros.device,start.device)
        pe_start = self.pe(zeros, start)
        #print(pe_start.device)
        pe_goal = self.pe(zeros, goal)
        return torch.cat([x, pe_start, pe_goal], dim=1)

    def _remove_auxiliary_heads(self):
        if hasattr(self, "aux_heads"):
            del self.aux_heads

    @property
    def backbone(self) -> nn.Module:
        """
        Support SG load backbone when training.
        """
        return self.encoder.backbone

    def forward(self, x,start,goal):
        x = self.pe_forward(x, start, goal)
        feats,orin_map = self.encoder(x)
        if self.use_aux_heads:
            enc_feats = feats[:-1]
        x = self.decoder(feats,orin_map)
        x = self.seg_head(x)
        if not self.use_aux_heads:
            x = torch.sigmoid(x)
            return x
        aux_feats = [aux_head(feat) for feat, aux_head in zip(enc_feats, self.aux_heads)]
        return tuple([x] + aux_feats)

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for training:
            - Different lr for backbone and the rest, if `multiply_head_lr` key is in `training_params`.
        """
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        multiply_lr_params, no_multiply_params = self._separate_lr_multiply_params()
        param_groups = [{"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
                        {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"}]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct,
                            total_batch: int) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        for param_group in param_groups:
            param_group['lr'] = lr
            if param_group["name"] == "multiply_lr_params":
                param_group['lr'] *= multiply_head_lr
        return param_groups

    def _separate_lr_multiply_params(self):
        """
        Separate backbone params from the rest.
        :return: iterators of groups named_parameters.
        """
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if "encoder.backbone" in name:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()

    def prep_model_for_conversion(self, input_size: Union[tuple, list], stride_ratio: int = 32, **kwargs):
        super().prep_model_for_conversion(input_size, **kwargs)
        if isinstance(self.encoder.context_module, SPPM):
            self.encoder.context_module.prep_model_for_conversion(input_size=input_size, stride_ratio=stride_ratio)


class PPLiteSegB(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "in_channels", 3),
                                 out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=get_param(arch_params, "num_classes"),
                         backbone=backbone,
                         projection_channels_list=[96, 128, 128],
                         sppm_inter_channels=128,
                         sppm_out_channels=128,
                         sppm_pool_sizes=[1, 2, 4],
                         sppm_upsample_mode="bilinear",
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[128, 96, 64],
                         decoder_upsample_mode="bilinear",
                         head_scale_factor=8,
                         head_upsample_mode="bilinear",
                         head_mid_channels=64,
                         dropout=get_param(arch_params, "dropout", 0.),
                         use_aux_heads=get_param(arch_params, "use_aux_heads", False),
                         aux_hidden_channels=[32, 64, 64],
                         aux_scale_factors=[8, 16, 32])


class PPLiteSegT_test(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC1Backbone(in_channels=get_param(arch_params, "in_channels", 3),
                                 out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=get_param(arch_params, "num_classes"),
                         backbone=backbone,
                         projection_channels_list=[64, 128, 128],
                         sppm_inter_channels=128,
                         sppm_out_channels=128,
                         sppm_pool_sizes=[1, 2, 4],
                         sppm_upsample_mode="bilinear",
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[128, 64, 32],
                         decoder_upsample_mode="bilinear",
                         head_scale_factor=8,
                         head_upsample_mode="bilinear",
                         head_mid_channels=32,
                         dropout=get_param(arch_params, "dropout", 0.),
                         use_aux_heads=get_param(arch_params, "use_aux_heads", False),
                         aux_hidden_channels=[32, 64, 64],
                         aux_scale_factors=[8, 16, 32])

class PPLiteSegB_H(PPLiteSegBase):
    def __init__(self, arch_params: HpmStruct):
        backbone = STDC2Backbone(in_channels=get_param(arch_params, "in_channels", 3),
                                 out_down_ratios=[8, 16, 32])
        super().__init__(num_classes=get_param(arch_params, "num_classes"),
                         backbone=backbone,
                         projection_channels_list=[192, 512, 512],
                         sppm_inter_channels=512,
                         sppm_out_channels=512,
                         sppm_pool_sizes=[1, 2, 4],
                         sppm_upsample_mode="bilinear",
                         align_corners=False,
                         decoder_up_factors=[1, 2, 2],
                         decoder_channels=[512, 192, 128],
                         decoder_upsample_mode="bilinear",
                         head_scale_factor=8,
                         head_upsample_mode="bilinear",
                         head_mid_channels=128,
                         dropout=get_param(arch_params, "dropout", 0.),
                         use_aux_heads=get_param(arch_params, "use_aux_heads", False),
                         aux_hidden_channels=[64, 128, 128],
                         aux_scale_factors=[8, 16, 32])


if __name__ == '__main__':
    params = {"in_channels": 3, "num_classes": 1}
    arch_params = HpmStruct(**params)
    model = PPLiteSegB_H(arch_params)
    data_sample = torch.randn(2, 3, 512, 512)
    output = model(data_sample)

    print(output.size())
    # ua1=UAFM(64,64,32,2)

    # in1=torch.randn(2,64,32,32)
    # in2=torch.randn(2,64,64,64)

    # out=ua1(in1,in2)
    # print(out.size())