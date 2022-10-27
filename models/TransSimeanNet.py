from inspect import GEN_CLOSED
from turtle import forward
from unittest import skip
from cv2 import norm
import cv2
from numpy.core.arrayprint import printoptions
import torch
from torch import nn
import torch.nn.functional as F
# import network.mynn as mynn

# from vessel_Net import VisionTransformer
# import vit_seg_configs as configs
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import copy
import logging
import math
from os.path import join as pjoin
from torch._C import Size
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from collections import OrderedDict
from scipy import ndimage
import ml_collections
from torch.nn.modules.conv import _ConvNd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = mynn.Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = mynn.Norm2d(planes)
        self.downsample = downsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = './R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print(attention_scores.size())
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x,m, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x) 
        # print('ssss',x.size()) # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        # print('eee',x.size())
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print('sss',x.size())
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings,m, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output,m, features = self.embeddings(input_ids)
        # print(embedding_output.size())
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # print('x',encoded.size())

        return encoded, attn_weights, m, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
            

        else:
            skip_channels=[0,0,0,0]
        # print(skip_channels)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        # print('features:')
        # print(features[0].size())
        # print(features[1].size())
        # print(features[2].size())
        x = self.conv_more(x)
        # print('2',x.size())

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=512, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.conv_more1=Conv2dReLU(
            512,
            512,
            3,
            padding=1,
            use_batchnorm=True
        )
        self.conv_more2=Conv2dReLU(
            512,
            512,
            3,
            padding=1,
            use_batchnorm=True
        )
        self.conv_more3=Conv2dReLU(
            512,
            512,
            3,
            padding=1,
            use_batchnorm=True
        )
        self.pool=nn.MaxPool2d(2, 2)

    def forward(self, x):
        # print('1',x.size())
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights,m, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        skip1 = self.pool(self.conv_more1(x))
        # print('3',skip1.size())
        skip2 = self.pool(self.conv_more2(skip1))
        x = self.pool(self.conv_more3(skip2))
        return skip1,skip2,x,m,features

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

# CONFIGS = {
#     'ViT-B_16': configs.get_b16_config(),
#     'ViT-B_32': configs.get_b32_config(),
#     'ViT-L_16': configs.get_l16_config(),
#     'ViT-L_32': configs.get_l32_config(),
#     'ViT-H_14': configs.get_h14_config(),
#     'R50-ViT-B_16': configs.get_r50_b16_config(),
#     'R50-ViT-L_16': configs.get_r50_l16_config(),
#     'testing': configs.get_testing(),
# }

class SEDown(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.se = SE(in_channels)
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.gate_conv(x)
        se = self.se(x)
        x = x*se
        return self.down(x)




class GEC(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,map_size = 1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GEC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = SEDown(in_channels+map_size)
        # self.se = SE(in_channels+1)

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class FusionModel(nn.Module):
    def __init__(self,in_dim,reduction_dim=256):
        super().__init__()
        self.vgg_block = VGGBlock(512+512,512,512)
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
        self.img_conv2 = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self,x,skip,edge):
        
        x_size = x.size()
        img_features = self.img_pooling(x)
        # print(img_features.size())
        
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear',align_corners=True)
        
        out = self.img_conv2(x)
        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out,edge_features),1)

        x = self.vgg_block(torch.cat((x,skip),1))
        out = torch.cat((x,out),1)
        out = self.up(out)
        return out
        


class TransSimUNet(nn.Module):
    def __init__(self,config):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        
        self.transformer=VisionTransformer(config,img_size=512, num_classes=21843, zero_head=False, vis=False)
        # self.transformer.load_from(config.pretrained_path)
        self.feature0_conv=nn.Conv2d(512,256,1,1)

        self.feature1_conv=nn.Conv2d(256,128,1,1)
        self.feature2_conv=nn.Conv2d(64,64,1,1)


        self.num_classes = 1
        self.input_channels = 3
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sa0 = SA(nb_filter[0])
        self.sa1 = SA(nb_filter[1])
        self.sa2 = SA(nb_filter[2])
        self.sa3 = SA(nb_filter[3])
        self.sa4 = SA(nb_filter[4])
        self.sigmoid = nn.Sigmoid()
        # self.se = SE(3,16)


        # self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.conv0_0 = Conv_layer(self.input_channels,nb_filter[0])
        
        # self.conv0_0 = Bottleneck(self.input_channels,nb_filter[0])
        self.se0 = SE(nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_0 = BasicBlock(nb_filter[0],nb_filter[1],downsample=nn.Conv2d(nb_filter[0],nb_filter[1],1,1))
        self.se1 = SE(nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_0 = BasicBlock(nb_filter[1],nb_filter[2],downsample=nn.Conv2d(nb_filter[1],nb_filter[2],1,1))
        # self.conv2_0 = Bottleneck(nb_filter[1])
        self.se2 = SE(nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_0 = Bottleneck(nb_filter[2],nb_filter[3]//4,downsample=nn.Conv2d(nb_filter[2],nb_filter[3],1,1))
        self.se3 = SE(nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv4_0 = Bottleneck(nb_filter[3],nb_filter[4]//4,downsample=nn.Conv2d(nb_filter[3],nb_filter[4],1,1))
       
        self.se4 = SE(nb_filter[4])
        # self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se5 = SE(nb_filter[4])
        # self.conv6_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv6_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se6 = SE(nb_filter[4])
        # self.conv7_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv7_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se7 = SE(nb_filter[4])

        self.fusion_model = FusionModel(512)

        
        self.map_attention1 = nn.Conv2d(64,1,1)
        self.map_attention1_1 = nn.Conv2d(64,1,1)
        self.d1 = nn.Conv2d(64,32,1)
        self.map_attention2 = nn.Conv2d(128,1,1)
        self.map_attention2_1 = nn.Conv2d(256,1,1)
        self.d2 = nn.Conv2d(32,16,1)
        self.map_attention3 = nn.Conv2d(512,1,1)
        self.map_attention3_1 = nn.Conv2d(512,1,1)
        self.d3 = nn.Conv2d(16,8,1)
        self.map_attention4 = nn.Conv2d(512,1,1)
        self.map_attention4_1 = nn.Conv2d(512,1,1)
        self.d4 = nn.Conv2d(8,4,1)
        self.d5 = nn.Conv2d(4,1,1)
        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.gate1 = GEC(32,32,map_size=2)
        self.gate2 = GEC(16,16,map_size=2)
        self.gate3 = GEC(8,8,map_size=2)
        self.gate4 = GEC(4,4,map_size=2)
        

        self.se_model1 = BasicBlock(32, 32, stride=1, downsample=None)
        self.se_model1_1 = BasicBlock(32, 32, stride=1, downsample=None)
        self.se_model2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.se_model3 = BasicBlock(16, 16, stride=1, downsample=None)
        self.se_model4 = BasicBlock(8, 8, stride=1, downsample=None)
        self.se_model5 = BasicBlock(4, 4, stride=1, downsample=None)

        self.conv6_1 = VGGBlock(nb_filter[4]+nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_2 = VGGBlock(nb_filter[4]+nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv4_3 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv3_4 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_5 = VGGBlock(nb_filter[2]+nb_filter[3]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_6 = VGGBlock(nb_filter[1]+nb_filter[2]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_7 = VGGBlock(nb_filter[0]+nb_filter[1]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        self.short_cut = nn.Sequential()  #将输入与剩余输出匹配

        

    def forward(self, input):
        x_size = input.size()
        skip1,skip2,skip3,m,features=self.transformer(input)
        
        im_arr = input.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)/255
        canny = torch.from_numpy(canny).to(input.device).float()

        
        
        x0_0, bx = self.conv0_0(input)       #224
        # m1 = self.map_attention1(bx)
        bx = self.se_model1(bx)
        _tbx = self.se_model1_1(m)
        # bx = self.gate1(bx,m1)
        # bx = self.se_model2(bx)
        bx = self.d1(torch.cat((bx,_tbx),1)) #32



        # se0_1 = self.se0(x0_0)
        # se0_2 = x0_0*se0_1
        # x0_0 = x0_0+self.short_cut(se0_2)
        # print(x0_0.size())
        # x0_0 = F.relu(x0_0)
        # sa0_1 = self.sa0(x0_0)
        # sa0_2 = x0_0*sa0_1
        # x0_0 = x0_0+self.short_cut(sa0_2)

        x1_0 = self.conv1_0(self.pool(x0_0))    # b 64 256 256

        m1 = self.map_attention1(x1_0)
        m1_1 = self.map_attention1_1(features[2])
        m1 = F.interpolate(m1,x_size[2:],mode='bilinear', align_corners=True)
        m1_1 = F.interpolate(m1_1,x_size[2:],mode='bilinear', align_corners=True)

        bx = self.gate1(bx,torch.cat((m1,m1_1),1))
        bx = self.se_model2(bx)
        bx = self.d2(bx) # 16

        # se1_1 = self.se1(x1_0)
        # se1_2 = x1_0*se1_1
        # x1_0 = x1_0+self.short_cut(se1_2)
        # x1_0 = F.relu(x1_0)
        # sa1_1 = self.sa1(x1_0)
        # sa1_2 = x1_0*sa1_1
        # x1_0 = x1_0+self.short_cut(sa1_2)

        x2_0 = self.conv2_0(self.pool(x1_0))    #64

        m2 = self.map_attention2(x2_0)
        m2_1 = self.map_attention2_1(features[1])

        m2 = F.interpolate(m2,x_size[2:],mode='bilinear', align_corners=True)
        m2_1 = F.interpolate(m2_1,x_size[2:],mode='bilinear', align_corners=True)

        bx = self.gate2(bx,torch.cat((m2,m2_1),1))
        bx = self.se_model3(bx)
        bx = self.d3(bx) # 8
        
        # se2_1 = self.se2(x2_0)
        # se2_2 = x2_0*se2_1
        # x2_0 = x2_0+self.short_cut(se2_2)
        # x2_0 = F.relu(x2_0)
        # sa2_1 = self.sa2(x2_0)
        # sa2_2 = x2_0*sa2_1
        # x2_0 = x2_0+self.short_cut(sa2_2)

        x3_0 = self.conv3_0(self.pool(x2_0))    #32 64
        
        # se3_1 = self.se3(x3_0)
        # se3_2 = x3_0*se3_1
        # x3_0 = x3_0+self.short_cut(se3_2)
        # x3_0 = F.relu(x3_0)
        # sa3_1 = self.sa3(x3_0)
        # sa3_2 = x3_0*sa3_1
        # x3_0 = x3_0+self.short_cut(sa3_2)

        x4_0 = self.conv4_0(self.pool(x3_0))    #16 32
        
        # se4_1 = self.se4(x4_0)
        # se4_2 = x4_0*se4_1
        # x4_0 = x4_0+self.short_cut(se4_2)


        x5_0 = self.conv5_0(self.pool(x4_0))    #8  16 

        
        # se5_1 = self.se5(x5_0)
        # se5_2 = x5_0*se5_1
        # x5_0 = x5_0+self.short_cut(se5_2)

        x6_0 = self.conv6_0(self.pool(x5_0))    #4 8
        m3 = self.map_attention3(x5_0)
        m3_1 = self.map_attention3_1(skip1)
        m3 = F.interpolate(m3,x_size[2:],mode='bilinear', align_corners=True)
        m3_1 = F.interpolate(m3_1,x_size[2:],mode='bilinear', align_corners=True)
        bx = self.gate3(bx,torch.cat((m3,m3_1),1))
        bx = self.se_model4(bx)
        bx = self.d4(bx) # 4
        # se6_1 = self.se6(x6_0)
        # se6_2 = x6_0*se6_1
        # x6_0 = x6_0+self.short_cut(se6_2)

        
        x7_0 = self.conv7_0(self.pool(x6_0))    #2
        m4 = self.map_attention4(x6_0)
        m4_1 = self.map_attention4_1(skip2)
        m4 = F.interpolate(m4,x_size[2:],mode='bilinear', align_corners=True)
        m4_1 = F.interpolate(m4_1,x_size[2:],mode='bilinear', align_corners=True)
        bx = self.gate4(bx,torch.cat((m4,m4_1),1))
        # bx = self.gate4(bx,m4)
        # bx = self.gate5(bx,m4_1)
        bx = self.se_model5(bx)
        bx = self.d5(bx)
        edge = self.sigmoid(bx)

        

        cat = torch.cat((bx, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        
        fusion_out = self.fusion_model(x7_0,skip2,acts)
        
        
        x6_1 = self.conv6_1(torch.cat([x6_0, fusion_out], 1))
        x5_2 = self.conv5_2(torch.cat([x5_0, self.up(torch.cat([x6_1,skip1],1))], 1))
        x4_3 = self.conv4_3(torch.cat([x4_0, self.up(x5_2)], 1))
        x3_4 = self.conv3_4(torch.cat([x3_0, self.up(x4_3)], 1))
        # print(22222222222,x3_4.size())
        x2_5 = self.conv2_5(torch.cat([x2_0, self.up(torch.cat([x3_4,self.feature0_conv(features[0])],1))], 1))
        # print(3,x2_5.size())
        x1_6 = self.conv1_6(torch.cat([x1_0, self.up(torch.cat([x2_5,self.feature1_conv(features[1])],1))], 1))
        # print(3,x1_6.size())
        x0_7 = self.conv0_7(torch.cat([x0_0, self.up(torch.cat([x1_6,self.feature2_conv(features[2])],1))], 1))

        output = self.final(x0_7)
        # print(output.size())
        return output,edge

class TransSimUNetDeskip(nn.Module):
    def __init__(self,config):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        
        self.transformer=VisionTransformer(config,img_size=512, num_classes=21843, zero_head=False, vis=False)
        # self.transformer.load_from(config.pretrained_path)
        self.feature0_conv=nn.Conv2d(512,256,1,1)

        self.feature1_conv=nn.Conv2d(256,128,1,1)
        self.feature2_conv=nn.Conv2d(64,64,1,1)


        self.num_classes = 1
        self.input_channels = 3
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sa0 = SA(nb_filter[0])
        self.sa1 = SA(nb_filter[1])
        self.sa2 = SA(nb_filter[2])
        self.sa3 = SA(nb_filter[3])
        self.sa4 = SA(nb_filter[4])
        # self.se = SE(3,16)


        self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        
        # self.conv0_0 = Bottleneck(self.input_channels,nb_filter[0])
        self.se0 = SE(nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_0 = BasicBlock(nb_filter[0],nb_filter[1],downsample=nn.Conv2d(nb_filter[0],nb_filter[1],1,1))
        self.se1 = SE(nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_0 = BasicBlock(nb_filter[1],nb_filter[2],downsample=nn.Conv2d(nb_filter[1],nb_filter[2],1,1))
        # self.conv2_0 = Bottleneck(nb_filter[1])
        self.se2 = SE(nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_0 = Bottleneck(nb_filter[2],nb_filter[3]//4,downsample=nn.Conv2d(nb_filter[2],nb_filter[3],1,1))
        self.se3 = SE(nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv4_0 = Bottleneck(nb_filter[3],nb_filter[4]//4,downsample=nn.Conv2d(nb_filter[3],nb_filter[4],1,1))
       
        self.se4 = SE(nb_filter[4])
        # self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se5 = SE(nb_filter[4])
        # self.conv6_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv6_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se6 = SE(nb_filter[4])
        # self.conv7_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv7_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se7 = SE(nb_filter[4])

        


        self.conv6_1 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_2 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv4_3 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv3_4 = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_5 = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_6 = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_7 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        self.short_cut = nn.Sequential()  #将输入与剩余输出匹配

    def forward(self, input):
        skip1,skip2,skip3,features=self.transformer(input)
        x0_0 = self.conv0_0(input)       #224
        # se0_1 = self.se0(x0_0)
        # se0_2 = x0_0*se0_1
        # x0_0 = x0_0+self.short_cut(se0_2)
        # print(x0_0.size())
        # x0_0 = F.relu(x0_0)
        # sa0_1 = self.sa0(x0_0)
        # sa0_2 = x0_0*sa0_1
        # x0_0 = x0_0+self.short_cut(sa0_2)

        x1_0 = self.conv1_0(self.pool(x0_0))    #128
        # se1_1 = self.se1(x1_0)
        # se1_2 = x1_0*se1_1
        # x1_0 = x1_0+self.short_cut(se1_2)
        # x1_0 = F.relu(x1_0)
        # sa1_1 = self.sa1(x1_0)
        # sa1_2 = x1_0*sa1_1
        # x1_0 = x1_0+self.short_cut(sa1_2)

        x2_0 = self.conv2_0(self.pool(x1_0))    #64
        # se2_1 = self.se2(x2_0)
        # se2_2 = x2_0*se2_1
        # x2_0 = x2_0+self.short_cut(se2_2)
        # x2_0 = F.relu(x2_0)
        # sa2_1 = self.sa2(x2_0)
        # sa2_2 = x2_0*sa2_1
        # x2_0 = x2_0+self.short_cut(sa2_2)

        x3_0 = self.conv3_0(self.pool(x2_0))    #32
        # se3_1 = self.se3(x3_0)
        # se3_2 = x3_0*se3_1
        # x3_0 = x3_0+self.short_cut(se3_2)
        # x3_0 = F.relu(x3_0)
        # sa3_1 = self.sa3(x3_0)
        # sa3_2 = x3_0*sa3_1
        # x3_0 = x3_0+self.short_cut(sa3_2)

        x4_0 = self.conv4_0(self.pool(x3_0))    #16
        # se4_1 = self.se4(x4_0)
        # se4_2 = x4_0*se4_1
        # x4_0 = x4_0+self.short_cut(se4_2)


        x5_0 = self.conv5_0(self.pool(x4_0))    #8
        # se5_1 = self.se5(x5_0)
        # se5_2 = x5_0*se5_1
        # x5_0 = x5_0+self.short_cut(se5_2)

        x6_0 = self.conv6_0(self.pool(x5_0))    #4
        # se6_1 = self.se6(x6_0)
        # se6_2 = x6_0*se6_1
        # x6_0 = x6_0+self.short_cut(se6_2)

        
        x7_0 = self.conv7_0(self.pool(x6_0))    #2
        # se7_1 = self.se7(x7_0)
        # se7_2 = x7_0*se7_1
        # x7_0 = x7_0+self.short_cut(se7_2)

        # x8_0 = self.conv8_0(self.pool(x7_0))    #2
        # se8_1 = self.se8(x8_0)
        # se8_2 = x8_0*se8_1
        # x8_0 = x8_0+self.short_cut(se8_2)
        # x4_0 = F.relu(x4_0)
        # sa4_1 = self.sa4(x4_0)
        # sa4_2 = x4_0*sa4_1
        # x4_0 = x4_0+self.short_cut(sa4_2)
        # print(1111111111111,x7_0.size())
        
        
        x6_1 = self.conv6_1(self.up(torch.cat([x7_0,skip2],1)))
        x5_2 = self.conv5_2(self.up(x6_1))
        x4_3 = self.conv4_3(self.up(x5_2))
        x3_4 = self.conv3_4(self.up(x4_3))
        # print(22222222222,x3_4.size())
        x2_5 = self.conv2_5(self.up(x3_4))
        # print(3,x2_5.size())
        x1_6 = self.conv1_6(self.up(x2_5))
        # print(3,x1_6.size())
        x0_7 = self.conv0_7(self.up(x1_6))

        output = self.final(x0_7)
        # print(output.size())
        return output

class Conv_layer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.norm = nn.BatchNorm2d(out_channel)
    def forward(self,x):
        m = self.conv(x)
        x = self.norm(m)
        x = self.relu(x)
        return x,m

class TransSimUNetDeglobal(nn.Module):
    def __init__(self,config):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        
        self.transformer=VisionTransformer(config,img_size=512, num_classes=21843, zero_head=False, vis=False)
        # self.transformer.load_from(config.pretrained_path)
        self.feature0_conv=nn.Conv2d(512,256,1,1)

        self.feature1_conv=nn.Conv2d(256,128,1,1)
        self.feature2_conv=nn.Conv2d(64,64,1,1)


        self.num_classes = 1
        self.input_channels = 3
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sa0 = SA(nb_filter[0])
        self.sa1 = SA(nb_filter[1])
        self.sa2 = SA(nb_filter[2])
        self.sa3 = SA(nb_filter[3])
        self.sa4 = SA(nb_filter[4])
        # self.se = SE(3,16)


        # self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        
        
        # self.conv0_0 = Bottleneck(self.input_channels,nb_filter[0])
        self.se0 = SE(nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_0 = BasicBlock(nb_filter[0],nb_filter[1],downsample=nn.Conv2d(nb_filter[0],nb_filter[1],1,1))
        self.se1 = SE(nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_0 = BasicBlock(nb_filter[1],nb_filter[2],downsample=nn.Conv2d(nb_filter[1],nb_filter[2],1,1))
        # self.conv2_0 = Bottleneck(nb_filter[1])
        self.se2 = SE(nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_0 = Bottleneck(nb_filter[2],nb_filter[3]//4,downsample=nn.Conv2d(nb_filter[2],nb_filter[3],1,1))
        self.se3 = SE(nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv4_0 = Bottleneck(nb_filter[3],nb_filter[4]//4,downsample=nn.Conv2d(nb_filter[3],nb_filter[4],1,1))
       
        self.se4 = SE(nb_filter[4])
        # self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se5 = SE(nb_filter[4])
        # self.conv6_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv6_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se6 = SE(nb_filter[4])
        # self.conv7_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv7_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se7 = SE(nb_filter[4])

        


        self.conv6_1 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_2 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv4_3 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv3_4 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_5 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_6 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_7 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        self.short_cut = nn.Sequential()  #将输入与剩余输出匹配

    def forward(self, input):
        skip1,skip2,skip3,features=self.transformer(input)
        x0_0 = self.conv0_0(input)       #224
        

        x1_0 = self.conv1_0(self.pool(x0_0))    #128
        # se1_1 = self.se1(x1_0)
        # se1_2 = x1_0*se1_1
        # x1_0 = x1_0+self.short_cut(se1_2)
        # x1_0 = F.relu(x1_0)
        # sa1_1 = self.sa1(x1_0)
        # sa1_2 = x1_0*sa1_1
        # x1_0 = x1_0+self.short_cut(sa1_2)

        x2_0 = self.conv2_0(self.pool(x1_0))    #64
        # se2_1 = self.se2(x2_0)
        # se2_2 = x2_0*se2_1
        # x2_0 = x2_0+self.short_cut(se2_2)
        # x2_0 = F.relu(x2_0)
        # sa2_1 = self.sa2(x2_0)
        # sa2_2 = x2_0*sa2_1
        # x2_0 = x2_0+self.short_cut(sa2_2)

        x3_0 = self.conv3_0(self.pool(x2_0))    #32
        # se3_1 = self.se3(x3_0)
        # se3_2 = x3_0*se3_1
        # x3_0 = x3_0+self.short_cut(se3_2)
        # x3_0 = F.relu(x3_0)
        # sa3_1 = self.sa3(x3_0)
        # sa3_2 = x3_0*sa3_1
        # x3_0 = x3_0+self.short_cut(sa3_2)

        x4_0 = self.conv4_0(self.pool(x3_0))    #16
        # se4_1 = self.se4(x4_0)
        # se4_2 = x4_0*se4_1
        # x4_0 = x4_0+self.short_cut(se4_2)


        x5_0 = self.conv5_0(self.pool(x4_0))    #8
        # se5_1 = self.se5(x5_0)
        # se5_2 = x5_0*se5_1
        # x5_0 = x5_0+self.short_cut(se5_2)

        x6_0 = self.conv6_0(self.pool(x5_0))    #4
        # se6_1 = self.se6(x6_0)
        # se6_2 = x6_0*se6_1
        # x6_0 = x6_0+self.short_cut(se6_2)

        
        x7_0 = self.conv7_0(self.pool(x6_0))    #2
        # se7_1 = self.se7(x7_0)
        # se7_2 = x7_0*se7_1
        # x7_0 = x7_0+self.short_cut(se7_2)

        # x8_0 = self.conv8_0(self.pool(x7_0))    #2
        # se8_1 = self.se8(x8_0)
        # se8_2 = x8_0*se8_1
        # x8_0 = x8_0+self.short_cut(se8_2)
        # x4_0 = F.relu(x4_0)
        # sa4_1 = self.sa4(x4_0)
        # sa4_2 = x4_0*sa4_1
        # x4_0 = x4_0+self.short_cut(sa4_2)
        # print(1111111111111,x7_0.size())
        
        
        x6_1 = self.conv6_1(self.up(torch.cat([x7_0,skip2],1)))
        x5_2 = self.conv5_2(self.up(torch.cat([x6_1,skip1],1)))
        x4_3 = self.conv4_3(torch.cat([x4_0, self.up(x5_2)], 1))
        x3_4 = self.conv3_4(torch.cat([x3_0, self.up(x4_3)], 1))
        # print(22222222222,x3_4.size())
        x2_5 = self.conv2_5(torch.cat([x2_0, self.up(x3_4)], 1))
        # print(3,x2_5.size())
        x1_6 = self.conv1_6(torch.cat([x1_0, self.up(x2_5)], 1))
        # print(3,x1_6.size())
        x0_7 = self.conv0_7(torch.cat([x0_0, self.up(x1_6)], 1))

        output = self.final(x0_7)
        # print(output.size())
        return output

class TransSimUNetDelocal(nn.Module):
    def __init__(self,config):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        
        self.transformer=VisionTransformer(config,img_size=512, num_classes=21843, zero_head=False, vis=False)
        # self.transformer.load_from(config.pretrained_path)
        self.feature0_conv=nn.Conv2d(512,256,1,1)

        self.feature1_conv=nn.Conv2d(256,128,1,1)
        self.feature2_conv=nn.Conv2d(64,64,1,1)


        self.num_classes = 1
        self.input_channels = 3
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sa0 = SA(nb_filter[0])
        self.sa1 = SA(nb_filter[1])
        self.sa2 = SA(nb_filter[2])
        self.sa3 = SA(nb_filter[3])
        self.sa4 = SA(nb_filter[4])
        # self.se = SE(3,16)


        self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        
        # self.conv0_0 = Bottleneck(self.input_channels,nb_filter[0])
        self.se0 = SE(nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_0 = BasicBlock(nb_filter[0],nb_filter[1],downsample=nn.Conv2d(nb_filter[0],nb_filter[1],1,1))
        self.se1 = SE(nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_0 = BasicBlock(nb_filter[1],nb_filter[2],downsample=nn.Conv2d(nb_filter[1],nb_filter[2],1,1))
        # self.conv2_0 = Bottleneck(nb_filter[1])
        self.se2 = SE(nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_0 = Bottleneck(nb_filter[2],nb_filter[3]//4,downsample=nn.Conv2d(nb_filter[2],nb_filter[3],1,1))
        self.se3 = SE(nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv4_0 = Bottleneck(nb_filter[3],nb_filter[4]//4,downsample=nn.Conv2d(nb_filter[3],nb_filter[4],1,1))
       
        self.se4 = SE(nb_filter[4])
        # self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se5 = SE(nb_filter[4])
        # self.conv6_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv6_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se6 = SE(nb_filter[4])
        # self.conv7_0 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv7_0 = Bottleneck(nb_filter[4],nb_filter[4]//4)
       
        self.se7 = SE(nb_filter[4])

        


        self.conv6_1 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv5_2 = VGGBlock(nb_filter[4]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv4_3 = VGGBlock(nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv3_4 = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_5 = VGGBlock(nb_filter[3]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_6 = VGGBlock(nb_filter[2]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_7 = VGGBlock(nb_filter[1]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        self.short_cut = nn.Sequential()  #将输入与剩余输出匹配

    def forward(self, input):
        skip1,skip2,skip3,features=self.transformer(input)
        x0_0 = self.conv0_0(input)       #224

        

        x1_0 = self.conv1_0(self.pool(x0_0))    #128
        # se1_1 = self.se1(x1_0)
        # se1_2 = x1_0*se1_1
        # x1_0 = x1_0+self.short_cut(se1_2)
        # x1_0 = F.relu(x1_0)
        # sa1_1 = self.sa1(x1_0)
        # sa1_2 = x1_0*sa1_1
        # x1_0 = x1_0+self.short_cut(sa1_2)

        x2_0 = self.conv2_0(self.pool(x1_0))    #64
        # se2_1 = self.se2(x2_0)
        # se2_2 = x2_0*se2_1
        # x2_0 = x2_0+self.short_cut(se2_2)
        # x2_0 = F.relu(x2_0)
        # sa2_1 = self.sa2(x2_0)
        # sa2_2 = x2_0*sa2_1
        # x2_0 = x2_0+self.short_cut(sa2_2)

        x3_0 = self.conv3_0(self.pool(x2_0))    #32
        # se3_1 = self.se3(x3_0)
        # se3_2 = x3_0*se3_1
        # x3_0 = x3_0+self.short_cut(se3_2)
        # x3_0 = F.relu(x3_0)
        # sa3_1 = self.sa3(x3_0)
        # sa3_2 = x3_0*sa3_1
        # x3_0 = x3_0+self.short_cut(sa3_2)

        x4_0 = self.conv4_0(self.pool(x3_0))    #16
        # se4_1 = self.se4(x4_0)
        # se4_2 = x4_0*se4_1
        # x4_0 = x4_0+self.short_cut(se4_2)


        x5_0 = self.conv5_0(self.pool(x4_0))    #8
        # se5_1 = self.se5(x5_0)
        # se5_2 = x5_0*se5_1
        # x5_0 = x5_0+self.short_cut(se5_2)

        x6_0 = self.conv6_0(self.pool(x5_0))    #4
        # se6_1 = self.se6(x6_0)
        # se6_2 = x6_0*se6_1
        # x6_0 = x6_0+self.short_cut(se6_2)

        
        x7_0 = self.conv7_0(self.pool(x6_0))    #2
        # se7_1 = self.se7(x7_0)
        # se7_2 = x7_0*se7_1
        # x7_0 = x7_0+self.short_cut(se7_2)

        # x8_0 = self.conv8_0(self.pool(x7_0))    #2
        # se8_1 = self.se8(x8_0)
        # se8_2 = x8_0*se8_1
        # x8_0 = x8_0+self.short_cut(se8_2)
        # x4_0 = F.relu(x4_0)
        # sa4_1 = self.sa4(x4_0)
        # sa4_2 = x4_0*sa4_1
        # x4_0 = x4_0+self.short_cut(sa4_2)
        # print(1111111111111,x7_0.size())
        
        
        x6_1 = self.conv6_1(self.up(torch.cat([x7_0,skip2],1)))
        x5_2 = self.conv5_2(self.up(torch.cat([x6_1,skip1],1)))
        x4_3 = self.conv4_3(self.up(x5_2))
        x3_4 = self.conv3_4(self.up(x4_3))
        # print(22222222222,x3_4.size())
        x2_5 = self.conv2_5(self.up(torch.cat([x3_4,self.feature0_conv(features[0])],1)))
        # print(3,x2_5.size())
        x1_6 = self.conv1_6(self.up(torch.cat([x2_5,self.feature1_conv(features[1])],1)))
        # print(3,x1_6.size())
        x0_7 = self.conv0_7(self.up(torch.cat([x1_6,self.feature2_conv(features[2])],1)))

        output = self.final(x0_7)
        # print(output.size())
        return output


    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    #same的空洞卷积
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

#dilation 空洞的间隔



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout=nn.Dropout(0.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)


        if self.downsample is not None:
            #是否改变通道和size的下采样
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout=nn.Dropout(0.2)
        self.se=SE(width)
        self.short_cut=nn.Sequential()

    def forward(self, x):
        identity = x

        #第一层卷积不改变步长，改变通道
        se = self.se(x)
        se3_2 = x*se
        out = x+self.short_cut(se3_2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #深度可分离卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # se = self.se(out)
        # se3_2 = out*se
        # out = out+self.short_cut(se3_2)

        out = self.conv3(out)
        out = self.bn3(out)
        out=self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout_rate = 0.2

    def forward(self, x):
        out = self.conv1(x)
        # mid = out
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate)

        return out

class SE(nn.Module):
    # def __init__(self,in_chnls,ratio):
    def __init__(self, in_chnls):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))   #B*C*1*1
        # self.compress = nn.Conv2d(in_chnls,in_chnls//ratio,1,1,0)


        self.trans = nn.Conv2d(1,1,1)       #转化成Q,K,V
        # self.trans = nn.Conv2d(in_chnls,in_chnls//ratio,1)
        self.softmax = nn.Softmax(3)

        # self.excitation = nn.Conv2d(in_chnls//ratio,in_chnls,1,1,0)

    def forward(self,x):
        out = self.squeeze(x)
        # out = self.compress(out)

        out = out.permute(0,2,3,1).contiguous() #b*1*1*c
        #slefattention
        q = self.trans(out)
        k = self.trans(out)
        v = self.trans(out)
        q = q.permute(0,1,3,2).contiguous()  #B*1*C*1
        # print(q.shape)
        # k = k.permute(0,2,3,1).contiguous()  #B*1*1*C
        v = v.permute(0,1,3,2).contiguous()  #B*1*C*1
        # a = self.softmax(q@k)
        a = self.softmax(q*v)
        # print(a.shape)
        out = a*v    #B*1*C*1
        # print(out.shape)
        out = out.permute(0,2,1,3).contiguous()

        out = F.relu(out)
        # out = self.excitation(out)
        # print(out.shape)
        return torch.sigmoid(out)


class SA(nn.Module):
    def __init__(self,in_channels,patch=2):
        super(SA,self).__init__()
        self.dc = nn.Conv2d(in_channels,1,1)   #B*1*H*W   降通道
        self.avgpool = nn.AvgPool2d(patch,patch)  #降低分辨率
        self.trans = nn.Conv2d(1,1, 1)  # 转化成Q,K,V
        self.up = nn.Upsample(scale_factor=patch, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(1)

    def forward(self,x):
        b,c,h,w = x.shape
        x1 = self.dc(x)    #b*1*h*w
        x2 = self.avgpool(x1)   #b*1*(h/p)*(w/p)
        q = self.trans(x2)
        k = self.trans(x2)
        v = self.trans(x2)
        q = q.view(b,1,-1).contiguous() #b*1*(hw/p2)
        q = q.permute(0,2,1).contiguous()  #b*(hw/p2)*1
        k = k.view(b,1,-1).contiguous() #b*1*(hw/p2)
        v = v.view(b,1,-1).contiguous() #b*1*(hw/p2)
        v = v.permute(0, 2, 1).contiguous()  # b*(hw/p2)*1
        a = self.softmax(q@k)
        x3 = a@v    #b*(hw/p2)*1
        x3 = x3.permute(0,2,1).contiguous()   #b*1(hw/p2)
        x3 = F.relu(x3)
        out = x3.view(b,1,h//2,-1) #b*1*(h/p)*(w/p)
        out = self.up(out)   #b*1*h*w


        return  torch.sigmoid(out)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        
        self.conv_layer = Conv_layer(3,32)
        
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(32, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x,m = self.conv_layer(x)
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x,m, features[::-1]


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-L_16': get_r50_l16_config(),
    'testing': get_testing(),
}

print('use edge module')
# model=TransSimUNet(CONFIGS['R50-ViT-B_16'])
# x=torch.randn(2,3,512,512)
# y=model(x)
# print('label')
# print(y[0].size())
# print('edge')
# print(y[1].size())
