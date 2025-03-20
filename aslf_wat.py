import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Net(nn.Module):
    def __init__(self, angRes, angRes_out, factor):
        super(Net, self).__init__()
        channels = 64
        self.channels = channels
        self.angRes = angRes
        self.angRes_out = angRes_out
        self.factor = factor
        layer_num = 6#6

        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.epiFeatureRebuild_AS = EpiFeatureRebuild_AS(self.angRes, self.angRes_out, self.factor, channels, feat_unfold=False)
        ##################### Initial Convolution #####################
        self.FeaExtract = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.DeepFeaExt = CascadedBlocks(layer_num, channels, angRes)

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = self.make_layer(layer_num=layer_num)

        ####################### UP Sampling ###########################
        self.DownSample = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(channels // 4, channels // 4 // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(channels // 4 // 4, 1, kernel_size=1, stride=1,
                      padding=0, bias=False),
        )

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(EPIX_Trans(self.angRes, self.channels, self.MHSA_params))
            layers.append(SA_Epi_Trans(self.angRes, self.channels, self.MHSA_params))
        layers.append(
            nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, lr, patchsize, ang_factor):


        x_mv = LFsplit(lr, self.angRes)
        buffer = self.FeaExtract(x_mv)
        buffer = self.DeepFeaExt(buffer)

        # EPIXTrans
        buffer = self.altblock(buffer) + buffer


        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.angRes, v=self.angRes)
        buffer = self.epiFeatureRebuild_AS(buffer, patchsize, ang_factor)
        buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w')

        b, c, n, h, w = buffer.shape
        buffer = self.DownSample(buffer).view(b, 1, ang_factor * ang_factor, h, w)  # n == angRes * angRes
        out = FormOutput(buffer)
        return out

class EpiFeatureRebuild_AS(nn.Module):
    def __init__(self, angRes_in, angRes_out, factor, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.factor = factor
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_Epi(self, epi, patchsize, ang_factor):
        bhu, c, v, w = epi.shape

        # Res_in x w --> Res_out x W
        coord = make_coord([ang_factor, patchsize]).cuda() \
            .unsqueeze(0).expand(epi.shape[0], ang_factor * patchsize, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, ang_factor, patchsize)

        return output_epi

    def forward(self, x, patchsize, ang_factor):
        batch_size, channel, u, v, h, w= x.shape
        U, V = ang_factor, ang_factor

        # 2 x 2 x h x w --> 2 x 7 x h x W
        horizontal_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b u h) c v w'), patchsize, ang_factor)
        x = rearrange(horizontal_x, '(b u h) c v w -> b c u v h w', b=batch_size, h=h)
        # 2 x 7 x h x W --> 7 x 7 x H x W
        vertical_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b v w) c u h'), patchsize, ang_factor)
        output = rearrange(vertical_x, '(b v w) c u h -> b c u v h w', b=batch_size, w=patchsize)

        return output

class EpiFeatureRebuild_A(nn.Module):
    def __init__(self, angRes_in, angRes_out, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_Epi(self, epi):
        buh, c, v, w = epi.shape

        # 2 x W --> 7 x W
        coord = make_coord([self.angRes_out, w]).cuda() \
            .unsqueeze(0).expand(epi.shape[0], w * self.angRes_out, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, self.angRes_out, w)

        # buh, c, angRes_out, w
        return output_epi

    def forward(self, x):
        batch_size, channel, u, v, h, w = x.shape

        # 2 x 2 x H x W --> 2 x 7 x H x W
        horizontal_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b u h) c v w'))
        x = rearrange(horizontal_x, '(b u h) c v w -> b c u v h w', b=batch_size, h=h)
        # 2 x 7 x H x W --> 7 x 7 x H x W
        vertical_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b v w) c u h'))
        output = rearrange(vertical_x, '(b v w) c u h -> b c u v h w', b=batch_size, w=w)

        return output
class EpiFeatureRebuild_S(nn.Module):
    def __init__(self, angRes_in, factor, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes_in = angRes_in
        self.factor = factor
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_Epi(self, epi):
        bhu, c, w, v = epi.shape

        # 5 x w --> 5 x W
        coord = make_coord([self.factor*w, v]).cuda() \
            .unsqueeze(0).expand(epi.shape[0], v * self.factor*w, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, self.factor*w, v)

        return output_epi

    def forward(self, x):
        batch_size, channel, h, w, u, v = x.shape

        # 2 x 2 x h x w --> 2 x 2 x H x w
        horizontal_x = self.query_Epi(rearrange(x, 'b c h w u v -> (b h u) c w v'))
        x = rearrange(horizontal_x, '(b h u) c w v -> b c h w u v', b=batch_size, u=u)
        # 2 x 2 x H x w --> 2 x 2 x H x W
        vertical_x = self.query_Epi(rearrange(x, 'b c h w u v -> (b w v) c h u'))
        output = rearrange(vertical_x, '(b w v) c h u -> b c h w u v', b=batch_size, v=v)

        return output


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class SA_Epi_CrossAttention_Trans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(SA_Epi_CrossAttention_Trans, self).__init__()
        self.emb_dim = emb_dim
        self.sa_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.epi_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.sa_norm = nn.LayerNorm(emb_dim)
        self.epi_norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels//2, bias=False)

    def forward(self, buffer):
        b, c, u, v, h, w = buffer.shape

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        sa_token = token[:, :, :c//2]
        epi_token = token[:, :, c//2:]

        epi_token_short_cut = epi_token

        sa_token = self.sa_linear_in(sa_token)
        epi_token = self.epi_linear_in(epi_token)

        sa_token_norm = self.sa_norm(sa_token)
        epi_token_norm = self.epi_norm(epi_token)
        sa_token = self.attention(query=epi_token_norm,
                                  key=sa_token_norm,
                                  value=sa_token,
                                  need_weights=False)[0] + sa_token

        sa_token = self.feed_forward(sa_token) + sa_token
        sa_token = self.linear_out(sa_token)

        buffer = torch.cat((sa_token, epi_token_short_cut), 2)
        buffer = buffer.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer

###################---------Patch_Embedding---------###################
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size   (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans   (int): Number of input image channels. Default: 3.
        embed_dim  (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):     #[1, 180, 64, 64]
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c  #[1, 180, 64, 64]->[1, 180, 4096]->[1, 4096, 180]
        if self.norm is not None:
            x = self.norm(x)  #[1, 4096, 180]
        return x
###################---------Patch_Embedding---------###################


###############------Rectangular_Window_Partition------###############
def window_partition(x, window_size):   #[1, 64, 64, 180]
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)    #[1, 64//16, 16, 64//16, 16, 180]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c) #[1, 64//16, 64//16, 16, 16, 180]->#[16, 16, 16, 180]
    return windows
###############------Rectangular_Window_Partition------###############

class CFAT(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=180,
                 depths=(8, 8, 8, 8, 8),
                 num_heads=(6, 6, 6, 6, 6),
                 window_size=16,
                 shift_size = (0, 0, 8, 8, 16, 16, 24, 24),  #changed to tuple
                 interval=(0, 2, 0, 2, 0),
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(CFAT, self).__init__()

        ##Arguments
        self.window_size = window_size
        self.shift_size = shift_size   #changed to tuple
        self.overlap_ratio = overlap_ratio
        self.upscale = upscale
        self.upsampler = upsampler
        self.num_layers = len(depths)  #5
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans

        ######----Patch_Embedding----######
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=embed_dim,
                                      embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        ######----Patch_Embedding----######

        ######----Absolute_Position_Embedding----######
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) #(1, 4096, 180)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        ######----Absolute_Position_Embedding----######

    ######----Attention_Mask_for_HAB(SW-MSA)----######
    def calculate_mask(self, x_size, shift_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    ######----Attention_Mask_for_HAB----######

    ###############------Triangular_Window_Mask------###############
    def triangle_masks(self, x):
        ws = 2 * self.window_size
        rows = torch.arange(ws).unsqueeze(1).repeat(1, ws)
        cols = torch.arange(ws).unsqueeze(0).repeat(ws, 1)

        upper_triangle_mask = (cols > rows) & (rows + cols < ws)
        right_triangle_mask = (cols >= rows) & (rows + cols >= ws)
        bottom_triangle_mask = (cols < rows) & (rows + cols >= ws - 1)
        left_triangle_mask = (cols <= rows) & (rows + cols < ws - 1)

        return [upper_triangle_mask.to(x.device), right_triangle_mask.to(x.device), bottom_triangle_mask.to(x.device),
                left_triangle_mask.to(x.device)]

    ###############------Triangular_Window_Mask------###############

    def forward_feature(self, x): #[b, c, uh, vw]
        x_size = (x.shape[2], x.shape[3])

        attn_mask = tuple([self.calculate_mask(x_size, shift_size).to(x.device) for shift_size in
                           (8, 16, 24)])  # [16, 256, 256]   #changed to tuple
        triangular_masks = tuple(self.triangle_masks(x))  # [16, 256, 256]   #changed to tuple

        params = {'attn_mask': attn_mask, 'triangular_masks': triangular_masks,
                  'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        ##Embed$$Unembed
        x = self.patch_embed(x)  # [1, 180, 64, 64]->[1, 4096, 180]
        if self.ape:
            x = x + self.absolute_pos_embed  # [1, 4096, 180]
        x = self.pos_drop(x)  # [1, 4096, 180]
        for layer in self.layers:
            x = layer(x, x_size, params)  # [1, 4096, 180]
        x = self.norm(x)  # b seq_len c     #[1, 4096, 180]
        x = self.patch_unembed(x, x_size)  # [1, 4096, 180]->[1, 180, 64, 64]

        return x



class SA_Epi_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(SA_Epi_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = SA_Epi_CrossAttention_Trans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut


        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut
        # shortcut = buffer

        return buffer


class EpiXTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiXTrans, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, maxdisp: int = 6):
        attn_mask = torch.zeros([h, w, h, w])
        [ii, jj] = torch.meshgrid(torch.arange(h), torch.arange(w))
        # [ii, jj] = torch.meshgrid(torch.arange(h), torch.arange(w))

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii - i).abs() * maxdisp >= (jj - j).abs()] = 1
                # temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.reshape(h * w, h * w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = epi_token.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class EPIX_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(EPIX_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = EpiXTrans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut

        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut
        # shortcut = buffer

        return buffer


class SA_Conv(nn.Module):
    def __init__(self, ch, angRes):
        super(SA_Conv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, E_ch = ch, ch // 4, ch // 2  #A_ch=ch//4
        self.angRes = angRes
        self.spaconv = SpatialConv(ch)
        self.angconv = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.SA_fuse = nn.Sequential(
            nn.Conv3d(in_channels=S_ch + A_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.2, inplace=True), #0.1->0.2
            nn.Conv3d(ch, ch//2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))#改ch//2
        self.Epi_fuse = nn.Sequential(
            nn.Conv3d(in_channels= E_ch + E_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.2, inplace=True),#0.1->0.2
            nn.Conv3d(ch, ch//2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))#改

    # epi并行
    def forward(self, x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)

        sa_out = self.SA_fuse(torch.cat((s_out, a_out), 1))

        epih_in = x.contiguous().view(b, c, an, an, h, w)  # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)

        # epiv_in = epih_in.permute(0,2,1,3,5,4)
        epiv_in = epih_in.permute(0, 1, 3, 2, 5, 4)  # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 5, 4).reshape(b, -1, n, h, w)

        epi_out = self.Epi_fuse(torch.cat((epih_out, epiv_out), 1))

        out = torch.cat((sa_out, epi_out), 1)

        return out + x  # out.contiguous().view(b,n,c,h,w) + x


class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
            nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, fm):
        return self.spaconv_s(fm)


class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch * angRes * angRes, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angRes * angRes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.PixelShuffle(angRes)
        )
        # self.an = angRes

    def forward(self, fm):
        b, c, n, h, w = fm.shape
        a_in = fm.contiguous().view(b, c * n, 1, h, w)
        out = self.angconv(a_in).view(b, -1, n, h, w)  # n == angRes * angRes
        return out


class EPiConv(nn.Module):
    def __init__(self, ch, angRes, EPIChannel):
        super(EPiConv, self).__init__()
        self.epi_ch = EPIChannel
        self.epiconv = nn.Sequential(
            nn.Conv3d(ch, EPIChannel, kernel_size=(1, angRes, angRes // 2 * 2 + 1), stride=1,
                      padding=(0, 0, angRes // 2), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(EPIChannel, angRes * EPIChannel, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                      bias=False),  # ksize maybe (1,1,angRes//2*2+1) ?
            nn.LeakyReLU(0.2, inplace=True),
            # PixelShuffle1D(angRes),
        )
        # self.an = angRes

    def forward(self, fm):
        b, c, u, v, h, w = fm.shape

        epih_in = fm.permute(0, 1, 2, 4, 3, 5).reshape(b, c, u * h, v, w)
        epih_out = self.epiconv(epih_in)  # (b,self.epi_ch*v, u*h, 1, w)
        epih_out = epih_out.reshape(b, self.epi_ch, v, u, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.epi_ch, u * v,
                                                                                                  h, w)
        return epih_out


class CascadedBlocks(nn.Module):
    '''
    Hierarchical feature fusion
    '''

    def __init__(self, n_blocks, channel, angRes):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(SA_Conv(channel, angRes))
        self.body = nn.Sequential(*body)
        # self.conv = nn.Conv2d(channel, channel, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.conv = nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1)

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3, 4)
        # b, n, c, h, w = x.shape
        buffer = x
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)
            # buffer = self.conv(buffer.contiguous().view(b*n, c, h, w))
        buffer = self.conv(buffer) + x
        # buffer = buffer.contiguous().view(b,n, c, h, w) + x
        return buffer  # buffer.permute(0, 2, 1, 3, 4)


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st.permute(0, 2, 1, 3, 4)

def FormOutput(x_sv):
    x_sv = x_sv.permute(0, 2, 1, 3, 4)
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out



if __name__ == "__main__":
    net = Net(2,5, 4).cuda()

    input = torch.randn(1, 1, 64, 64).cuda()
    out = net(input,64,5)
    print(out.shape)
    total = sum([param.nelement() for param in net.parameters()])
