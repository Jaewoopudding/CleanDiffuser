import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import FourierEmbedding


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
    
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, (4, 3), (2, 1), (1, 1))

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, (2, 1), 1)

    def forward(self, x):
        return self.conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)


class MixerBlock(nn.Module):
    def __init__(self, height, width, dim, dim_out, *, embed_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(embed_dim, dim_out)
        )

        self.ln1 = nn.LayerNorm(dim * width)
        self.block1 = FeedForward(dim * width, 4*dim)

        self.ln2 = nn.LayerNorm(height)
        self.block2 = FeedForward(height, 4*height)

        self.output_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        bs, ch, height, width = x.size()
        x_first = x.permute(0, 2, 1, 3).reshape(bs, height, ch * width)
        x_first = self.ln1(x_first) 
        x_first = self.block1(x_first)
        x_first = x_first.view(bs, height, ch, width).permute(0, 2, 1, 3)
        h = x_first
        x_second = x.permute(0, 1, 3, 2).contiguous()
        x_second = self.ln2(x_second)
        x_second = self.block2(x_second).permute(0, 1, 3, 2).contiguous()
        h = h + x_second
        h = self.output_conv(h)
        h += self.mlp(time_emb)[:, :, None, None]
        return h + self.res_conv(x)


class MixerUnet(BaseNNDiffusion):
    def __init__(
        self,
        horizon,
        d_in,
        cond_dim,
        embed_dim=32,
        model_dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        # condition_dropout : float = 0.25, ##TODO
        # use_dropout : bool = True,
        # force_dropout : bool = False,
        timestep_emb_type: str = "fourier"
    ):
        super().__init__(embed_dim, timestep_emb_type) ##TODO=embed_dim == dim
        self.channels = channels

        self.cond_dim = cond_dim 

        # self.use_dropout = use_dropout
        # self.force_dropout = force_dropout

        dims = [channels, *map(lambda m: model_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.map_emb = nn.Sequential(
            nn.Linear(embed_dim, model_dim * 4), nn.Mish(),
            nn.Linear(model_dim * 4, model_dim))


        # time_dim = dim
        # self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        # sinu_pos_emb = FourierEmbedding(learned_sinusoidal_dim, random_fourier_features) ## TODO
        # fourier_dim = learned_sinusoidal_dim # + 1

        # self.time_mlp = nn.Sequential(
        #     sinu_pos_emb,
        #     nn.Linear(fourier_dim, time_dim * 4),
        #     nn.Mish(),
        #     nn.Linear(time_dim * 4, time_dim),
        # )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # if cond_dim > 0:
        #     self.cond_mlp = nn.Sequential(
        #         nn.Linear(cond_dim, model_dim * 2),
        #         nn.Mish(),
        #         nn.Linear(model_dim * 2, model_dim),
        #     )
        #     self.mask_dist = torch.distributions.Bernoulli(probs=1-condition_dropout)

        #     time_dim = 2 * model_dim
        is_seq = (horizon %2**3 == 0) 

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            if not is_seq:
                downsample_last = nn.Identity()
            else:
                downsample_last = Downsample(dim_out)

            self.downs.append(nn.ModuleList([
                MixerBlock(horizon, d_in, dim_in, dim_out, embed_dim=model_dim),
                MixerBlock(horizon, d_in, dim_out, dim_out, embed_dim=model_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                downsample_last if not is_last else nn.Identity()
            ]))

            if is_seq and (not is_last):
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = MixerBlock(horizon, d_in, mid_dim, mid_dim, embed_dim=model_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = MixerBlock(horizon, d_in, mid_dim, mid_dim, embed_dim=model_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            if not is_seq:
                upsample_last = nn.Identity()
            else:
                upsample_last = Upsample(dim_in)

            self.ups.append(nn.ModuleList([
                MixerBlock(horizon, d_in, dim_out * 2, dim_in, embed_dim=model_dim),
                MixerBlock(horizon, d_in, dim_in, dim_in, embed_dim=model_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                upsample_last if not is_last else nn.Identity()
            ]))

            if is_seq and (not is_last):
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Block(model_dim, model_dim),
            nn.Conv2d(model_dim, channels, 1)
        )


    def forward(self, x, time, cond): 
        
    # def forward(self, ##TODO
    #             x: torch.Tensor, noise: torch.Tensor,
    #             condition: Optional[torch.Tensor] = None):
    #     """
    #     Input:
    #         x:          (b, horizon, in_dim)
    #         noise:      (b, )
    #         condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

    #     Output:
    #         y:          (b, horizon, in_dim)
    #     """

    #     x = x.permute(0, 2, 1)

    #     emb = self.map_noise(noise) ## Time_embedding
    #     if condition is not None:
    #         emb = emb + condition
    #     emb = self.map_emb(emb)
        emb = self.map_noise(time)
        if cond is not None:
            emb  = emb + cond
        emb = self.map_emb(emb)
        h = []

        # t = self.time_mlp(time)

        # if self.cond_dim > 0:
        #     if cond is not None:
        #         cond = self.cond_mlp(cond)
        #         if self.use_dropout : 
        #             mask = self.mask_dist.sample(sample_shape=(cond.size(0),1)).to(cond.device)
        #             cond = cond * mask
                
        #         if self.force_dropout:
        #             cond = 0*cond
        #     else:
        #         cond = torch.zeros_like(t).to(x.device)
        #     t = torch.cat([cond, t], dim=-1)
        x = x[:, None]

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, emb) ## TODO 원래 emb은 위의 t였음
            x = resnet2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = rearrange(x, 'b 1 t h -> b t h')
        return x