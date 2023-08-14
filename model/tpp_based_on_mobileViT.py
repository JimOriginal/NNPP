import torch
import torch.nn as nn
from einops import rearrange
from model.pe import GaussianRelativePE

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel_in,channel_out, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel_in, channel_in, kernel_size)
        self.conv2 = conv_1x1_bn(channel_in, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel_in)
        self.conv4 = conv_nxn_bn(2 * channel_in, channel_out, kernel_size)
    
    def forward(self, x):
        

        # Local representations
        x = self.conv1(x)
        y = x.clone()
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class pp_MobileViT(nn.Module):
    def __init__(self, image_size, expansion=4, kernel_size=3, patch_size=(32, 32)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
 
        L = [2, 4, 3]
        dims = [64, 128, 128, 64]
        self.pe = GaussianRelativePE(256)
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], 2, 3,16, kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], 4, 16,8, kernel_size, patch_size, int(dims[1]*2)))
        self.mvit.append(MobileViTBlock(dims[2], 3, 8,4, kernel_size, patch_size, int(dims[2]*2)))
        self.mvit.append(MobileViTBlock(dims[3], 3, 4,1, kernel_size, patch_size, int(dims[3]*2)))

    def pe_forward(self, x, start, goal):
        zeros = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        #print("dd",zeros.device,start.device)
        pe_start = self.pe(zeros, start)
        #print(pe_start.device)
        pe_goal = self.pe(zeros, goal)
        return torch.cat([x, pe_start, pe_goal], dim=1)

    def forward(self, img,start,goal):
        x = self.pe_forward(img,start,goal)

        x = self.mvit[0](x)
       
        x = self.mvit[1](x)
        x = self.mvit[2](x)
        x = self.mvit[3](x)

        x=torch.sigmoid(x)
        return x


# def mobilevit_xxs():
#     dims = [64, 80, 96]
#     channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
#     return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


# def mobilevit_xs():
#     dims = [96, 120, 144]
#     channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
#     return MobileViT((256, 256), dims, channels, num_classes=1000)


# def mobilevit_s():
#     dims = [144, 192, 240]
#     channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
#     return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # map = torch.randn(5, 1, 256, 256)
    # start=torch.randn(5, 2)
    # goal=torch.randn(5, 2)
    # path=torch.randn(5, 256,256)
    # #out = MobileViTBlock(64, 2, 32, 3, (2, 2), 128)(img)
    
    model=pp_MobileViT((256,256))
    

    # out=model(map,start,goal)
    # print(out.shape)
   
    # # vit = mobilevit_xxs()
    # # out = vit(img)
    # # print(out.shape)
    print(count_parameters(model))

    # vit = mobilevit_xs()
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    # vit = mobilevit_s()
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))