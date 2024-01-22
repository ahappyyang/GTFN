import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from einops import rearrange, repeat


def gain_neighborhood_band(x_train, band, band_patch, patch_all):
    nn = band_patch // 2
    pp = (patch_all) // 2
    x_train_band = torch.zeros((x_train.shape[0], patch_all*band_patch, band),dtype=float)#64*27*200

    x_train_band[:,nn*patch_all:(nn+1)*patch_all,:] = x_train

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch_all:(i+1)*patch_all,:i+1] = x_train[:,:,band-i-1:]
            x_train_band[:,i*patch_all:(i+1)*patch_all,i+1:] = x_train[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train[:,0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch_all:(nn+i+2)*patch_all,:band-i-1] = x_train[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch_all:(nn+i+2)*patch_all,band-i-1:] = x_train[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train[:,0:1,(i+1):]
    return x_train_band


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask


        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x

class ViT(nn.Module):
    def __init__(self, n_gcn, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='CAF'):
        super().__init__()

        patch_dim = n_gcn
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask = None):

        x=x.to(torch.float32)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        pos=self.pos_embedding[:, :(n + 1)]
        x += pos
        x = self.dropout(x)


        x = self.transformer(x, mask)

        x = self.to_latent(x[:,0])
        x = self.mlp_head(x)

        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))


    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(2)
        batch,l=D.shape
        D1=torch.reshape(D, (batch * l,1))
        D1=D1.squeeze(1)
        D2=torch.pow(D1, -0.5)
        D2=torch.reshape(D2,(batch,l))
        D_hat=torch.zeros([batch,l,l],dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cuda()

    def forward(self, H, A ):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)

        (batch, l, c) = H.shape
        H1 = torch.reshape(H,(batch*l, c)) 
        H2 = self.BN(H1)
        H=torch.reshape(H2,(batch,l, c)) 
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))#点乘
        A_hat = I + A_hat
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))#矩阵相乘
        output = self.Activition(output)
        return output


class neigh_Conv(nn.Module):
    def __init__(self, channel, neigh_number):
        super(neigh_Conv, self).__init__()
        self.neigh_Branch = nn.Sequential()
        self.neigh_number=neigh_number
        for i in range(channel-neigh_number+1):
            self.neigh_Branch.add_module('neigh_Branch' + str(i), nn.Conv2d(neigh_number, 1, kernel_size = (1,1), stride=1))

    def forward(self, x):
        batch,c,w,h = x.shape
        for i in range(c-self.neigh_number+1):
            if i==0:
                A=self.neigh_Branch[i](x[:,i:i+self.neigh_number,:,:])
            if i>0:
                B= self.neigh_Branch[i](x[:, i:i + self.neigh_number, :, :])
                A = torch.cat((A,B),1)
        return A

class neigh_Conv2(nn.Module):
    def __init__(self, channel, neigh_number):
        super(neigh_Conv2, self).__init__()
        self.neigh_Branch = nn.Sequential()
        self.neigh_number=neigh_number
        for i in range(channel):
            self.neigh_Branch.add_module('neigh_Branch' + str(i), nn.Conv2d(neigh_number, 1, kernel_size = (1,1), stride=1))

    def forward(self, x):
        batch,c,w,h = x.shape
        start=int((self.neigh_number-1)/2)#3 1
        end = int(c-1-start)#c-1
        for i in range(c):
            self_c = x[:, i, :, :]
            self_c=self_c.unsqueeze(1)
            if i==0:
                A=self_c+self.neigh_Branch[i](x[:,i:i+self.neigh_number,:,:])#[64 1 21 1]
            if i>0:
                if i<start:
                    B= self_c + self.neigh_Branch[i](x[:, 0:self.neigh_number, :, :])  # [64 1 21 1]
                if i>=start and i<=end:
                    B= self_c + self.neigh_Branch[i](x[:, (i-start):(i-start+ self.neigh_number), :, :])  # [64 1 21 1]
                if i>end:
                    B= self_c + self.neigh_Branch[i](x[:, c-self.neigh_number:c , :, :])  # [64 1 21 1]
                A = torch.cat((A,B),1)
        return A


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)



class GCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int):
        super(GCN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        layers_count = 4
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                if i==0:
                    self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(self.channel, 128))
                else:
                    self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64))
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        self.ca = ChannelAttention(64)
        self.neigh_C = neigh_Conv2(64,3)
        self.BN = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor,A: torch.Tensor,indexs_train):
        (batch,h, w, c) = x.shape
        _, in_num=indexs_train.shape

        H = torch.reshape(x,(batch,h*w, c))
        for i in range(len(self.GCN_Branch)):
            H = self.GCN_Branch[i](H, A)


        _, _, c_gcn=H.shape
        gcn_out = torch.zeros((batch, in_num, c_gcn),dtype=float)
        gcn_out = gcn_out.type(torch.cuda.FloatTensor)
        for i in range(batch):
            gcn_out[i]=H[i][indexs_train[i]]


        gcn_out = gcn_out.transpose(1, 2)
        gcn_out = gcn_out.unsqueeze(3)
        gcn_out = self.ca(gcn_out) * gcn_out
        gcn_out = self.neigh_C(gcn_out)
        gcn_out = gcn_out.squeeze(3)
        gcn_out = self.BN(gcn_out)
        gcn_out = gcn_out.transpose(1, 2)


        tr_in=gcn_out.transpose(1,2)
        return tr_in.cuda()

