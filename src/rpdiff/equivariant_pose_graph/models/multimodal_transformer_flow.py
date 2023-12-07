#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#from pointnets.models.pointnet2 import PN2Encoder, PN2EncoderParams
from torch.distributions import Categorical

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, emb_dims=512, emb_nn='dgcnn'):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_dims = emb_dims
        self.emb_nn_name = emb_nn
        if emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - \
            points.mean(dim=2, keepdim=True)
    
        points_embedding = self.emb_nn(
            points_dmean)  # B, emb_dims, num_points

        return points_embedding

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    # src, dst (num_dims, num_points)
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512, input_dims=3, num_heads=1, conditioning_size=0, last_relu=True):
        super(DGCNN, self).__init__()
        self.num_heads = num_heads
        self.conditioning_size = conditioning_size
        self.last_relu = last_relu

        self.conv1 = nn.Conv2d(input_dims*2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        
        if self.num_heads == 1:
            self.conv5 = nn.Conv2d(512 + self.conditioning_size, emb_dims, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm2d(emb_dims)
        else:
            if self.conditioning_size > 0:
                raise NotImplementedError("Conditioning not implemented for multi-head DGCNN")
            self.conv5s = nn.ModuleList([nn.Conv2d(512, emb_dims, kernel_size=1, bias=False) for _ in range(self.num_heads)])
            self.bn5s = nn.ModuleList([nn.BatchNorm2d(emb_dims) for _ in range(self.num_heads)])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        # bn5 defined above            

    def forward(self, x, conditioning=None):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        if self.conditioning_size == 0:
            assert conditioning is None
            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            assert conditioning is not None
            x = torch.cat((x1, x2, x3, x4, conditioning[:,:,:,None]), dim=1)
            # x = torch.cat((x1, x2, x3, x4, torch.tile(conditioning[:,:,None,:], (1, 1, num_points, 1))), dim=1)

        if self.num_heads == 1:
            x = self.bn5(self.conv5(x)).view(batch_size, -1, num_points)
        else:
            x = [bn5(conv5(x)).view(batch_size, -1, num_points) for bn5, conv5 in zip(self.bn5s, self.conv5s)]

        if self.last_relu:
            if self.num_heads == 1:
                x = F.relu(x)
            else:
                x = [F.relu(head) for head in x]
        return x

class DGCNNClassification(nn.Module):
    # Reference: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py#L88-L153

    def __init__(self, emb_dims=512, input_dims=3, num_heads=1, dropout=0.5, output_channels=40):
        super(DGCNNClassification, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.dropout=dropout
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_dims*2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, self.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)

        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        if self.num_heads == 1:
            self.linear3 = nn.Linear(256, self.output_channels)
        else:
            self.linear3s = nn.ModuleList([nn.Linear(256, self.output_channels) for _ in range(self.num_heads)])

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x).squeeze()
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        if self.num_heads == 1:
            x = self.linear3(x)[:,:,None]
        else:
            x = [linear3(x)[:,:,None] for linear3 in self.linear3s]
        return x

class Transformer(nn.Module):
    def __init__(self, emb_dims=512, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4, return_attn=False, bidirectional=True):
        super(Transformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.return_attn = return_attn
        self.bidirectional = bidirectional
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(
                                        attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        src_embedding = self.model(
            tgt, src, None, None).transpose(2, 1).contiguous()
        src_attn = self.model.decoder.layers[-1].src_attn.attn

        if(self.bidirectional):
            tgt_embedding = self.model(
                src, tgt, None, None).transpose(2, 1).contiguous()
            tgt_attn = self.model.decoder.layers[-1].src_attn.attn

            if(self.return_attn):
                return src_embedding, tgt_embedding, src_attn, tgt_attn
            return src_embedding, tgt_embedding

        if(self.return_attn):
            return src_embedding, src_attn
        return src_embedding


class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]
        batch_size = action_points.size(0)

        d_k = action_embedding.size(1)
        scores = torch.matmul(action_embedding.transpose(
            2, 1).contiguous(), anchor_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        action_corr = torch.matmul(
            anchor_points, scores.transpose(2, 1).contiguous())

        action_centered = action_points - \
            action_points.mean(dim=2, keepdim=True)

        action_corr_centered = action_corr - \
            action_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(
            action_centered, action_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(action_points.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, action_points.mean(dim=2, keepdim=True)
                         ) + action_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


class PointNet(nn.Module):
    def __init__(self, layer_dims=[3, 64, 64, 64, 128, 512]):
        super(PointNet, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(nn.Conv1d(
                layer_dims[j], layer_dims[j+1],
                kernel_size=1, bias=False))
            norms.append(nn.BatchNorm1d(layer_dims[j+1]))

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        for bn, conv in zip(self.norms, self.convs):
            x = F.relu(bn(conv(x)))
        return x


class MLPHead(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
            nn.Conv1d(emb_dims//8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input):
        action_embedding = input[0]
        embedding = action_embedding
        flow = self.proj_flow(embedding)
        return flow


class MLPHeadWeight(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
            nn.Conv1d(emb_dims//8, 4, kernel_size=1, bias=False),
        )

    def forward(self, *input):
        action_embedding = input[0]
        embedding = action_embedding
        flow = self.proj_flow(embedding)
        return flow

class ResidualMLPHead(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512, pred_weight=True, residual_on=True):
        super(ResidualMLPHead, self).__init__()

        self.emb_dims = emb_dims
        if self.emb_dims < 10:
            self.proj_flow = nn.Sequential(
                PointNet([emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )
        else:
            self.proj_flow = nn.Sequential(
                PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(emb_dims//8, 3, kernel_size=1, bias=False),
            )
        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

        self.residual_on = residual_on

    def forward(self, *input, scores=None, return_flow_component=False, return_embedding=False):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]

        if(scores is None):
            if(len(input) <= 4):
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(action_query.transpose(
                2, 1).contiguous(), anchor_key) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)
        corr_points = torch.matmul(
            anchor_points, scores.transpose(2, 1).contiguous())
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = action_embedding  # B,512,N
        residual_flow = self.proj_flow(embedding)  # B,3,N

        if self.residual_on:
            flow = residual_flow + corr_flow
        else:
            flow = corr_flow

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        if(return_flow_component):
            return {
                'full_flow': corr_flow_weight,
                'residual_flow': residual_flow,
                'corr_flow': corr_flow,
                'corr_points': corr_points,
                'scores': scores,
                }
        return corr_flow_weight

class MLP(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(emb_dims, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, emb_dims)

    def forward(self, x):

        # x = [batch size, emb_dims, num_points]
        batch_size, _, num_points = x.shape
        x = x.permute(0, -1, -2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        h_1 = F.relu(self.input_fc(x))
        # batch size*num_points, 100
        h_2 = F.relu(self.hidden_fc(h_1))

        # batch size*num_points, output dim
        y_pred = self.output_fc(h_2)
        # batch size, num_points, output dim
        y_pred = y_pred.view(batch_size, num_points, -1)
        # batch size, emb_dims, num_points
        y_pred = y_pred.permute(0, 2, 1)

        return y_pred


class Multimodal_ResidualFlow_DiffEmbTransformer(nn.Module):
    EMB_DIMS_BY_CONDITIONING = {
        'pos_delta_l2norm': 1,
        "uniform_prior_pos_delta_l2norm": 1,
        # 'latent_z': 1, # Make the dimensions as close as possible to the ablations we're comparing this against
        # 'latent_z_1pred': 1, # Same
        # 'latent_z_1pred_10d': 10, # Same
        'latent_z_linear': 512,
        'latent_z_linear_internalcond': 512,
        'pos_delta_vec': 1,
        'pos_onehot': 1,
        'pos_loc3d': 3,
    }

    # Number of heads that the DGCNN should output
    NUM_HEADS_BY_CONDITIONING = {
        'pos_delta_l2norm': 1,
        "uniform_prior_pos_delta_l2norm": 1,
        # 'latent_z': 2, # One for mu and one for var
        # 'latent_z_1pred': 2, # Same
        # 'latent_z_1pred_10d': 2, # Same
        'latent_z_linear': 2,
        'latent_z_linear_internalcond': 2,
        'pos_delta_vec': 1,
        'pos_onehot': 1,
        'pos_loc3d': 1,
    }

    DEPRECATED_CONDITIONINGS = ["latent_z", "latent_z_1pred", "latent_z_1pred_10d"]

    TP_INPUT_DIMS = {
        'pos_delta_l2norm': 3 + 1,
        'uniform_prior_pos_delta_l2norm': 3 + 1,
        # Not implemented because it's dynamic. Also this isn't used anymore
        # 'latent_z_linear': 3 + cfg.latent_z_linear_size,
        'latent_z_linear_internalcond': 3,
        'pos_delta_vec': 3 + 3,
        'pos_onehot': 3 + 1,
        'pos_loc3d': 3 + 3,
        "latent_3d_z": 3 + 3,
    }

    def __init__(self, residualflow_diffembtransformer, gumbel_temp=0.5, freeze_residual_flow=False, center_feature=False, freeze_z_embnn=False,
                 division_smooth_factor=1, add_smooth_factor=0.05, conditioning="pos_delta_l2norm", latent_z_linear_size=40,
                 taxpose_centering="mean"):
        super(Multimodal_ResidualFlow_DiffEmbTransformer, self).__init__()

        assert taxpose_centering in ["mean", "z"]
        assert conditioning not in self.DEPRECATED_CONDITIONINGS, f"This conditioning {conditioning} is deprecated and should not be used"
        assert conditioning in self.EMB_DIMS_BY_CONDITIONING.keys()

        self.latent_z_linear_size = latent_z_linear_size
        self.conditioning = conditioning
        self.taxpose_centering = taxpose_centering
        # if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
        #     assert not freeze_residual_flow and not freeze_z_embnn, "Prob didn't want to freeze residual flow or z embnn when using latent_z_linear"

        self.tax_pose = residualflow_diffembtransformer
        self.return_flow_component = self.tax_pose.return_flow_component

        self.emb_dims = self.EMB_DIMS_BY_CONDITIONING[self.conditioning]
        self.num_emb_heads = self.NUM_HEADS_BY_CONDITIONING[self.conditioning]
        # Point cloud with class labels between action and anchor
        if self.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            self.emb_nn_objs_at_goal = DGCNN(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False)
        else:
            self.emb_nn_objs_at_goal = DGCNNClassification(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, dropout=0.5, output_channels=self.latent_z_linear_size)
        # TODO
        self.freeze_residual_flow = freeze_residual_flow
        self.center_feature = center_feature
        self.freeze_z_embnn = freeze_z_embnn
        self.freeze_embnn = self.tax_pose.freeze_embnn
        self.gumbel_temp = gumbel_temp

        self.division_smooth_factor = division_smooth_factor
        self.add_smooth_factor = add_smooth_factor

        # TODO GET RID OF THIS
        self.force_val_to_train_sample = False
        if self.force_val_to_train_sample:
            self._temp_translation_sample_action = None
            self._temp_translation_sample_anchor = None
            self._temp_stuff = {}

    def get_dense_translation_point(self, points, ref, conditioning):
        """
            points- point cloud. (B, 3, num_points)
            ref- one hot vector (or nearly one-hot) that denotes the reference point
                     (B, num_points)

            Returns:
                dense point cloud. Each point contains the distance to the reference point (B, 3 or 1, num_points)
        """
        assert ref.ndim == 2
        assert torch.allclose(ref.sum(axis=1), torch.full((ref.shape[0], 1), 1, dtype=torch.float, device=ref.device))
        num_points = points.shape[2]
        reference = (points*ref[:,None,:]).sum(axis=2)
        if conditioning in ["pos_delta_l2norm", "uniform_prior_pos_delta_l2norm"]:
            dense = torch.norm(reference[:, :, None] - points, dim=1, keepdim=True)
        elif conditioning == "pos_delta_vec":
            dense = reference[:, :, None] - points
        elif conditioning == "pos_loc3d":
            dense = reference[:,:,None].repeat(1, 1, 1024)
        elif conditioning == "pos_onehot":
            dense = ref[:, None, :]
        else:
            raise ValueError(f"Conditioning {conditioning} probably doesn't require a dense representation. This function is for" \
                                + "['pos_delta_l2norm', 'pos_delta_vec', 'pos_loc3d', 'pos_onehot', 'uniform_prior_pos_delta_l2norm']")
#        dense = reference[:, :, None] - points
#        dense = torch.abs(reference[:, :, None] - points)
#        dense = torch.exp(-dense / 0.1)
#        dense = torch.exp(-dense.abs() / 0.1)
        return dense, reference
    
    def add_conditioning(self, goal_emb, action_points, anchor_points, conditioning):
        for_debug = {}

        if conditioning in ['pos_delta_l2norm', 'pos_delta_vec', 'pos_loc3d', 'pos_onehot', 'uniform_prior_pos_delta_l2norm']:

            goal_emb = (goal_emb + self.add_smooth_factor) / self.division_smooth_factor

            # Only handle the translation case for now
            goal_emb_translation = goal_emb[:,0,:]

            goal_emb_translation_action = goal_emb_translation[:, :action_points.shape[2]]
            goal_emb_translation_anchor = goal_emb_translation[:, action_points.shape[2]:]

            translation_sample_action = F.gumbel_softmax(goal_emb_translation_action, self.gumbel_temp, hard=True, dim=-1)
            translation_sample_anchor = F.gumbel_softmax(goal_emb_translation_anchor, self.gumbel_temp, hard=True, dim=-1)
            
            # This is the only line that's different among the 3 different conditioning schemes in this category
            dense_trans_pt_action, ref_action = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, action_points, translation_sample_action, conditioning=self.conditioning)
            dense_trans_pt_anchor, ref_anchor = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, anchor_points, translation_sample_anchor, conditioning=self.conditioning)

            action_points_and_cond = torch.cat([action_points] + [dense_trans_pt_action], axis=1)
            anchor_points_and_cond = torch.cat([anchor_points] + [dense_trans_pt_anchor], axis=1)

            for_debug = {
                'dense_trans_pt_action': dense_trans_pt_action,
                'dense_trans_pt_anchor': dense_trans_pt_anchor,
                'trans_pt_action': ref_action,
                'trans_pt_anchor': ref_anchor,
                'trans_sample_action': translation_sample_action,
                'trans_sample_anchor': translation_sample_anchor,
            }
        elif conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Do the reparametrization trick on the predicted mu and var

            # Here, the goal emb has 2 heads. One for mean and one for variance
            goal_emb_mu = goal_emb[0]
            goal_emb_logvar = goal_emb[1]

            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu
            
            goal_emb = reparametrize(goal_emb_mu, goal_emb_logvar)

            for_debug = {
                'goal_emb_mu': goal_emb_mu,
                'goal_emb_logvar': goal_emb_logvar,
            }

            if conditioning == "latent_z_linear":
                action_points_and_cond = torch.cat([action_points] + [torch.tile(goal_emb, (1, 1, action_points.shape[-1]))], axis=1)
                anchor_points_and_cond = torch.cat([anchor_points] + [torch.tile(goal_emb, (1, 1, anchor_points.shape[-1]))], axis=1)
            elif conditioning == "latent_z_linear_internalcond":
                # The cond will be added in by TAXPose
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug['goal_emb'] = goal_emb
            else:
                raise ValueError("Why is it here?")
        else:
            raise ValueError(f"Conditioning {conditioning} does not exist. Choose one of: {list(self.EMB_DIMS_BY_CONDITIONING.keys())}")

        return action_points_and_cond, anchor_points_and_cond, for_debug

    def forward(self, *input, mode="forward"):
        # Forward pass goes through all of the model
        # Inference will use a sample from the prior if there is one
        #     - ex: conditioning = latent_z_linear_internalcond
        assert mode in ['forward', 'inference']

        action_points = input[0].permute(0, 2, 1)[:, :3] # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        if input[2] is None:
            mode = "inference"

        if mode == "forward":
            goal_action_points = input[2].permute(0, 2, 1)[:, :3]
            goal_anchor_points = input[3].permute(0, 2, 1)[:, :3]

            # mean center point cloud before DGCNN
            if self.center_feature:
                mean_goal = torch.cat([goal_action_points, goal_anchor_points], axis=-1).mean(dim=2, keepdim=True)
                goal_action_points_dmean = goal_action_points - \
                                    mean_goal
                goal_anchor_points_dmean = goal_anchor_points - \
                                    mean_goal
                action_points_dmean = action_points - \
                        action_points.mean(dim=2, keepdim=True)
                anchor_points_dmean = anchor_points - \
                        anchor_points.mean(dim=2, keepdim=True)
            else:
                goal_action_points_dmean = goal_action_points
                goal_anchor_points_dmean = goal_anchor_points
                action_points_dmean = action_points
                anchor_points_dmean = anchor_points

            goal_points_dmean = torch.cat([goal_action_points_dmean, goal_anchor_points_dmean], axis=2)

            if self.freeze_z_embnn:
                with torch.no_grad():
                    if self.num_emb_heads > 1:
                        goal_emb = [a.detach() for a in self.emb_nn_objs_at_goal(goal_points_dmean)]
                    else:
                        goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean).detach()
            else:
                goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean)


            action_points_and_cond, anchor_points_and_cond, for_debug = self.add_conditioning(goal_emb, action_points, anchor_points, self.conditioning)
        elif mode == "inference":
            action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug  = self.sample(action_points, anchor_points)
        else:
            raise ValueError(f"Unknown mode {mode}")

        tax_pose_conditioning_action = None
        tax_pose_conditioning_anchor = None
        if self.conditioning == "latent_z_linear_internalcond":
            tax_pose_conditioning_action = torch.tile(for_debug['goal_emb'], (1, 1, action_points.shape[-1]))
            tax_pose_conditioning_anchor = torch.tile(for_debug['goal_emb'], (1, 1, anchor_points.shape[-1]))

        if self.taxpose_centering == "mean":
            # Use TAX-Pose defaults
            action_center = action_points[:, :3].mean(dim=2, keepdim=True)
            anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
        elif self.taxpose_centering == "z":
            action_center = for_debug['trans_pt_action'][:,:,None]
            anchor_center = for_debug['trans_pt_anchor'][:,:,None]
        else:
            raise ValueError(f"Unknown self.taxpose_centering: {self.taxpose_centering}")

        if self.freeze_residual_flow:
            with torch.no_grad():
                flow_action = self.tax_pose(action_points_and_cond.permute(0, 2, 1), anchor_points_and_cond.permute(0, 2, 1),
                                                conditioning_action=tax_pose_conditioning_action,
                                                conditioning_anchor=tax_pose_conditioning_anchor,
                                                action_center=action_center,
                                                anchor_center=anchor_center)
        else:
            flow_action = self.tax_pose(action_points_and_cond.permute(0, 2, 1), anchor_points_and_cond.permute(0, 2, 1),
                                            conditioning_action=tax_pose_conditioning_action,
                                            conditioning_anchor=tax_pose_conditioning_anchor,
                                            action_center=action_center,
                                            anchor_center=anchor_center)
    

        ########## LOGGING ############

        # Change goal_emb here to be what is going to be logged. For the latent_z conditioning, we just log the mean
        if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            goal_emb = goal_emb[0]

        if self.tax_pose.return_flow_component:
            if self.freeze_residual_flow:
                flow_action['flow_action'] = flow_action['flow_action'].detach()
                flow_action['flow_anchor'] = flow_action['flow_anchor'].detach()
            flow_action = {
                **flow_action, 
                'goal_emb': goal_emb,
                **for_debug,
            }
        else:
            if self.freeze_residual_flow:
                flow_action = (flow_action[0].detach(), flow_action[1].detach(), *flow_action[2:])

            if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"] and mode == "forward":
                # These are for the loss
                heads = {
                    k: for_debug[k] for k in ['goal_emb_mu', 'goal_emb_logvar']
                }
                flow_action = (
                    *flow_action, 
                    goal_emb,
                    heads
                )
            else:
                flow_action = (*flow_action, goal_emb)

        return flow_action
    
    def sample(self, action_points, anchor_points):
        if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Take a SINGLE sample z ~ N(0,1)
            for_debug = {}
            goal_emb_action = None
            goal_emb_anchor = None
            if self.conditioning == "latent_z_linear":
                goal_emb = torch.tile(torch.randn((action_points.shape[0], self.emb_dims, 1)).to(action_points.device), (1, 1, action_points.shape[-1]))
                action_points_and_cond = torch.cat([action_points, goal_emb], axis=1)
                anchor_points_and_cond = torch.cat([anchor_points, goal_emb], axis=1)
            elif self.conditioning == "latent_z_linear_internalcond":
                goal_emb = torch.randn((action_points.shape[0], self.latent_z_linear_size, 1)).to(action_points.device)
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug['goal_emb'] = goal_emb
            else:
                raise ValueError("Why is it here?")
        elif self.conditioning in ['uniform_prior_pos_delta_l2norm']:
            # sample from a uniform prior
            N_action, N_anchor, B = action_points.shape[-1], anchor_points.shape[-1], action_points.shape[0]
            translation_sample_action = F.one_hot(torch.randint(N_action, (B,)), N_action).float().cuda()
            translation_sample_anchor = F.one_hot(torch.randint(N_anchor, (B,)), N_anchor).float().cuda()

            dense_trans_pt_action, ref_action = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, action_points, translation_sample_action, conditioning=self.conditioning)
            dense_trans_pt_anchor, ref_anchor = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, anchor_points, translation_sample_anchor, conditioning=self.conditioning)

            action_points_and_cond = torch.cat([action_points] + [dense_trans_pt_action], axis=1)
            anchor_points_and_cond = torch.cat([anchor_points] + [dense_trans_pt_anchor], axis=1)

            goal_emb = None

            for_debug = {
                'dense_trans_pt_action': dense_trans_pt_action,
                'dense_trans_pt_anchor': dense_trans_pt_anchor,
                'trans_pt_action': ref_action,
                'trans_pt_anchor': ref_anchor,
                'trans_sample_action': translation_sample_action,
                'trans_sample_anchor': translation_sample_anchor,
            }
        else:
            raise ValueError(f"Sampling not supported for conditioning {self.conditioning}. Pick one of the latent_z_xxx conditionings")
        return action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug

class Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(nn.Module):
    def __init__(self, residualflow_embnn, encoder_type="2_dgcnn", sample_z=True, shuffle_for_pzX=False):
        super(Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX, self).__init__()
        self.residflow_embnn = residualflow_embnn

        # Use the other class definition so that it matches between classes
        self.conditioning = self.residflow_embnn.conditioning
        self.num_emb_heads = self.residflow_embnn.num_emb_heads
        self.emb_dims = self.residflow_embnn.emb_dims
        self.taxpose_centering = self.residflow_embnn.taxpose_centering
        self.freeze_residual_flow = self.residflow_embnn.freeze_residual_flow
        self.freeze_z_embnn = self.residflow_embnn.freeze_z_embnn
        self.freeze_embnn = self.residflow_embnn.freeze_embnn

        self.shuffle_for_pzX = shuffle_for_pzX

        #assert self.conditioning not in ['uniform_prior_pos_delta_l2norm']

        # assert self.conditioning not in ["latent_z_linear", "latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear_internalcond"], "Latent z conditioning does not need a p(z|X) because it's regularized to N(0,1)"

        # Note: 1 DGCNN probably loses some of the rotational invariance between objects
        assert encoder_type in ["1_dgcnn", "2_dgcnn"]
        
        # disable smoothing
        self.add_smooth_factor = 0.05
        self.division_smooth_factor = 1.0
        self.gumbel_temp = self.residflow_embnn.gumbel_temp

        self.encoder_type = encoder_type
        self.sample_z = sample_z

        if self.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            if self.encoder_type == "1_dgcnn":
                self.p_z_cond_x_embnn = DGCNN(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False)
            elif self.encoder_type == "2_dgcnn":
                self.p_z_cond_x_embnn_action = DGCNN(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False)
                self.p_z_cond_x_embnn_anchor = DGCNN(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False)
            else:
                raise ValueError()
        else:
            if self.encoder_type == "1_dgcnn":
                self.p_z_cond_x_embnn = DGCNNClassification(emb_dims=self.emb_dims, num_heads=self.num_emb_heads)
            elif self.encoder_type == "2_dgcnn":
                self.p_z_cond_x_embnn_action = DGCNNClassification(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, dropout=0.5, output_channels=self.residflow_embnn.latent_z_linear_size)
                self.p_z_cond_x_embnn_anchor = DGCNNClassification(emb_dims=self.emb_dims, num_heads=self.num_emb_heads, dropout=0.5, output_channels=self.residflow_embnn.latent_z_linear_size)
            else:
                raise ValueError()

        self.center_feature = self.residflow_embnn.center_feature

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)[:, :3] # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        # mean center point cloud before DGCNN
        if self.residflow_embnn.center_feature:
            action_points_dmean = action_points - \
                action_points.mean(dim=2, keepdim=True)
            anchor_points_dmean = anchor_points - \
                anchor_points.mean(dim=2, keepdim=True)
        else:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points

        if self.shuffle_for_pzX:
            action_shuffle_idxs = torch.randperm(action_points_dmean.size()[2])
            anchor_shuffle_idxs = torch.randperm(anchor_points_dmean.size()[2])
            action_points_dmean = action_points_dmean[:,:,action_shuffle_idxs]
            anchor_points_dmean = anchor_points_dmean[:,:,anchor_shuffle_idxs]

        def prepare(arr, is_action):
            if self.shuffle_for_pzX:
                shuffle_idxs = action_shuffle_idxs if is_action else anchor_shuffle_idxs
                return arr[:,:,torch.argsort(shuffle_idxs)]
            else:
                return arr

        if self.encoder_type == "1_dgcnn":
            goal_emb_cond_x = self.p_z_cond_x_embnn(torch.cat([action_points_dmean, anchor_points_dmean], dim=-1))
            goal_emb_cond_x_action = prepare(goal_emb_cond_x[:, :, :action_points_dmean.shape[-1]])
            goal_emb_cond_x_anchor = prepare(goal_emb_cond_x[:, :, action_points_dmean.shape[-1]:])
        elif self.encoder_type == "2_dgcnn":
            # Sample a point
            goal_emb_cond_x_action = self.p_z_cond_x_embnn_action(action_points_dmean)
            goal_emb_cond_x_anchor = self.p_z_cond_x_embnn_anchor(anchor_points_dmean)

            if self.num_emb_heads > 1:
                goal_emb_cond_x = [
                    torch.cat([prepare(action_head, True), prepare(anchor_head, False)], dim=-1)
                        for action_head, anchor_head in zip(goal_emb_cond_x_action, goal_emb_cond_x_anchor)
                ]
            else:
                goal_emb_cond_x = torch.cat([prepare(goal_emb_cond_x_action, True), prepare(goal_emb_cond_x_anchor, False)], dim=-1)
        else:
            raise ValueError()
        
        action_points_and_cond, anchor_points_and_cond, for_debug = Multimodal_ResidualFlow_DiffEmbTransformer.add_conditioning(self, goal_emb_cond_x, action_points, anchor_points, self.conditioning)
        
        tax_pose_conditioning_action = None
        tax_pose_conditioning_anchor = None
        if self.conditioning == "latent_z_linear_internalcond":
            tax_pose_conditioning_action = torch.tile(for_debug['goal_emb'][:,:,0][:,:,None], (1, 1, action_points.shape[-1]))
            tax_pose_conditioning_anchor = torch.tile(for_debug['goal_emb'][:,:,1][:,:,None], (1, 1, anchor_points.shape[-1]))

        if self.taxpose_centering == "mean":
            # Use TAX-Pose defaults
            action_center = action_points[:, :3].mean(dim=2, keepdim=True)
            anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
        elif self.taxpose_centering == "z":
            action_center = for_debug['trans_pt_action'][:,:,None]
            anchor_center = for_debug['trans_pt_anchor'][:,:,None]
        else:
            raise ValueError(f"Unknown self.taxpose_centering: {self.taxpose_centering}")

        # Unpermute the action and anchor point clouds to match how tax pose is written
        flow_action = self.residflow_embnn.tax_pose(action_points_and_cond.permute(0, 2, 1), anchor_points_and_cond.permute(0, 2, 1),
                                                    conditioning_action=tax_pose_conditioning_action,
                                                    conditioning_anchor=tax_pose_conditioning_anchor,
                                                    action_center=action_center,
                                                    anchor_center=anchor_center)

        if self.residflow_embnn.tax_pose.return_flow_component:
            # If the demo is available, run p(z|Y)
            if input[2] is not None:
                # Inputs 2 and 3 are the objects in demo positions
                # If we have access to these, we can run the pzY network
                pzY_results = self.residflow_embnn(*input)
                goal_emb = pzY_results['goal_emb']
            else:
                goal_emb = None

            flow_action = {
                **flow_action,
                'goal_emb': goal_emb,
                'goal_emb_cond_x': goal_emb_cond_x,
                **for_debug,
            }
        else:
            # If the demo is available, run p(z|Y)
            if input[2] is not None:
                # Inputs 2 and 3 are the objects in demo positions
                # If we have access to these, we can run the pzY network
                pzY_results = self.residflow_embnn(*input)
                goal_emb = pzY_results[2]
            else:
                goal_emb = None

            flow_action = (*flow_action, goal_emb, goal_emb_cond_x)
            # TODO put everything into a dictionary later. Getting outputs from a tuple with a changing length is annoying
            if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
                flow_action = (
                    *flow_action, 

                    # These are for the loss
                    {
                        k: for_debug[k] for k in ['goal_emb_mu', 'goal_emb_logvar']
                    }
                )


        return flow_action




