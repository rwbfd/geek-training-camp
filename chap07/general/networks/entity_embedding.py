# coding = 'utf-8'
import logging

import torch
import torch.nn as nn

from ...tabular.tab_data_opt import TabDataOpt

EPS = 1e-5


class EntityEmbeddingVLayer(nn.Module):
    """
    This embedding layer uses vicinity information to smooth over the continuous variables.
    If normal embedding layer is to be desired, please use EntityEmbeddingLayer (without V).
    """

    def __init__(self, num_level, emdedding_dim, centroid, name):
        super(EntityEmbeddingVLayer, self).__init__()
        self.embedding = nn.Embedding(num_level, emdedding_dim)
        self.centroid = torch.tensor(centroid).detach_().unsqueeze(1)
        self.name = name
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x must be batch_size times 1
        """
        x = x.unsqueeze(1)
        x = x.unsqueeze(2)
        d = 1.0 / ((x - self.centroid).abs() + EPS)
        w = self.softmax(d.squeeze(2))
        v = torch.mm(w, torch.transpose(self.embedding.weight, 0, 1))
        return v


class EntityEmbeddingLayer(nn.Module):
    def __init__(self, num_level, embedding_dim, name):
        super(EntityEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_level, embedding_dim)
        self.name = name

    def forward(self, x):
        return self.embedding(x)


class EntityDenseLayer(nn.Module):
    def __init__(self, data_pd, opt: TabDataOpt, mlp=None):
        super(EntityDenseLayer, self).__init__()
        self.opt = opt
        self.mlp = mlp

        if len(self.opt.conti_vars) and mlp is None:
            logging.error("""
            The options have specified the using of continuous variables. Yet no MLP network is specified.
            """)

        if self.opt.dis_vars_vic is not None and self.opt.centroids is not None:
            self.entity_v = nn.ModuleDict()
            for name in self.opt.dis_vars_vic:
                self.entity_v[name] = EntityEmbeddingVLayer(len(self.opt.centroids[name]), self.opt.num_dim,
                                                            centroid[name], name)

        if self.opt.dis_vars_entity is not None:
            self.entity = nn.ModuleDict()
            for name in self.opt.dis_vars_entity:
                self.entity[name] = EntityEmbeddingLayer(len(data_pd[name].unique()), self.opt.num_dim, name)

    def forward(self, x, **kwargs):
        result = list()
        if self.opt.dis_vars_vic is not None:
            for name in self.opt.dis_vars_vic:
                result.append(torch.unsqueeze(self.entity_v[name](x[name]), 1))

        if self.opt.dis_vars_entity is not None:
            for name in self.opt.dis_vars_entity:
                result.append(torch.unsqueeze(self.entity[name](x[name]), 1))

        if self.opt.conti_vars is not None:  # TODO: This part must be tested.
            pass

        return torch.cat(result, dim=1)
