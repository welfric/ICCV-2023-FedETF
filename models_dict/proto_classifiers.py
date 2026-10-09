
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import copy
import math
import numpy as np


class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.proto = M.cuda()

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def load_proto(self, proto):
        self.proto = copy.deepcopy(proto)

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d
        return target


class Semantic_Classifier(nn.Module):
    '''
    FedSem classifier: anchors T_lambda (C x d) built once on the server, see fedsem.py
    '''
    def __init__(self, feat_in, num_classes, learnable=False):
        super(Semantic_Classifier, self).__init__()
        anchors = torch.zeros(num_classes, feat_in)
        if learnable:
            # control: unfreezing T_lambda gives a learnable head, trained and averaged like the other weights
            self.anchors = nn.Parameter(anchors)
        else:
            # frozen: a non-persistent buffer is neither optimized nor in the state_dict, so aggregation never touches it
            self.register_buffer('anchors', anchors, persistent=False)

    def load_anchors(self, anchors):
        self.anchors.data.copy_(anchors)

    def forward(self, feature):
        # B, d -> B, C: entry k is t_k^T f
        return torch.matmul(feature, self.anchors.T)


