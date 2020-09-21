# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNet201(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super().__init__()
        base_model = models.densenet201(pretrained=pretrained)

        layers = list(base_model.children())[:-1]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)
        in_features = base_model.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }
