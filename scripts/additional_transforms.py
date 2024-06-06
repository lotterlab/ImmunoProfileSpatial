import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from datetime import date

import torch
import torch_geometric

from data import get_feature_names

from utils import get_all_graph_labels, EDGE_TYPES, EXPRESSION_MARKERS
from utils import CELL_TYPES_VERSION8_2, CELL_TYPE_LABELS_VERSION8_2, EXPRESSION_MARKERS

import random


   
def get_feature_names(features, cell_types=CELL_TYPES_VERSION8_2, cell_features=EXPRESSION_MARKERS, cell_morphology_features=None, expression_features=None): # expression_markers=EXPRESSION_MARKERS,
    '''adapted from /space-gm/data.py to comply with the structure of the ImmunoProfile data'''
    
    """ helper fn for getting feature names """
    # print(cell_morphology_features)
    # print(expression_features)
    feat_names = []
    for feat in features:
        # print(feat)
        if feat in ["size", "distance", "cell_type", "edge_type"]:
            feat_names.append(feat)
        elif feat == "center_coord":
            feat_names.extend(["center_coord-x", "center_coord-y"])
        elif feat == "expression":
            feat_names.extend(["expression-%s" %
                               exp for exp in cell_features])
        elif feat == "cell_morphology":
            feat_names.extend(["cell_morphology-%s" %
                               exp for exp in cell_morphology_features])
        elif feat == "expression_features":
            feat_names.extend(["expression_features-%s" %
                               exp for exp in expression_features])
        elif feat == "neighborhood_composition":
            feat_names.extend(["neighborhood_composition-%s" %
                ct for ct in sorted(cell_types.keys(), key=lambda x: cell_types[x])])
        elif feat == "neighborhood_expression":
            feat_names.extend(["neighborhood_expression-%s" %
                exp for exp in expression_features])
        else:
            print(feat)
            raise ValueError("Feature not recognized")
    return feat_names
    


class CODEXNodeTransform(object):
    '''adapted from space-gm/transforms.py to comply with the structure of the Immuno Profile data'''

    """ Base transformer object """
    def __init__(self, 
                 node_features=None, 
                 edge_features=None,
                 cell_types=CELL_TYPES_VERSION8_2,
                 # expression_markers=EXPRESSION_MARKERS,
                 cell_features = EXPRESSION_MARKERS,
                 cell_morphology_features = None,
                 use_selected_morphology_features = None,
                 expression_features = None,
                 use_neighbor_node_features=None,
                 use_center_node_features=None,
                 use_edge_features=None,):
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.cell_types = cell_types
        # self.expression_markers = expression_markers
        self.cell_features = cell_features
        self.cell_morphology_features = cell_morphology_features
        self.expression_features = expression_features
        self.node_feature_names = get_feature_names(node_features, cell_types=self.cell_types, cell_features=self.cell_features, cell_morphology_features=self.cell_morphology_features, expression_features=self.expression_features)
        print('all node features: ', self.node_feature_names)
        self.edge_feature_names = get_feature_names(edge_features, cell_types=self.cell_types, cell_features=self.cell_features, cell_morphology_features=self.cell_morphology_features, expression_features=self.expression_features)
        print('edge features: ', self.edge_feature_names)
        self.use_neighbor_node_features = use_neighbor_node_features if not use_neighbor_node_features is None else self.node_features
        print('neighbor features: ', self.use_neighbor_node_features)
        self.use_center_node_features = use_center_node_features if not use_center_node_features is None else self.node_features
        print('center node features: ', self.use_center_node_features)
        self.use_edge_features = use_edge_features if not use_edge_features is None else self.edge_features
        self.use_selected_morphology_features = use_selected_morphology_features 
        print('selected_morphology_features:', self.use_selected_morphology_features)
        
        self.center_node_feature_masks = list()
        self.neighbor_node_feature_masks = list()
        print(len(self.node_feature_names))

        if not self.use_selected_morphology_features is None:
            assert 'cell_morphology' in self.use_center_node_features
            self.use_center_node_features.remove('cell_morphology')

            assert 'cell_morphology' in self.use_neighbor_node_features
            self.use_neighbor_node_features.remove('cell_morphology')
            
        for name in self.node_feature_names:
            add = False
            for feat in self.use_center_node_features:
                if name.startswith(feat):
                    # self.center_node_feature_masks.append(1)
                    add = True
                    break
            if not self.use_selected_morphology_features is None:
                for feat in self.use_selected_morphology_features:
                    # print(feat)
                    if name == feat:
                        # print('select')
                        add = True
            if add:
                self.center_node_feature_masks.append(1)
            else:
                self.center_node_feature_masks.append(0)

        for name in self.node_feature_names:
            add = False
            for feat in self.use_neighbor_node_features:
                if name.startswith(feat):
                    # self.center_node_feature_masks.append(1)
                    add = True
                    break
            if not self.use_selected_morphology_features is None:
                for feat in self.use_selected_morphology_features:
                    # print(feat)
                    if name == feat:
                        # print('select')
                        add = True
            if add:
                self.neighbor_node_feature_masks.append(1)
            else:
                self.neighbor_node_feature_masks.append(0)

        self.center_node_feature_masks = \
            torch.from_numpy(
                np.array(self.center_node_feature_masks).reshape((-1,))).float()
        self.neighbor_node_feature_masks = \
            torch.from_numpy(
                np.array(self.neighbor_node_feature_masks).reshape((1, -1))).float()


    def __call__(self, data):
        data = deepcopy(data)
        if "center_node_index" in data:
            center_node_feat = data.x[data.center_node_index].detach().data.clone()
        else:
            center_node_feat = None
        data = self.transform_neighbor_node(data)
        data = self.transform_center_node(data, center_node_feat)

        return data

        
    def transform_neighbor_node(self, data):
        data.x = data.x * self.neighbor_node_feature_masks
        return data


    def transform_center_node(self, data, center_node_feat=None):
        if center_node_feat is None:
            return data
        assert "center_node_index" in data
        center_node_feat = center_node_feat * self.center_node_feature_masks
        data.x[data.center_node_index] = center_node_feat
        return data
