import os
import numpy as np
import random
import torch

from copy import deepcopy

from data_IP import get_feature_names

from utils import CELL_TYPES_VERSION8_2, EXPRESSION_MARKERS

class CODEXNodeTransform(object):
    '''adapted from space-gm/transforms.py to comply with the structure of the Immuno Profile data'''

    """ Base transformer object """
    def __init__(self, 
                 node_features=None, # classes of features for the nodes (can be filtered) 
                 edge_features=None, # edge_type (self, distant), edge_distance: distance between the centers of connected cells
                 cell_types=CELL_TYPES_VERSION8_2,
                 cell_features = EXPRESSION_MARKERS, # specify expression features (markers)
                 cell_morphology_features = None, # all cell morphology features
                 use_selected_morphology_features = None, # if you do not want to use all cell morphology features, specify the cell morphology features that should not be masked
                 expression_features = None,
                 use_neighbor_node_features=None, # feature classes for all neighborh nodes (all nodes except the center node/cell)
                 use_center_node_features=None, # feature classes for the center node/cell (if None: all node_features will be used)
                 use_edge_features=None, # specify the edge features that should not be masked (if none: all edge_features will be used)
                 ):
        
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
