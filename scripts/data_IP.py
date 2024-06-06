#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy of space-gm/data.py 
code adapted to work with the structure of the Immuno Profile data 
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
from scipy.stats import rankdata
# from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from joblib import cpu_count, delayed, Parallel

import torch_geometric as tg
import torch
from torch_geometric.data import Dataset
from torch_geometric.data.dataloader import DataLoader
from torch_geometric.utils import subgraph
from torch.utils.data import RandomSampler, WeightedRandomSampler
# from torch_geometric.data.dataloader import DataLoader

from graph_build import plot_codex_graph
from utils import get_clinical_data_table, get_sample_info, EDGE_TYPES
from utils import CELL_TYPES_VERSION8_2, CELL_TYPE_FREQ_VERSION8_2, EXPRESSION_MARKERS


def process_expression(expression_vec, method='linear', **kwargs):
    """ Expression preprocessing fn """
    expression_vec = np.array(expression_vec)
    if method == 'linear':
        assert "upper_bound" in kwargs
        assert "lower_bound" in kwargs
        lb = kwargs['lower_bound']
        ub = kwargs['upper_bound']
        expression_vec = np.clip(expression_vec, lb, ub)
        expression_vec = (expression_vec - lb)/(ub - lb)
        return list(expression_vec)
    elif method == 'log':
        assert "upper_bound" in kwargs
        assert "lower_bound" in kwargs
        lb = kwargs['lower_bound']
        ub = kwargs['upper_bound']
        expression_vec = np.clip(np.log(expression_vec + 1e-9), lb, ub)
        expression_vec = (expression_vec - lb)/(ub - lb)
        return list(expression_vec)
    elif method == 'rank':
        expression_vec = rankdata(expression_vec, method='min')
        num_exp = len(expression_vec)
        expression_vec = (expression_vec - 1)/(num_exp - 1)
        return list(expression_vec)
    elif method == 'binary':
        assert "gm" in kwargs
        gm = kwargs["gm"]
        expression_vec = gm.predict_proba(expression_vec.reshape((-1, 1)))[:, 1]
        return list(expression_vec)
    elif method == 'scale':
        # scales to range 0-1 (no clipping)
        max = np.max(expression_vec)
        min = np.min(expression_vec)
        expression_vec = (expression_vec - min)/(max - min)
        return list(expression_vec)
    elif method== 'conserve':
        # return the expression as it is (because I had already normalized everything)
        return(list(expression_vec))


def get_neighbor_composition(G, 
                             node_ind, 
                             neighborhood_size=10, 
                             cell_types=CELL_TYPES_VERSION8_2,
                             **kwargs):
    """ Calculate the composition vector of k-nearest neighboring cells """
    center_pos = G.nodes[node_ind]["center_coord"]
    
    def node_dist(center_pos, neighbor_pos):
        return (center_pos[0] - neighbor_pos[0]) ** 2 + \
            (center_pos[1] - neighbor_pos[1]) ** 2
    radius = 1
    neighbors = {}
    while len(neighbors) < min(len(G) - 1, 2*neighborhood_size):
        radius += 1
        ego_g = nx.ego_graph(G, node_ind, radius=radius)
        neighbors = {n: feat_dict["center_coord"] for n, feat_dict in ego_g.nodes.data()}
        
    closest_neighbors = sorted(neighbors.keys(), key=lambda x: node_dist(center_pos, neighbors[x]))
    # assert closest_neighbors[0] == node_ind
    closest_neighbors = closest_neighbors[1:(neighborhood_size+1)]
    
    feat_ar = np.zeros((len(cell_types),))
    for n in closest_neighbors:
        cell_type = cell_types[G.nodes[n]["cell_type"]]
        feat_ar[cell_type] += 1
    feat_ar = list(feat_ar / feat_ar.sum())
    return feat_ar

def get_neighbor_expression(G,
                            node_ind,
                            neighborhood_size=10,
                            expression_markers=['PD1', 'PDL1', 'FOXP3', 'TUMOR', 'CD8'], # true experssion markers (not cell features)
                            **kwargs):
    
    """ Calculate the expression composition vector of k-nearest neighboring cells """
    # new additon (not part of the original space gm code)
    center_pos = G.nodes[node_ind]["center_coord"]
    
    def node_dist(center_pos, neighbor_pos):
        return (center_pos[0] - neighbor_pos[0]) ** 2 + \
            (center_pos[1] - neighbor_pos[1]) ** 2
    radius = 1
    neighbors = {}
    # grab the neighbor nodes
    while len(neighbors) < min(len(G) - 1, 2*neighborhood_size):
        radius += 1
        ego_g = nx.ego_graph(G, node_ind, radius=radius)
        neighbors = {n: feat_dict["center_coord"] for n, feat_dict in ego_g.nodes.data()}
        
    closest_neighbors = sorted(neighbors.keys(), key=lambda x: node_dist(center_pos, neighbors[x]))
    # assert closest_neighbors[0] == node_ind
    closest_neighbors = closest_neighbors[1:(neighborhood_size+1)]
    # print(closest_neighbors)
    expr_feat = np.zeros((len(closest_neighbors),len(expression_markers)))
    for i in range(len(closest_neighbors)):
        # add the expression features for the closest neighbors
        expr_feat[i,:] = [G.nodes[closest_neighbors[i]]['expression'][a] for a in expression_markers]
        # print(closest_neighbors[i])
    expr_feat = list(np.mean(expr_feat, axis=0))
    return expr_feat
    

def process_feature(G, key, node_ind=None, edge_ind=None, **kwargs):
    """ wrapper fn for generating node/edge features """
    # Node features
    # print(key)
    if key == "size":
        v = np.log(G.nodes[node_ind][key])
        v = np.clip((v - 4.5)/(7.5 - 4.5), 0, 1)
        return [v]
    elif key == "center_coord":
        v = list(G.nodes[node_ind][key])
        return v
    elif key == "cell_type":
        assert "cell_types" in kwargs
        v = [kwargs["cell_types"][G.nodes[node_ind][key]]]
        return v
    elif key == "expression":
        # for both expression_markers and cell_morphology
        val = G.nodes[node_ind]["expression"]
        v = []
        assert "cell_features" in kwargs
        for marker in kwargs["cell_features"]:
            #print(marker)
            #print(val)
            if val is None:
                v.append(0.)
            else:
                v.append(val[marker])
        v = process_expression(v, **kwargs)
        return v
    elif key == "cell_morphology":
        # select specified cell morphology features (no markers!)
        val = G.nodes[node_ind]["expression"]
        # print(val)
        v = []
        assert 'cell_morphology_features' in kwargs
        for marker in kwargs['cell_morphology_features']:
            if val is None:
                v.append(0.)
            else:
                v.append(val[marker])
        v = process_expression(v, **kwargs)
        return v
    elif key == 'expression_features':
        # select specified expression markers only
        # no scaling
        val = G.nodes[node_ind]["expression"]
        v = []
        assert 'expression_features' in kwargs
        for marker in kwargs['expression_features']:
            if val is None:
                v.append(0.)
            else:
                v.append(val[marker])
        return v
    elif key == "neighborhood_composition":
        assert "neighborhood_size" in kwargs
        assert "cell_types" in kwargs
        v = get_neighbor_composition(G, node_ind, **kwargs)
        return v
    elif key == "neighborhood_expression":
        assert "neighborhood_size" in kwargs
        v = get_neighbor_expression(G, node_ind, **kwargs)
        return v

    # Edge features
    elif key == "distance":
        v = np.log(G.edges[edge_ind][key] + 1e-5)
        v = np.clip((v - 2.)/(5. - 2.), 0, 1)
        return [v]
    elif key == "edge_type":
        v = [EDGE_TYPES[G.edges[edge_ind][key]]]
        return v
    else:
        raise ValueError("Feature not recognized")


def get_feature_names(features, cell_types=CELL_TYPES_VERSION8_2, cell_features=EXPRESSION_MARKERS, cell_morphology_features=None, expression_features=None): # expression_markers=EXPRESSION_MARKERS,
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
            raise ValueError("Feature not recognized")
    return feat_names


def nx_to_tg_graph(G,
                   node_features=["cell_type", 
                                  "size", 
                                  "expression", 
                                  "neighborhood_composition",
                                  "neighborhood_expression", 
                                  "center_coord"],
                   edge_features=["edge_type", 
                                  "distance"],
                   **kwargs):
    """ Build tensorgraphs from nx graph """
    assert "cell_types" in kwargs, "Cell type version should be specified"
    data_list = []
    for inds in nx.connected_components(G):
        if len(inds) >=100:
            sub_G = G.subgraph(inds)
            mapping = {n: i for i, n in enumerate(sorted(sub_G.nodes))}
            sub_G = nx.relabel.relabel_nodes(sub_G, mapping)
            assert np.all(sub_G.nodes == np.arange(len(sub_G)))
            
            data = {"x": [], "edge_attr": [], "edge_index": []}
            for node_ind in sub_G.nodes:
                feat_val = []
                for key in node_features:
                    # print(key)
                    feat_val.extend(process_feature(sub_G, key, node_ind=node_ind, **kwargs))
                    # print(feat_val)
                data["x"].append(feat_val)

            for edge_ind in sub_G.edges:
                feat_val = []
                for key in edge_features:
                    feat_val.extend(process_feature(sub_G, key, edge_ind=edge_ind, **kwargs))
                data["edge_attr"].append(feat_val)
                data["edge_index"].append(edge_ind)
                data["edge_attr"].append(feat_val)
                data["edge_index"].append(tuple(reversed(edge_ind)))

            for key, item in data.items():
                data[key] = torch.tensor(item)
            data['edge_index'] = data['edge_index'].t().long()
            data = tg.data.Data.from_dict(data)
            data.num_nodes = sub_G.number_of_nodes()
            data_list.append(data)
    return data_list


def k_hop_subgraph(node_idx, 
                   num_hops, 
                   edge_index, 
                   edge_type_mask=None,
                   relabel_nodes=False,
                   num_nodes=None, 
                   flow='source_to_target'):
    """ A customized k-hop subgraph fn that could filter for edge_type """

    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    edge_type_mask = torch.ones_like(edge_mask) if edge_type_mask is None else edge_type_mask

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    next_root = node_idx

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[next_root] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        next_root = col[edge_mask * edge_type_mask] # use nodes connected with mask=True to span

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def get_graph_splits(dataset,
                     split='random',
                     cv_k=5,
                     seed=12,
                     id_mapping=None):
    """ Define train/valid split """
    splits = {}
    aq_ids = set([dataset.get_full(i).aq_id for i in range(dataset.N)])
    _aq_ids = sorted(aq_ids)
    if split == 'random':
        if not seed is None:
            np.random.seed(seed)
        if id_mapping is None:
            id_mapping = {aq_id: aq_id for aq_id in _aq_ids}
        
        # `_ids` could be sample ids / patient ids / certain properties
        _ids = sorted(set(list(id_mapping.values())))
        np.random.shuffle(_ids)
        cv_shard_size = len(_ids) / cv_k
        for i, aq_id in enumerate(_aq_ids):
            splits[aq_id] = _ids.index(id_mapping[aq_id]) // cv_shard_size
            
    elif split == 'coverslip':
        sample_df = get_sample_info()
        coverslip_mapping = dict(
            zip(sample_df['ACQUISITION_ID'], sample_df["COVERSLIP_ID"]))

        unique_coverslips = sorted(
            set(coverslip_mapping[aq_id] for aq_id in aq_ids))
        for aq_id in aq_ids:
            coverslip_id = coverslip_mapping[aq_id]
            splits[aq_id] = unique_coverslips.index(coverslip_id)
            
    elif split == 'cs_fold':
        # Only used for UPMC
        cs_to_fold = {
            70: 1,
            71: 2,
            72: 0,
            73: -1,
            74: 0,
            75: 1,
            76: 2,
        }
        sample_df = get_sample_info()
        coverslip_mapping = dict(
            zip(sample_df['ACQUISITION_ID'], sample_df["COVERSLIP_ID"]))
        for aq_id in aq_ids:
            coverslip_id = coverslip_mapping[aq_id]
            splits[aq_id] = cs_to_fold[coverslip_id]
    
    else:
        raise ValueError("split mode not recognized")
    
    split_inds = []
    for i in range(dataset.N):
        split_inds.append(splits[dataset.get_full(i).aq_id])
    return split_inds


class CODEXGraphDataset(Dataset):
    """ Main dataset structure for model training / inference """
    def __init__(self,
                 root,
                 transform=[],
                 pre_transform=None,
                 raw_folder_name='graph',
                 processed_folder_name='tg_graph',
                 subsample_neighbor_size=0,
                 node_features=["cell_type", "size", "expression", "neighborhood_composition", "center_coord"],
                 edge_features=["edge_type", "distance"],
                 cell_types=CELL_TYPES_VERSION8_2,
                 cell_type_freq=CELL_TYPE_FREQ_VERSION8_2,
                 expression_markers=EXPRESSION_MARKERS,
                 subgraph_source='on-the-fly', # 'save', 'chunk_save', 'on-the-fly'
                 subgraph_allow_distant_edge=True,
                 subgraph_size_limit=0,
                 sampling_avoid_unassigned=True,
                 **kwargs):
        self.root = root
        self.raw_folder_name = raw_folder_name
        self.processed_folder_name = processed_folder_name
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # If to subsample local graphs, 0 = no subsampling
        self.subsample_neighbor_size = subsample_neighbor_size

        # Feature names
        self.node_features = node_features
        self.edge_features = edge_features
        self.cell_types = cell_types
        self.cell_type_freq = cell_type_freq
        self.expression_markers = expression_markers

        self.node_feature_names = get_feature_names(node_features, cell_types=self.cell_types)
        self.edge_feature_names = get_feature_names(edge_features, cell_types=self.cell_types)
        
        self.process_kwargs = kwargs
        self.process_kwargs['cell_types'] = self.cell_types
        self.process_kwargs['expression_markers'] = self.expression_markers
        
        super(CODEXGraphDataset, self).__init__(root, None, pre_transform)
        self.transform = transform
        self.subgraph_source = subgraph_source
        self.subgraph_allow_distant_edge = subgraph_allow_distant_edge
        self.subgraph_size_limit = subgraph_size_limit
        
        self.N = len(self.processed_paths)
        self.sampling_freq = {self.cell_types[ct]: 1./self.cell_type_freq[ct] for ct in self.cell_types}
        self.sampling_freq = torch.from_numpy(np.array([self.sampling_freq[i] for i in range(len(self.sampling_freq))]))
        # Avoid sampling unassigned cell
        if sampling_avoid_unassigned:
            self.sampling_freq[self.cell_types['Unassigned']] = 0.
        self.cached_data = {}
        
    def set_indices(self, inds):
        """ Limit sampling to `inds` """
        self._indices = inds
        return
    
    def set_subgraph_source(self, subgraph_source):
        assert subgraph_source in ['save', 'chunk_save', 'on-the-fly']
        self.subgraph_source = subgraph_source
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.gpkl')])

    @property
    def processed_file_names(self):
        # Only files for full graphs
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.gpt') and not 'hop' in f])

    def len(self):
        return len(self.processed_paths)

    def process(self):

        def process_graph(raw_path):
            # for raw_path in self.raw_paths:
            aq_id = os.path.splitext(os.path.split(raw_path)[-1])[0]
            if not os.path.exists(os.path.join(self.processed_dir, '%s.0.gpt' % aq_id)):
                G = nx.read_gpickle(raw_path)
                data_list = nx_to_tg_graph(G,
                                            node_features=self.node_features,
                                            edge_features=self.edge_features,
                                            **self.process_kwargs)
                for i, d in enumerate(data_list):
                    d["aq_id"] = aq_id
                    d["subgraph_id"] = i
                    if not self.pre_transform is None:
                        for transform_fn in self.pre_transform:
                            d = transform_fn(d)
                    torch.save(d, os.path.join(self.processed_dir,
                                                '%s.%d.gpt' % (d.aq_id, d.subgraph_id)))

        # print(self.raw_paths)
        '''for raw_path in self.raw_paths:
            # print(raw_path)
            aq_id = os.path.splitext(os.path.split(raw_path)[-1])[0]
            if os.path.exists(os.path.join(self.processed_dir, '%s.0.gpt' % aq_id)):
                continue
            
            G = nx.read_gpickle(raw_path)
            data_list = nx_to_tg_graph(G,
                                       node_features=self.node_features,
                                       edge_features=self.edge_features,
                                       **self.process_kwargs)
            for i, d in enumerate(data_list):
                d["aq_id"] = aq_id
                d["subgraph_id"] = i
                if not self.pre_transform is None:
                    for transform_fn in self.pre_transform:
                        d = transform_fn(d)
                torch.save(d, os.path.join(self.processed_dir,
                                           '%s.%d.gpt' % (d.aq_id, d.subgraph_id)))'''
        
            
        Parallel(n_jobs=cpu_count()-2)(delayed(process_graph)(raw_path) for raw_path in self.raw_paths)
            
        return


    def save_all_subgraphs_to_chunk(self):

        def save_one(idx,p):
            """ save individual n-hop subgraphs to file (one file per sample) """
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]
            neighbor_graph_path = p.replace('.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
            if not os.path.exists(neighbor_graph_path):
                subgraphs = []
                for node_i in range(n_nodes):
                    subgraphs.append(self.get_subgraph(idx, node_i))
                torch.save(subgraphs, neighbor_graph_path)

        Parallel(n_jobs=cpu_count()-2)(delayed(save_one)(idx,p) for idx,p in enumerate(self.processed_paths))
        
        # for idx, p in enumerate(self.processed_paths):    
        return


    def save_all_subgraphs(self):
        """ (deprecated) save individual n-hop subgraph to file (one file per subgraph)  """
        for idx, p in enumerate(self.processed_paths):
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]

            sub_graph_folder = os.path.join(os.path.split(p)[0], '%d-hop_neighborgraph' % self.subsample_neighbor_size)
            os.makedirs(sub_graph_folder, exist_ok=True)
            for node_i in range(n_nodes):
                neighbor_graph_path = os.path.join(
                    sub_graph_folder,
                    os.path.split(p)[1].replace('.gpt', '.%d-hop.%d.gpt' % (self.subsample_neighbor_size, node_i)))
                if not os.path.exists(neighbor_graph_path):
                    sub_g = self.get_subgraph(idx, node_i)
                    torch.save(sub_g, neighbor_graph_path)
        return


    def pick_center(self, data):
        """ Random sample center nodes, cell type balanced """
        cell_types = data["x"][:, 0].long()
        
        freq = self.sampling_freq.gather(0, cell_types)
        freq = freq / freq.sum()
        center_node_ind = np.random.choice(np.arange(len(freq)), p=freq.cpu().data.numpy())
        return center_node_ind


    def load_to_cache(self, idx, subgraphs=True):
        data = torch.load(self.processed_paths[idx])
        self.cached_data[idx] = data
        
        if subgraphs or self.subgraph_source == 'chunk_save':
            neighbor_graph_path = self.processed_paths[idx].replace('.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
            neighbor_graphs = torch.load(neighbor_graph_path)
            for j, ng in enumerate(neighbor_graphs):
                self.cached_data[(idx, j)] = ng
        
    
    def clear_cache(self):
        del self.cached_data
        self.cached_data = {}
        return


    def get_full(self, idx):
        """ Read entire sample """
        if idx in self.cached_data:
            return self.cached_data[idx]
        else:
            data = torch.load(self.processed_paths[idx])
            self.cached_data[idx] = data
            return data


    def get(self, idx):
        """ Read an n-hop subgraph from sample """
        data = self.get_full(idx)
        if self.subsample_neighbor_size == 0:
            return data
        else:
            center_ind = self.pick_center(data)
            if (idx, center_ind) in self.cached_data:
                return self.cached_data[(idx, center_ind)]
            
            if self.subgraph_source == 'on-the-fly':
                return self.get_subgraph(idx, center_ind)
            elif self.subgraph_source == 'save':
                return self.get_saved_subgraph(idx, center_ind)
            elif self.subgraph_source == 'chunk_save':
                return self.get_saved_subgraph_from_chunk(idx, center_ind)


    def get_saved_subgraph_from_chunk(self, idx, center_ind):
        """ Read subgraph from chunk file, use after calling `save_all_subgraphs_to_chunk` """
        full_graph_path = self.processed_paths[idx]
        neighbor_graph_path = full_graph_path.replace('.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
        if not os.path.exists(neighbor_graph_path):
            print("Subgraph save %s not found" % neighbor_graph_path)
            return self.get_subgraph(idx, center_ind)
        
        neighbor_graphs = torch.load(neighbor_graph_path)
        for j, ng in enumerate(neighbor_graphs):
            self.cached_data[(idx, j)] = ng
        return self.cached_data[(idx, center_ind)]


    def get_saved_subgraph(self, idx, center_ind):
        """ (deprecated) Read subgraph from individual file, use after calling `save_all_subgraphs` """
        full_graph_path = self.processed_paths[idx]
        neighbor_graph_path = os.path.join(
            os.path.split(full_graph_path)[0],
            '%d-hop_neighborgraph' % self.subsample_neighbor_size,
            os.path.split(full_graph_path)[1].replace('.gpt', '.%d-hop.%d.gpt' % (self.subsample_neighbor_size, center_ind)))
        if not os.path.exists(neighbor_graph_path):
            print("Subgraph save %s not found" % neighbor_graph_path)
            return self.get_subgraph(idx, center_ind)
        
        neighbor_graph = torch.load(neighbor_graph_path)
        self.cached_data[(idx, center_ind)] = neighbor_graph
        return neighbor_graph


    def get_subgraph(self, idx, center_ind):
        """ Generate subgraph on the fly """
        data = self.get_full(idx)
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind), 
                                       self.subsample_neighbor_size, 
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]
        
        if self.subgraph_size_limit > 0:
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(self.node_feature_names) if n.startswith('center_coord')]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_size_limit)]
        
        sub_x = data.x[sub_node_inds]
        sub_edge_index, sub_edge_attr = subgraph(sub_node_inds, 
                                                 data.edge_index, 
                                                 edge_attr=data.edge_attr, 
                                                 relabel_nodes=True)
        
        relabeled_node_ind = list(sub_node_inds.numpy()).index(center_ind)
        
        sub_data = {'center_node_index': relabeled_node_ind,
                    'original_center_node': center_ind,
                    'x': sub_x,
                    'edge_index': sub_edge_index,
                    'edge_attr': sub_edge_attr}
        for k in data:
            if not k[0] in sub_data:
                sub_data[k[0]] = k[1]
                
        sub_data = tg.data.Data.from_dict(sub_data)
        self.cached_data[(idx, center_ind)] = sub_data
        return sub_data


    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        for transform_fn in self.transform:
            data = transform_fn(data)
        return data


    def plot_subgraph(self, idx, center_ind, n=None):
        """ Plot neighborhood around node `center_ind` as voronoi """
        
        n = self.subsample_neighbor_size if n is None else n
        
        data = self.get_full(idx)
        nx_graph = nx.read_gpickle(self.raw_paths[idx])
        assert self.cell_types[nx_graph.nodes[center_ind]['cell_type']] == \
            data.x[center_ind, 0].item()
    
        # Same procedure as get_subgraph
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind), 
                                       n, 
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]
        
        if self.subgraph_size_limit > 0:
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(self.node_feature_names) if n.startswith('center_coord')]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_size_limit)]
            
        sub_node_inds = sub_node_inds.data.numpy().astype(int)
        G = nx_graph.subgraph(sub_node_inds)
        x_c, y_c = G.nodes[center_ind]['center_coord']
    
        plot_codex_graph(G, cell_types=self.cell_types)
        xmin, xmax = plt.gca().xaxis.get_data_interval()
        ymin, ymax = plt.gca().yaxis.get_data_interval()
        scale = max(x_c - xmin, xmax - x_c, y_c - ymin, ymax - y_c) * 1.05
        plt.xlim(x_c - scale, x_c + scale)
        plt.ylim(y_c - scale, y_c + scale)
        plt.plot([x_c], [y_c], 'x', markersize=5, color='k')



class InfDataLoader(DataLoader):
    def __len__(self):
        return int(1e10)


class CODEXSubgraphSampler(object):
    def __init__(self,
                 dataset,
                 selected_inds=None,
                 batch_size=64,
                 num_graphs_per_segment=32,
                 steps_per_segment=1000,
                 num_workers=None,
                 seed=None,
                 **kwargs):
        self.dataset = dataset
        self.selected_inds = list(dataset.indices()) if selected_inds is None else list(selected_inds)
        self.dataset.set_indices(self.selected_inds)
        
        self.batch_size = batch_size
        self.num_graphs_per_segment = num_graphs_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        
        self.graph_inds_q = []
        self.fill_queue(seed=seed)
        
        self.step_counter = 0
        self.data_iter = None
        print("Initiate data loader, subgraph source: %s" % self.dataset.subgraph_source)
        self.get_new_segment()
        
    
    def fill_queue(self, seed=None):
        if not seed is None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.graph_inds_q))
        np.random.shuffle(fill_inds)
        self.graph_inds_q.extend(fill_inds)
    
    
    def get_new_segment(self):
        if self.num_graphs_per_segment <= 0:
            self.dataset.set_indices(self.selected_inds)
        else:
            graph_inds_in_segment = self.graph_inds_q[:self.num_graphs_per_segment]
            self.graph_inds_q = self.graph_inds_q[self.num_graphs_per_segment:]
            if len(self.graph_inds_q) < self.num_graphs_per_segment:
                self.fill_queue()

            self.dataset.clear_cache()
            self.dataset.set_indices(graph_inds_in_segment)
            for ind in graph_inds_in_segment:
                self.dataset.load_to_cache(ind, subgraphs=True)
        
        sampler = RandomSampler(self.dataset, replacement=True, num_samples=int(1e10))
        loader = InfDataLoader(self.dataset, 
                               batch_size=self.batch_size, 
                               sampler=sampler, 
                               num_workers=self.num_workers)
        self.data_iter = iter(loader)
        self.step_counter = 0
    
    
    def __iter__(self):
        return self

    
    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            self.get_new_segment()
        if not len(set(self.dataset.indices()) - set(self.selected_inds)) == 0:
            self.get_new_segment()
        batch = next(self.data_iter)
        self.step_counter += 1
        return batch

class CODEXSubgraphSampler_IP(object):
    def __init__(self,
                 dataset,
                 selected_inds=None,
                 batch_size=64,
                 num_graphs_per_segment=32,
                 steps_per_segment=1000,
                 num_workers=None,
                 seed=None,
                 **kwargs):
        self.dataset = dataset
        
        # modify the transforms (s.t. only the label is grabbed 
        # but the data is not automatically masked)
        # print(self.dataset.transform)
        # print(len(self.dataset.transform))
        self.masking_transform = self.dataset.transform[0]
        self.dataset.transform = [self.dataset.transform[1]]
        # print(self.dataset.transform)

        self.selected_inds = list(dataset.indices()) if selected_inds is None else list(selected_inds)
        self.dataset.set_indices(self.selected_inds)
        
        self.batch_size = batch_size
        self.num_graphs_per_segment = num_graphs_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        
        self.graph_inds_q = []
        self.fill_queue(seed=seed)
        
        self.step_counter = 0
        self.data_iter = None
        print("Initiate data loader, subgraph source: %s" % self.dataset.subgraph_source)
        
        self.get_new_segment()

    def fill_queue(self, seed=None):
        if not seed is None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.graph_inds_q))
        np.random.shuffle(fill_inds)
        self.graph_inds_q.extend(fill_inds)
    
    
    def get_new_segment(self):
        if self.num_graphs_per_segment <= 0:
            self.dataset.set_indices(self.selected_inds)
        else:
            graph_inds_in_segment = self.graph_inds_q[:self.num_graphs_per_segment]
            self.graph_inds_q = self.graph_inds_q[self.num_graphs_per_segment:]
            if len(self.graph_inds_q) < self.num_graphs_per_segment:
                self.fill_queue()

            self.dataset.clear_cache()
            self.dataset.set_indices(graph_inds_in_segment)
            for ind in graph_inds_in_segment:
                self.dataset.load_to_cache(ind, subgraphs=True)
        print(self.dataset.transform)
        sampler = RandomSampler(self.dataset, replacement=True, num_samples=int(1e10))
        loader = InfDataLoader(self.dataset, 
                               batch_size=self.batch_size, 
                               sampler=sampler, 
                               num_workers=self.num_workers)
        self.data_iter = iter(loader)
        self.step_counter = 0
    
    
    def __iter__(self):
        return self

    
    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            self.get_new_segment()
        if not len(set(self.dataset.indices()) - set(self.selected_inds)) == 0:
            self.get_new_segment()
        batch = next(self.data_iter)
        batch_coordinates = [batch[i].x[:,-2:].numpy() for i in range(len(batch.to_data_list()))]
        batch_ct = [batch[i].x[:,0].numpy() for i in range(len(batch.to_data_list()))] # celltypes
        # apply masking transformation
        batch = self.masking_transform(batch)
        self.step_counter += 1
        return batch, batch_coordinates, batch_ct
  
class CODEXSubgraphWeightedSampler_K(object):
    def __init__(self,
                 dataset,
                 weights_dict,
                 label_path,
                 selected_inds=None,
                 batch_size=64,
                 num_graphs_per_segment=32,
                 steps_per_segment=1000,
                 num_workers=None,
                 seed=None,
                 **kwargs):
        self.dataset = dataset
        # get the weights
        weights = list()
        label_df = pd.read_csv(label_path)
        for i in range(dataset.N):
            label_row = label_df[label_df["aq_id"] == dataset[i].aq_id]
    
            t = label_row['tumor_type'].item()
            weights.append(weights_dict[t])
        self.weights = weights
        self.selected_inds = list(dataset.indices()) if selected_inds is None else list(selected_inds)
        self.dataset.set_indices(self.selected_inds)
        
        self.batch_size = batch_size
        self.num_graphs_per_segment = num_graphs_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        
        self.graph_inds_q = []
        self.fill_queue(seed=seed)
        
        self.step_counter = 0
        self.data_iter = None
        print("Initiate data loader, subgraph source: %s" % self.dataset.subgraph_source)
        self.get_new_segment()
        
    
    def fill_queue(self, seed=None):
        if not seed is None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.graph_inds_q))
        np.random.shuffle(fill_inds)
        self.graph_inds_q.extend(fill_inds)
    
    
    def get_new_segment(self):
        if self.num_graphs_per_segment <= 0:
            self.dataset.set_indices(self.selected_inds)
        else:
            graph_inds_in_segment = self.graph_inds_q[:self.num_graphs_per_segment]
            self.graph_inds_q = self.graph_inds_q[self.num_graphs_per_segment:]
            if len(self.graph_inds_q) < self.num_graphs_per_segment:
                self.fill_queue()

            self.dataset.clear_cache()
            self.dataset.set_indices(graph_inds_in_segment)
            for ind in graph_inds_in_segment:
                self.dataset.load_to_cache(ind, subgraphs=True)
        
        # sampler = RandomSampler(self.dataset, replacement=True, num_samples=int(1e10))
        sampler = WeightedRandomSampler(weights=self.weights,replacement=True, num_samples=(self.dataset.N)*5000)
        loader = InfDataLoader(self.dataset, 
                               batch_size=self.batch_size, 
                               sampler=sampler, 
                               num_workers=self.num_workers)
        self.data_iter = iter(loader)
        self.step_counter = 0
    
    
    def __iter__(self):
        return self

    
    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            self.get_new_segment()
        if not len(set(self.dataset.indices()) - set(self.selected_inds)) == 0:
            self.get_new_segment()
        batch = next(self.data_iter)
        self.step_counter += 1
        return batch


class CODEXSubgraphWeightedSampler_K_old(object):
    def __init__(self,
                 dataset,
                 weights_dict,
                 label_path,
                 selected_inds=None,
                 batch_size=64,
                 num_graphs_per_segment=32,
                 steps_per_segment=1000,
                 num_workers=None,
                 seed=None,
                 **kwargs):
        self.dataset = dataset
        # get the weights
        weights = list()
        label_df = pd.read_csv(label_path)
        for i in range(dataset.N):
            label_row = label_df[label_df["aq_id"] == dataset[i].aq_id]
    
            t = label_row['tumor_type'].item()
            weights.append(weights_dict[t])
        self.weights = weights
        # modify the transforms (s.t. only the label is grabbed 
        # but the data is not automatically masked)
        # print(self.dataset.transform)
        # print(len(self.dataset.transform))
        self.masking_transform = self.dataset.transform[0]
        self.dataset.transform = [self.dataset.transform[1]]
        # print(self.dataset.transform)

        self.selected_inds = list(dataset.indices()) if selected_inds is None else list(selected_inds)
        self.dataset.set_indices(self.selected_inds)
        
        self.batch_size = batch_size
        self.num_graphs_per_segment = num_graphs_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        
        self.graph_inds_q = []
        self.fill_queue(seed=seed)
        
        self.step_counter = 0
        self.data_iter = None
        print("Initiate data loader, subgraph source: %s" % self.dataset.subgraph_source)
        
        self.get_new_segment()

        
        

    def fill_queue(self, seed=None):
        if not seed is None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.graph_inds_q))
        np.random.shuffle(fill_inds)
        self.graph_inds_q.extend(fill_inds)
    
    
    def get_new_segment(self):
        if self.num_graphs_per_segment <= 0:
            self.dataset.set_indices(self.selected_inds)
        else:
            graph_inds_in_segment = self.graph_inds_q[:self.num_graphs_per_segment]
            self.graph_inds_q = self.graph_inds_q[self.num_graphs_per_segment:]
            if len(self.graph_inds_q) < self.num_graphs_per_segment:
                self.fill_queue()

            self.dataset.clear_cache()
            self.dataset.set_indices(graph_inds_in_segment)
            for ind in graph_inds_in_segment:
                self.dataset.load_to_cache(ind, subgraphs=True)
        print(self.dataset.transform)
        # sampler = RandomSampler(self.dataset, replacement=True, num_samples=int(1e10))
        sampler = WeightedRandomSampler(weights=self.weights,replacement=True, num_samples=(self.dataset.N)*10)
        loader = InfDataLoader(self.dataset, 
                               batch_size=self.batch_size, 
                               sampler=sampler, 
                               num_workers=self.num_workers)
        self.data_iter = iter(loader)
        self.step_counter = 0
    
    
    def __iter__(self):
        return self

    
    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            self.get_new_segment()
        if not len(set(self.dataset.indices()) - set(self.selected_inds)) == 0:
            self.get_new_segment()
        batch = next(self.data_iter)
        batch_coordinates = [batch[i].x[:,-2:].numpy() for i in range(len(batch.to_data_list()))]
        batch_ct = [batch[i].x[:,0].numpy() for i in range(len(batch.to_data_list()))] # celltypes
        # apply masking transformation
        batch = self.masking_transform(batch)
        self.step_counter += 1
        return batch, batch_coordinates, batch_ct
  


class CODEXGraphDataset_IP(Dataset):
    
    """ Main dataset structure for model training / inference """
    def __init__(self,
                 root,
                 ROI_ids = None,
                 transform=[],
                 pre_transform=None,
                 raw_folder_name='graph',
                 processed_folder_name='tg_graph',
                 subsample_neighbor_size=0,
                 node_features=["cell_type", "size", "expression", "neighborhood_composition", "center_coord"],
                 edge_features=["edge_type", "distance"],
                 cell_types=CELL_TYPES_VERSION8_2,
                 cell_type_freq=CELL_TYPE_FREQ_VERSION8_2,
                 # expression_markers=EXPRESSION_MARKERS,
                 cell_features = EXPRESSION_MARKERS,
                 cell_morphology_features = [],
                 expression_features = [],
                 subgraph_source='on-the-fly', # 'save', 'chunk_save', 'on-the-fly'
                 subgraph_allow_distant_edge=True,
                 subgraph_size_limit=0,
                 sampling_avoid_unassigned=True,
                 equal_sampling_celltypes = True,
                 **kwargs):
        self.root = root
        self.ROI_ids = ROI_ids
        self.raw_folder_name = raw_folder_name
        self.processed_folder_name = processed_folder_name
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # If to subsample local graphs, 0 = no subsampling
        self.subsample_neighbor_size = subsample_neighbor_size
        
        # Feature names
        self.node_features = node_features
        self.edge_features = edge_features
        self.cell_types = cell_types
        self.cell_type_freq = cell_type_freq
        # self.expression_markers = expression_markers
        self.cell_features = cell_features
        self.cell_morphology_features = cell_morphology_features
        self.expression_features = expression_features
        # print(self.cell_morphology_features)
        # print(self.expression_features)
        self.node_feature_names = get_feature_names(node_features, cell_types=self.cell_types, cell_features=self.cell_features, cell_morphology_features=self.cell_morphology_features, expression_features=self.expression_features)
        self.edge_feature_names = get_feature_names(edge_features, cell_types=self.cell_types, cell_features=self.cell_features, cell_morphology_features=self.cell_morphology_features, expression_features=self.expression_features)
        
        self.process_kwargs = kwargs
        self.process_kwargs['cell_types'] = self.cell_types
        # self.process_kwargs['expression_markers'] = self.expression_markers
        self.process_kwargs['cell_features'] = self.cell_features
        self.process_kwargs['cell_morphology_features'] = self.cell_morphology_features
        self.process_kwargs['expression_features'] = self.expression_features
        
        super(CODEXGraphDataset_IP, self).__init__(root, None, pre_transform)
        self.transform = transform
        self.subgraph_source = subgraph_source
        self.subgraph_allow_distant_edge = subgraph_allow_distant_edge
        self.subgraph_size_limit = subgraph_size_limit
        
        self.N = len(self.processed_paths)
        self.equal_sampling_celltypes = equal_sampling_celltypes
        print(self.equal_sampling_celltypes)
        self.sampling_freq = {self.cell_types[ct]: 1./self.cell_type_freq[ct] for ct in self.cell_types}
        print('sampling frequencies:', self.sampling_freq)
        if self.equal_sampling_celltypes:
            print('apply equal sampling')
            self.sampling_freq = torch.from_numpy(np.array([self.sampling_freq[i] for i in range(len(self.sampling_freq))]))
        else:
            print('no celltype sampling')
            self.sampling_freq = torch.from_numpy(np.array([1 for i in range(len(self.sampling_freq))]))
        print(self.sampling_freq)
        # Avoid sampling unassigned cells
        if sampling_avoid_unassigned:
            print('not sampling unassigned')
            self.sampling_freq[self.cell_types['Unassigned']] = 0.
            print(self.sampling_freq)
        self.cached_data = {}
        
    def set_indices(self, inds):
        """ Limit sampling to `inds` """
        self._indices = inds
        return
    
    def set_subgraph_source(self, subgraph_source):
        assert subgraph_source in ['save', 'chunk_save', 'on-the-fly']
        self.subgraph_source = subgraph_source
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.gpkl')])

    @property
    def processed_file_names(self):
        # Only files for full graphs
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.gpt') and not 'hop' in f])

    def len(self):
        return len(self.processed_paths)

    def process(self):

        def process_graph(raw_path):
            # for raw_path in self.raw_paths:
            aq_id = os.path.splitext(os.path.split(raw_path)[-1])[0]
            if not os.path.exists(os.path.join(self.processed_dir, '%s.0.gpt' % aq_id)):
                G = nx.read_gpickle(raw_path)
                data_list = nx_to_tg_graph(G,
                                            node_features=self.node_features,
                                            edge_features=self.edge_features,
                                            **self.process_kwargs)
                for i, d in enumerate(data_list):
                    d["aq_id"] = aq_id
                    d["subgraph_id"] = i
                    if not self.pre_transform is None:
                        for transform_fn in self.pre_transform:
                            d = transform_fn(d)
                    torch.save(d, os.path.join(self.processed_dir,
                                                '%s.%d.gpt' % (d.aq_id, d.subgraph_id)))

        # print(self.raw_paths)
        if self.ROI_ids is None:
            worklist = self.raw_paths
        else: 
            worklist = [os.path.join(self.raw_dir, filename +'.gpkl') for filename in self.ROI_ids]
        
        print('len worklist:', len(worklist))
        print(worklist)
        for raw_path in worklist:
            # print(raw_path)
            process_graph(raw_path)
        
        # Parallel(n_jobs=cpu_count()-2)(delayed(process_graph)(raw_path) for raw_path in worklist)
            
        return
    
    def save_all_subgraphs_to_chunk(self):

        def save_one(idx,p):
            """ save individual n-hop subgraphs to file (one file per sample) """
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]
            neighbor_graph_path = p.replace('.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
            if not os.path.exists(neighbor_graph_path):
                subgraphs = []
                for node_i in range(n_nodes):
                    subgraphs.append(self.get_subgraph(idx, node_i))
                torch.save(subgraphs, neighbor_graph_path)

        Parallel(n_jobs=cpu_count()-2)(delayed(save_one)(idx,p) for idx,p in enumerate(self.processed_paths))
        
        # for idx, p in enumerate(self.processed_paths):    
        return


    def save_all_subgraphs(self):
        """ (deprecated) save individual n-hop subgraph to file (one file per subgraph)  """
        for idx, p in enumerate(self.processed_paths):
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]

            sub_graph_folder = os.path.join(os.path.split(p)[0], '%d-hop_neighborgraph' % self.subsample_neighbor_size)
            os.makedirs(sub_graph_folder, exist_ok=True)
            for node_i in range(n_nodes):
                neighbor_graph_path = os.path.join(
                    sub_graph_folder,
                    os.path.split(p)[1].replace('.gpt', '.%d-hop.%d.gpt' % (self.subsample_neighbor_size, node_i)))
                if not os.path.exists(neighbor_graph_path):
                    sub_g = self.get_subgraph(idx, node_i)
                    torch.save(sub_g, neighbor_graph_path)
        return


    def pick_center(self, data):
        """ Random sample center nodes, cell type balanced """
        cell_types = data["x"][:, 0].long()
        
        freq = self.sampling_freq.gather(0, cell_types)
        freq = freq / freq.sum()
        center_node_ind = np.random.choice(np.arange(len(freq)), p=freq.cpu().data.numpy())
        return center_node_ind


    def load_to_cache(self, idx, subgraphs=True):
        data = torch.load(self.processed_paths[idx])
        self.cached_data[idx] = data
        
        if subgraphs or self.subgraph_source == 'chunk_save':
            neighbor_graph_path = self.processed_paths[idx].replace('.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
            neighbor_graphs = torch.load(neighbor_graph_path)
            for j, ng in enumerate(neighbor_graphs):
                self.cached_data[(idx, j)] = ng
        
    
    def clear_cache(self):
        del self.cached_data
        self.cached_data = {}
        return


    def get_full(self, idx):
        """ Read entire sample """
        if idx in self.cached_data:
            return self.cached_data[idx]
        else:
            data = torch.load(self.processed_paths[idx])
            self.cached_data[idx] = data
            return data


    def get(self, idx):
        """ Read an n-hop subgraph from sample """
        data = self.get_full(idx)
        if self.subsample_neighbor_size == 0:
            return data
        else:
            center_ind = self.pick_center(data)
            if (idx, center_ind) in self.cached_data:
                return self.cached_data[(idx, center_ind)]
            
            if self.subgraph_source == 'on-the-fly':
                return self.get_subgraph(idx, center_ind)
            elif self.subgraph_source == 'save':
                return self.get_saved_subgraph(idx, center_ind)
            elif self.subgraph_source == 'chunk_save':
                return self.get_saved_subgraph_from_chunk(idx, center_ind)


    def get_saved_subgraph_from_chunk(self, idx, center_ind):
        """ Read subgraph from chunk file, use after calling `save_all_subgraphs_to_chunk` """
        full_graph_path = self.processed_paths[idx]
        neighbor_graph_path = full_graph_path.replace('.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
        if not os.path.exists(neighbor_graph_path):
            print("Subgraph save %s not found" % neighbor_graph_path)
            return self.get_subgraph(idx, center_ind)
        
        neighbor_graphs = torch.load(neighbor_graph_path)
        for j, ng in enumerate(neighbor_graphs):
            self.cached_data[(idx, j)] = ng
        return self.cached_data[(idx, center_ind)]


    def get_saved_subgraph(self, idx, center_ind):
        """ (deprecated) Read subgraph from individual file, use after calling `save_all_subgraphs` """
        full_graph_path = self.processed_paths[idx]
        neighbor_graph_path = os.path.join(
            os.path.split(full_graph_path)[0],
            '%d-hop_neighborgraph' % self.subsample_neighbor_size,
            os.path.split(full_graph_path)[1].replace('.gpt', '.%d-hop.%d.gpt' % (self.subsample_neighbor_size, center_ind)))
        if not os.path.exists(neighbor_graph_path):
            print("Subgraph save %s not found" % neighbor_graph_path)
            return self.get_subgraph(idx, center_ind)
        
        neighbor_graph = torch.load(neighbor_graph_path)
        self.cached_data[(idx, center_ind)] = neighbor_graph
        return neighbor_graph


    def get_subgraph(self, idx, center_ind):
        """ Generate subgraph on the fly """
        data = self.get_full(idx)
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind), 
                                       self.subsample_neighbor_size, 
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]
        
        if self.subgraph_size_limit > 0:
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(self.node_feature_names) if n.startswith('center_coord')]
            # print(self.node_feature_names)
            assert len(coord_feature_inds) == 2
            # print(data.x.shape)
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_size_limit)]
        
        sub_x = data.x[sub_node_inds]
        sub_edge_index, sub_edge_attr = subgraph(sub_node_inds, 
                                                 data.edge_index, 
                                                 edge_attr=data.edge_attr, 
                                                 relabel_nodes=True)
        
        relabeled_node_ind = list(sub_node_inds.numpy()).index(center_ind)
        
        sub_data = {'center_node_index': relabeled_node_ind,
                    'original_center_node': center_ind,
                    'x': sub_x,
                    'edge_index': sub_edge_index,
                    'edge_attr': sub_edge_attr}
        for k in data:
            if not k[0] in sub_data:
                sub_data[k[0]] = k[1]
                
        sub_data = tg.data.Data.from_dict(sub_data)
        self.cached_data[(idx, center_ind)] = sub_data
        return sub_data


    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        for transform_fn in self.transform:
            data = transform_fn(data)
        return data


    def plot_subgraph(self, idx, center_ind, n=None):
        """ Plot neighborhood around node `center_ind` as voronoi """
        
        n = self.subsample_neighbor_size if n is None else n
        
        data = self.get_full(idx)
        nx_graph = nx.read_gpickle(self.raw_paths[idx])
        assert self.cell_types[nx_graph.nodes[center_ind]['cell_type']] == \
            data.x[center_ind, 0].item()
    
        # Same procedure as get_subgraph
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind), 
                                       n, 
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]
        
        if self.subgraph_size_limit > 0:
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(self.node_feature_names) if n.startswith('center_coord')]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_size_limit)]
            
        sub_node_inds = sub_node_inds.data.numpy().astype(int)
        G = nx_graph.subgraph(sub_node_inds)
        x_c, y_c = G.nodes[center_ind]['center_coord']
    
        plot_codex_graph(G, cell_types=self.cell_types)
        xmin, xmax = plt.gca().xaxis.get_data_interval()
        ymin, ymax = plt.gca().yaxis.get_data_interval()
        scale = max(x_c - xmin, xmax - x_c, y_c - ymin, ymax - y_c) * 1.05
        plt.xlim(x_c - scale, x_c + scale)
        plt.ylim(y_c - scale, y_c + scale)
        plt.plot([x_c], [y_c], 'x', markersize=5, color='k')







if __name__ == '__main__':

    dataset_kwargs = {
        'transform': [],
        'pre_transform': None,
        'raw_folder_name': 'graph',
        'processed_folder_name': 'tg_graph_linear_exp',
        'subsample_neighbor_size': 3,
        'node_features': ["cell_type", "size", "expression", "neighborhood_composition", "center_coord"],
        'edge_features': ["edge_type", "distance"],
        'subgraph_source': 'on-the-fly',
        'subgraph_allow_distant_edge': True,
        'subgraph_size_limit': 3 * 55. + 35.,
    }
    
    
    # %% UPMC - Version 8.2
    process_kwargs = {
        "method": "linear",
        "upper_bound": 18,
        "neighborhood_size": 10,
    }
    dataset_kwargs.update(process_kwargs)
    
    os.makedirs("data/version8.2/tg_graph_linear_exp", exist_ok=True)
    dataset_kwargs['cell_types'] = CELL_TYPES_VERSION8_2
    dataset_kwargs['cell_type_freq'] = CELL_TYPE_FREQ_VERSION8_2
    # dataset_kwargs['expression_markers'] = EXPRESSION_MARKERS
    dataset_kwargs['cell_features'] = EXPRESSION_MARKERS
    dataset_kwargs['sampling_avoid_unassigned'] = True # True for 8.2, False for 8
    dataset = CODEXGraphDataset("data/version8.2", **dataset_kwargs)
    dataset.save_all_subgraphs_to_chunk()

