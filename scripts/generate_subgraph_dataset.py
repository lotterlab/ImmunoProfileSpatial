'''
initial preprocessing of the dataset
save full graphs and subgraphs for training/validation/test separately

date created: 05/01/2023
updated: 
- 05/05/2023: split 'expression' into cell morphology and true expression features

'''

import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import torch

from joblib import cpu_count, delayed, Parallel
from pathlib import Path
from pycrumbs import tracked
from copy import deepcopy

import sys
sys.path.append(os.path.join(os.getcwd(),'../space-gm'))
from transform import CODEXAddGraphLabel
from additional_transforms import CODEXNodeTransform, get_feature_names
# from data import CODEXGraphDataset_K

from torch_geometric.data import Dataset
from data_IP import nx_to_tg_graph, CODEXGraphDataset_IP

paths = json.load(open('../configs/paths.json'))

def get_all_graph_labels():
    '''adapted from data.py CODEXGraphDataset to comply with the structure of our data organization'''
    return pd.read_csv(paths['label_path'])


# @tracked(directory_parameter='record_directory') # uncomment to track parameters using pycrumbs
def generate_dataset(root: Path,
                     # dataset: str,
                     exp_name: str,
                     target_label_name: str,
                     roi_per_case: str,
                     record_directory: Path):

    # import the paths 
    json_path = os.path.join(root, 'configs', 'paths.json')
    paths = json.load(open(json_path))

    print(paths) 

    # import cell and marker settings 
    f = open(paths['marker_path'])
    markers = json.load(f)

    
    CELL_TYPES = markers['CELL_TYPES']
    CELL_TYPE_FREQ = markers['CELL_TYPE_FREQ']
    EXPRESSION_MARKERS = [e.upper() for e in markers['EXPRESSION_MARKERS']]
    CELL_MORPHOLOGY_FEATURES = [e.upper() for e in markers['CELL_MORPHOLOGY_FEATURES']]
    EXPRESSION_FEATURES = [e.upper() for e in markers['EXPRESSION_FEATURES']]

    '''
    DATASET GENERATION
    '''
    # IMPORT SPLITS
    f = open(paths["dataset_splits_dict"])
    # f = open(os.path.join('/lotterlab/users/khoebel/mIP/data/NSCLC', exp_name, "experiment_split_no_cv.json"))
    splits = json.load(f)

    if list(splits.keys()) == ['full_split_dicts', 'all_ROI_lists', 'single_ROI_lists']:
        cross_validation = False
        print("no cv")
    else:
        cross_validation = True
        print('cross-validation')

    if cross_validation:
        splits = splits['0']  

    if roi_per_case == 'single':
        roi_dict = splits['single_ROI_lists']
    elif roi_per_case == 'all':
        roi_dict = splits['all_ROI_lists']
    # print(splits.keys())
    
    # GENERATE LIST OF ROIS FOR THIS EXPERIMENT 
    if cross_validation:
        # generate a 'development' dataset (training + validation)
        # and a separate test dataset 

        development_list = deepcopy(roi_dict['training'])
        development_list.extend(roi_dict['validation'])

        print(len(roi_dict['training']), len(roi_dict['validation']), len(development_list))
        roi_dict = {'development': development_list,
                    'test': roi_dict['test']}
    
    for k in roi_dict.keys():
        print(k, len(roi_dict[k]))

    
    # DATASET PARAMETERS
    
    # load in dataset_kwargs and process_kwargs
    dataset_kwargs_json = json.load(open(paths['dataset_kwargs_path']))
    dataset_kwargs = dataset_kwargs_json['dataset_kwargs']
    process_kwargs = dataset_kwargs_json['process_kwargs']
    
    # add additional information that is stored in other files
    dataset_kwargs['pre_transform'] = None
    dataset_kwargs['raw_folder_name'] = paths['nx_graph_root'] 
    dataset_kwargs['cell_types'] = CELL_TYPES
    dataset_kwargs['cell_type_freq'] = CELL_TYPE_FREQ
    # dataset_kwargs['expression_markers'] = EXPRESSION_MARKERS
    dataset_kwargs['cell_features'] = EXPRESSION_MARKERS 
    dataset_kwargs['expression_features'] = EXPRESSION_FEATURES
    dataset_kwargs['cell_morphology_features'] = CELL_MORPHOLOGY_FEATURES
    
    dataset_kwargs.update(process_kwargs)

    try: 
        use_selected_morphology_features = dataset_kwargs['use_selected_morphology_features']
    except:
        use_selected_morphology_features = None

    transform = [
        CODEXNodeTransform(  # This transformation defines feature mask: only features given in `use_XXX_node_features` will be used for model training/evaluation
            node_features=dataset_kwargs['node_features'], 
            edge_features=dataset_kwargs['edge_features'],
            cell_types=dataset_kwargs['cell_types'],
            cell_features = dataset_kwargs['cell_features'],
            expression_features = dataset_kwargs['expression_features'],
            cell_morphology_features =  dataset_kwargs['cell_morphology_features'],
            use_neighbor_node_features=dataset_kwargs['use_neighbor_node_features'],  # Only use cell type and cell size in this experiment -> K: removed size feature
            use_center_node_features=dataset_kwargs['use_center_node_features'],
            use_selected_morphology_features=use_selected_morphology_features), 
        CODEXAddGraphLabel(graph_labels=[target_label_name])# , class_weight={index: value for index, value in enumerate(dataset_kwargs['class_weight'])}),  # Add phenotype label (primary outcome) to the subgraph
    ]

    available_rois = [a.split('.')[0] for a in os.listdir(paths['nx_graph_root'])]
    
    
    # generate datasets
    for k in roi_dict.keys():
        print(k)
        dataset_kwargs['processed_folder_name'] = k
        print(len(roi_dict[k]))
        ROI_ids = list(set(roi_dict[k]).intersection(set(available_rois)))
        print(len(ROI_ids))
        dataset = CODEXGraphDataset_IP(paths["dataset_root"], ROI_ids=ROI_ids, transform=transform, **dataset_kwargs) # K: generates a dataset with subgraphs that reads in the data on the fly
        dataset.save_all_subgraphs_to_chunk()
        


if __name__ == "__main__": 

    root = '..'
    dataset = 'endometrial'
    exp_name = 'cox_survival'
    target_label_name = 'survival_cox'
    roi_per_case = 'all' # 'single', 'all' uses either only one single roi per case ('single') or all rois that are available for a specific case ('all')

    # record_directory = os.path.join(root, 'data', dataset, exp_name)
    generate_dataset(root,
                     # dataset,
                     exp_name,
                     target_label_name,
                     roi_per_case,
                     record_directory=None)
    
