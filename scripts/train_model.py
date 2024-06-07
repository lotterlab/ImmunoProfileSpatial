'''
train a space-gm model on the Immuno Profile data
'''

# from audioop import bias
import json
import numpy as np
import os
import torch

from pycrumbs import tracked

from utils_spatial import get_loss_fct
from additional_transforms import CODEXNodeTransform
from data_IP import CODEXGraphDataset_IP

import sys
sys.path.append(os.path.join(os.getcwd(),'../space-gm'))
print(sys.path)
from transform import CODEXAddGraphLabel
from models import GNN_pred 
from train import evaluate_subgraph_whole, save_model_weight, train_subgraph


def roi_ids_to_idx(sel_roi_ids: list, # selected roi_ids 
                   dev_rois_sorted: list): # all roi_ids in the development dataset - sorted
                    #excl_ROI_idx = [] # indices of ROIs to exclude, e.g. no label available
                    #     ):
    # returns a list of the indices of the specificed ROIs 
    
    idx = list() # list to store the indices
    # find the indices of the ROIs in overlap
    for roi in sel_roi_ids:
        try:
            ind = dev_rois_sorted.index(roi)
            idx.append(ind)
        except:
            pass
    
    return idx


# @tracked(directory_parameter='record_directory') # uncomment to track function call using pycrumbs
def train(root, 
          dataset_name, 
          exp_name,
          target_label_name,
          # dataset_kwargs_path,
          # train_kwargs_path,
          device,
          record_directory,
          cv_split=None,
          roi_per_case = 'all_ROI_lists',
          pretrain_weight=None,
          fc_size = None,
          ):
    
    # import the paths 
    json_path = os.path.join(root, 'configs', 'paths.json')
    paths = json.load(open(json_path))

    print(paths) 

    # import cell and marker settings 
    f = open(paths['marker_path'])
    markers = json.load(f)

    # print(markers)
    CELL_TYPES = markers['CELL_TYPES']
    CELL_TYPE_FREQ = markers['CELL_TYPE_FREQ']
    EXPRESSION_MARKERS = [e.upper() for e in markers['EXPRESSION_MARKERS']]
    CELL_MORPHOLOGY_FEATURES = [e.upper() for e in markers['CELL_MORPHOLOGY_FEATURES']]
    EXPRESSION_FEATURES = [e.upper() for e in markers['EXPRESSION_FEATURES']]
    
    

    '''
    LOAD TRAINING/VALIDATION DATASETS
    '''
    
    # load in dataset_kwargs and process_kwargs
    dataset_kwargs_json = json.load(open(paths['dataset_kwargs_path']))
    dataset_kwargs = dataset_kwargs_json['dataset_kwargs']
    process_kwargs = dataset_kwargs_json['process_kwargs']
    
    # add additional information that is stored in other files
    dataset_kwargs['pre_transform'] = None
    dataset_kwargs['raw_folder_name'] = paths['nx_graph_root'] 
    dataset_kwargs['cell_types'] = CELL_TYPES
    dataset_kwargs['cell_type_freq'] = CELL_TYPE_FREQ   
    dataset_kwargs['cell_features'] = EXPRESSION_MARKERS 
    dataset_kwargs['expression_features'] = EXPRESSION_FEATURES
    dataset_kwargs['cell_morphology_features'] = CELL_MORPHOLOGY_FEATURES
    dataset_kwargs['sampling_avoid_unassigned'] = bool(dataset_kwargs['sampling_avoid_unassigned'])
    
    dataset_kwargs.update(process_kwargs)
    

    try: 
        use_selected_morphology_features = dataset_kwargs['use_selected_morphology_features']
    except:
        use_selected_morphology_features = None

    transform = [
        CODEXNodeTransform(  # This transformation defines feature mask: only features given in `use_XXX_node_features` will be used for model training/evaluation
            node_features=dataset_kwargs['node_features'], 
            edge_features=dataset_kwargs['edge_features'],
            cell_types=CELL_TYPES,
            cell_features =dataset_kwargs['cell_features'],
            cell_morphology_features = dataset_kwargs['cell_morphology_features'],
            expression_features = dataset_kwargs['expression_features'],
            use_neighbor_node_features=dataset_kwargs['use_neighbor_node_features'],  
            use_center_node_features=dataset_kwargs['use_center_node_features'],
            use_selected_morphology_features = use_selected_morphology_features), 
        CODEXAddGraphLabel(graph_labels=[target_label_name]),  # Add phenotype label (primary outcome) to the subgraph
    ]
    print(dataset_kwargs['class_weight'])
    print({index: value for index, value in enumerate(dataset_kwargs['class_weight'])})
    
    f = open(paths["dataset_splits_dict"])
    splits = json.load(f)

    if list(splits.keys()) == ['full_split_dicts', 'all_ROI_lists', 'single_ROI_lists']:
        cross_validation = False
    else:
        cross_validation = True

    print('cross validation:', cross_validation)

    if cross_validation:
        # assert that a split number is gives as training parameter 
        assert cv_split is not None, "cross validation split undefined"
        # grab all roi_ids in the development dataset
        # dev_rois = os.listdir(os.path.join(paths['dataset_root'], exp_name,'development'))
        dev_rois = os.listdir(os.path.join(paths['experiment_root'],'development')) # updated version
        dev_rois = np.unique([a.split('.')[0] for a in dev_rois if a[-3:]=='gpt'])
        dev_rois = sorted(dev_rois)
    indices = dict()
        
    # load training and validation datasets
    dataset_dict = dict()

    # for k in ['training', 'validation']: # commented out for simplified example set up
    for k in ['training']:
        if cross_validation:
            sel_roi_ids = splits[str(cv_split)][roi_per_case][k]
            # select ROIs for which a graph is available
            roi_idx = roi_ids_to_idx(sel_roi_ids, dev_rois)
            print(k, len(roi_idx))
            indices[k] = roi_idx
            
            dataset_kwargs['processed_folder_name'] = os.path.join(exp_name,'development')  

        else: 
            dataset_kwargs['processed_folder_name'] = os.path.join(exp_name,k)

        temp_dataset = CODEXGraphDataset_IP(paths["dataset_root"], 
                                           transform=transform, 
                                           **dataset_kwargs) # K: generates a dataset with subgraphs that reads in the data on the fly
        temp_dataset.save_all_subgraphs_to_chunk()
        dataset_dict[k] = temp_dataset
        
        if not cross_validation:
            indices[k] = np.arange(len(temp_dataset))
    
    failed = dict()
    for  k, v in dataset_dict.items():
        f = list()
        n_graphs = len(v) # grab the number of items in the dataset 
        for i in range(n_graphs):
            try:
                a = v[i]
            except:
                f.append(i)
        failed[k] = f

    for key, f in failed.items():
        print('n failed indices',key, len(f))
    
    
    
    '''
    TRAINING
    '''
    
    # set training parameters
    ### Model and Training parameters ###
    # load in dataset_kwargs and process_kwargs
    train_kwargs_json = json.load(open(paths['train_kwargs_path']))
    model_kwargs = train_kwargs_json['model_kwargs']
    train_kwargs = train_kwargs_json['train_kwargs']
    
    model_kwargs['num_feat'] = dataset_dict['training'][0].x.shape[1] - 1  # exclude the cell type column -> K: don't understanding this ... yet
    model_kwargs['num_node_type'] = 6+1 # len(CELL_TYPES) + 1
        
    train_kwargs['graph_task_criterion'] = get_loss_fct(train_kwargs['graph_task_criterion'])
    train_kwargs['node_task_criterion'] = None
    train_kwargs['evaluate_fn'] = [evaluate_subgraph_whole, save_model_weight] # K: specify the functions to evaluate the model
    train_kwargs['evaluate_on_train'] = bool(train_kwargs['evaluate_on_train'])
    
    os.makedirs(paths['model_save_root'], exist_ok=True)

    save_paths = {
        'score_file':  os.path.join(paths['model_save_root'], paths['training_log_name']), # original: .txt file
        'model_folder': os.path.join(paths['model_save_root'], paths['checkpoint_name']),
    }
    train_kwargs.update(save_paths)

    # initialize GNN
    torch.manual_seed(seed)
    model = GNN_pred(**model_kwargs)

    if not fc_size is None:
        # generate new FC layers
        modules = list()
        for i in range(len(fc_size)-1):
           modules.append(torch.nn.Linear(fc_size[i], fc_size[i+1]))
           modules.append(torch.nn.LeakyReLU())
        modules.append(torch.nn.Linear(fc_size[-1],1))

        fc_sequ = torch.nn.Sequential(*modules)
        model.graph_pred_module = fc_sequ
        
    print(model)
    
    if not pretrain_weight is None:
        print('load pretrained model', pretrain_weight)
        model.load_state_dict(torch.load(pretrain_weight, map_location=device))

    # train
    print('training model on split', str(cv_split))
    model = train_subgraph(model, 
                           dataset_dict['training'],
                           device,
                           train_inds=np.arange(10), # select the samples used for training (by index)
                           valid_inds= np.arange(10), # select the samples for validation (by index)
                           **train_kwargs)
     
  
if __name__ == "__main__": 

    root = '..'
    dataset_name =  None
    exp_name = 'test'
    target_label_name = 'is_ned' # using a pseudo binary label for demonstration purposes, use 'survival_cox' for cox survival labels (status and time)
    device = 'cuda:0'
    seed = 0

    record_directory = ... # directory for the pycrumbs record
    pretrain_weight = None
    fc_size = [512,128,32] # specify the size of the fully connected layers, set to None for space-gm defaults
    roi_per_case = 'all_ROI_lists' # 'single_ROI_lists' # for each case use either all available ROIs or a randomly selected single ROI 
         
    train(root,
        dataset_name,
        exp_name,
        target_label_name,
        device, 
        record_directory = None,
        cv_split = None,
        roi_per_case = roi_per_case,
        pretrain_weight=pretrain_weight,
        fc_size = fc_size,
        )