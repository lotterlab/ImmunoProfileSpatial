'''
utility functions for set-up and analysis of mIP project

date created: 2023/05/15
'''
import json
import numpy as np
import os
import pandas as pd
import random
import sys 
import torch
sys.path.append('/lotterlab/users/khoebel/mIP/baseline_repos/space-gm')

from models import BinaryCrossEntropy, CoxSGDLossFn
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d


paths = json.load(open('../configs/paths.json'))

def get_all_graph_labels():
    '''adapted from data.py CODEXGraphDataset to comply with the structure of our data organization'''
    return pd.read_csv(paths['label_path'])

def get_loss_fct(loss_str):
    if loss_str == 'BinaryCrossEntropy':
        return BinaryCrossEntropy()
    
    elif loss_str =='CoxLoss':
        return CoxSGDLossFn()
    
    else: 
        print("loss function not recognized")

def categorise_cell(row, output_type='str'): 
    # categorises cells based on their expression markers
    # return type can be 'str' or 'int' 
    # 1/'tumor/pdl1+': pdl1+ tumor cells
    # 2/'tumor/pdl1-': pdl1- tumor cells
    # 4/'cd8/pd1+'/: pd1 postive cd8 cells
    # 3/'cd8/pd1-': pd1 - cd8 cells 
    # 5/'pdl1 other': any pd-l1+ cell other than tumor cells
    # 6/'pd1 other': any pd-1+ cell other than cd8 cells
    # 7/'foxp3 any': any cell that is foxp3 positive (Treg)
    
    #0/'undef': cells that do not stain for any of the 5 markers 

    if (row['pdl1']==1) and (row['tumor']==1) :
        # pdl1 postive tumor
        temp_type = {'str': 'tumor/pdl1+', 'int': 2}   
    elif (row['pdl1']!=1) and (row['tumor']==1):
        # pdl1 negative tumor
        temp_type = {'str': 'tumor/pdl1-', 'int': 1}
    elif (row['pd1']==1) and (row['cd8']==1):
        # pd1 positive cd8
        temp_type = {'str': 'cd8/pd1+', 'int': 4}
    elif (row['pd1']!=1) and (row['cd8']==1):
        # pd1 negative cd8
        temp_type = {'str': 'cd8/pd1-', 'int': 3}
    elif (row['pd1']==1) and (row['cd8']!=1):
        # any pd1 positive cell (other than cd8+)
        temp_type = {'str': 'pd1_other', 'int': 6} 
    elif (row['foxp3']==1) :
        # any foxp3 positive cell
        temp_type = {'str': 'foxp3_any', 'int': 7}
    elif (row['pdl1']==1) and (row['tumor']!=1):
        # any pdl1 positive cell (other than tumor)
        temp_type = {'str': 'pdl1_other', 'int': 5}
    else:
        temp_type = {'str': 'Unassigned', 'int': 0}
    
    return temp_type[output_type]


def points_from_xy(row, x_name='cell_x', y_name='cell_y'):
    # combines x and y (sep columns) to an array
    
    point = np.array([row[x_name], row[y_name]])
    return point


def roi_indices(test_out_dict, aq_id): 
    # identify all indices of a specified aq_id

    return list(np.where(np.asarray(test_out_dict['aq_id'])==aq_id)[0])


def map_dataloader_to_dict(i, j, test_out_dict, dataloader):
    
    aq_id = dataloader[i].aq_id
    temp_idx = roi_indices(test_out_dict,aq_id)
    
    dict_index = temp_idx[j]
    
    return dict_index


def map_dict_to_dataloader(dict_index, test_out_dict, aq_id_index_dict=None, aq_id_type='str'):
    assert aq_id_type in ['str', 'int'], 'aq_id_type invalid'

    aq_id = test_out_dict['aq_id'][dict_index]
    if aq_id_type == 'str':
        assert aq_id_index_dict is not None, 'aq_id_index_dict needs to be provided to obtain index of aq_id'
        i = aq_id_index_dict[aq_id] # aq_id index 
    elif aq_id_type == 'int':
        i = aq_id

    # identify all the subgraphs that belong to the selected aq_id 
    temp_idx = roi_indices(test_out_dict,aq_id)
    j = temp_idx.index(dict_index) # subgraph index
    
    return i, j 

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def plot_subgraph_data_obj(data_obj, plot_type=None, cmap=None, ax=None):
    x, y = data_obj.x[:, -2], data_obj.x[:, -1]
    celltypes = data_obj.x[:, 0]
    
    if ax is None:
        _, ax = plt.subplots()
        
    if plot_type == 'Delaunay': 
        tri = Delaunay(data_obj.x[:,-2:])
        ax.triplot(x, y, tri.simplices, c='gray')
    
    ax.scatter(x, y,
               c=celltypes , 
               cmap=cmap, vmin=0, vmax=(cmap.N-1))
    plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    # plt.title('pred '+str(np.round(test_out_dict['preds'][k][0],5)))
    return ax


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



'''
subgraph plotting fuctions
'''

def subgraph_scatter(test_out_dict, index, ax, cmap):
    coord_list = test_out_dict['coord_list'][index]
    color_list = [int(a) for a in test_out_dict['ct_list'][index]]
    ax.scatter(coord_list[:,0],
                coord_list[:,1], 
                c=color_list, 
                cmap=cmap, vmin=0, vmax=(cmap.N-1))
    plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    
    ax.axis('off')
    # plt.title('pred '+str(np.round(test_out_dict['preds'][k][0],5)))


def subgraph_Delaunay_plot(test_out_dict, index, ax, cmap):
    coord_list = test_out_dict['coord_list'][index]
    color_list = color_list = [int(a) for a in test_out_dict['ct_list'][index]]
    
    tri = Delaunay(coord_list)
    ax.triplot(coord_list[:,0], coord_list[:,1], tri.simplices, c='gray')
    ax.scatter(coord_list[:,0],
                        coord_list[:,1], 
                        c=color_list, 
                        cmap=cmap, vmin=0, vmax=(cmap.N-1))
    

def plot_colored_voronoi(df, test_out_dict, index, ax, cmap, plot_margin = 150, temp_ct=None):
    aq_id = test_out_dict['aq_id'][index]
    # grab cell coordinates for all cells in the roi (aq_id)
    roi_coords = df.xs(aq_id, level=1, axis=0, drop_level=False)[['cell_x', 'cell_y']]
    
    # add color column and add the cell types for cell in the selected subgraph
    roi_coords['color'] = [8]*roi_coords.shape[0]
    temp_coords = test_out_dict['coord_list'][index]
    if temp_ct is None:
        temp_ct = test_out_dict['ct_list'][index]
    for k in range(len(temp_ct)):
        # find the right entry in roi_coords
        roi_coords.loc[(roi_coords['cell_x']==temp_coords[k][0]) & (roi_coords['cell_y']==temp_coords[k][1]), 'color'] = int(temp_ct[k])
    
    
    # combine the x and y coordinates into points (for the Voronoi generation)
    roi_coords['point'] = roi_coords.apply(lambda row: points_from_xy(row), axis=1)
    
    # generate Voronoi diagram
    vor = Voronoi(np.vstack(roi_coords['point']))
    
    # plot
    colors = roi_coords['color'].to_list()
    # fig = plt.figure()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
    
    # fill with colors
    for vr in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[vr]] # grab the region for point i (need to use the point region mapping) 

        if not -1 in region:
            polygon = [vor.vertices[vr] for vr in region]
            ax.fill(*zip(*polygon), color=cmap(colors[vr]))# colors[cell_type]) # the color mapping uses the id (here index) of the point
    
    # determine optimal plotting area
    max_dist = np.max(np.max(temp_coords, axis=0)-np.min(temp_coords, axis=0))
    d = max_dist/2+plot_margin 
    
    # define plotting area symmetrically around center_node
    center_node_coord = temp_coords[test_out_dict['center_cell_id'][index]]
    x_lim = [center_node_coord[0]-d, center_node_coord[0]+d]
    y_lim = [center_node_coord[1]-d, center_node_coord[1]+d]
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    


