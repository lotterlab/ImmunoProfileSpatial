'''
generic processing for the cell data into a format that is suitable for training a space-gm 
- generation of one csv file for each ROI (in a format that resembels what is required by space-gm)
- generation of Voronoi polygons for each ROI
- converts polygons to an nx graph for each ROI
'''

import json
import multiprocessing
import numpy as np
import networkx as nx
import os
import pandas as pd
import pickle
import random
import tqdm

from pycrumbs import tracked

import sys
sys.path.append(os.path.join(os.getcwd(),'../space-gm'))
from graph_build import calculate_voronoi_from_coords, read_raw_voronoi, read_cell_data, build_graph_from_voronoi, build_polygon_to_cell_mapping, get_edge_type 
from utils import load_sample_phenotypes

from utils_spatial import categorise_cell

'''
Helper functions
'''
    
def convert_region_label(row):
    if row['region_label'] == 'InnerTumor':
        return 1
    elif row['region_label'] == 'InnerMargin':
        return 2
    elif row['region_label'] == 'OuterMargin':
        return 3
    elif row['region_label'] == 'OuterStroma':
        return 4
    else:
        return 0
    

def add_clinical_information(roi_df, clinical_feature_dict):
    # to add clinical features such as gender, tumor subtype as feature to the graph nodes
    n_rows = roi_df.shape[0]
    
    for k, v in clinical_feature_dict.items():
        temp_col = np.ones(n_rows)*v
        roi_df[k] = temp_col
    
    return roi_df

def collect_clinical_features(case_id, 
                              cell_features,
                              clinical_df, 
                              feature_dict):
    
    # only supports NSCLC
    sel_clinical_features = [f  for f in cell_features if f in feature_dict.keys()]
    
    feature_val_dict = dict() # dictionary to store numerical return values 
    for f in sel_clinical_features:
        # grab information
    
        try:
            value = clinical_df.loc[clinical_df['case_id']==int(case_id)][f].values[0]
        except:
            value = np.nan
            
        # convert to numerical value
        try: 
            num_value = feature_dict[f][value]
        except:
            num_value = 0

        # add to return dictionary 
        feature_val_dict[f] = num_value
    
    return feature_val_dict    

def normalize(x, mean, std):
    # zero mean unit variance normalization
    # use median and interquartile range for data with outliers
    
    return (x - mean)/std


def quartile_norm(x):
    # normalization centered around the mean
    # scaled using the interquartile range
    x_no_nans = x[~np.isnan(x)] # remove potential nans
    median = np.median(x_no_nans)
    iq_range = np.percentile(x_no_nans, 75) - np.percentile(x_no_nans,25)
    
    return normalize(x, median, iq_range)


'''
Preprocessing functions
'''
def convert_roi(roi_df, features_to_normalize):

    conversion_sucessful = True

    # select the required columns 
    roi_df = roi_df.reset_index()
    expression_markers = ['pd1', 'pdl1', 'foxp3', 'tumor', 'cd8', 'region_label']
    select_features = ['sample_id', 'roi_id', 'cell_id', 'cell_x', 'cell_y' ]
    select_features.extend(expression_markers)
    for f in features_to_normalize:
        select_features.append(f+'_norm')
    roi_df = roi_df[select_features].copy()
    
    # generate cell type column
    roi_df['CLUSTER_LABEL'] = roi_df.apply(lambda row: categorise_cell(row, output_type='str'), axis=1) # use the CLUSTER_LABEL column name in agreement with the SPACE GM code
    
    # convert the string region_label into an integer
    roi_df['region_label'] = roi_df.apply(lambda row: convert_region_label(row), axis=1) 

    # test if the roi contains any nans (either pd nans or )
    for marker in ['pd1', 'pdl1', 'foxp3', 'tumor', 'cd8']:
        if roi_df[marker].isnull().values.any():
            roi_df[marker].fillna(False,inplace=True)
            roi_df[marker] = roi_df[marker].astype(int)
            
    # rename select columns to agree with space gm requirements 
    column_name_dict = {"cell_id": "CELL_ID",
                        'sample_id': 'STUDY_ID', 
                        'roi_id': 'ACQUISITION_ID', 
                        'cell_x': 'X', 
                        'cell_y': 'Y'}
    roi_df = roi_df.rename(columns=column_name_dict)
    
    return roi_df, conversion_sucessful


# @tracked(directory_parameter='save_root') # uncomment to use pycrumbs for tracking
def parquet_to_ROI_csv(parquet_path:str, 
                       save_root:str,
                       survival_path = None
                       ):
    
    os.makedirs(os.path.join(save_root, 'cell_data'), exist_ok=True)

    # load features for normalization
    markers = json.load(open(os.path.join(save_root, 'markers.json')))
    features_to_normalize = markers['features_to_normalize']
    
    # read in the information for all ROIs
    if parquet_path[-3:] == 'csv':
        df = pd.read_csv(parquet_path, index_col=[0,1,2]) # read in csv with multi-index
    else:
        df = pd.read_parquet(parquet_path)

    # list of all available ROIs
    roi_ids = np.unique(df.index.get_level_values(1).to_numpy())
    print(len(roi_ids))
    
    # preprocess the dataframe 
    # normalize selected features (defined in features_to_normalize)
    for col in features_to_normalize:
        df[(col+'_norm')] = quartile_norm(df[col].to_numpy())

    # read in the survival_dataframe containing the clincal features
    if survival_path is not None:
        survival_df = pd.read_csv(survival_path)   
        # convert pandas boolean columns to string (for easier lookup of boolean clinical features)
        survival_df['post_chemo'] = survival_df['post_chemo'].map({True: 'True', False: 'False'})
        survival_df['post_icb'] = survival_df['post_icb'].map({True: 'True', False: 'False'})
  

    ROI_generation_failed = {"ROI":"case_id"}
    sample_ROI_dict = dict()
    for roi in tqdm.tqdm(roi_ids):
        temp_roi = df.xs(roi, level=1, axis=0, drop_level=False).copy()
        roi_df, conversion_successful = convert_roi(temp_roi, features_to_normalize)
        case_id = f"{roi_df['STUDY_ID'].to_list()[0]}"
        
        if conversion_successful:
            try:
                feature_dict=markers['CLINICAL_FEATURES_DICT']
            except:
                feature_dict = None
            
            if feature_dict is not None:
            # add clinical information to roi_df
                clinical_information = collect_clinical_features(case_id=case_id,
                                                                cell_features=markers['CELL_MORPHOLOGY_FEATURES'],
                                                                clinical_df=survival_df,
                                                                feature_dict=markers['CLINICAL_FEATURES_DICT']
                                                                ) 
                
                roi_df = add_clinical_information(roi_df, clinical_information)
            roi_df.to_csv(os.path.join(save_root, 'cell_data', roi+'.cell_data.csv'))

            if case_id in sample_ROI_dict.keys():
                sample_ROI_dict[case_id].append(f"{roi}")
            else:
                sample_ROI_dict[case_id] = [f"{roi}"]

        else:
            ROI_generation_failed[f"{roi}"] = f"{roi_df['STUDY_ID'].to_list()[0]}"


    # save dicts to json
    failed_json = json.dumps(ROI_generation_failed, indent=4)
    roi_case_json = json.dumps(sample_ROI_dict)
    
    # Writing to sample.json
    with open(os.path.join(save_root,"ROI_gen_failed.json"), "w") as outfile:
        outfile.write(failed_json)

    with open(os.path.join(save_root,"sample_roi_dict.json"), "w") as outfile:
        outfile.write(roi_case_json)

def load_voronoi(voronoi_file,
                 cell_data_file=None,
                 EXPRESSION_MARKERS = [],
                 qc_thr=0.15,
                 radius_relaxation=0.1,
                 voronoi_img_output=None,
                 graph_img_output=None,
                 figsize=10,
                 unique_node_types=None # CELL_TYPES for plotting
                 ):
    
    # adapted from space-gm/graph_build.py load_voronoi 
    aq_id = os.path.split(voronoi_file)[-1].split('.')[0]

    # Load voronoi
    polygons = read_raw_voronoi(voronoi_file)

    # Load cell expression, labels
    cell_data = read_cell_data(aq_id, cell_data_file=cell_data_file)
    del cell_data['UNNAMED: 0']
    
    # Build initial graph
    G = build_graph_from_voronoi(polygons, radius_relaxation=radius_relaxation)

    # Match polygon with cell id
    polygon_to_cell_mapping = build_polygon_to_cell_mapping(G, polygons, cell_data)
    
    # Add coordinates, size, cell type to node feature
    unassigned_ct = 0
    properties = {}
    for k in polygon_to_cell_mapping:
        cell_id, center_coord = polygon_to_cell_mapping[k]
        p = {"cell_id": cell_id, "center_coord": center_coord}
        
        row = cell_data[cell_data["CELL_ID"] == cell_id]
        # p["size"] = row["SIZE"].item() # K: commented out the cell size dict entry
        
        row = cell_data[cell_data["CELL_ID"] == cell_id]
        if len(row) == 1:
            p["cell_type"] = row["CLUSTER_LABEL"].item()
        else:
            p["cell_type"] = "Unassigned"
            unassigned_ct += 1
        properties[k] = p
    if unassigned_ct > 0:
        print("\tCannot find cell type for %d nodes" % unassigned_ct)
        
    nx.set_node_attributes(G, properties)

    # Add distance, edge type (by thresholding) to edge feature
    edge_properties = get_edge_type(G)
    nx.set_edge_attributes(G, edge_properties)
    
    # Add expression data to node features
    expression_missing_ct = 0
    if True: # not expression_file is None:
        expressions = {}
        # expression_data = pd.read_csv(expression_file)
        for node in G.nodes:
            cell_id = polygon_to_cell_mapping[node][0]
            row = cell_data[cell_data['CELL_ID'] == cell_id]
            if row.shape[0] == 0:
                expressions[node] = {"expression": None}
                expression_missing_ct += 1
            else:
                expression_dict = {k: v for k, v in dict(row.iloc[0]).items() if k in [m.upper() for m in EXPRESSION_MARKERS]}
                expressions[node] = {"expression": expression_dict}
        nx.set_node_attributes(G, expressions)
    # Reference plots
    try:
        sample_labels = load_sample_phenotypes(aq_id)
        size = (sample_labels['WIDTH'], sample_labels['HEIGHT'])
        assert not (size[0] == size[0] and size[1] == size[1])
    except Exception as e:
        print(e)
        size = None
    
    '''
    size = [1500,1500] 
    if not voronoi_img_output is None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize*size[1]/size[0]))
        plot_codex_voronoi(polygons, 
                           polygon_to_cell_mapping, 
                           size=size)
        plt.savefig(voronoi_img_output, dpi=300, bbox_inches='tight')
    if not graph_img_output is None:
        # `plot_codex_graph` does not set figure size
        plt.clf()
        plt.figure(figsize=(figsize, figsize*size[1]/size[0]))
        plot_codex_graph(G, 
                         unique_node_types=unique_node_types, 
                         size=size,
                         cell_types=unique_node_types)
        plt.savefig(graph_img_output, dpi=300, bbox_inches='tight')'''
    
    if expression_missing_ct < qc_thr * len(G):
        return G, True
    else:
        # Too many cells are filtered out, density of connection will be affected
        print("QC removed too many cells, remaining %d/%d" % (expression_missing_ct, len(G)))
        return G, False


def generate_voronoi(aq_id, raw_data_root, polygon_root): 
    # use cell data to create voronoi polygons
    # save them to a .pkl file 
    
    # read in cell data
    cell_data = pd.read_csv(os.path.join(raw_data_root, "%s.cell_data.csv" % aq_id))
    cell_data.columns = [col.upper() for col in cell_data.columns] # For backward compatibility
        
    # generate and save Voronoi polygons
    polygons = calculate_voronoi_from_coords(cell_data.X, cell_data.Y)
    pickle.dump(polygons, open(os.path.join(polygon_root, aq_id+'.pkl'), 'wb'))


def generate_cell_graph(aq_id, raw_data_root, polygon_root, nx_graph_root, CELL_TYPES, EXPRESSION_MARKERS, save_img=False, graph_outputs=None):
    # adapted from space-gm/graph_build.py to generate the cell graph based on information in a single csv file 
    
    cell_data_file = os.path.join(raw_data_root, "%s.cell_data.csv" % aq_id)  # cell features
    if save_img:
        graph_img_output = os.path.join(graph_outputs, "%s.graph.jpg" % aq_id) 
        voronoi_img_output = os.path.join(graph_outputs, "%s.voronoi.jpg" % aq_id) 
        print('generated image output paths')
    else: 
        graph_img_output = None
        voronoi_img_output = None
        
    voronoi_file = os.path.join(polygon_root, "%s.pkl" % aq_id)  # Path to voronois
    
    # generate Voronoi graph and save 
    G, flag = load_voronoi(
        voronoi_file,
        cell_data_file=cell_data_file,
        EXPRESSION_MARKERS = EXPRESSION_MARKERS,
        radius_relaxation=0.0,
        qc_thr=0.15,
        unique_node_types=CELL_TYPES, 
        graph_img_output=graph_img_output, 
        voronoi_img_output=voronoi_img_output) 
    
    assert flag  # QC flag
    nx.write_gpickle(G, os.path.join(nx_graph_root, "%s.gpkl" % aq_id))


def csv_to_graph_ROI(inputs): 
    # uses the information stored in the csv file to generate an nx graph
    # with all provided features 

    aq_id, raw_data_root, polygon_root, nx_graph_root, graph_outputs, CELL_TYPES, EXPRESSION_MARKERS = inputs

    # generate and save polygons for all ROIss
    save_img = False
    '''# save Voronois and graph plot for randomly selected ROIs
    if random.uniform(0,1) < 0.5:
            save_img = True'''

    if not os.path.exists(os.path.join(nx_graph_root, "%s.gpkl" % aq_id)):
        try:
            # generate and save polygons
            generate_voronoi(aq_id, raw_data_root, polygon_root)
        
            # generate graph based on voronoi polygons 
            generate_cell_graph(aq_id, raw_data_root, polygon_root, nx_graph_root, CELL_TYPES, EXPRESSION_MARKERS, save_img=save_img, graph_outputs=graph_outputs)

        except:
            # record aq_ids (ROI ids) for cases for which no graph is generated
            return aq_id
    

# @tracked(directory_parameter='data_root') # uncomment to use pycrumbs for tracking
def csv_to_graph_dataset(data_root):
    
    raw_data_root = os.path.join(data_root,'cell_data')
    polygon_root = os.path.join(data_root,'polygons')
    nx_graph_root = os.path.join(data_root,'graph')
    graph_outputs = os.path.join(data_root, 'graph_plots')
    
    os.makedirs(polygon_root, exist_ok=True)
    os.makedirs(nx_graph_root, exist_ok=True)
    os.makedirs(graph_outputs, exist_ok=True)

    # load cell types and markers
    markers = json.load(open(os.path.join(data_root,'markers.json')))
    CELL_TYPES = markers['CELL_TYPES']
    CELL_FEATURES = markers['CELL_FEATURES']
    CELL_FEATURES = [e.upper() for e in CELL_FEATURES]
    print(CELL_FEATURES)

    aq_ids = [a.split('.')[0] for a in os.listdir(raw_data_root)]
    print('number of ROIs to convert', len(aq_ids))

    inputs_list = [(aq_id,
                    raw_data_root, 
                        polygon_root, 
                        nx_graph_root, 
                        graph_outputs, 
                        CELL_TYPES, 
                        CELL_FEATURES) for aq_id in aq_ids]
    with multiprocessing.Pool() as pool:
        failed = pool.map(csv_to_graph_ROI, inputs_list)

    failed = [f"{i}" for i in failed if i is not None]
    failed_json = json.dumps({"graph_gen_failed": failed})
    
    with open(os.path.join(data_root,"graph_gen_failed.json"), "w") as outfile:
        outfile.write(failed_json)



if __name__ == "__main__":

    parquet_path = '../data/nsclc_samples.csv'
    survival_path = None

    save_root = '../data'
    
    parquet_to_ROI_csv(parquet_path, 
                       save_root,
                       survival_path=survival_path
                       )
    print('finished csv generation ... working on graph generation now')
    csv_to_graph_dataset(save_root)