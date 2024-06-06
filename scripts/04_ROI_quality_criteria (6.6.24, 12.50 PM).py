'''
identifies what ROIs fulfill a defined quality criteria
all cases and ROIs that do fulfill the criteria are written to a dictionary 

date created: 2023/04/27
'''

import json
import numpy as np
import os
import pandas as pd

from pycrumbs import tracked


def comp_inner_tumor_ratio(test_roi):
    # computes the ratio of cells considered Inner Tumor vs all other regions
    try:
        inner_tumor_count = test_roi.region_label.value_counts()['InnerTumor']
        inner_tumor_ratio = inner_tumor_count/test_roi.shape[0] # total number of cells in the ROI
    except:
        inner_tumor_ratio = 0 
    
    return inner_tumor_ratio


def test_addional_qc_criteria(roi, save_root, additional_criteria_accepted_values):

    additional_qc_passed = True
    # organization of additonal_criteria_accpeted_values:
    # key: name of the column that the selection is based on
    # list of values: accepted values - if the column value is not in the list 
    # then the ROI does not pass the QC step 
    
    # read in the cell data csv
    try: 
        cell_data_df = pd.read_csv(os.path.join(save_root, 'cell_data', str(roi)+'.cell_data.csv'))
    except: 
        # if no cell data df exists -> QC not passed because information not available
        additional_qc_passed = False 
    
    if additional_qc_passed:
        for k, v in additional_criteria_accepted_values.items():
            # grab the cell_data value for the current column
            temp_value = np.unique(cell_data_df[k].values)
            assert len(temp_value) == 1
            if not temp_value in v:
                # if only one of the additonal criteria is not fulfilled then the QC step is not passed
                additional_qc_passed = False 

    return additional_qc_passed 

@tracked(directory_parameter='save_root')
def inner_tumor_qc(parquet_path, save_root, threshold=0.95,  
                   additional_criteria_accepted_values=None):
    # identifies all ROIs with an inner tumor cell ratio at or above the defined threshold
    
    df = pd.read_parquet(parquet_path)
    roi_ids = np.unique(df.index.get_level_values(1).to_numpy())

    qc_passed_dict = dict()

    for roi in roi_ids:
        # grab the subdataframe for the specified ROI 
        test_roi = df.xs(roi, level=1, axis=0, drop_level=False).copy()

        # compute the ratio of inner tumor within the ROI
        inner_tumor_ratio = comp_inner_tumor_ratio(test_roi)

        qc_passed = False
        # test if the inner tumor ratio is at or above the defined quality ratio
        if inner_tumor_ratio >= threshold:
            qc_passed = True
        
        # if additonal qc criteria are provided test whether the roi passes
        if qc_passed & (additional_criteria_accepted_values is not None):
            qc_passed = test_addional_qc_criteria(roi, save_root, additional_criteria_accepted_values)

        if qc_passed:
            # add case and roi id to the dictionary with all rois that fulfill the quality criterion
            case_id = np.unique(test_roi.index.get_level_values(0))
            assert len(case_id) == 1
            case_id  = f"{case_id[0]}"
            
            if case_id in qc_passed_dict.keys():
                qc_passed_dict[case_id].append(f"{roi}")
            else:
                qc_passed_dict[case_id] = [f"{roi}"]

    # save dict to json

    qc_passed_json = json.dumps(qc_passed_dict)
    
    # Writing to sample.json
    if additional_criteria_accepted_values is None:
        outfile_name = 'QC_inner_tumor_'+str(threshold)+".json"
    else: 
        additonal_criteria = ''
        for k in additional_criteria_accepted_values.keys():
            additonal_criteria = additonal_criteria = '_' + k
        outfile_name = 'QC_inner_tumor_'+str(threshold)+additonal_criteria+".json"

    with open(os.path.join(save_root, outfile_name), "w") as outfile:
        outfile.write(qc_passed_json)


def identifyTSI(roi_df, stroma_criterion=0.2, tumor_criterion=0.3, print_density=False):

    cell_count = roi_df.shape[0]
    try:
        tumor_density = roi_df.region_label.value_counts()['InnerTumor']/cell_count
    except:
        tumor_density = 0
    
    try:
        stroma_density = roi_df.region_label.value_counts()['OuterStroma']/cell_count
    except:
        stroma_density = 0

    if print_density:
        print('tumor:', tumor_density, ', stroma:', stroma_density) 
    
    if ((tumor_density >= tumor_criterion) and (stroma_density >= stroma_criterion)):
        # roi fulfills the QC criteria
        return True
    else:
        # roi does not fulfill the QC criteria
        return False


@tracked(directory_parameter='save_root')
def tumor_stroma_interface_qc(parquet_path, save_root, stroma_threshold=0.1):
    # identifies all ROIs with an inner tumor cell ratio at or above the defined threshold
    
    df = pd.read_parquet(parquet_path)
    roi_ids = np.unique(df.index.get_level_values(1).to_numpy())

    qc_passed_dict = dict()

    for roi in roi_ids:
        # grab the subdataframe for the specified ROI 
        test_roi = df.xs(roi, level=1, axis=0, drop_level=False).copy()

        tsi_criterion_passed = identifyTSI(test_roi, 
                                           stroma_criterion=stroma_threshold, 
                                           tumor_criterion=0.2, 
                                           print_density=False)

        # test if the inner tumor ratio is at or above the defined quality ratio
        if tsi_criterion_passed:
            # add case and roi id to the dictionary with all rois that fulfill the quality criterion
            
            case_id = np.unique(test_roi.index.get_level_values(0))
            assert len(case_id) == 1
            case_id  = f"{case_id[0]}"
            
            if case_id in qc_passed_dict.keys():
                qc_passed_dict[case_id].append(f"{roi}")
            else:
                # print(case_id)
                qc_passed_dict[case_id] = [f"{roi}"]

    print('number of cases that passed:', len(list(qc_passed_dict.keys())))
    # save dict to json
    qc_passed_json = json.dumps(qc_passed_dict)
    
    # Writing to sample.json
    with open(os.path.join(save_root, 'QC_tumor_stroma_interface_'+str(stroma_threshold)+".json"), "w") as outfile:
        outfile.write(qc_passed_json)


if __name__ == "__main__": 
    parquet_path = '/ksg-images/cells/non-small_cell_lung_cancer.parquet'
    # parquet_path = '/ksg-images/cells/colorectal_cancer.parquet'
    # parquet_path = '/ksg-images/cells/breast_cancer.parquet'
    # parquet_path = '/ksg-images/cells/bladder_cancer.parquet'
    # parquet_path = '/ksg-images/cells/esophagogastric_cancer.parquet'
    # parquet_path = '/ksg-images/cells/melanoma.parquet'
    # parquet_path = '/ksg-images/cells/pancreatic_cancer.parquet'
    # parquet_path = '/ksg-images/cells/head_and_neck_cancer.parquet'
    # parquet_path = '/ksg-images/cells/endometrial_cancer.parquet'
    
    parquet_dict = {'CRC': '/ksg-images/cells/colorectal_cancer.parquet',
                     'breast': '/ksg-images/cells/breast_cancer.parquet',
                     'bladder': '/ksg-images/cells/bladder_cancer.parquet',
                     'esophagus': '/ksg-images/cells/esophagogastric_cancer.parquet',
                     'melanoma': '/ksg-images/cells/melanoma.parquet',
                     'pancreatic': '/ksg-images/cells/pancreatic_cancer.parquet',
                     'headandneck': '/ksg-images/cells/head_and_neck_cancer.parquet',
                     'endometrial': '/ksg-images/cells/endometrial_cancer.parquet'
                     }

   
    for tumor_type, parquet_path in parquet_dict.items():
        print(tumor_type, parquet_path)

        save_root = f'/lotterlab/users/khoebel/mIP/data/{tumor_type}/generic' 
        print(save_root)
        # additional_criteria_accepted_values = {'joao_lung_stg':[1,2,3,4]}
        inner_tumor_qc(parquet_path, save_root, threshold=0.90, 
                    # additional_criteria_accepted_values=additional_criteria_accepted_values
                    )
        # tumor_stroma_interface_qc(parquet_path, save_root, stroma_threshold=0.1)