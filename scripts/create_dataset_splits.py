'''
creates a split of all available cases (and reads in the ROIs of these cases as well)
for a given parquet file
saves everthing into a dictionary 
'''

import json
import numpy as np
import os
import pandas as pd
# import pickle
import random

from itertools import chain
from pycrumbs import tracked

def stratification_case_ids(stratification_df_path: str,
                            tumor_metamain: str, # name of the tumor that the dataset is generated for to filter the outcome dataframe by 
                            stratification_feature: str
                            ):
    # returns the case ids for stratification 

    # read in the dataframe 
    strat_df = pd.read_csv(stratification_df_path)
    strat_df = strat_df.loc[strat_df['oncotree_metamain']==tumor_metamain]

    # split the dataframe according to the stratification feature 
    pos_case_ids = np.unique(strat_df.loc[strat_df[stratification_feature]==True]['case_id'].to_numpy())
    neg_case_ids = np.unique(strat_df.loc[strat_df[stratification_feature]==False]['case_id'].to_numpy())

    # return the case ids for positive/negative strata
    return pos_case_ids, neg_case_ids

@tracked(directory_parameter='pkl_root',
         seed_parameter='seed')
def split(parquet_path:str, 
          ratio:list, 
          pkl_root:str, 
          stratify_by = None,
          seed=0, 
          n_cross_val_splits = None,
          return_output = False,
          save = True
          ):
    
    '''
    optional parameters: 
    stratify_by: [positive_case_ids, negative_case_ids] - case_ids in the positive and negative strata

    '''

    # generate a dictionary that contains all available cases
    # and the corresponding ROIs
    
    assert sum(ratio) == 1 

    df = pd.read_parquet(parquet_path)
    
    # create dictionary that links cases with their available ROIs
    sample_ROI_dict = create_case_ROI_dict(df)
    # print(sample_ROI_dict)

    all_cases = np.unique(df.index.get_level_values(0).to_numpy())
    
    if stratify_by is None: 
        # if no case ids are given for stratification
        # use all case ids in one strata
        stratify_by = all_cases
        print(len(stratify_by))
    
    else:
        # are there case_ids that are not contained in either of the strata? 
        difference = set(all_cases) - set(list(chain(*stratify_by)))
        if not difference==0: 
            # add the cases that are not within the specified strata as an additional one  
            stratify_by.append(list(difference))

    training_case_ids, validation_case_ids, test_case_ids = list(), list(), list()
    
    dev_stratify_by = [list() for i in stratify_by] # the positive, negative, no_info strata only for the cases in the training and validation splits
    # for cross-validation 
    for i in range(len(stratify_by)):
        
        # sort and shuffle
        temp_ids = stratify_by[i]
        temp_ids.sort()
        random.shuffle(temp_ids)
        
        # split
        training_case_ids.extend(temp_ids[:int(len(temp_ids)*ratio[0])])
        validation_case_ids.extend(temp_ids[int(len(temp_ids)*ratio[0]):int(len(temp_ids)*(ratio[0]+ratio[1]))])
        test_case_ids.extend(temp_ids[int(len(temp_ids)*(ratio[0]+ratio[1])):])
        
        # record the strata for the training and validation cases from split 0 
        # for cross validation 
        dev_stratify_by[i] = temp_ids[:int(len(temp_ids)*(ratio[0]+ratio[1]))]
    print('train 0', len(training_case_ids))
    print('val 0', len(validation_case_ids))
    assert len(set(training_case_ids).intersection(set(validation_case_ids))) == 0 
    assert len(set(validation_case_ids).intersection(set(test_case_ids))) == 0 

    # create separate case:ROI dictionaries for the dataset splits
    outdict = {
        "splits": {
            "training": {key.item(): sample_ROI_dict[key] for key in training_case_ids if key in sample_ROI_dict},
            "validation": {key.item(): sample_ROI_dict[key] for key in validation_case_ids if key in sample_ROI_dict},
            "test": {key.item(): sample_ROI_dict[key] for key in test_case_ids if key in sample_ROI_dict}
        },
        "n_cases": {
            "training": len(training_case_ids),
            "validation": len(validation_case_ids),
            "test": len(test_case_ids)
        }
    }

    '''cross validation'''
    # if "cross-validation" is performed - add n_cross_val_splits cross validation splits 
    # create n random splits of the developent data into training and validation (not conventional n-fold cross-validation!)
    if not n_cross_val_splits is None: 

        cross_val_splits_dict = { str(0): 
                                    {"training": {key.item(): sample_ROI_dict[key] for key in training_case_ids if key in sample_ROI_dict},
                                     "validation": {key.item(): sample_ROI_dict[key] for key in validation_case_ids if key in sample_ROI_dict},
                                     "test": {key.item(): sample_ROI_dict[key] for key in test_case_ids if key in sample_ROI_dict}
                                    }
                                 }
        # combine the training and validation cases 
        
        for i in range(1, n_cross_val_splits):
            print('cross validation split', i)
            # shuffle and randomly split them 
            for k in dev_stratify_by:
                random.shuffle(k)
            
            temp_training_case_ids, temp_validation_case_ids = list(), list()
            # selecte positive/negative cases (stratification)
            for temp_ids in dev_stratify_by:
        
                # shuffle
                random.shuffle(temp_ids)
                
                # split
                split_ind = int(len(temp_ids)*(ratio[0]/(ratio[0]+ratio[1])))
                temp_training_case_ids.extend(temp_ids[:split_ind])
                temp_validation_case_ids.extend(temp_ids[split_ind:])
                
            assert len(set(temp_training_case_ids).intersection(set(temp_validation_case_ids))) == 0 
            
            cross_val_splits_dict[str(i)] = {"training": {key.item(): sample_ROI_dict[key] for key in temp_training_case_ids if key in sample_ROI_dict},
                                            "validation": {key.item(): sample_ROI_dict[key] for key in temp_validation_case_ids if key in sample_ROI_dict},
                                            "test": {key.item(): sample_ROI_dict[key] for key in test_case_ids if key in sample_ROI_dict}
                                            }

        outdict["cross_val_splits"] = cross_val_splits_dict

    split_json = json.dumps(outdict)
    
    # Writing to sample.json
    if save:
        with open(os.path.join(pkl_root, 'split.json'), "w") as outfile:
            outfile.write(split_json)

    if return_output:
        # only for testing purposes
        return outdict


def create_case_ROI_dict(df):
    # creates a dictionary with all cases and the corresponding ROI ids
    # dict = {case_id : [list of all ROI ids for the case]}
    case_ROI_dict = dict()

    case_ids = np.unique(df.index.get_level_values(0).to_numpy())
    case_ids.sort() 

    for case in case_ids:
        # grab all available ROI ids
        temp_df = df.xs(case, level=0, axis=0, drop_level=False)
        
        temp_rois = np.unique(temp_df.index.get_level_values(1).to_numpy()).tolist()
        temp_rois.sort()
        temp_rois = [f"{r}" for r in temp_rois]

        case_ROI_dict[case] = temp_rois

    return case_ROI_dict

def test_reproducibility(parquet_path, ratio, pkl_root, stratify_by, seed, n_cross_val_splits):
    out1 = split(
        parquet_path, 
        ratio,
        pkl_root,
        stratify_by=stratify_by,
        seed=seed,
        n_cross_val_splits=n_cross_val_splits,
        return_output=True
    )

    out2 = split(
        parquet_path, 
        ratio,
        pkl_root,
        stratify_by=stratify_by,
        seed=seed,
        n_cross_val_splits=n_cross_val_splits,
        return_output=True
    )

    print(out1==out2)

    print(out1.keys())

    if n_cross_val_splits is not None:
        for i in range(n_cross_val_splits):
            print(i)
            print(out1['cross_val_splits'][str(i)]==out2['cross_val_splits'][str(i)])
        pass


if __name__ == "__main__": 
    parquet_path = ... # path to cell file 
    
    ratio = [.60,.15,.25]
    pkl_root = ... # path to dir to save the split.json file
    seed = 0

    # n_cross_val_splits = 5
    n_cross_val_splits = None

    stratification_df_path = ... # path to spreadsheet with survival informaiton for stratification 
    tumor_metamain =  'Non-Small Cell Lung Cancer'
    stratification_feature = 'survival_status' # name of the survival status column for stratification 

    pos_ids, neg_ids = stratification_case_ids(stratification_df_path,
                            tumor_metamain, # name of the tumor that the dataset is generated for to filter the outcome dataframe by 
                            stratification_feature
                            )
    split(
        parquet_path, 
        ratio,
        pkl_root,
        stratify_by=[pos_ids,neg_ids],
        seed = seed,
        n_cross_val_splits=n_cross_val_splits
    )

    # test_reproducibility(parquet_path, ratio, pkl_root, [pos_ids,neg_ids], seed, n_cross_val_splits)
    
    

    

