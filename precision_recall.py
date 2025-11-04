import sys
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

def evaluate_quality(real_path, syn_path, info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)


    ''' Special treatment for default dataset and CoDi model '''

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')


    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()

    # cat_syn_data_np = np.array
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    def check_int(s):
        if s[0] in ('-', '+'):
            return s[1:].isdigit()
        return s.isdigit()

    def normalize_to_int(real, syn):
        # out = syn.astype(str)
        # for i in range(out.shape[1]):
        #     for j in range(out.shape[0]):
        #         if check_int(real[j, i]) and not check_int(syn[j, i]):
        #             out[j, i] = str(int(float(syn[j, i])))
        # return out
        out = syn.astype(str)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i,j][-2:] == '.0':
                    out[i,j] = str(int(float(syn[i,j])))
        return out
    
    cat_syn_data_np = normalize_to_int(cat_real_data_np, cat_syn_data_np)
    
    # ensure categorical columns are strings and normalized
    def normalize_cat(arr):
        # arr is 2D numpy array of categorical columns
        out = arr.astype(str)
        # strip whitespace (and lowercase?) to avoid trivial mismatches
        for i in range(out.shape[1]):
            out[:, i] = np.char.strip(out[:, i])
            # out[:, i] = np.char.lower(out[:, i])
        return out
    
    cat_real_data_np = normalize_cat(cat_real_data_np)
    cat_syn_data_np = normalize_cat(cat_syn_data_np)

    # quick diagnostic: show unseen categories in synthetic data
    for col_idx in range(cat_real_data_np.shape[1]):
        real_uni = set(np.unique(cat_real_data_np[:, col_idx]))
        syn_uni = set(np.unique(cat_syn_data_np[:, col_idx]))
        unseen = syn_uni - real_uni
        if unseen:
            print(f"Warning: unseen categories in synthetic column {col_idx}: {sorted(list(unseen))}")
            print(f"Real categories: {sorted(list(real_uni))}")
            print(f"Synthetic categories: {sorted(list(syn_uni))}")

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis = 1)).astype(float)
    # le_real_num = pd.DataFrame(num_real_data_np).astype(float)
    # le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)


    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis = 1)).astype(float)
    # le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
    # le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)

    # Check for nan
    if le_syn_data.isnull().values.any():
        nan_coordinate = np.isnan(le_syn_data.to_numpy()).nonzero()
        nan_row = np.unique(nan_coordinate[0])
        print(f"Synthetic data contains NaN at row {nan_row}: ")
        print(le_syn_data.iloc[nan_row])
        return None, None


    np.set_printoptions(precision=4)

    print('=========== All Features ===========')
    print('Data shape: ', le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision
    # qual_score = np.mean(list(qual_res.values()))

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive'] ))

    Alpha_Precision_all = qual_res['delta_precision_alpha_naive']
    Beta_Recall_all = qual_res['delta_coverage_beta_naive']

    return Alpha_Precision_all, Beta_Recall_all


real_path = sys.argv[1]
syn_path = sys.argv[2]
info_path = sys.argv[3]

Alpha_Precision_all, Beta_Recall_all = evaluate_quality(real_path, syn_path, info_path)