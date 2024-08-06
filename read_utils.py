import os

import numpy as np
import pandas as pd
import psycopg2

from autoqild import *

__all__ = ['MAE', 'MSE', 'NMAE', 'NMSE', 'columns_dict', 'learner_dict', 'dataset_dict', 'color_palette',
           'color_palette_dict', 'markers', 'markers_dict', 'get_synthetic_dataset_results', 'create_combined_dataset',
           'get_reduced_dataframe', 'learner_names', 'create_directory_safely', 'detection_methods', 'ild_metrics',
           'create_custom_order', 'get_synthetic_dataset_results', 'get_real_dataset_results',
           'create_combined_real_dataset']

MAE = "Mean absolute error"
MSE = "Mean squared error"
NMAE = "Normalized Mean \n absolute error"
NMSE = "Normalized Mean \n squared error"
columns_dict = {
    MID_POINT_MI_ESTIMATION.lower(): 'Mid-point',
    LOG_LOSS_MI_ESTIMATION.lower(): 'Log-loss',
    'cal_log-loss': "Cal Log-loss",
    LOG_LOSS_MI_ESTIMATION_PLATT_SCALING.lower(): 'PS Cal Log-loss',
    LOG_LOSS_MI_ESTIMATION_ISOTONIC_REGRESSION.lower(): 'IR Cal Log-loss',
    LOG_LOSS_MI_ESTIMATION_BETA_CALIBRATION.lower(): 'Beta Cal Log-loss',
    LOG_LOSS_MI_ESTIMATION_TEMPERATURE_SCALING.lower(): 'TS Cal Log-loss',
    PC_SOFTMAX_MI_ESTIMATION_HISTOGRAM_BINNING.lower(): 'HB Cal Log-loss'
}
learner_dict = {
    MULTI_LAYER_PERCEPTRON: "MLP",
    AUTO_GLUON: "AutoGluon",
    AUTO_GLUON_STACK: "AutoGluonStack",
    RANDOM_FOREST: "RF",
    TABPFN: "TabPFN",
    TABPFN_VAR: "TabPFN_V",
    GMM_MI_ESTIMATOR: "GMM Baseline",
    MINE_MI_ESTIMATOR: "MINE Baseline",
    MINE_MI_ESTIMATOR_HPO: "MINE Baseline"
}
dataset_dict = {
    SYNTHETIC_DATASET: "Mixed MVN Noise",
    SYNTHETIC_DISTANCE_DATASET: "Mixed MVN Distance",
    SYNTHETIC_IMBALANCED_DATASET: "Mixed MVN Noise Imbalanced",
    SYNTHETIC_DISTANCE_IMBALANCED_DATASET: "Mixed MVN Distance Imbalanced"
}
detection_methods = {
    'paired-t-test': 'PTT-Majority',
    'fishers-exact-mean': 'FET-Mean',
    'fishers-exact-median': 'FET-Median',
    'estimated_mutual_information': 'Direct MI Estimation',
    'mid_point_mi': 'Mid-point',
    'log_loss_mi': 'Log-loss',
    'log_loss_mi_isotonic_regression': 'IR Cal Log-loss',
    'log_loss_mi_platt_scaling': 'PS Cal Log-loss',
    'log_loss_mi_beta_calibration': 'Beta Cal Log-loss',
    'log_loss_mi_temperature_scaling': 'TS Cal Log-loss',
    'log_loss_mi_histogram_binning': 'HB Cal Log-loss',
    'p_c_softmax_mi': 'PCSoftmaxMI'
}
generation_methods = {
    'balanced': 'Balanced',
    'binary': 'Binary-class \n imbalanced',
    'multiple': 'Multi-class \n imbalanced',
    'single': 'Multi-class \n imbalanced'
}
connect_params = {
    "dbname": "autosca",
    "user": "autoscaadmin",
    "password": "THcRuCHjBcLjDYMps3jGVckN",
    "host": "vm-prithag04.cs.uni-paderborn.de",
    "port": 5432
}
ild_metrics = [ACCURACY, F_SCORE, MCC, INFORMEDNESS, FPR, FNR]

color_palette = ['#A50026', '#A50026', '#D62728', '#FF9896', '#FF9896', '#FF9896', '#FF9896', '#FF9896', '#FF9896',
                 # Shades of red
                 '#006D2C', '#006D2C', '#2CA02C', '#98DF8A', '#98DF8A', '#98DF8A', '#98DF8A', '#98DF8A', '#98DF8A',
                 # Shades of green
                 '#08519C', '#08519C', '#1F77B4', '#AEC7E8', '#AEC7E8', '#AEC7E8', '#AEC7E8', '#AEC7E8', '#AEC7E8',
                 # Shades of blue
                 '#000000', '#7F7F7F', '#C7C7C7']  # Shades of gray
markers = ['o', 'o', 's', 'H', 'H', 'H', 'H', 'H', 'H',
           'v', 'v', '^', '<', '<', '<', '<', '<', '<',
           'x', 'x', '+', 'X', 'X', 'X', 'X', 'X', 'X',
           '|', 'd', '*']


def get_real_dataset_results(table_name):
    connection = psycopg2.connect(**connect_params)
    sql_query = f"SELECT * from results.{table_name}"
    combined_results = pd.read_sql(sql_query, connection)
    return combined_results


def create_combined_real_dataset(table_name, filter_results=True):
    df = get_real_dataset_results(table_name)
    if filter_results:
        condition1 = df['base_detector'].isin([RANDOM_FOREST, AUTO_GLUON_STACK, TABPFN_VAR])
        condition2 = (df['detection_method'] == 'fishers-exact-mean')  # False
        df = df[~(condition1 | condition2)]
    condition = (df['base_detector'] == 'mlp') & (df['detection_method'] != 'p_c_softmax_mi')
    combined_results = df[~condition]
    threshold_condition = "n_hypothesis_threshold" in combined_results.columns
    if threshold_condition:
        group = ['n_hypothesis_threshold', 'delay', 'imbalance', 'base_detector', 'detection_method']
    else:
        group = ['delay', 'imbalance', 'base_detector', 'detection_method']
    columns_new = ["Dataset", 'Base Learner', "Detection Method", "Detection Technique", "Imbalance", "Delay",
                   "Threshold", "Time"]
    for col in ild_metrics:
        columns_new.extend([col, col + '-Std'])
    data = []
    for (values), dataset_df in combined_results.groupby(group):
        if threshold_condition:
            n_hypothesis_threshold, delay, imbalance, base_detector, detection_method = values
        else:
            delay, imbalance, base_detector, detection_method = values
            n_hypothesis_threshold = 1
        one_row = [f"Timing {delay} micro-seconds", learner_dict[base_detector], detection_methods[detection_method],
                   f"{learner_dict[base_detector]}-{detection_methods[detection_method]}",
                   imbalance, int(delay), n_hypothesis_threshold]

        if detection_method == 'p_c_softmax_mi':
            if learner_dict[base_detector] == 'MLP':
                one_row[1] = 'PC-Softmax Baseline'
                one_row[2] = detection_methods['estimated_mutual_information']
            else:
                continue

        if one_row[2] == detection_methods['estimated_mutual_information']:
            one_row[3] = one_row[1]

        time = np.mean(dataset_df['evaluation_time'].values)
        one_row.append(time)

        for col in ild_metrics:
            mean = np.mean(dataset_df[col.lower()].values)
            std = np.std(dataset_df[col.lower()].values)
            one_row.extend([mean, std])

        data.append(one_row)

    final_df = pd.DataFrame(data, columns=columns_new)
    if filter_results:
        custom_order = [learner_dict[AUTO_GLUON], learner_dict[TABPFN_VAR], learner_dict[GMM_MI_ESTIMATOR],
                        learner_dict[MINE_MI_ESTIMATOR], 'PC-Softmax Baseline']
    else:
        custom_order = [learner_dict[AUTO_GLUON_STACK], learner_dict[AUTO_GLUON], learner_dict[TABPFN],
                        learner_dict[TABPFN_VAR], learner_dict[GMM_MI_ESTIMATOR], learner_dict[MINE_MI_ESTIMATOR],
                        'PC-Softmax Baseline']
    detectors_order = [detection_methods['mid_point_mi'], detection_methods['log_loss_mi'],
                       detection_methods['log_loss_mi_isotonic_regression'],
                       detection_methods['log_loss_mi_platt_scaling'],
                       detection_methods['log_loss_mi_beta_calibration'],
                       detection_methods['log_loss_mi_temperature_scaling'],
                       detection_methods['log_loss_mi_histogram_binning'], detection_methods['paired-t-test'],
                       detection_methods['fishers-exact-mean'], detection_methods['fishers-exact-median'],
                       detection_methods['estimated_mutual_information']]

    techniques_order = [f"{learner}-{col}" for learner in custom_order if 'Baseline' not in learner for col in
                        detectors_order]
    techniques_order += ['GMM Baseline', 'MINE Baseline', 'PC-Softmax Baseline']
    final_df['Detection Technique'] = pd.Categorical(final_df['Detection Technique'], categories=techniques_order,
                                                     ordered=True)
    final_df.sort_values(['Detection Technique', 'Imbalance', 'Delay'], inplace=True)
    final_df['Detection Technique'] = final_df['Detection Technique'].cat.remove_unused_categories()

    return final_df


def get_synthetic_dataset_results():
    connection = psycopg2.connect(**connect_params)
    sql_query = "SELECT * from results.automl_results"
    auto_ml_df = pd.read_sql(sql_query, connection)
    sql_query = "SELECT * from results.mutual_information_results"
    mutual_info_df = pd.read_sql(sql_query, connection)
    columns = ['learner', 'dataset', 'fold_id', 'n_classes', 'n_features', 'noise', 'flip_y', 'gen_type']
    mutual_info_df.sort_values(columns, inplace=True)
    auto_ml_df.sort_values(columns, inplace=True)
    combined_results = pd.concat([auto_ml_df, mutual_info_df])
    combined_results['noise'] = combined_results['noise'].fillna(-1.0)
    combined_results['flip_y'] = combined_results['flip_y'].fillna(-1.0)
    combined_results['gen_type'] = combined_results['gen_type'].fillna('balanced')
    combined_results['imbalance'] = combined_results['imbalance'].fillna(-1.0)
    combined_results.loc[combined_results["dataset"].str.contains("imbalanced", case=False) & (
            combined_results["n_classes"] == 2), 'gen_type'] = 'binary'
    combined_results.sort_values('gen_type', inplace=True)
    return combined_results


# cols.sort()
def create_custom_order():
    ls = ["AutoGluon", "TabPFN", "MLP"]
    cols = list(columns_dict.values())
    custom_order = []
    for learner in ls:
        if 'Baseline' in learner:
            continue
        custom_order.append(learner)
        for col in cols:
            learner_name = f"{learner} {col}"
            custom_order.append(learner_name)
    custom_order = custom_order + ['GMM Baseline', 'MINE Baseline', 'PC-Softmax Baseline']
    return custom_order


learner_names = create_custom_order()
color_palette_dict = dict(zip(learner_names, color_palette))
markers_dict = dict(zip(learner_names, markers))


def clean_array(arr):
    # Calculate the maximum value (excluding inf and nan)
    indicies = np.where(np.logical_or(np.isnan(arr), np.isinf(arr)))[0]

    max_value = 0  # np.nanmax(arr[np.isfinite(arr)])

    # Replace inf and nan values with the maximum value
    arr_cleaned = np.where(np.logical_or(np.isnan(arr), np.isinf(arr)), max_value, arr)
    return arr_cleaned


def get_values(y_true, y_pred, time, n_classes):
    y_true = clean_array(y_true)
    y_pred = clean_array(y_pred)
    mae = np.around(np.nanmean(np.abs(y_true - y_pred)), 8)
    mse = np.around(np.nanmean((y_true - y_pred) ** 2), 8)
    time = np.mean(clean_array(time))
    time = np.around(time, 4)
    y_true_norm = y_true / np.log2(n_classes)
    y_pred_norm = y_pred / np.log2(n_classes)
    nmae = np.around(np.nanmean(np.abs(y_true_norm - y_pred_norm)), 8)
    nmse = np.around(np.nanmean((y_true_norm - y_pred_norm) ** 2), 8)
    return mae, mse, nmae, nmse, time


def get_values_std(y_true, y_pred, n_classes):
    y_true = clean_array(y_true)
    y_pred = clean_array(y_pred)
    mae = np.around(np.nanstd(np.abs(y_true - y_pred)), 8)
    mse = np.around(np.nanstd((y_true - y_pred) ** 2), 8)
    y_true_norm = y_true / np.log2(n_classes)
    y_pred_norm = y_pred / np.log2(n_classes)
    nmae = np.around(np.nanstd(np.abs(y_true_norm - y_pred_norm)), 8)
    nmse = np.around(np.nanstd((y_true_norm - y_pred_norm) ** 2), 8)
    return mae, mse, nmae, nmse


def sort_dataframe(df):
    learner_order = create_custom_order()
    gen_type_order = [generation_methods['balanced'], generation_methods['binary'], generation_methods['single']]
    df['Learner'] = pd.Categorical(df['Learner'], categories=learner_order, ordered=True)
    df['Generation Type'] = pd.Categorical(df['Generation Type'], categories=gen_type_order, ordered=True)

    # Sort the DataFrame based on the custom order
    df.sort_values(['Learner', 'Generation Type'], inplace=True)
    df['Learner'] = df['Learner'].cat.remove_unused_categories()
    df['Generation Type'] = df['Generation Type'].cat.remove_unused_categories()
    return df


def create_combined_dataset():
    combined_results = get_synthetic_dataset_results()
    columns_new = ["Dataset", 'Learner', "Flip Percentage", "Distance", "Noise", "Classes", "Features",
                   "Generation Type", "Imbalance", MAE, MSE, NMAE, NMSE, "Time"]
    data = []
    for dataset, dataset_df in combined_results.groupby('dataset'):
        dataset_name = dataset_dict[dataset]
        group = ['flip_y', 'noise', 'n_classes', 'n_features', 'gen_type', 'imbalance']
        for (values), filter_df in dataset_df.groupby(group):
            flip_y, noise, n_classes, n_featrues, gen_type, imbalance = values
            gen_type = generation_methods[gen_type]
            noise = np.round(noise, 1)
            for (learner), learner_df in filter_df.groupby('learner'):
                y_true = np.array(learner_df['mcmcbayesmi'].values)
                time = learner_df['evaluation_time'].values
                learners = ['auto_gluon', 'tab_pfn']  # + ["mlp"]
                if learner in learners:
                    for column in columns_dict.keys():
                        if column in list(learner_df.columns):
                            y_pred = learner_df[column].values
                            # if np.any(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred)):
                            # print(dataset, values, learner, column)
                            learner_name = f"{learner_dict[learner]} {columns_dict[column]}"
                            mae, mse, nmae, nmse, time = get_values(y_true, y_pred, time, n_classes)
                            one_row = [dataset_name, learner_name, flip_y, 1 - noise, noise, n_classes, n_featrues,
                                       gen_type, imbalance, mae, mse, nmae, nmse, time]
                            data.append(one_row)
                if learner == 'mlp':
                    y_pred = learner_df['pcsoftmaxmi'].values
                    learner_name = "PC-Softmax Baseline"
                    # if np.any(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred)):
                    # print(dataset, values, learner, column)
                    mae, mse, nmae, nmse, time = get_values(y_true, y_pred, time, n_classes)
                    one_row = [dataset_name, learner_name, flip_y, 1 - noise, noise, n_classes, n_featrues, gen_type,
                               imbalance, mae, mse, nmae, nmse, time]
                    data.append(one_row)
                if "mi_estimator" in learner:
                    y_pred = learner_df['estimatedmutualinformation'].values
                    # if np.any(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred)):
                    # print(dataset, values, learner, column)
                    learner_name = learner_dict[learner]
                    mae, mse, nmae, nmse, time = get_values(y_true, y_pred, time, n_classes)
                    one_row = [dataset_name, learner_name, flip_y, 1 - noise, noise, n_classes, n_featrues, gen_type,
                               imbalance, mae, mse, nmae, nmse, time]
                    data.append(one_row)
    df = pd.DataFrame(data, columns=columns_new)
    df = sort_dataframe(df)
    # print(f"Learners {df['Learner'].unique()}")
    # print(f"Generation Types {df['Generation Type'].unique()}")
    return df


filter_cases = ['best_of_ll', 'best_of_all', 'best_of_cal_ll']


def filter_best_results(cat_df, filter_case):
    result = cat_df.groupby(['Learner'])[NMAE].agg(['mean', 'std']).reset_index()
    result_df = pd.DataFrame(result)
    filter_learners = []
    learners = ['AutoGluon', "TabPFN"]  # + ["MLP"]
    for learner in learners:
        sub_df = result_df[result_df['Learner'].str.contains(learner)]
        print(sub_df.to_string(index=False))
        if filter_case == 'best_of_ll':
            sub_df = sub_df[sub_df['Learner'].str.contains("Log-loss")]
            filter_learners.append(f"{learner} {columns_dict['midpointmi']}")
            strings_to_remove = ['Platt ', 'IR ', 'Beta ', 'TS ', 'HB ', 'Cal ']
            strings_to_remove = []
        if filter_case == 'best_of_cal_ll':
            sub_df = sub_df[sub_df['Learner'].str.contains("Cal Log-loss")]
            filter_learners.append(f"{learner} {columns_dict['midpointmi']}")
            filter_learners.append(f"{learner} {columns_dict['loglossmi']}")
            # strings_to_remove = ['Platt ', 'IR ', 'Beta ', 'TS ', 'HB ']
            strings_to_remove = []
        if filter_case == 'best_of_all':
            strings_to_remove = []
        # Get the learner with the minimum mean absolute error
        min_error_learner = sub_df.loc[sub_df['mean'].idxmin()]

        # Check if there are duplications based on mean error
        duplicated_learners = sub_df.loc[sub_df['mean'].duplicated()]

        # If there are duplications, choose the learner with the lower standard deviation (std)
        if not duplicated_learners.empty:
            dup = duplicated_learners.sort_values(['Learner', 'std'], ascending=[False, True])
            print(f"Duplicate Entries {dup.to_string(index=False)}")
            chosen_learner = dup.iloc[0]
        else:
            chosen_learner = min_error_learner
        # Print the learner with the minimum mean absolute error
        # print("Learner with the minimum mean absolute error:", min_error_learner)
        filter_learners.append(chosen_learner['Learner'])
    filter_learners = filter_learners + ['GMM Baseline', 'MINE Baseline', 'PC-Softmax Baseline']
    # print(f"Best Learners {filter_learners}")
    cat_df = cat_df[cat_df['Learner'].isin(filter_learners)]
    cat_df['Learner'] = cat_df['Learner'].astype("category")

    # Remove unused categories from the 'Learner' column
    cat_df['Learner'] = cat_df['Learner'].cat.remove_unused_categories()
    if filter_case in ['best_of_ll', 'best_of_cal_ll']:
        # Remove the strings from the column
        cat_df['Learner'] = cat_df['Learner'].str.replace('|'.join(strings_to_remove), '', regex=True)
    else:
        for learner in learners:
            cat_df['Learner'] = np.where(cat_df['Learner'].str.contains(learner), learner, cat_df['Learner'])
    cat_df = sort_dataframe(cat_df)
    print(f"Best Learners Renamed {cat_df['Learner'].unique()}")
    return cat_df


def get_reduced_dataframe(df, datasets=[], filter_case='best_of_all'):
    ds = [dataset_dict[d] for d in datasets]
    dataset_df = df[df['Dataset'].isin(ds)]
    if filter_case is not None:
        categories = list(dataset_df['Generation Type'].unique())
        dfs = []
        for category in categories:
            cat = category.replace(" \n ", ' ')
            print(f"************ Category {cat} ************")
            cat_df = dataset_df[dataset_df['Generation Type'] == category]
            filter_df = filter_best_results(cat_df, filter_case)
            dfs.append(filter_df)
        result_df = pd.concat(dfs, axis=0)
    else:
        result_df = pd.DataFrame.copy(dataset_df)
    return result_df


def create_directory_safely(path, is_file_path=False):
    try:
        if is_file_path:
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(str(e))
