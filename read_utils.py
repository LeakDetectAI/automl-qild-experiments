import os
import re

import matplotlib.path as mpath
import numpy as np
import pandas as pd
import psycopg2
from autoqild import *
import logging
import inspect
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

__all__ = ['MAE', 'MSE', 'NMAE', 'NMSE', 'columns_dict', 'learner_dict', 'dataset_dict', 'color_palette',
           'color_palette_dict', 'markers', 'markers_dict', 'get_synthetic_dataset_results',
           'create_combined_synthetic_dataset', 'get_reduced_dataframe', 'learner_names', 'create_directory_safely',
           'detection_methods', 'ild_metrics', 'create_custom_order', 'get_synthetic_dataset_results',
           'get_real_dataset_results', 'create_combined_real_dataset', 'setup_logging', 'generation_methods',
           'filter_real_dataset', 'camel_to_words', 'real_parameters', 'noise_column', 'imb_column', 'thre_column',
           'delay_column', 'delay_value', 'fp_column', 'classes_column', 'features_column', 'noise_column',
           'gen_type_column', 'detection_technique', 'convert_xlabels', 'ptt_mv', 'ptt_rg', 'fet_mean', 'bright_pallet_configs',
           'palette_configs', 'my_palplot', 'legend_params', 'fig_params']

legend_params = dict(frameon=True, markerscale=0.4, labelspacing=.2, columnspacing=0.5, handletextpad=0.5, handlelength=1,
                     borderpad=0.4, edgecolor='k', fancybox=True, shadow=True)
fig_params = dict(dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.02, facecolor='white'}

def my_palplot(pal, size=1, ax=None):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot
    ax :
        an existing axes to use
    """

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n = len(pal)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    # Ensure nice border between colors
    ax.set_xticklabels(["" for _ in range(n)])
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())


# Create a list of palette configurations with titles
palette_configs = [
    {"title": "Blueberry to Maroon", 'palette': sns.diverging_palette(240, 10, n=12, center="light", s=99, l=30, as_cmap=False)},
    {"title": "Teal to Coral", "palette": sns.diverging_palette(160, 15, n=12, center="light", s=85, l=40, as_cmap=False)},
    {"title": "Turquoise to Pale Orange", "palette": sns.diverging_palette(160, 25, n=12, center="light", s=80, l=45, as_cmap=False)},
    {"title": "Dusty Blue to Blush Pink", "palette": sns.diverging_palette(220, 10, n=12, center="light", s=70, l=50, as_cmap=False)},
    {"title": "Muted Cyan to Warm Beige", "palette": sns.diverging_palette(200, 30, n=12, center="light", s=65, l=55, as_cmap=False)},
    {"title": "Blue-Green to White to Coral", "palette": sns.diverging_palette(140, 0, n=12, center="light", s=70, l=50, as_cmap=False)}
]
bright_pallet_configs = [
    {"title": "Lavender to Pale Peach", "palette": sns.diverging_palette(270, 25, n=12, center="light", s=70, l=60, as_cmap=False)},
    {"title": "Aqua to Light Rose", "palette": sns.diverging_palette(180, 0, n=12, center="light", s=75, l=55, as_cmap=False)},
    {"title": "Cool Mint to Sandstone", "palette": sns.diverging_palette(120, 35, n=12, center="light", s=65, l=50, as_cmap=False)},
    {"title": "Classic Coolwarm", "palette": sns.color_palette("coolwarm", as_cmap=False)},
    {"title": "Brown to Blue-Green (BrBG)", "palette": sns.diverging_palette(220, 20, n=12, center="light", s=90, l=60, as_cmap=False)},
    {"title": "Spectral Palette", "palette": sns.color_palette("Spectral", n_colors=12, as_cmap=False)},
    {"title": "Purple to Orange (PuOr)", "palette": sns.diverging_palette(250, 50, n=12, center="light", s=90, l=60, as_cmap=False)},
    {"title": "Red-Yellow-Blue (RdYlBu)", "palette": sns.diverging_palette(220, 10, n=12, center="light", s=90, l=65, as_cmap=False)},
    {"title": "Blue-Green to Red-Purple", "palette": sns.diverging_palette(210, 350, n=12, center="light", s=80, l=55, as_cmap=False)}
]


def create_rgb_colormap():
    tab10 = plt.get_cmap("tab10")

    # Extract the required Red, Green, and Blue colors from Tab10
    red = tab10(0)  # Tab10 Red
    green = tab10(2)  # Tab10 Green
    blue = tab10(1)  # Tab10 Blue

    # Create a custom colormap with Red-Green-Blue transition
    return mcolors.LinearSegmentedColormap.from_list("RedGreenBlue", [red, green, blue])


delay_column = 'Delay'
thre_column = 'Threshold'
fp_column = "Flip Percentage ($\epsilon$)"
classes_column = "Classes ($M$)"
features_column = "Input Dimensions ($d$)"
imb_column = 'Class Imbalance ($r$)'
noise_column = "Noise Level ($\epsilon$)"
gen_type_column = 'Generation Type'
detection_technique = "Detection Technique"

noises = [0.0, 0.5, 1.0]
title_noises = {}
for fp in noises:
    if fp == 1.0:
        title_noises[fp] = f"Non-vulnerable $\epsilon={fp}$"
    else:
        title_noises[fp] = f"Vulnerable $\epsilon={fp}$"

imbalances = [0.1, 0.3, 0.5]
titles_imbalances = {}
for imb in imbalances:
    if imb == 0.5:
        titles_imbalances[imb] = "Balanced"
    else:
        titles_imbalances[imb] = f"Class Imbalance\n$r= {imb}$"
thresholds = [1, 5, 7]
titles_thresholds = {}
for threshold in thresholds:
    titles_thresholds[threshold] = fr'ILD with Rejection Threshold = {threshold}, $\tau \geq {threshold}$'
delay_value = 25
titles_delays = {'less': fr'Time Delay $\leq$ ${delay_value}$ $\mu$-seconds',
                 'more': fr'Time Delay $\geq$ ${delay_value}$ $\mu$-seconds'}
real_parameters = {thre_column: titles_thresholds, imb_column: titles_imbalances, delay_column: titles_delays,
                   noise_column: title_noises}

MAE = "Mean Absolute Error"
MSE = "Mean Squared Error"
NMAE = "Normalized Mean Absolute Error"
NMSE = "Normalized Mean Squared Error"

columns_dict = {
    MID_POINT_MI_ESTIMATION.lower(): 'Mid-Point',
    LOG_LOSS_MI_ESTIMATION.lower(): 'Log-Loss',
    'cal_log-loss': "Cal Log-Loss",
    LOG_LOSS_MI_ESTIMATION_PLATT_SCALING.lower(): 'PS Cal Log-Loss',
    LOG_LOSS_MI_ESTIMATION_ISOTONIC_REGRESSION.lower(): 'IR Cal Log-Loss',
    LOG_LOSS_MI_ESTIMATION_BETA_CALIBRATION.lower(): 'Beta Cal Log-Loss',
    LOG_LOSS_MI_ESTIMATION_TEMPERATURE_SCALING.lower(): 'TS Cal Log-Loss',
    PC_SOFTMAX_MI_ESTIMATION_HISTOGRAM_BINNING.lower(): 'HB Cal Log-Loss',
    'paired-t-test': 'PTT-Majority',
    'paired-t-test-random': 'PTT-Random',
    'fishers-exact-median': 'FET-Mean',
    'fishers-exact-mean': 'FET-Median'
}
detection_methods = {
    'paired-t-test-random': 'PTT-Random',
    'paired-t-test': 'PTT-Majority',
    'fishers-exact-mean': 'FET-Mean',
    'fishers-exact-median': 'FET-Median',
    'estimated_mutual_information': 'Direct MI Estimation',
    'mid_point_mi': 'Mid-Point',
    'log_loss_mi': 'Log-Loss',
    'log_loss_mi_isotonic_regression': 'IR Cal Log-Loss',
    'log_loss_mi_platt_scaling': 'PS Cal Log-Loss',
    'log_loss_mi_beta_calibration': 'Beta Cal Log-Loss',
    'log_loss_mi_temperature_scaling': 'TS Cal Log-Loss',
    'log_loss_mi_histogram_binning': 'HB Cal Log-Loss',
    'p_c_softmax_mi': 'PCSoftmaxMI'
}
ptt_mv = detection_methods['paired-t-test']
ptt_rg = detection_methods['paired-t-test-random']
fet_mean = detection_methods['fishers-exact-mean']

learner_dict = {
    MULTI_LAYER_PERCEPTRON: "MLP",
    AUTO_GLUON: "AutoGluon",
    AUTO_GLUON_STACK: "AutoGluonStack",
    RANDOM_FOREST: "RF",
    TABPFN: "TabPFN_S",
    TABPFN_VAR: "TabPFN",
    GMM_MI_ESTIMATOR: "GMM \\textsc{Baseline}",
    MINE_MI_ESTIMATOR: "MINE \\textsc{Baseline}",
    MINE_MI_ESTIMATOR_HPO: "MINE \\textsc{Baseline}",
    PC_SOFTMAX_MI_ESTIMATION: "\\textsc{PC-Softmax Baseline}"
}


def convert_xlabels(name):
    if "baseline" not in name.lower():
        name = "\\textsc{{{}}}".format(name)
    return name


def transform_dict(learner_dict):
    for k in learner_dict.keys():
        learner_dict[k] = convert_xlabels(learner_dict[k])
    return learner_dict


# learner_dict = transform_dict(learner_dict)
detection_methods = transform_dict(detection_methods)
columns_dict = transform_dict(columns_dict)

dataset_dict = {
    SYNTHETIC_DATASET: "MVN Perturbation Dataset",
    SYNTHETIC_DISTANCE_DATASET: "MVN Proximity Dataset",
    SYNTHETIC_IMBALANCED_DATASET: "MVN Perturbation Imbalanced Dataset",
    SYNTHETIC_DISTANCE_IMBALANCED_DATASET: "MVN Proximity Imbalanced Dataset"
}

generation_methods = {
    'balanced': 'Balanced',
    'binary': 'Binary-class Imbalanced',
    'multiple': 'Multi-class Imbalanced',  # Majority
    'single': 'Multi-class Imbalanced'  # Minority
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
                 # '#C49C94', '#8C564B', '#8C564B',
                 # Shades of brown
                 '#ff7f0e', '#ffbb78', '#8c564b', '#c49c94',
                 # Shades of orange and blue

                 '#006D2C', '#006D2C', '#2CA02C', '#98DF8A', '#98DF8A', '#98DF8A', '#98DF8A', '#98DF8A', '#98DF8A',
                 # Shades of green
                 # '#FFBB78', '#FF7F0E', '#FF7F0E',
                 # Shades of orange
                 '#9467bd', '#c5b0d5', '#1f77b4', '#aec7e8',
                 # Shades of orange and blue

                 '#08519C', '#08519C', '#1F77B4', '#AEC7E8', '#AEC7E8', '#AEC7E8', '#AEC7E8', '#AEC7E8', '#AEC7E8',
                 # Shades of blue
                 '#9467BD', '#9467BD', '#C5B0D5', '#C5B0D5',
                 # Shades of purple
                 '#000000', '#7F7F7F', '#C7C7C7'
                 # Shades of gray
                 ]
star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
cut_star = mpath.Path(
    vertices=np.concatenate([circle.vertices, star.vertices[::-1, ...]]),
    codes=np.concatenate([circle.codes, star.codes]))
markers = ['o', 'o', 's', 'H', 'H', 'H', 'H', 'H', 'H', '8', '8', 'p', 'p',
           'x', 'x', '+', 'X', 'X', 'X', 'X', 'X', 'X', star, star, cut_star, cut_star,
           'v', 'v', '^', '<', '<', '<', '<', '<', '<', '>', '>', '1', '1',
           '<', 'v', '^']


def camel_to_words(camel_case_string):
    # Use regular expression to split camelCase
    words = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', camel_case_string)
    # Join the words with spaces
    return ' '.join(words)


# cols.sort()
def create_custom_order():
    ls = [learner_dict[AUTO_GLUON], learner_dict[TABPFN_VAR], learner_dict[MULTI_LAYER_PERCEPTRON]]
    cols = list(columns_dict.values())
    custom_order = []
    for learner in ls:
        if '\\textsc{Baseline}' in learner:
            continue
        custom_order.append(learner)
        for col in cols:
            learner_name = f"{learner} {col}"
            custom_order.append(learner_name)
    custom_order = custom_order + [learner_dict[GMM_MI_ESTIMATOR], learner_dict[MINE_MI_ESTIMATOR],
                                   learner_dict[PC_SOFTMAX_MI_ESTIMATION]]
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


def get_max_mi_value(n_classes, gen_type, imbalance):
    # print("******************************************************")
    # print(n_classes, gen_type, imbalance)
    if gen_type == 'balanced':
        max_value = np.log2(n_classes)
    else:
        class_distribution = {str(i): 500 for i in range(n_classes)}
        if gen_type == 'single' or gen_type == 'binary':
            class_distribution = generate_samples_per_class(n_classes, 500, imbalance, 'single',
                                                            None, verbose=0)
        elif gen_type == 'multiple':
            class_distribution = generate_samples_per_class(n_classes, 500, imbalance, 'multiple',
                                                            None, verbose=0)
        # print(class_distribution)
        counts = np.array(list(class_distribution.values()))
        total_instances = np.sum(counts)

        probabilities = counts / total_instances
        max_value = -np.sum(probabilities * np.log2(probabilities))
        if max_value < 1.0:
            max_value = 1.0
    # print(max_value, np.log2(n_classes))
    # max_value = np.log2(n_classes)
    return max_value


def get_values(y_true, y_pred, time, max_value, n_classes):
    y_true = clean_array(y_true)
    y_pred = clean_array(y_pred)
    mae = np.around(np.nanmean(np.abs(y_true - y_pred)), 8)
    mse = np.around(np.nanmean((y_true - y_pred) ** 2), 8)
    time = np.mean(clean_array(time))
    time = np.around(time, 4)
    y_true_norm = y_true / max_value
    y_pred_norm = y_pred / max_value
    nmae = np.around(np.nanmean(np.abs(y_true_norm - y_pred_norm)), 8)
    if nmae > 1.0:
        nmae = nmae / np.log2(n_classes)
    nmse = np.around(np.nanmean((y_true_norm - y_pred_norm) ** 2), 8)
    return mae, mse, nmae, nmse, time


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
    combined_results['learner'] = combined_results['learner'].replace(TABPFN, TABPFN_VAR)
    return combined_results


def create_combined_synthetic_dataset():
    combined_results = get_synthetic_dataset_results()
    columns_new = ["Dataset", 'Learner', fp_column, "Distance", noise_column, classes_column, features_column,
                   gen_type_column, imb_column, MAE, MSE, NMAE, NMSE, "Time"]
    data = []
    for dataset, dataset_df in combined_results.groupby('dataset'):
        dataset_name = dataset_dict[dataset]
        group = ['flip_y', 'noise', 'n_classes', 'n_features', 'gen_type', 'imbalance']
        for (values), filter_df in dataset_df.groupby(group):
            flip_y, noise, n_classes, n_featrues, gen_type, imbalance = values
            max_value = get_max_mi_value(n_classes, gen_type, imbalance)
            gen_type = generation_methods[gen_type]
            noise = np.round(noise, 1)
            for (learner), learner_df in filter_df.groupby('learner'):
                y_true = np.array(learner_df['mcmcbayesmi'].values)
                time = learner_df['evaluation_time'].values
                learners = [AUTO_GLUON, TABPFN_VAR]  # + [MULTI_LAYER_PERCEPTRON]
                if learner in learners:
                    for column in columns_dict.keys():
                        if column in list(learner_df.columns):
                            y_pred = learner_df[column].values
                            # if np.any(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred)):
                            # print(dataset, values, learner, column)
                            learner_name = f"{learner_dict[learner]} {columns_dict[column]}"
                            # print(n_classes, gen_type, imbalance, max_value)
                            # print(y_true, y_pred)
                            mae, mse, nmae, nmse, time = get_values(y_true, y_pred, time, max_value, n_classes)
                            one_row = [dataset_name, learner_name, flip_y, 1 - noise, noise, n_classes, n_featrues,
                                       gen_type, imbalance, mae, mse, nmae, nmse, time]
                            data.append(one_row)
                if learner == MULTI_LAYER_PERCEPTRON:
                    y_pred = learner_df['pcsoftmaxmi'].values
                    learner_name = learner_dict[PC_SOFTMAX_MI_ESTIMATION]
                    # if np.any(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred)):
                    # print(dataset, values, learner, column)
                    mae, mse, nmae, nmse, time = get_values(y_true, y_pred, time, max_value, n_classes)
                    one_row = [dataset_name, learner_name, flip_y, 1 - noise, noise, n_classes, n_featrues, gen_type,
                               imbalance, mae, mse, nmae, nmse, time]
                    data.append(one_row)
                if "mi_estimator" in learner:
                    y_pred = learner_df['estimatedmutualinformation'].values
                    # if np.any(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred)):
                    # print(dataset, values, learner, column)
                    learner_name = learner_dict[learner]
                    mae, mse, nmae, nmse, time = get_values(y_true, y_pred, time, max_value, n_classes)
                    one_row = [dataset_name, learner_name, flip_y, 1 - noise, noise, n_classes, n_featrues, gen_type,
                               imbalance, mae, mse, nmae, nmse, time]
                    data.append(one_row)
    df = pd.DataFrame(data, columns=columns_new)
    df = sort_dataframe(df)
    # print(f"Learners {df['Learner'].unique()}")
    # print(f"Generation Types {df['Generation Type'].unique()}")
    return df


def get_real_dataset_results(table_name):
    connection = psycopg2.connect(**connect_params)
    sql_query = f"SELECT * from results.{table_name}"
    df = pd.read_sql(sql_query, connection)
    numeric_df = df.select_dtypes(include=[np.number])
    if (np.isinf(numeric_df).any(axis=1)).any():
        df = df.replace([float('inf'), -float('inf')], 0)

    return df


def create_combined_real_dataset(table_name, filter_results=True):
    df = get_real_dataset_results(table_name)
    if df['base_detector'].str.contains(TABPFN).any() and df['base_detector'].str.contains(TABPFN_VAR).any():
        df['base_detector'] = df['base_detector'].replace(TABPFN, TABPFN_VAR)
    if filter_results:
        condition1 = df['base_detector'].isin([RANDOM_FOREST, AUTO_GLUON_STACK])
        # condition2 = (df['detection_method'] == 'fishers-exact-mean')
        # condition2 = (df['detection_method'] == 'paired-t-test-random')
        df = df[~(condition1)]
        # df = df[~(condition1 | condition2)]
    condition = (df['base_detector'] == MULTI_LAYER_PERCEPTRON) & (df['detection_method'] != 'p_c_softmax_mi')
    combined_results = df[~condition]
    threshold_condition = "n_hypothesis_threshold" in combined_results.columns

    if 'server' in combined_results.columns:
        columns_new = ["Server", 'Base Learner', "Detection Method", "Detection Technique", imb_column, "Dataset",
                       "Threshold", "Time"]
        first_column = 'server'
    else:
        columns_new = ["Dataset", 'Base Learner', "Detection Method", "Detection Technique", imb_column, "Delay",
                       "Threshold", "Time"]
        first_column = 'delay'
    if threshold_condition:
        group = ['n_hypothesis_threshold', first_column, 'imbalance', 'base_detector', 'detection_method']
    else:
        group = [first_column, 'imbalance', 'base_detector', 'detection_method']

    for col in ild_metrics:
        columns_new.extend([col, col + '-Std'])
    columns_new.extend(["BalancedErrorRate", "BalancedErrorRate" + '-Std'])
    data = []
    for (values), dataset_df in combined_results.groupby(group):
        if threshold_condition:
            n_hypothesis_threshold, fs_value, imbalance, base_detector, detection_method = values
        else:
            fs_value, imbalance, base_detector, detection_method = values
            n_hypothesis_threshold = 1
        if first_column == 'delay':
            one_row = [f"Timing {fs_value} micro-seconds", learner_dict[base_detector], detection_methods[detection_method],
                       f"{learner_dict[base_detector]} {detection_methods[detection_method]}",
                       imbalance, int(fs_value), n_hypothesis_threshold]
        else:
            server = re.sub(r'-\d{4}-\d{2}-\d{2}', '', str(fs_value))
            if "open" not in server.lower():
                server = server.title()
                if "Net" in server:
                    server = "NetScalerGCM"
            one_row = [server, learner_dict[base_detector], detection_methods[detection_method],
                       f"{learner_dict[base_detector]} {detection_methods[detection_method]}",
                       imbalance, fs_value, n_hypothesis_threshold]
        if detection_method == 'p_c_softmax_mi':
            if learner_dict[base_detector] == learner_dict[MULTI_LAYER_PERCEPTRON]:
                one_row[1] = learner_dict[PC_SOFTMAX_MI_ESTIMATION]
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
        mean = np.mean(1 - dataset_df[INFORMEDNESS.lower()].values)
        std = np.std(1 - dataset_df[INFORMEDNESS.lower()].values)
        one_row.extend([mean, std])
        data.append(one_row)

    final_df = pd.DataFrame(data, columns=columns_new)
    if filter_results:
        custom_order = [learner_dict[AUTO_GLUON], learner_dict[TABPFN_VAR], learner_dict[GMM_MI_ESTIMATOR],
                        learner_dict[MINE_MI_ESTIMATOR], learner_dict[PC_SOFTMAX_MI_ESTIMATION]]
    else:
        custom_order = [learner_dict[RANDOM_FOREST], learner_dict[AUTO_GLUON_STACK], learner_dict[AUTO_GLUON],
                        learner_dict[TABPFN], learner_dict[TABPFN_VAR], learner_dict[GMM_MI_ESTIMATOR],
                        learner_dict[MINE_MI_ESTIMATOR], learner_dict[PC_SOFTMAX_MI_ESTIMATION]]
    detectors_order = [detection_methods['mid_point_mi'], detection_methods['log_loss_mi'],
                       detection_methods['log_loss_mi_isotonic_regression'],
                       detection_methods['log_loss_mi_platt_scaling'],
                       detection_methods['log_loss_mi_beta_calibration'],
                       detection_methods['log_loss_mi_temperature_scaling'],
                       detection_methods['log_loss_mi_histogram_binning'], detection_methods['paired-t-test'],
                       detection_methods['paired-t-test-random'],
                       detection_methods['fishers-exact-mean'], detection_methods['fishers-exact-median'],
                       detection_methods['estimated_mutual_information']]
    techniques_order = [f"{learner} {col}" for learner in custom_order if '\\textsc{Baseline}' not in learner for col in
                        detectors_order]
    techniques_order += [learner_dict[GMM_MI_ESTIMATOR], learner_dict[MINE_MI_ESTIMATOR],
                         learner_dict[PC_SOFTMAX_MI_ESTIMATION]]
    final_df['Detection Technique'] = pd.Categorical(final_df['Detection Technique'], categories=techniques_order,
                                                     ordered=True)
    if first_column == 'delay':
        final_df.sort_values(['Detection Technique', imb_column, 'Delay'], inplace=True)
    else:
        final_df.sort_values(['Detection Technique', imb_column, 'Server'], inplace=True)
    final_df['Detection Technique'] = final_df['Detection Technique'].cat.remove_unused_categories()

    return final_df


def filter_real_dataset(real_df, imbalances, best_learners_balanced, best_learners_imbalanced, remove_ptt_r=True,
                        remove_ptt_mv=False, remove_fet=False, logger=None, verbose=0):
    dfs = []
    detection_column = "Detection Technique"
    for imbalance in imbalances:
        if verbose:
            logger.info(f"************ Imbalance {imbalance} ************")
        cat_df = real_df[real_df[imb_column] == imbalance]
        result = cat_df.groupby([detection_column])[ACCURACY].agg(['mean', 'std']).reset_index()
        result_df = pd.DataFrame(result)
        filter_learners = []
        learners = [learner_dict[AUTO_GLUON], learner_dict[TABPFN_VAR]]  # + ["MLP"]
        for learner in learners:
            sub_df = result_df[result_df[detection_column].str.contains(learner)]
            sub_df = sub_df[sub_df[detection_column].str.contains("Cal Log-Loss")]
            if verbose:
                logger.info(f"Cal Log-Loss entries \n {sub_df.to_string(index=False)}")
            # Get the learner with the minimum mean absolute error
            min_error_learner = sub_df.loc[sub_df['mean'].idxmax()]
            logger.info(min_error_learner[detection_column])
            # Check if there are duplications based on mean error
            duplicated_learners = sub_df.loc[sub_df['mean'].duplicated()]
            # If there are duplications, choose the learner with the lower standard deviation (std)
            # Check if any duplicated entry has a mean greater than the max value
            duplicated_learners = duplicated_learners[duplicated_learners['mean'] > sub_df['mean'].max()]

            if not duplicated_learners.empty:
                logger.info(duplicated_learners[detection_column])
                dup = duplicated_learners.sort_values([detection_column, 'std'], ascending=[False, True])
                chosen_learner = dup.iloc[0]
            else:
                chosen_learner = min_error_learner
            if verbose:
                logger.info(f"Chosen Detector Best on Leakage dataset {chosen_learner[detection_column]}")
        techniques = list(result_df[detection_column].unique())
        filter_learners = [technique for technique in techniques if "Cal Log-Loss" not in str(technique)]
        if imbalance == 0.5:
            filter_learners.extend(best_learners_balanced)
        else:
            filter_learners.extend(best_learners_imbalanced)
        filter_df = cat_df[cat_df[detection_column].isin(filter_learners)]
        filter_df[detection_column] = filter_df[detection_column].astype("category")
        filter_df[detection_column] = filter_df[detection_column].cat.remove_unused_categories()
        if verbose:
            logger.info(f"Chosen Detectors {list(filter_df[detection_column].unique())}")
        dfs.append(filter_df)
    result_df = pd.concat(dfs, axis=0)
    # print(result_df[detection_column].unique())
    if remove_ptt_mv:
        result_df = result_df[~result_df[detection_column].str.contains(ptt_mv)]
    if remove_ptt_r:
        result_df = result_df[~result_df[detection_column].str.contains(ptt_rg)]
    if remove_fet:
        result_df = result_df[~result_df[detection_column].str.contains(fet_mean)]
    return result_df


filter_cases = ['best_of_ll', 'best_of_all', 'best_of_cal_ll']


def filter_best_results(cat_df, filter_case, logger, verbose, remove_cal=False):
    pd.set_option('display.float_format', '{:.20f}'.format)
    result = cat_df.groupby(['Learner'])[NMAE].agg(['mean', 'std']).reset_index()
    result_df = pd.DataFrame(result)
    filter_learners = []
    learners = [learner_dict[AUTO_GLUON], learner_dict[TABPFN_VAR]]  # + ["MLP"]
    for learner in learners:
        sub_df = result_df[result_df['Learner'].str.contains(learner)]
        if verbose:
            logger.info(f"{learner} entries \n {sub_df.to_string(index=False)}")
        if filter_case == 'best_of_ll':
            sub_df = sub_df[sub_df['Learner'].str.contains("Log-Loss")]
            filter_learners.append(f"{learner} {columns_dict['midpointmi']}")
            strings_to_remove = ['IR ', 'Beta ', 'TS ', 'HB ', 'PS ', 'Cal ']
            strings_to_remove = []
        if filter_case == 'best_of_cal_ll':
            sub_df = sub_df[sub_df['Learner'].str.contains("Cal Log-Loss")]
            filter_learners.append(f"{learner} {columns_dict['midpointmi']}")
            filter_learners.append(f"{learner} {columns_dict['loglossmi']}")
            if remove_cal:
                strings_to_remove = ['IR ', 'Beta ', 'TS ', 'HB ', 'PS ']
            else:
                strings_to_remove = []
        if filter_case == 'best_of_all':
            strings_to_remove = []
        # logger.info(f"Sub \n {sub_df.to_string(index=False)}")
        # Get the learner with the minimum mean absolute error
        min_error_learner = sub_df.loc[sub_df['mean'].idxmin()]

        # Check if there are duplications based on mean error
        duplicated_learners = sub_df[sub_df['mean'] == sub_df['mean'].min()]
        if len(duplicated_learners) == 1:
            duplicated_learners = sub_df[sub_df['mean'] < sub_df['mean'].min()]
        if not duplicated_learners.empty:
            dup = duplicated_learners.sort_values(['std'], ascending=[True])
            if verbose:
                logger.info(f"Duplicate Entries Sorted \n {dup.to_string(index=False)}")
            chosen_learner = dup.iloc[0]
        else:
            chosen_learner = min_error_learner
        # Print the learner with the minimum mean absolute error
        # print("Learner with the minimum mean absolute error:", min_error_learner)
        filter_learners.append(chosen_learner['Learner'])
    filter_learners = filter_learners + [learner_dict[GMM_MI_ESTIMATOR], learner_dict[MINE_MI_ESTIMATOR],
                                         learner_dict[PC_SOFTMAX_MI_ESTIMATION]]
    # print(f"Best Learners {filter_learners}")
    cat_df = cat_df[cat_df['Learner'].isin(filter_learners)]
    cat_df['Learner'] = cat_df['Learner'].astype("category")

    # Remove unused categories from the 'Learner' column
    cat_df['Learner'] = cat_df['Learner'].cat.remove_unused_categories()
    if verbose:
        logger.info(f"Best Learners {cat_df['Learner'].unique()}")
    if filter_case in ['best_of_ll', 'best_of_cal_ll']:
        # Remove the strings from the column
        cat_df['Learner'] = cat_df['Learner'].str.replace('|'.join(strings_to_remove), '', regex=True)
    else:
        for learner in learners:
            cat_df['Learner'] = np.where(cat_df['Learner'].str.contains(learner), learner, cat_df['Learner'])
    cat_df = sort_dataframe(cat_df)
    if verbose:
        logger.info(f"Best Learners Renamed {cat_df['Learner'].unique()}")
    return cat_df


def get_reduced_dataframe(df, datasets=[], filter_case='best_of_all', only_binary=False, logger=None, verbose=0):
    ds = [dataset_dict[d] for d in datasets]
    dataset_df = df[df['Dataset'].isin(ds)]
    if filter_case is not None:
        categories = list(dataset_df['Generation Type'].unique())
        dfs = []
        for category in categories:
            cat = category.replace("\n", ' ')
            if verbose:
                logger.info(f"************ Category {cat} ************")
            cat_df = dataset_df[dataset_df['Generation Type'] == category]
            if category == generation_methods['balanced'] and only_binary:
                cat_df = cat_df[cat_df[classes_column] == 2]
            filter_df = filter_best_results(cat_df, filter_case, logger, verbose)
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


def setup_logging(log_path=None, level=logging.INFO):
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "logs", "logs.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    # Create and configure the logger
    logging.basicConfig(filename=log_path, level=level,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logger = logging.getLogger("SetupLogging")  # root logger
    logger.info("log file path: {}".format(log_path))
    return logger
