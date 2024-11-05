import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import copy

# used in big run manip

def only_group_by_mice(session_list, values):
    """
    NOTE - this assumes that sessnames are structured like: RR20240320_G-2024_05_06.pkl
    """

    groups = {}
    for i, sessname in enumerate(session_list):
        key = sessname.split('_')[1][0]  # Extracting the identifying character after the first underscore
        if key not in groups:
            groups[key] = []
        groups[key].append(values[i]) #groups[key] corresponds to a mouse
    group_keys = list(groups.keys())
    grouped_values = list(groups.values())
    return group_keys, grouped_values

# VERY USEFUL
def group_by_mice(session_list, values, mode='scalar'):
    groups = {}
    for i,sessname in enumerate(session_list):
        key = sessname.split('_')[1][0]  # Extracting the identifying character after the first underscore
        if key not in groups:
            groups[key] = []
        groups[key].append(values[i]) #groups[key] corresponds to a mouse
    group_keys = list(groups.keys())
    grouped_values = list(groups.values())
    stacked_vals = [np.vstack(vals) for vals in grouped_values]
    print('stack shape: ', stacked_vals[0].shape)
    means_v2 = [np.squeeze(np.nanmean(stack, axis=0, keepdims=True)) for stack in stacked_vals]
    print('mean elem shape: ', len(means_v2), means_v2[0].shape)
    return group_keys, grouped_values, means_v2

#JUST COPIED IN

def generate_statistics(stats_dic):
    key_labels = []
    n_list = []
    std_list = []
    mean_list = []
    for key in stats_dic:
        param_vals = stats_dic[key]  # vector of mean velocities

        n = len(param_vals)
        std = np.nanstd(param_vals)
        mean = np.nanmean(param_vals)
        key_labels.append(key)
        n_list.append(n)
        std_list.append(std)
        mean_list.append(mean)
    n_list = np.array(n_list)
    std_list = np.array(std_list)
    mean_list = np.array(mean_list)
    sem_list = std_list / (n_list ** 0.5)
    output_dic = {'key_labels': key_labels, 'n_list': n_list,
                  'std_list': std_list, 'mean_list': mean_list,
                  'sem_list': sem_list}
    return output_dic

# stats_dic is specific to a parameter
def plot_stats(stats_dic, errorbars='confidence_interval', title='', ylim=None):
    stats = generate_statistics(stats_dic)
    # Plot scatter here
    plt.figure(figsize=(12, 4))
    plt.bar(stats['key_labels'], stats['mean_list'])

    if errorbars == 'confidence_interval':
        t_crit_vals = []
        for n in stats['n_list']:
            t_crit_vals.append(compute_t_critval(n, margin=0.05, two_tail=True))
        t_crit_vals = np.array(t_crit_vals)
        plt.errorbar(stats['key_labels'], stats['mean_list'], yerr=t_crit_vals * stats['sem_list'], fmt="o", color="r")
    if errorbars == 'sem':
        plt.errorbar(stats['key_labels'], stats['mean_list'], yerr=stats['sem_list'], fmt="o", color="r")
    plt.xticks(rotation=-45)
    plt.title(title + ', error bars show: ' + errorbars)
    if ylim != None:
        plt.ylim(ylim)

## OLDER



def compute_t_critval(sample_size, margin=0.05, two_tail=True):
    print('if two sample t test, provide sample_size = n1+n2-1, this func subtracts one')
    if two_tail:
        t_crit_val = scipy.stats.t.ppf(margin / 2, sample_size - 1)
    else:
        t_crit_val = scipy.stats.t.ppf(margin, sample_size - 1)
    return abs(t_crit_val)

# alternative : {‘two-sided’, ‘less’, ‘greater’},
def compute_t_statistic(data_group1, data_group2, equal_var=True, alternative='two-sided'):
    print('n1: ', len(data_group1), ', n2: ', len(data_group2))
    var1 = np.var(data_group1)
    var2 = np.var(data_group2)
    print('var1: ', var1, 'var2:', var2, 'variance ratio: ', max(var2/var1, var1/var2))
    # Perform the two sample t-test with equal variances
    ttest_res = stats.ttest_ind(a=data_group1, b=data_group2, equal_var=equal_var, alternative=alternative)
    print(ttest_res)
    return ttest_res