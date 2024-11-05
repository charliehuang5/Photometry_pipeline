import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statistics_helper as sh
import copy

def bootstrap_hist(control_arr, exp_group, title=''):
    """
    Summary
    -----
    plotting function for bootstrapping data.
    :param control_arr: dimension : Num iterations x num mice <- this is the bootstrap data
    :param exp_group: vector of length num mice <- this is the experimental data (10 scalars for 10 mice)
    """
    arr_means = np.mean(control_arr, axis=1)
    plt.figure(figsize=(20,15))
    plt.title(title)
    plt.hist(arr_means, bins=100)
    exp_mean = np.mean(exp_group)
    plt.axvline(exp_mean, c='r')


""" 
BARCHARTS AND PARSING
"""

def plot_barchart(group1, group2, group_names, title='', label=['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K'], ylim=None):
    """
    Plots barcharts and also prints out t statistic/p val from paired two sided t test (ommiting nan values)
    - also prints out (group1 variance)/(group2 variance) (should be ideally be between 0.25 and 4)

    :param group1: list containing scalar values
    :param group2: list containing scalar values
    :param group_names: names of the groups you are using barchart to compare
    :param label: default contains the 10 mice, change this if plotting different set of mice
    :param ylim: ylimit
    :return:
    """
    two_groups = np.array([group1, group2])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(group_names, two_groups, '-o', label=label)
    plt.bar(group_names, np.mean(two_groups, axis=1), color='0.3', alpha=0.4)  # alpha is transparency
    plt.legend()
    plt.title(title)

    print('\n2 tailed paired t stat: rew vs unrew')
    print('variance ratio: ', np.nanvar(two_groups[0, :]) / np.nanvar(two_groups[1, :]))
    ttest_res = scipy.stats.ttest_rel(two_groups[0, :], two_groups[1, :], alternative='two-sided', nan_policy='omit')
    print(ttest_res)
    pval = ttest_res[1]
    pval_title = str(round(pval, 5))
    plt.title(title + ' ' + str(pval_title))
    if ylim != None:
        plt.ylim(ylim)
    return fig


def means_per_group(group):
    """
    nanmean (mean and ignore nan) of each list within group. In framework this function is used in following way:
    group a list of lists: outerlist being mice ['A','B','C'...] and inner lists corresponding to
    session averages [day1_avg, day2_avg, ...]

    :param group: list of lists of scalars
    :return: new_metric_list: list of scalars
    """
    print("group is provided as a list of lists, mouse x sess avgs")
    new_metric_list = [np.nanmean(g) for g in group]
    return new_metric_list

def generate_groups_scalar(sess_list, sess_2_metric, key=None):
    """
    Generate mouse grouped list of lists, when the metric is a scalar:
    ie: mouse A: [x0, x1, x2 ...] mouse B: [y0, y1, y2 ...] where x0, x1 are scalars for diff sessions

    :param
    - sess_2_metric - dictionary mapping session name to metric

    :return:
    - mouse_IDs - the corresponding mouse IDs corresponding to grouped_scalars
    - grouped_scalars - list of lists, (grouped by mouse ID)
    """
    metric_list = []
    for sessname in sess_list:
        metric = sess_2_metric[sessname]
        if key is not None: # which means metric is a dictionary
            metric = metric[key] # this is a scalar
        metric_list.append(metric)
    mouse_IDs, grouped_scalars = sh.only_group_by_mice(sess_list, metric_list)
    return mouse_IDs, grouped_scalars

def generate_groups_vector(sess_list, sess_2_metric, key=None):
    """
    Generate mouse grouped list of lists, when the metric is a scalar:
    ie: mouse A: [[x0, x0_1, x0_2, ...], [...], [...]] mouse B: [[y0, y0_1, y0_2, ...], [...], [...]]

    First takes mean per session such that each session has a scalar. Then groups these scalars (one val per session) by
    mouse ID

    :param
    - sess_2_metric - dictionary mapping session name to metric

    :return:
    - mouse_IDs - the corresponding mouse IDs corresponding to grouped_vectormeans
    - grouped_vectormeans - list of lists, (grouped by mouse ID)
    """
    metric_list = []
    for sessname in sess_list:
        metric = sess_2_metric[sessname]
        if key is not None: # which means metric is a dictionary
            metric = metric[key] # this is a vector
        metric_list.append(np.nanmean(metric))  # append the value (averaged over trials)
    mouse_IDs, grouped_vectormeans = sh.only_group_by_mice(sess_list, metric_list)
    return mouse_IDs, grouped_vectormeans

def generate_groups_matrix(sess_list, sess_2_metric, col_ind, key=None):
    """
    Same thing as above, but the metric is a matrix (the column should correspond to region ie: DCN, Thal)
    so there is additional param:
    :param
    - sess_2_metric - dictionary mapping session name to metric
    :param col_ind:
    - index of column of interest (often times I'll save a ..._ref key in output_dic's that saves column labels

    Is more or less the same as generate_groups_vector but first accessing a column vector from the metric matrix

    :return:
    - mouse_IDs - the corresponding mouse IDs corresponding to grouped_vectormeans
    - grouped_vectormeans - list of lists, (grouped by mouse ID)
    """
    print('generate groups_vector does not nan account')
    metric_list = []
    for sessname in sess_list:
        metric = sess_2_metric[sessname]
        if key is not None: # which means metric is a dictionary
            metric = metric[key] # this is a matric
        metric = metric[:, col_ind] # now it's a vector
        metric_list.append(np.nanmean(metric))  # append the value (averaged over trials)
    mouse_IDs, grouped_vectormeans = sh.only_group_by_mice(sess_list, metric_list)
    return mouse_IDs, grouped_vectormeans



# def generate_metric_dic(sess_list, metric_func, metric_parameters):
#     # Functions below are not being used
#     dic = {}
#     for sessname in sess_list:
#         dic[sessname] = metric_func(metric_parameters)
#     return dic
#
#
# def metric_func(metric_paramters):
#     pass




