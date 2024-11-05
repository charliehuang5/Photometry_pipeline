import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import seaborn as sns
import math
import scipy.signal
import data_analysis_helperfuncs.behav_data_analysis as bd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import dlc_helper as dh

# from scipy.ndimage import maximum_filter1d
# from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d

## Visualizations

#main one
def plot_single_speed_trial(df, key_list, clus_bound, title='', padding=1000, legend_flag = True):
    dfdata = df[key_list].to_numpy()
    plt.figure(figsize=(10,6))
    lineObjects = plt.plot(dfdata)
    if legend_flag:
        plt.legend(iter(lineObjects), key_list)
    plt.xlim([clus_bound[0] - padding, clus_bound[1] + padding])
    plt.axvline(clus_bound[0], c='0.1')
    plt.axvline(clus_bound[1], c='0.1')
    plt.title(title)

# temp
def plot_single_speed_trial_andderiv(df, key_list, clus_bound, title='', padding=1000, legend_flag=True):
    dfdata = df[key_list].to_numpy()
    plt.figure(figsize=(10,6))
    plt.plot(dfdata)
    plt.xlim([clus_bound[0] - padding, clus_bound[1] + padding])
    plt.axvline(clus_bound[0], c='0.1')
    plt.axvline(clus_bound[1], c='0.1')

    plt.figure(figsize=(10, 6))
    dfdata_deriv = np.diff(dfdata, axis=0)
    plt.plot(dfdata_deriv)
    plt.xlim([clus_bound[0] - padding, clus_bound[1] + padding])
    plt.axvline(clus_bound[0], c='0.1')
    plt.axvline(clus_bound[1], c='0.1')
    plt.title(title + 'deriv, bounds: ' + str(clus_bound))

def plot_single_speed_trial_vec(vec, label, clus_bound, title='', padding=1000):
    plt.figure(figsize=(10,6))
    plt.plot(vec, label=label)
    plt.xlim([clus_bound[0] - padding, clus_bound[1] + padding])
    plt.axvline(clus_bound[0], c='0.1')
    plt.axvline(clus_bound[1], c='0.1')
    plt.legend()
    plt.title(title + ', bounds: '+ str(clus_bound))
## Computations

# new March 29 2024

# def gen_radians_mat()

# function: plots and computes trial bounds based on speed thresholding, uses low pass filtering and
#   user defined parameters for min trial length and continuity of above threshold speed
# spec_dic should be formatted:
# spec_dic = {"thresh" : float, "filter_param" : int, "order": int, ""clus_max_interval": int, "clus_min_range": int}
# filter_param is sigma for gaussian filter (which it is currently using)
# filter_param is window size for other filters (which is not currently written in)
def compute_threshed_disp(displac_vec, lookrange, spec_dic, plot=True):
    func_bool = set(["thresh", "filter_param", "order", "clus_max_interval", "clus_min_range"]) == set(spec_dic.keys())
    assert func_bool, "lack of full set of keys - spec_dic"
    # plot1: original and filtered
    trans_points, deriv, filtered_rads = derivative_filter(displac_vec, spec_dic["thresh"],
                                                           filter_param=spec_dic["filter_param"],
                                                           order=spec_dic["order"])
    assert len(trans_points) > 0, "no points from velocity filter, check your threshold"

    # plot2
    if plot:
        plt.figure(figsize=(15, 8))
        plt.plot(deriv)
        plt.axhline(spec_dic['thresh'], c='r')

        # plot3 original and transition points
        plt.figure(figsize=(20, 8))
        plt.plot(displac_vec)
        plt.xlim(lookrange)
    clus_bounds = find_clusters(trans_points, spec_dic["clus_max_interval"], spec_dic["clus_min_range"])
    print("num of trans_points: ", len(trans_points))

    if plot:
        plt.plot(filtered_rads)
        for tpoint in trans_points:
            plt.axvline(tpoint, c='r', alpha=0.01)
        for bound_pair in clus_bounds:
            if not isinstance(bound_pair[0], int):
                print(bound_pair)
                print(type(bound_pair[0]))

            plt.axvline(bound_pair[0], c='g')
        plt.xlim(lookrange)
    # plt.ylim([0,5])
    return trans_points, clus_bounds


# disp_vec is some displacement vector
def derivative_filter(disp_vec, vel_thresh, filter_param=80, order=1):
    plt.figure(figsize=(15, 8))
    plt.plot(disp_vec, label='og')
    # plt.figure()
    filtered = gaussian_filter1d(disp_vec, filter_param)
    plt.plot(filtered, label='filtered')
    plt.legend()
    plt.title("original and filtered")

    deriv = np.abs(np.diff(filtered, n=order))

    return np.where(deriv > vel_thresh)[0], deriv, filtered


# Used because there will be a bunch of points/derivatives above a threshold so we want to find the
# beginning and end of these cluster
# max_spacing - maximum distance between neighboring points
# min_distance - min distance for end-beginning of a cluster we consider (so we don't add too tiny clusters)
def find_clusters(vec, max_spacing, min_distance):
    clusters = []
    current_cluster = [vec[0]]
    for i in range(1, len(vec)):
        if vec[i] - vec[i - 1] <= max_spacing:  # a tpoint must be less than max_difference away from the previous point
            current_cluster.append(vec[i])
        else:
            if current_cluster[-1] - current_cluster[0] >= min_distance:  # so the bounds have to be greater than min
                odd_offset = int((current_cluster[-1] - current_cluster[0]) % 2)
                clusters.append([int(current_cluster[0]), int(current_cluster[-1])-odd_offset])
            current_cluster = [vec[i]] # if dist is greater than max_spacing, we reset current_cluster
                                       # with its starting elem being vec[i]
    if current_cluster[-1] - current_cluster[0] >= min_distance:
        odd_offset = int((current_cluster[-1] - current_cluster[0]) % 2)
        clusters.append([int(current_cluster[0]), int(current_cluster[-1])-odd_offset])  # Add the last cluster

    return clusters


