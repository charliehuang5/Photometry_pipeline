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
from scipy.ndimage import maximum_filter1d

# NEW 5/17 - for determining reward waves

def determine_rew_wavelist(day_dic, forward_window):
    waves = day_dic['waves']
    behav_mat = day_dic['behav_mat']
    rew = behav_mat[:, 9]
    rew_wavelist = []
    for wave in waves:
        if wave[-2].startswith('unrewarded'):
            continue
        rew_period = rew[wave[0]:wave[0] + forward_window]
        rew_trans = np.where(np.diff(rew_period) == 1)[0]
        assert len(rew_trans) > 0
        rew_diff = rew_trans[0]
        rew_wavelist.append(wave[0] + rew_diff)
    return rew_wavelist


#note that data should be UNFILTERED and it is the RAW manip data
#it includes the transition!!! and is not filtered out by the time (but we have that data in metadata)

#trash: wave_type = bd.determine_waveform_type_ver3(data[ec + i: ec + i + trial_front_half, :])

def wave_form_identifier(data, bounds):
    robo = data[:, 10]
    pairs = []
    zero_to_one = np.where(np.diff(robo) == 1)[0]


    if robo[bounds[0]] == 1:
        zero_to_one = np.hstack([bounds[0], zero_to_one])

    for i, zto in enumerate(zero_to_one):
        # base qualifications
        if not (bounds[0] <= zto < bounds[1] and robo[zto] <= 1):
            continue
        end_index = zto + 1
        while end_index < bounds[1] and robo[end_index] != 3:
            end_index += 1
        one_to_zero = np.where(np.diff(robo[zto:end_index]) == -1)[0]
        if len(one_to_zero) == 1: # there should have only been one transition from 1 to 0
            otz = one_to_zero[0] + zto
            pairs.append([zto, otz])
    return pairs

def number_waveforms_modern(data, metadata, spec_trials=None, sing_trial=None, max_back = 0, trial_front_half=1000, winsize=3, vel_threshes=[0.005,0.005], mini_window=True, plot=False):
    waves = []
    wavedata = []
    type_track = {'rewarded_success': 0, 'unrewarded_success': 0, 'rewarded_failure': 0, 'unrewarded_failure': 0, 'gs_ds': 0, 'double_suc_overlap': 0}
    robo = data[:, 10]
    assert robo.max() == 5, "max robo value should be 5 because data should be unfiltered and including the 'find home' "

    mtp = metadata['manip_trans_points']
    bounds = [mtp[0] + max_back, mtp[1] - trial_front_half] # adds some padding before and after
    pairs_culled = wave_form_identifier(data, bounds)
    print('number of waves: ', len(pairs_culled))
    for count, pair in enumerate(pairs_culled):
        # print('# ', count)
        # print(pair)
        if sing_trial and count != sing_trial:
            continue
        i = pair[0]
        j = pair[1]
        if i > mtp[1] or j > mtp[1]: #if any one of our 1-0 trans points are over the transition point, we exit
            print("We are done")
            break
        if spec_trials is not None:
            if count not in spec_trials:
                continue
        # print('i,j: ', i, j)
        data_subset = data[i:j, :]
        if plot:
            plt.figure()
            plt.title('trial ' + str(count))
            plt.plot(data_subset[:, 0], label='x')
            plt.plot(data_subset[:, 1], label='y')
            plt.plot(data_subset[:, 10], label='robo', c='m')
        ec = det_earliest_cross(data_subset)
        if not ec:
            continue

        print('count: ' + str(count) + ' LOOKA HERE: ' + str(j))
        reward_state = bd.determine_waveform_type_ver3(data[j:j+trial_front_half,:], mode='reward')
        bt_subset = data[i:i + ec+1, :]
        if reward_state == 'rewarded':
            tpoints = backtrack_modern(bt_subset, vel_threshes[0], winsize=winsize)
        elif reward_state == 'unrewarded':
            tpoints = backtrack_modern(bt_subset, vel_threshes[1], winsize=winsize)
        z = ec
        used_cand = False
        if plot:
            plt.plot(ec, 1, 'o', c='tab:brown')
            # for tpoint in tpoints:
            #     plt.axvline(tpoint, c='c', alpha = 0.7)
            # plt.xlim([800,1000])
            # plt.ylim([0,2])
        if len(tpoints) > 0:
            z, used_cand = determine_move_init(tpoints, ec)
        trial_init = z + i

        success_state = bd.determine_waveform_type_ver3(data[trial_init:j], mode='distance')
        wave_type = reward_state + '_' + success_state

        type_track[wave_type] += 1
        waves.append([trial_init, [i,j], wave_type, used_cand])
        wavedata.append([z, i])
        if plot:
            plt.title('trial ' + str(count) + ' ' + str(wave_type) + ' ' + str(used_cand))
            plt.axvline(z, c='tab:red')
            plt.legend()
            if mini_window:
                plt.xlim([z-50, ec + 50]) #Chmod 3/6 - revert this back when done testing
                # plt.xlim([z-200, ec+150]) #revert this back
                pass
    return waves, type_track#, wavedata


def determine_move_init(tpoints, ec, max_stall=150):
    diffs = np.diff(tpoints)
    bool_diff = np.where(diffs > max_stall)[0]
    cand = tpoints[0]
    if len(bool_diff) > 0:
        ind = bool_diff[-1] + 1
        cand = tpoints[ind]
    # print('looka here: ')
    # print('ec ', ec, ', cand ', cand)
    # print('dif: ', ec-cand)
    if ec - cand >= max_stall:
        options = np.where((ec-tpoints) <= max_stall)[0]
        if len(options) > 0:
            return tpoints[options[0]], ' earliest option below max stall from ec'
        return ec, ' set to ec'
    else:
        return cand, ' used first candidate'


def backtrack_modern(data_subset, vel_thresh, winsize=3):
    dist_mat = np.linalg.norm(data_subset[:, :2], axis=1)
    diff_mat = np.diff(dist_mat)
    filtered = maximum_filter1d(diff_mat, size=winsize)
    # plt.plot(filtered, label='filtered')
    # plt.axhline(vel_thresh, c='c', alpha = 0.5)
    tpoints = np.where(filtered > vel_thresh)[0]
    return tpoints

# what is the earliest cross?
# Ans: earliest point in which the DISTANCE is greater than dist_min
# applying secondary restraint of max_stall to account for small fluctuations over the dist_min
def det_earliest_cross(data, dist_min=1.0, max_stall = 150):
    dist_mat = np.linalg.norm(data[:, :2], axis=1)
    cond_mat = dist_mat > dist_min
    cond_mat = cond_mat.astype(int) #so that we don't count negative crosses in the next line
    crosses = np.where(np.diff(cond_mat) == 1)
    # print('printing crosses: ', crosses)
    crosses = crosses[0]
    if len(crosses) == 0:
        return False
    
    between_dists = np.where(np.diff(crosses) > max_stall)[0]
    if len(between_dists) > 0:
        # print('used_max stall in det_earliest_cross')
        ind = between_dists[-1] + 1 #takes the last one with big stall
        return crosses[ind]
    else:
        return crosses[0]
    