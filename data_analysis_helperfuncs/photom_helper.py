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

#relevant = 0,1,6(framecounter), 8(lick),9(reward),10- robot state, 11- cam frame

#new: added on May 30th
def shuffle_cube_trials(cube):
    """
    Used to generate control random cubes
    :param cube: np array (TIME X PARAM X TRIALS)
    :return: shuffled cube: np array where regions are now shuffled relative to each other
    """
    num_trials = cube.shape[2]
    og_trial_inds = np.arange(num_trials)
    new_seq_dcn = np.random.permutation(og_trial_inds)
    new_seq_thal = np.random.permutation(og_trial_inds)
    new_seq_snr = np.random.permutation(og_trial_inds)
    new_dcn_slice = np.expand_dims(cube[:,[1],new_seq_dcn], axis=1)
    new_thal_slice = np.expand_dims(cube[:,[2],new_seq_thal], axis=1)
    new_snr_slice = np.expand_dims(cube[:,[3],new_seq_snr], axis=1)
    old_ch1_slice = cube[:,[0],:]
    shuffled_cube = np.hstack([old_ch1_slice, new_dcn_slice, new_thal_slice, new_snr_slice])
    return shuffled_cube

def convert_to_phot_frames(ind, combin_df):
    refpoint_framecount = int(combin_df.loc[ind]['frame_count_1'])
    return refpoint_framecount

def visualize_single_wave(behav_chunk, col_dic, title='', vline=None):
    plt.figure(figsize=(7,5))
    for reg in col_dic.keys():
        plt.plot(behav_chunk[:,col_dic[reg]], label=reg)
    plt.title(title)
    if vline is not None:
        plt.axvline(vline, alpha=0.5, c='r')
        plt.legend()

def save_cube(cube, subfolder, filename_base, cube_type = 'zscore', output_path = '/Users/charliehuang/Documents/python_work/data/Photometry/Outputs',
              behav = False):
    regions = ['CH1','DCN','Thal','SNr']
    cube_save_folder = output_path + '/Cube_slices'
    assert subfolder[0] == '/', "not proper subfolder"
    if not os.path.isdir(cube_save_folder + subfolder):
        os.mkdir(cube_save_folder + subfolder)
    for i in range(cube.shape[1]):
        reg = regions[i]
        filename = '/'+filename_base + '_' + reg + '_' + cube_type + '.csv'
        if behav:
            filename = '/'+filename_base + '_' + cube_type + '.csv'
        cube_slice = cube[:,i,:]
        save_path = cube_save_folder + subfolder + filename
        np.savetxt(save_path, cube_slice, delimiter=',')

def visualize_master_behavcube(cube, norm_window, frame_rate, time_offset, title= '', save_flag=False, save_path = '', save_title = '',heatmap=True):
    if save_flag:
        plt.ioff()
    mean_trace = np.mean(cube, axis=2)
    plt.figure(figsize=(15,8))
    time = (np.arange(cube.shape[0])/frame_rate) - time_offset
    plt.plot(time, mean_trace)
    plt.title(title)

    pre_move = norm_window[0]/frame_rate - time_offset
    move_init = norm_window[1]/frame_rate - time_offset
    plt.axvline(pre_move, c='r', alpha=0.5)
    plt.axvline(move_init, c='r', alpha=0.8)



def visualize_cube(cube, col_dic, time_offset, spec_params=[], indiv_traces=False, title='',
                   save_flag=False, save_path='', save_title='',
                   heatmap=True,
                   plot_3D=False, xlabel='', ylabel='', ylim=None, norm_window=None, p_BACK_WINDOW=150):
    if save_flag:
        plt.ioff()
    average_square = np.nanmean(cube, axis=2)
    params = list(col_dic.keys())
    avg_df = pd.DataFrame(data=average_square, columns=params)
    sem_square = scipy.stats.sem(cube, axis=2, ddof=0, nan_policy='omit')
    time = (np.arange(cube.shape[0]) / 30) - time_offset
    # plt.figure(figsize=(15, 8))
    half_flag = False
    half_label = ''
    if '-410' in params[1]:
        half_flag = True
        half_label = '_mean-subtracted'
        fig, ax = plt.subplots(2, int(len(params) / 2), figsize=(int(4 * len(params) / 2), 4))
    else:
        fig, ax = plt.subplots(2, int(len(params)), figsize=(5.5 * len(params), 10))
    fig.suptitle(title + half_label)
    for i, col in enumerate(params):
        if len(spec_params) and col not in spec_params:
            continue
        ind = i
        sub = 0

        if half_flag:
            ind = int(i / 2)
            sub = np.mean(avg_df[col])
        ax[0,ind].plot(time, avg_df[col] - sub, label=col)
        ax[0,ind].legend()
        # plt.plot(time, avg_df[col], label=col)
        ax[0,ind].fill_between(time, avg_df[col] - sub - sem_square[:, col_dic[col]],
                             avg_df[col] - sub + sem_square[:, col_dic[col]],
                             alpha=0.5)

        # ax[ind].axhline(0, c='0.3', alpha=0.4)
        if ylim != None:
            ax[0,ind].set_ylim(ylim)
        # else:
        #     maxval = np.max(avg_df)-sub
        #     ax[ind].set_ylim([-maxval, maxval])
        ax[0,ind].axvline(0, c='r')
        # ax[i].legend()
        ax[0,ind].set_title(col + ' ' + str(round(np.mean(avg_df[col]), 4)))
        ax[0,ind].set_xlabel(xlabel)
        ax[0,ind].set_ylabel(ylabel)

        if norm_window != None:
            ax[0,ind].axvline(norm_window[0] / 30 - time_offset, c='r', alpha=0.5)
            ax[0, ind].axvline(norm_window[1] / 30 - time_offset, c='r', alpha=0.5)

    # Heatmaps
    fig.suptitle(title + ' heatmaps')
    if heatmap:
        for i, col in enumerate(params):
            slice_mat = cube[:, i, :]
            if i == len(params) - 1:
                cbar_par = ax[1,-1]
            else:
                cbar_par = False
            if ylim == None: #YlGnBu
                g1 = sns.heatmap(slice_mat.T, cmap="Spectral", cbar=cbar_par, ax=ax[1,i])
            else:
                g1 = sns.heatmap(slice_mat.T, cmap="Spectral", cbar=cbar_par, ax=ax[1,i], vmin=ylim[0], vmax=ylim[1])
            ax[1,i].set_title(col)
            if norm_window != None:
                ax[1,i].vlines(norm_window, *ax[1,i].get_xlim(), colors='r', alpha=0.5)
            # ax[1, i].vlines(p_BACK_WINDOW, *ax[1, i].get_xlim(), colors='r', alpha=1)
            ax[1,i].set_ylabel('trials')
            ax[1,i].set_xlabel('frame')

    if save_flag:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + save_title)
        plt.close(fig)
    else:
        plt.show()
    if plot_3D:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(params[0])
        ax.set_ylabel(params[1])
        ax.set_zlabel(params[2])
        colors = np.arange(average_square.shape[0]) / average_square.shape[0]
        # Store the scatter plot object in a variable
        line = ax.plot(average_square[:, 0], average_square[:, 1], average_square[:, 2],
                       color='0.3', alpha=0.3)  # or any color you like
        scatter = ax.scatter(average_square[:, 0], average_square[:, 1], average_square[:, 2], c=colors, edgecolors='k',
                             cmap='RdBu')
        # Add colorbar, using the scatter plot object as the mappable for the color mapping
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Color scale label')  # You can customize the label here
        plt.title('3D: ' + title)
        plt.show()


