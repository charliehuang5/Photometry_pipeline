import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import seaborn as sns
import math
import scipy.signal
import cube_helper as ch
import behav_helper

### UNIVERSALS
base_path1 = 'test_behav_outputs/'
base_path2 = base_path1 + 'output_analysis/'

dic = {0: 'x', 1: 'y', 8: 'lick', 9: 'reward', 10: 'robot state'}
relev_params = list(dic.keys())
TEST_BACKTRACK = False

###

### TRAJECTORY analysis
def determine_reach_vector(wave):
    #the trajectory endpoint we will use to define vectors will be the "rest-point" corresponding to right before robot state
    #goes from 0 to 3:
    robo = wave[:, 10]
    trans_point = np.argwhere(robo == 3)[0] - 1

    return wave[trans_point, [0, 1]], trans_point
#a is desired vector
#max_err is magnitude of vector from a to max
def calc_max_theta(max_err, a=7):
    return 2*np.arcsin(0.5*max_err/a)
def calc_max_theta_2(max_err, a=7):
    print(max_err, a)
    print('angle: ', np.arctan(max_err/a))
    return np.arctan(max_err/a)
    # return 2*np.arcsin(0.5*max_err/np.linalg.norm(a))
def calc_angle(a, b):
    num = np.dot(a, b)
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return np.arccos(num/denom)
def rotation_matrix(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,-sin],[sin,cos]])
def generate_circle(radius, stepsize=0.1, center_x=0, center_y=0):
    x = np.arange(center_x, center_x + radius + stepsize, stepsize)
    y = np.sqrt(radius ** 2 - x ** 2)
    x = np.concatenate([x, -x[::-1],x, -x[::-1]])
    y = np.concatenate([y, y[::-1],-y, -y[::-1]])
    return x, y


### ALIGNMENT HELPER FUNCTIONS
def save_autobounded_file(file,path,skip_start=0, end=0):
    #just to show before and after
    visualize_behavior(file, base_path=path, plotname='original')
    bounded_data, bounds, _, _ = visualize_behavior(file, base_path=path, end=end, auto_bound=True, plotname='bounded',skip_start=skip_start)
    print(bounded_data.shape)
    save_name = str(bounds[0])+'_'+str(bounds[1])+'_'+file
    print(save_name)
    print(path + save_name)
    np.savetxt(path+save_name, bounded_data, delimiter='\t')

#WORKING ASSUMPTION: file is already autobounded
#camera frames should be 1 indexed
def calculate_offset(x1,x2,file,path,wave_alignment=[0,1],rounding=True):
    wave1, wave2 = visualize_behavior(file, base_path=path, mark_waves=True, wave_alignment=wave_alignment)
    y1 = wave1[0]
    y2 = wave2[0]
    print("vals for offset")
    print(y2, y1, y2-y1)
    print(x2, x1, x2-x1)
    r = (y2-y1)/(x2-x1)

    offset = r*x1-y1
    if rounding:
        offset = round(offset)
    print('OFFSET', offset, 'FRAME RATIO', r)
    return offset, r #r maps cam to behav

def calculate_offset_ca(ca_len,x0,xfinal,rounding=True):
    r = (xfinal-x0)/(ca_len) #r maps ca to cam
    offset = x0/r
    if rounding:
        offset = round(offset)
    return offset, r

def map_ca2cam(frame, r_ca, ca_offset, inverse=False, rounding=True):
    output = (frame+ca_offset)*r_ca
    if inverse:
        output = frame/r_ca - ca_offset
    if rounding:
        return round(output)
    else:
        return output

def map_cam2behav(frame, r_behav, behav_offset, inverse=False, rounding=True):
    output = frame*r_behav-behav_offset
    if inverse:
        output = (frame+behav_offset)/r_behav
    if rounding:
        return round(output)
    else:
        return output


def full_ca_map(calc_frame, ca_map_data, behav_map_data):
    cam_frame = map_ca2cam(calc_frame, ca_map_data[0], ca_map_data[1], rounding=False)
    behav_frame = map_cam2behav(cam_frame, behav_map_data[0], behav_map_data[1], rounding=True)
    return cam_frame, behav_frame

def add_ghost_rows(data,num_rows,file,path):
    ghosts = np.zeros([num_rows,11])
    new_np = np.vstack([ghosts,data])
    np.savetxt(path+'ghost_'+file, new_np, delimiter='\t')

#params= [[b_f_rate, b_offset], [c_f_rate, c_offset]]

def report_diff(uni_dir_window, example_wav, params):
    wav = example_wav[:2]
    center = wav[0]
    ex_point = [center, center + uni_dir_window]
    frames = []
    for point in ex_point:
        cam_frame = map_cam2behav(point, params[0][0], params[0][1], inverse=True, rounding=False)
        ca_frame = map_ca2cam(cam_frame, params[1][0], params[1][1], inverse=True, rounding=False)
        frames.append(ca_frame)
    diff = frames[1]-frames[0]
    return diff, round(diff), frames[0], round(frames[0])

def manip_cube_helper(wavs, trial_type='reward'):
    wav_dic = {}
    if trial_type == 'reward':
        wav_dic['success'] = []
        wav_dic['fail'] = []
        for wav in wavs:
            if wav[2] == 'success':
                wav_dic['success'].append(wav)
            elif wav[2] == 'unrewarded_failure':
                wav_dic['fail'].append(wav)
        return wav_dic
    else:
        print('trial type not supported yet')


#NEW - added 8/23
def gen_manip_cube(data, wavs, back_window=0, forward_window=0):
    manip_cube = np.zeros([forward_window + back_window, data.shape[1], len(wavs)])
    for i in range(len(wavs)):
        wav = wavs[i][:2]
        bounds = [wav[0] - back_window, wav[0] + forward_window]
        manip_cube[:,:,i] = data[bounds[0]:bounds[1], :]
    print('DIMENSIONS:  TIME x PARAM x TRIAL')
    print(manip_cube.shape)
    return manip_cube


#if waves_keep is not None, then waves_skip should be empty
def vis_waves_ca(data, wavs, ca_data, params, back_window=0, forward_window=0, rounding=True, composite_matrix=False, wavs_skip ={}, wavs_keep = None, height_heatmap = 3, save_wave = None, cmap_ran = [-0.15,0.15], date=''):
    start_flag = False
    comp_mat = None

    num_behav_params = 4
    comp_xy_mat = np.zeros([len(wavs), forward_window+back_window, num_behav_params])

    print(len(wavs))
    counter = 0
    for i in range(len(wavs)):
        if wavs_keep is not None:
            if i not in wavs_keep:
                continue
        if i in wavs_skip:
            continue

        _,back_dif,_,center_ca=report_diff(back_window, wavs[i], params)
        _,for_dif,_,_ = report_diff(forward_window, wavs[i], params)
        ca_bounds = [center_ca-back_dif,center_ca+for_dif]

        wav = wavs[i][:2]

        behav_bounds = [wav[0]-back_window, wav[0]+forward_window]
        x = np.arange(behav_bounds[1] - behav_bounds[0])

        if save_wave is not None: #This allows us to save specific trials with also more behavior visualization

            if counter not in set(save_wave):
                counter += 1
                continue
            else:
                fig, ax = plt.subplots(4, 2, gridspec_kw={'width_ratios': [20, 1]}, figsize=(25, 30))
                ax[0, 1].remove()
                ax[2, 1].remove()
                ax[3, 1].remove()
                ax[0, 0].margins(x=0)
                temp = ca_data[:, ca_bounds[0]:ca_bounds[1]]
                ca_data_subset = copy.deepcopy(temp)
                ca_data_subset = ca_data_subset.astype('float')
                pre_mov_norm = 50
                pre_mov_avgs = np.mean(ca_data_subset[:, :pre_mov_norm], axis=1)
                for i in range(len(pre_mov_avgs)):
                    ca_data_subset[i, :] = (ca_data_subset[i, :] - pre_mov_avgs[i]) / pre_mov_avgs[i]
                mean_trace = np.mean(ca_data_subset, axis=0)
                ax[0, 0].plot(np.arange(len(mean_trace)), mean_trace)
                sns.heatmap(ca_data_subset, cmap='coolwarm', ax=ax[1, 0], cbar_ax=ax[1, 1], vmin=cmap_ran[0], vmax=cmap_ran[1])  # , yticklabels = cell_labels)

                behav_x = data[behav_bounds[0]:behav_bounds[1], 0]
                behav_y = data[behav_bounds[0]:behav_bounds[1], 1]
                rew = data[behav_bounds[0]:behav_bounds[1], 9]
                robo = data[behav_bounds[0]:behav_bounds[1], 10]

                zeez = np.sqrt(np.square(behav_x) + np.square(behav_y))
                print('zeez shape: ', zeez.shape)
                vel = np.gradient(zeez)
                print(vel.shape)
                robo_filter = copy.deepcopy(robo)
                robo_filter[robo_filter > 1] = 0
                filtered_vel = np.multiply(vel, robo_filter)
                # ax[2, 0].set_ylim([-0.1, 1])
                ax[2, 0].margins(x=0)
                ax[2, 0].plot(filtered_vel, c='g', alpha=0.7)# filtered velocity
                # reward
                ax[2, 0].scatter(np.arange(len(rew)), np.zeros(len(rew)), 80, c='r', alpha=rew)

                ax[3, 0].axis('equal')

                ax[3, 0].set_xlim([-10, 10])
                ax[3, 0].set_ylim([-0.5, 7])
                circ_x, circ_y = generate_circle(5)
                ax[3, 0].plot(circ_x, circ_y, c='y')
                xpush = np.multiply(behav_x, robo_filter)
                ypush = np.multiply(behav_y, robo_filter)
                ax[3, 0].plot(xpush, ypush, marker='o', linestyle='None', alpha=0.5)
                basepath = 'C:/Users/MotionTracking/Documents/Charlie/behav_analysis/images/'
                plt.savefig(basepath+date+'_'+str(counter)+'.pdf')

        else:
            fig, ax = plt.subplots(4, 2,
                                   gridspec_kw={'height_ratios': [1, 1, height_heatmap, 1], 'width_ratios': [100, 5]},
                                   figsize=(15, 8))

            fig.suptitle('Wave #' + str(counter), fontsize=16)


            ax[0, 1].remove()  # remove unused upper right axes
            ax[1, 1].remove()

            behav_x = data[behav_bounds[0]:behav_bounds[1], 0]
            behav_y = data[behav_bounds[0]:behav_bounds[1], 1]
            comp_xy_mat[i,:,0] = behav_x
            comp_xy_mat[i,:,1] = behav_y
            comp_xy_mat[i,:,2] = data[behav_bounds[0]:behav_bounds[1], 9] #reward data for this window
            comp_xy_mat[i,:,3] = data[behav_bounds[0]:behav_bounds[1], 10] #robot state

            ax[0, 0].plot(x, behav_x)
            ax[0, 0].plot(x, behav_y)
            ax[0, 0].plot(x, data[behav_bounds[0]:behav_bounds[1], 9], c='r')
            ax[0, 0].margins(x=0)
            ax[1, 0].margins(x=0)

            # if i < 10:
            #     print(comp_behav_mat[i,:])
            ca_data_subset = ca_data[:, ca_bounds[0]:ca_bounds[1]]

            mean_trace = np.mean(ca_data_subset, axis=0)

            ax[1, 0].plot(np.arange(len(mean_trace)), mean_trace)

            sns.heatmap(ca_data_subset, cmap='mako', ax=ax[2, 0], cbar_ax=ax[2, 1])#, yticklabels = cell_labels)
            if composite_matrix:
                if not start_flag:
                    # print('subset shape: ', ca_data_subset.shape)
                    comp_mat = ca_data_subset
                    start_flag = True
                else:
                    # print('comp shape: ', comp_mat.shape)
                    # print('subset shape: ', ca_data_subset.shape)
                    comp_mat = np.dstack((comp_mat,ca_data_subset))
        counter += 1
    print(comp_xy_mat.shape)
    # comp_xy_mat = comp_xy_mat[~np.all(comp_xy_mat == 0, axis=1)]
    if wavs_skip is not None:
        print("shape before: ", comp_xy_mat.shape)
        comp_xy_mat = np.delete(comp_xy_mat, list(wavs_skip), axis=0)
        print("shape after: ", comp_xy_mat.shape)
    comp_behav = np.sqrt(np.square(comp_xy_mat[:,:,0])+np.square(comp_xy_mat[:,:,1]))
    return comp_mat, comp_behav, comp_xy_mat

#pre_mov_norm either None or a number (the calcium frame I'm upperbounding for
#pre_mov_norm defines an upper limit for pre-movement which we average our vals to be delt f/f
def plot_avg_percell(comp_mat, comp_behav, cell_labels, pre_mov_norm = None, plot_uni_average = False, cmap_range = None, sort=False):
    zmeans = np.mean(comp_mat, axis=2)
    ca_xrange = np.arange(comp_mat.shape[1])

    if sort:
        fig, ax = plt.subplots(4, 2, gridspec_kw={'width_ratios': [100, 5]}, figsize=(10, 15))
    else:
        fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [100, 5]}, figsize=(10, 15))
    for i in range(comp_behav.shape[0]):
        ax[0, 0].plot(comp_behav[i, :], alpha=0.3)
    ax[0, 0].plot(np.mean(comp_behav, axis=0), linewidth=3, c='k')
    # np.mean(comp_behav, axis = 0)
    ax[1, 1].remove()
    ax[0, 1].remove()
    ax[0, 0].margins(x=0)
    # print(ax[1,0].Axis.get_data_interval())
    ax[1, 0].margins(x=0)
    shifted_pos = ca_xrange + 0.5
    #just to be sure
    zmeans = zmeans.astype('float')
    if pre_mov_norm is not None:
        pre_mov_avgs = np.mean(zmeans[:,:pre_mov_norm], axis=1)
        for i in range(len(pre_mov_avgs)):
            zmeans[i,:] = (zmeans[i,:]-pre_mov_avgs[i])/pre_mov_avgs[i]
    alph = 1
    if plot_uni_average:
        ax[1, 0].plot(shifted_pos,np.mean(zmeans, axis=0), linewidth=3, c='k')
        alph = 0.15
    for i in range(zmeans.shape[0]):
        ax[1, 0].plot(shifted_pos,zmeans[i, :], alpha=alph)

    ax[1, 0].set_xlim([0,comp_mat.shape[1]])
    ax[1, 0].set_xticks(shifted_pos)
    ax[1, 0].set_xticklabels(ca_xrange)
    cmap_reversed = matplotlib.cm.get_cmap('coolwarm')

    if cmap_range is not None:
        sns.heatmap(zmeans, cmap=cmap_reversed, ax=ax[2, 0], cbar_ax=ax[2, 1], vmin=cmap_range[0], vmax=cmap_range[1])#, yticklabels=cell_labels)
    else:
        sns.heatmap(zmeans, cmap=cmap_reversed, ax=ax[2, 0], cbar_ax=ax[2, 1])#, yticklabels=cell_labels)
    if sort:
        permut = zmeans[np.argsort(np.argmax(zmeans, axis=1)), :]
        if cmap_range is not None:
            sns.heatmap(permut, cmap=cmap_reversed, ax=ax[3, 0], cbar_ax=ax[3, 1], vmin=cmap_range[0], vmax=cmap_range[1])#, yticklabels=cell_labels)
        else:
            sns.heatmap(permut, cmap=cmap_reversed, ax=ax[3, 0], cbar_ax=ax[3, 1])#, yticklabels=cell_labels) #just for code completion <- we shouldn't be using this line anyway
    #for now we will not care about permuting the cell_labels
    ## creates z-averaged and pre-movement normalized dataframe using zmeans and cell_labels
    df = pd.DataFrame(zmeans, index=cell_labels)
    return df

### GENERAL

# this function is now archaic. number_waveforms_modern from cube_helper is used outside directly in the notebook


#usage notes:
# - autobound should usually be true unless you are specifying manually start and end (ie: in single waveform analysis)
#parameter combos
#   - if u want to do wave by wave analysis, run vis behav w conserve indices as TRUE and markwaves. Then with the outputted list of waves, do bd.vis behav with start and end specified from that list
#trajec_analyze is either None or [ideal vector, max_error, rewarded_dist]
#ca_package = [ca_data, forward_mapping_params] in which forward_mapping_params =

def visualize_behavior(filename, start = 0, end = 0, scatter = False, params = [0,1], base_path = base_path2, conserve_indices = True, auto_bound = False, plotname = 'behavior', mark_waves = False, percent_show = 1, display = True,
                       trajec_analyze=None, wave_by_wave=False, only_bounds=False, wave_alignment=None, ca_package=None, fixed_waves=False, dist_min = 1.0, manual_overide_window = None, skip_start = 0, manip_trans_point = None):
    dic = {0: 'x', 1: 'y', 8: 'lick', 9: 'reward', 10: 'robot state'}
    print(base_path + filename)
    df = pd.read_csv(base_path + filename, sep='\t', lineterminator='\n', header = None)
    data = df.to_numpy()
    data = data[1:, :]
    if data is not None: #we can supply the data in directly and over-write it
        data = data
    end_bound = data.shape[0]
    if auto_bound:
        bounds = find_bounds(data[skip_start:, :])
        print("BOUNDS: ", bounds)
        start = bounds[0] + skip_start
        end_bound = bounds[1]
    if end != 0:
        end_bound = end
        #essentially if we specify end, then it is the auto_bound or our specification (higher priority)
        #otherwise the end_bound will be the default length of data
    if percent_show != 1:
        end_bound = int(percent_show * (end_bound-start) + start)
        print("new end_bound:", end_bound)
    x_indices = np.arange(start, end_bound)
    data = data[start:end_bound,:]
    if only_bounds:
        print('ONLY BOUNDS')
        return [start, end_bound], data
    waves = None
    wave_count = None
    if mark_waves:
        # waves, wave_count = number_waveforms(data, , params=params, fixed_waves=fixed_waves, dist_min=dist_min)
        #new 9/6
        waves, wave_count = ch.number_waveforms_modern(data)

        #commented out on 9/6
        # waves, wave_count = number_waveforms_simplified(data, dist_min=dist_min, plot=wave_by_wave)
        if wave_alignment is not None:
            # print('wav alignment', wave_alignment[0],wave_alignment[1])
            # print(waves[wave_alignment[0]],waves[wave_alignment[1]])
            return waves[wave_alignment[0]],waves[wave_alignment[1]]
    if display:
        plt.figure(figsize=[12, 6])
        for i in params:
            lab = i
            if i in dic.keys():
                lab = dic[i]
            if conserve_indices:
                if manual_overide_window is not None: # note 4/17 - mark waves should be False if manual_overide_window is on
                    plt.plot(x_indices[manual_overide_window[0]:manual_overide_window[1]], data[manual_overide_window[0]:manual_overide_window[1], i], label=lab)
                else:
                    plt.plot(x_indices, data[:, i], label=lab)
            else:
                if manual_overide_window is not None:
                    plt.plot(data[manual_overide_window[0]:manual_overide_window[1], i], label=lab)
                else:
                    plt.plot(data[:, i], label=lab)
        if mark_waves:
            for j in range(len(waves)):
                if not conserve_indices: # this is needed because number_waveforms(data) does not know the original indexing
                    plt.axvline(waves[j][0], color='r')
                    plt.axvline(waves[j][1], color='b')
                else:
                    plt.axvline(waves[j][0] + start, color='r')
                    plt.axvline(waves[j][1] + start, color='b')
        plt.legend()
        plt.title(plotname)
        if scatter and not mark_waves:
            print("need markwaves in order to plot scatter")
        if scatter and mark_waves:
            plt.figure()
            for k in range(len(waves)):
                plt.scatter(data[waves[k][0]:waves[k][1], 0], data[waves[k][0]:waves[k][1], 1], label=k)
            plt.legend()
            plt.title('trajectories')
    if conserve_indices and mark_waves:
        for d in range(len(waves)):
            waves[d] = [waves[d][0] + start, waves[d][1] + start, waves[d][2], waves[d][3]]
    # print('WAVECOUNT: ', wave_count)

    return data, [start, end_bound], waves, wave_count #10/24 changed second outputted variable from end to end_bound

# def plot_trajectories:
#     pass

#first tests whether there is a find home at the beginning
def find_bounds(data):
    robot_state = data[:,10]
    indices = [0,0]
    begin_flag = False
    transition_flag = True
    start_point = 0
    if robot_state[0] == 5: #We have a find-home at the beginning
        #skip the first 5000
        start_point = 5001
    for i in range(start_point,len(robot_state)):
        if robot_state[i] == 1 and not begin_flag: #marks the beginning
            indices[0] = i
            begin_flag = True
        if robot_state[i] == 0 and transition_flag:
            indices[1] = i
            transition_flag = False
        if robot_state[i] >= 1:
            transition_flag = True #if we pass a robot state pulse, then we can re-set the end index
    indices[1] = indices[1]-1
    return indices

###WAVE
def number_waveforms_simplified(data, trial_back_half = 1000, trial_front_half = 1000, dist_min=1.0, plot=False):
    robot_state = data[:, 10]
    print('numba waveforms shape: ', data.shape)

    prev_state = robot_state[0]
    waves = []
    type_track = {'success': 0, 'rewarded_failure': 0, 'unrewarded_failure': 0, 'gs_ds': 0, 'double_suc_overlap': 0}
    metadata = {'trial_back_half': trial_back_half, 'trial_front_half': trial_front_half}
    print(metadata)

    i = 0
    trial_count = 0

    while i < len(robot_state):
        # print(i)
        if prev_state == 1 and robot_state[i] == 0: #if currently on 0 and previously was 1
            trial_init, reached_dist_min = backtrack_simplified(data,i-1,dist_min=dist_min)
            if reached_dist_min and (trial_init + trial_front_half < len(robot_state)) and (trial_init-trial_back_half >= 0):
                wave_type = determine_waveform_type_ver3(data[trial_init: trial_init + trial_front_half,:])
                type_track[wave_type] += 1
                waves.append([trial_init, trial_init+trial_front_half, wave_type, ''])
                # i = trial_init + trial_front_half-1
                if plot:
                    dist = np.linalg.norm([data[trial_init, 0], data[trial_init, 1]])
                    ptitle = 'trial ' + str(trial_count) + ' ' + wave_type + ' ' + str(dist)

                    behav_helper.visualize_single_wave(data[trial_init-trial_back_half: trial_init + trial_front_half,:], title=ptitle)
                trial_count += 1
        prev_state = robot_state[i]
        i += 1
    return waves, type_track
    # theoretically dist_min can be 0 or close to 0 for behav data 8/31 onwards cuz I fix the drift


def backtrack_simplified(data, pos, dist_min=1.0, wiggle_room = 0.2, maximum_backtrack = 500):

    reached_dist_min = False
    i = pos
    earliest_one_cross = None
    trialtype = determine_waveform_type_ver3(data[pos:pos+500, :])


    while i >= 0:
        dist = np.linalg.norm([data[i, 0], data[i, 1]])

        #new
        if pos - i > maximum_backtrack:
            break

        if trialtype == 'success':
            if dist <=dist_min:
                reached_dist_min = True
                earliest_one_cross = i
                break
        elif trialtype == 'unrewarded_failure':
            if dist_min-wiggle_room < dist < dist_min+wiggle_room:
                reached_dist_min = True
                earliest_one_cross = i
        if data[i, 10] != 1:  # if while backtracking, we go from waiting for movement back to hold (the previous wave's return), we break
            break
        i -= 1
    return earliest_one_cross, reached_dist_min


# note: Jan 12 - I believe this is archaic now

#returns a list start and stop of the waveform
#[[start,end,type], [c,d]] where the first waveform is [a,b] and second is [c,d]
#type = 'rewarded success', 'unrewarded success', 'rewarded fail', 'unrewarded fail'

##NOTE the forward window is fine but I gotta adjust the backward window to be variable, so it moves back until when the movement was not going on
# change 10/25: now using back_window from the 1 to 0 robostate transition rather than 0 3 transition, changed back_window to 100
def number_waveforms(data, trajec_analyze, wave_by_wave, forward_window = 509, back_window = 250, params=relev_params, max_dist = 8, fixed_waves=False, dist_min=0.2):
    robot_state = data[:,10]
    print('numba waveforms shape: ', data.shape)
    i = 0
    prev_state = robot_state[0]
    waves = []
    type_track = {'success': 0, 'rewarded_failure': 0, 'unrewarded_failure': 0, 'gs_ds':0, 'double_suc_overlap': 0}
    backward_window = 0
    wavecount = 0
    deletedis = 0
    test_thing = False

    # total_geosuc_count = 0

    if wave_by_wave and trajec_analyze is None:
        print('INCOMPATIBLE PARAM COMBINATION')
        return

    if trajec_analyze is not None:
        max_theta = calc_max_theta(trajec_analyze[1])
        max_theta_deg = max_theta * 180/np.pi
    while i < len(robot_state):
        if prev_state == 1 and robot_state[i] == 0:
            if fixed_waves:
                backward_window = i-back_window
            else:
                backward_window = backtrack(data,i,dist_min=dist_min)
            # # print('ENTER BACKTRACK: ', i)
            # backward_window = backtrack(data, i)
            # # print('backward_window: ', backward_window)
            # # print('there')
        if prev_state == 0 and robot_state[i] == 3: #3 marks the go somewhere robot state
            # print('everywhere')
            # if fixed_waves:
            #     backward_window = i - back_window

            if TEST_BACKTRACK: #for testing backtrack_velocity
                if deletedis == 0:
                    print('trying new stuff')
                    deletedis += 1

                backward_window = backtrack_velocity(data,i,backward_window)
            print(backward_window)
            if robot_state[i+forward_window-1] != 1:
                # print('USING FORTRACK', i)
                forward_window = fortrack(robot_state,i)
                # print('forward window ', forward_window)
            waveform_index = [backward_window, i+forward_window]
            # print('wave index!!!!: ', waveform_index)
            if waveform_index[0] < 0:
                waveform_index[0] = 0
            # print(waveform_index)
            #NEW: 8/28 changed this from
            wave_type = determine_waveform_type_ver3(data[waveform_index[0]:waveform_index[1],:])
            geo_descrip = 'not applicable'
            if trajec_analyze is not None:
                vec, trans_point = determine_reach_vector(data[waveform_index[0]:waveform_index[1],:])
                theta = calc_angle(trajec_analyze[0], vec)
                if np.abs(theta) <= max_theta:
                    if np.linalg.norm(vec) >= trajec_analyze[2]:
                        geo_descrip = 'geo_suc-dist_suc'
                        # total_geosuc_count += 1
                        type_track['gs_ds'] += 1
                        if wave_type == 'success':
                            type_track['double_suc_overlap'] += 1
                    else:
                        geo_descrip = 'geo_suc-dist_fail'
                else:
                    if np.linalg.norm(vec) >= trajec_analyze[2]:
                        geo_descrip = 'geo_fail-dist_suc'
                    else:
                        geo_descrip = 'geo_fail-dist_fail'
            if wave_by_wave:
                # TODO: make the scatter plot to scale
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})
                fig.suptitle('Wave #' + str(wavecount) + ' with Max Error Angle: ' + str(round(max_theta_deg, 2)))
                ax1.title.set_text(wave_type)
                ax2.title.set_text(geo_descrip)


                single_wave = ax1.plot(data[waveform_index[0]:waveform_index[1], params]) #assuming we pass in
                ax1.legend(iter(single_wave), [dic[k] for k in params])

                # uncomment this below if wanna work with gradients
                # ax1.plot(np.gradient(data[waveform_index[0]:waveform_index[1], 0]), label='X vel')
                # ax1.plot(np.gradient(data[waveform_index[0]:waveform_index[1], 1]), label='Y vel')

                # ax1.plot(np.gradient(np.gradient(data[waveform_index[0]:waveform_index[1], 0])), label='X accel')
                # ax1.plot(np.gradient(np.gradient(data[waveform_index[0]:waveform_index[1], 1])), label='Y accel')
                # ax1.legend()


                #SCATTER PLOT
                theta_deg = theta*180/np.pi
                ax2.title.set_text(geo_descrip + ' Angle: ' + str(abs(round(theta_deg, 2))) + ' Dist: ' + str(round(np.linalg.norm(vec),2)))

                ax2.scatter(data[waveform_index[0]:waveform_index[0] + trans_point[0], 0], data[waveform_index[0]:waveform_index[0] + trans_point[0], 1])#, c=speeds, cmap='blue')
                ax2.scatter(trajec_analyze[0][0], trajec_analyze[0][1], c='r')

                #scatter: angle bounds
                bound_vec_cloc = rotation_matrix(max_theta)@trajec_analyze[0]
                bound_vec_c_cloc = rotation_matrix(-max_theta)@trajec_analyze[0]
                ax2.plot([0,bound_vec_cloc[0]],[0,bound_vec_cloc[1]], c = 'g')
                ax2.plot([0, bound_vec_c_cloc[0]], [0, bound_vec_c_cloc[1]], c='g')

                #scatter: additional marks
                circle1 = plt.Circle((10, 10), 1, color='b')
                circle1_5 = plt.Circle((10, 10), 1, color='r')
                circle2 = plt.Circle((10,8), 1, color='g')
                if wave_type == 'success':
                    ax2.add_patch(circle1)
                elif wave_type == 'rewarded_failure':
                    ax2.add_patch(circle1_5)
                if geo_descrip == 'geo_suc-dist_suc':
                    ax2.add_patch(circle2)
                x, y = generate_circle(5)
                # x1, y1 = generate_circle(1)
                ax2.plot(x, y, c='y')
                # ax2.plot(x1, y1, c='y')
                bound = max_dist+2
                ax2.set_xlim([-bound, bound])
                ax2.set_ylim([0, bound])
                # ax2.gca().set_aspect('equal', adjustable='box')
                # ax2.set(adjustable='box-forced', aspect='equal')
                # ax2.set_aspect('equal', 'box')

            type_track[wave_type] += 1

            wavecount += 1
            waveform_index.append(wave_type)
            waveform_index.append(geo_descrip)

            waves.append(waveform_index)
            i += forward_window #jump i to forward window
            if i >= len(robot_state):
                break
            prev_state = robot_state[i-1] #adjust previous robot state appropriately
            # print('new i', i, ' new prev state ',prev_state)
            # print(i,prev_state)
        else:
            prev_state = robot_state[i]
            i += 1
    # print('GEOSUC: ', total_geosuc_count)
    return waves, type_track
    # for i in range(len(robot_state)):
    #     if robot_state[i] == 3: #

# theoretically dist_min can be 0 or close to 0 for behav data 8/31 onwards cuz I fix the drift
def backtrack(data, pos, dist_min = 0.2, extra_back = 200):

    wait_state = False
    while pos >= 0:
        # if pos % 20 == 0:
        #     print(pos, data[pos,0], data[pos,1])

        #ORIGINAL
        # if abs(data[pos,0]) < dist_min and abs(data[pos,1]) < dist_min:
        #     break

        #THRESHOLDING USING MAGNITUDE OF DISPLACEMENT
        if np.linalg.norm([data[pos,0],data[pos,1]]) < dist_min:
            break

        if wait_state and data[pos, 10] == 0: #if while backtracking, we go from waiting for movement back to hold (the previous wave's return), we break
            pos += 1
            break
        if data[pos,10] == 1:
            wait_state = True
        pos -= 1
    # if pos - extra_back >= 0:
    #     pos -= extra_back
    return pos

#theoretically dist_min can be 0 or close to 0 for behav data 8/31 onwards cuz I fix the drift
def backtrack_velocity(data, pos, backtrack, dist_min = 0.2, extra_back = 200):
    wait_state = False
    x_vel = np.gradient(data[backtrack:pos,0])
    y_vel = np.gradient(data[backtrack:pos,1])
    speeds = [np.linalg.norm([x, y]) for x,y in zip(x_vel,y_vel)]
    max_speed_pos = np.argmax(speeds)
    if max_speed_pos - 100 >= 0:
        return backtrack + max_speed_pos - 100
    else:
        return backtrack + max_speed_pos

def fortrack(robostate,pos,forward_window=509):
    count = 0
    prev_state = robostate[pos]
    while count < forward_window:
        if prev_state == 0 and robostate[pos+count] == 1:
            break
        prev_state = robostate[pos+count]
        count += 1
    return count


### FOR NOW I'm using the inter-waveform index for reward as 460. This is subject to change when waveforms become variable window size as well as if we decide 460 captures too many
# false positives

#wave is a subset of the data: data[waveform_index[0]:waveform_index[1]
#wave length = data.shape[0]
#changing upper index to a negative from end (because beginning will now be variable)

# if mode is 'reward', our wave goes from j (one to zero) to j+trial front half
# if mode is 'distance', our wave will go from trial_init to j
def determine_waveform_type_ver3(wave, mode='reward', rewarded_length=5):
    if mode not in ['reward','distance']:
        print("dafuq is you doing")
        return None
    rew = wave[:,9]
    robo = wave[:,10]
    dist_mat = np.linalg.norm(wave[:, :2], axis=1)
    if mode == 'reward':
        zero_to_three = np.where(np.diff(robo) == 3)[0][0]
        if rew[:zero_to_three].max() == 1:
            return 'rewarded'
        return 'unrewarded'
    elif mode == 'distance':
        if dist_mat.max() >= rewarded_length:
            return 'success'
        else:
            return 'failure'


def determine_waveform_type_ver2(wave, dist_from_end = 799):
    # success vs fail
    rewcount, rewloc = pulse_characterizer(wave)
    # print('rewcount: ', rewcount)
    wavelength = wave.shape[0]
    if rewcount > 0:
        for loc in rewloc:
            # print(loc)
            if wavelength-loc > dist_from_end:
                return 'success'
        return 'rewarded_failure'
    return 'unrewarded_failure'

def pulse_characterizer(wav):
    rew = wav[:, 9]
    prev_rew = rew[0]
    rew_index = [0, 0]
    rewcount = 0
    locs = []
    for i in range(len(rew)):
        if prev_rew == 0 and rew[i] == 1:
            rewcount += 1
            rew_index[0] = i
            locs.append(i)
            # print('rew loc: ', i)
        elif prev_rew == 1 and rew[i] == 0:
            rew_index[1] = i
        prev_rew = rew[i]
    return rewcount, locs

### PLOTTING






def auto_run_behavis(path, save_data = False, plot_data = False):
    files = os.listdir(path)
    corresp = []
    mc4 = None
    td6 = None
    for file in files:
        parts = file.split('_')
        if len(parts) > 2:
            continue
        elif parts[0] != 'MC4' and parts[0] != 'TD6':
            continue
        print(file)
        count = visualize_behavior(file, base_path=path, params=relev_params, conserve_indices=True, auto_bound=True,
                              display=False, mark_waves=True)[3]
        date = file[9:12]
        corresp.append([file, count])
        nam = file[:3]
        new_df = pd.DataFrame(count, index = [int(date)])
        if nam == 'MC4':
            if mc4 is None:
                mc4 = new_df
            else:
                mc4 = pd.concat([mc4,new_df])
        elif nam == 'TD6':
            if td6 is None:
                td6 = new_df
            else:
                td6 = pd.concat([td6,new_df])
    dfs = [mc4, td6]
    print('made dfs')
    for df,i in zip(dfs,range(len(dfs))):
        df['total'] = df.apply(lambda row: row.success + row.unrewarded_failure + row.rewarded_failure, axis=1)
        df['ratio'] = df.apply(lambda row: row.success/row.total, axis=1)
        dfs[i] = df.sort_index()
    return dfs

def filter(path = '/Volumes/Chonkawonka/outputs/training/'):
    dates_univ_skip = {'0830', '0903', '0904', '0910', '0911', '0912', '0913', '0914'}
    files_ignore = {'MC4': {'0825', '0905'}, 'TD4': {'0825', '0826', '0827', '0828', '0829', '0902'},
                    'TD6': {'0825', '0826', '0827', '0905', '0825'}, 'MC2': {'0906', '0909', '0914'}}
    complet_ignore = {'TD4MC2_20220909.csv', 'MC4_20220906_2.csv', 'TD4_20220901_extra.csv'}
    compar_files = []
    for fil in os.listdir(path):
        if fil in complet_ignore:
            continue
        mouse = fil.split('_')[0]
        date = fil.split('_')[1][4:-4]
        if date in dates_univ_skip:
            continue
        if date in files_ignore[mouse]:
            continue
        compar_files.append(fil)
    compar_files.sort()
    MC2_obs = [2, 2, 5, 2, 1, 0]
    MC4_obs = [0, 9, 1, 0, 7, 8, 1, 0, 0]
    TD4_obs = [1, 37, 0, 0, 3]
    TD6_obs = [3, 7, 0, 2, 5, 0, 1, 0, 0]
    concat = MC2_obs + MC4_obs + TD4_obs + TD6_obs
    return compar_files, [MC4_obs, TD6_obs, MC2_obs, TD4_obs]

def ydn_parser(fil):
    push = None
    vertpush = [0, 8]
    horizpush = [8, 0]
    mouse = fil.split('_')[0]
    date = fil.split('_')[1][4:-4]
    max_err = 5.5
    push = vertpush
    if int(date) < 829:
        max_err = 2.5
    elif int(date) < 831:
        max_err = 3
    elif int(date) < 901:
        max_err = 5
    if int(date) >= 907 or (mouse == 'MC2' and int(date) > 831):
        push = horizpush
        max_err = 5.5
    return push, max_err, date, mouse


def ya_done_now(path='/Volumes/Chonkawonka/outputs/training/'):
    a = filter()
    files = a[0]
    obs = a[1]

    corresp = []
    mc4, mc2, td4, td6 = None, None, None, None
    for file in files:
        push,max_err,date,nam = ydn_parser(file)
        print('FILE: ', file)
        print(nam, date, push, max_err)

        count = visualize_behavior(file, base_path=path, params=relev_params, conserve_indices=True, auto_bound=True,
                              display=False, mark_waves=True, trajec_analyze=[push,max_err,5])[3]
        corresp.append([file, count])
        new_df = pd.DataFrame(count, index = [int(date)])
        if nam == 'MC4':
            if mc4 is None:
                mc4 = new_df
            else:
                mc4 = pd.concat([mc4,new_df])
        elif nam == 'TD6':
            if td6 is None:
                td6 = new_df
            else:
                td6 = pd.concat([td6,new_df])
        elif nam == 'MC2':
            if mc2 is None:
                mc2 = new_df
            else:
                mc2 = pd.concat([mc2,new_df])
        elif nam == 'TD4':
            if td4 is None:
                td4 = new_df
            else:
                td4 = pd.concat([td4,new_df])

    dfs = [mc4, td6, mc2, td4]
    # dfdic = {'mc4': 0, 'td6': 1, 'mc2': 2, 'td4': 3}

    print('made dfs')
    for df,i in zip(dfs,range(len(dfs))):
        df['total'] = df.apply(lambda row: row.success + row.unrewarded_failure + row.rewarded_failure, axis=1)
        df['rew suc/tot'] = df.apply(lambda row: row.success/row.total, axis=1)
        df['geo_suc/tot'] = df.apply(lambda row: row.gs_ds/row.total, axis=1)
        df['geo_suc/rew_suc'] = df.apply(lambda row: row.gs_ds/row.success, axis=1)
        df['overlap/geo_suc'] = df.apply(lambda row: row.double_suc_overlap/row.gs_ds, axis=1)
        df['obs'] = obs[i]
        df['error : geo - recorded'] = df.apply(lambda row: row.gs_ds - row.obs)
        df['error : rew - recorded'] = df.apply(lambda row: row.success - row.obs)

        # double_suc_overlap gs_ds
        dfs[i] = df.sort_index()
        dfs[i]['suc_obs'] = obs[i]


    return dfs




