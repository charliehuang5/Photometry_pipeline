import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import cv2
import seaborn as sns
import random

#relevant = 0,1,6(framecounter), 8(lick),9(reward),10- robot state, 11- cam frame

ref_dic = {'frame_count': 6}

#global
no_interpolate = {'nose', 'water_spout', 'light_on', 'tongue'}

def quality_assessment(df, bodyparts, title = '', bpoi = ['r_hand', 'r_elbow', 'mirror_r_hand', 'mirror_r_foot', 'r_foot_tip']):
    bodyparts.sort()
    avg_metrics(df, bodyparts)
    plt.title(title)
    for part in bpoi:
        body_part_visualize(df, part, 'likelihood', threshold=False)


def avg_metrics(df, bodyparts):
    liks, stds = [], []
    plt.figure()
    for i, part in enumerate(bodyparts):
        liks.append(np.mean(df[part + '_likelihood']))
        stds.append(np.std(df[part + '_likelihood']))
    plt.plot(bodyparts, liks, label='mean')
    plt.plot(bodyparts, stds, label='std')
    plt.legend()
    plt.xticks(rotation=45, ha='right')

def body_part_visualize(df, part, metric, threshold = False):
    plt.figure()
    print(np.mean(df[part+'_'+metric]))
    hue = part+'_'+metric
    if threshold:
        hue += '_outliers'
    ax = sns.scatterplot(data = df, x=part+'_x', y=part+'_y', hue=hue)
    plt.plot(np.mean(df[part+'_x']), np.mean(df[part+'_y']), marker = 'v',  markersize=20, c='r')
    plt.title(part + '_' + metric)
    plt.legend(loc=(1.04, 0))
    plt.show()

# filter part 1
# new: added 8/28 Note that dlc_df is not filtered
#new: made into its own function on 8/30 so I can apply the time filter after all is said and done
def filter_df_waves(key, ultra_dic, collist, trial_length = 1000):

    #generate filter
    manip_trans_points = ultra_dic[key]['metadata']['manip_trans_points']
    manip_data = ultra_dic[key]['manip_data']
    combin_df = ultra_dic[key]['combin_df']
    combin_np = combin_df.to_numpy()
    filter = np.zeros([manip_data.shape[0], 1])
    filter[manip_trans_points[0]:manip_trans_points[1]+1,:] = 1

    #filter manip data
    manip_data_filt = np.multiply(manip_data, filter)
    #filter combin_np and gen combin_df
    combin_np_filt = np.multiply(combin_np, filter)
    combin_df_filt = pd.DataFrame(data=combin_np_filt, columns=collist)

    waves = ultra_dic[key]['waves']
    wcube_all = ultra_dic[key]['wcube_all']
    summary = ultra_dic[key]['summary']
    waves_filtered = waves
    wcube_all_filtered = wcube_all
    summary_filtered = summary

    filtered_waves = waves
    first_update_done = False
    for i, wav in enumerate(waves):
        if wav[0] + trial_length >= manip_trans_points[1]:
            if not first_update_done:
                print('getting rid of waves including and after: ' + str(i))
                waves_filtered = waves[0:i] #not including i
                wcube_all_filtered = wcube_all[:,:,:i]
                first_update_done = True
            summary_filtered[wav[2]] -= 1
    return manip_data_filt, combin_df_filt, waves_filtered, wcube_all_filtered, summary_filtered

def load_synced_dfs(manip_path, cam_path):
    dlc_df, bodyparts = gen_dlc_df(cam_path)
    bodyparts.sort()
    # prepare the manipulandum data
    manip_df = pd.read_csv(manip_path, sep='\t', lineterminator='\n', header = None)
    manip_data = manip_df.to_numpy()[1:, :]

    # frame mapping from manipulandum to camera
    # using frame count
    manip_trans = determine_manip_trans(manip_data[:,ref_dic['frame_count']])
    print('manip_trans: ', manip_trans)
    cam_trans = determine_light_trans(dlc_df)
    print('cam_trans: ', cam_trans)
    cam_frames = []
    for i in range(manip_data.shape[0]):
        cam_frames.append(manip_to_cam(i, manip_trans, cam_trans))
    cam_frames = np.array([cam_frames]).T

    #add camera frames to manipulandum data
    manip_data = np.hstack([manip_data, cam_frames])
    print("cam_frames column has been added")
    print("manip_data should shape[1] == 12")
    metadata  = {'manip_trans_points': manip_trans, 'cam_trans_points': cam_trans, 'bodyparts': bodyparts}
    # relevant = 0,1,6(framecounter), 8(lick),9(reward),10- robot state, 11- cam frame

    manip_colnames = ['x','y','_','_','_','_','frame_count_1','frame_count_2', 'lick', 'reward', 'robot_state', 'cam_frame']
    dlcdf_colnames = list(dlc_df.columns)
    dlc_np = dlc_df.to_numpy()
    attach_np = np.zeros([manip_data.shape[0], len(dlcdf_colnames)])
    for i in range(manip_trans[0], manip_trans[1]):
        cam_frame = int(manip_data[i,-1])
        attach_np[i,:] = dlc_np[cam_frame,:]
    combin_np = np.hstack([manip_data, attach_np])

    combin_df = pd.DataFrame(data=combin_np, columns=manip_colnames+dlcdf_colnames)

    col_dic = {col:i for col, i in zip(combin_df.columns,np.arange(combin_df.shape[1]))}


    return combin_df, col_dic, dlc_df, manip_data, metadata

def gen_dlc_df(file_path, interp=True, thresh_nans=0.9, no_interpolate = {'nose', 'water_spout', 'light_on', 'tongue'}):
    df = pd.read_csv(file_path, skiprows=1)
    # return df
    rename_dic = {a: a.split('.')[0] for a in df.columns}
    df = df.rename(columns=rename_dic)
    df.columns = df.columns + '_' + df.iloc[0]
    df = df[1:]
    df.index = df['bodyparts_coords']
    df = df.iloc[:, 1:]
    df.index = df.index.astype('int')
    df = df.astype('float')
    bodyparts = list(set([word[0:word.rfind('_')] for word in df.columns]))
    if thresh_nans is not None:
        for part in bodyparts:
            setnans_lik_outliers(df, part, thresh=thresh_nans)
    if interp:
        interp_bodyparts = [part for part in bodyparts if part not in no_interpolate]
        parts_surrseq, parts_surrseq_interp = interpolate(df, 0.9, interp_bodyparts)

    return df, bodyparts

def setnans_lik_outliers(df, part, thresh=0.9):
    print('num of nans: ' + str(df[df[part + '_likelihood'] <= thresh].shape[0]))
    df.loc[df[part + '_likelihood'] <= thresh,part+ '_x'] = np.nan
    df.loc[df[part + '_likelihood'] <= thresh,part+ '_y'] = np.nan

# apply find_true_sequences_indices to each body part_likelihood with parameter threshold
# note: if index = -1, or arr_len then
def find_true_sequences_indices(arr):
    true_indices = np.where(arr)[0]
    sequences = []
    surr_sequences = []
    current_sequence = []

    for idx in true_indices:
        if len(current_sequence) == 0 or idx == current_sequence[-1] + 1:
            current_sequence.append(idx)
        else:
            surr = [current_sequence[0] - 1, current_sequence[-1] + 1]
            surr_sequences.append(surr)
            sequences.append(tuple(current_sequence))
            current_sequence = [idx]

    if len(current_sequence) > 0:
        surr = [current_sequence[0] - 1, current_sequence[-1] + 1]
        surr_sequences.append(surr)
        sequences.append(tuple(current_sequence))
    return sequences, surr_sequences

# len thresh is high cuz for radians we inteprolate everyting
def interpolate_radians(df, thresh, non_nan_index, len_thresh=1e6):
    parts_surrseq = {}
    parts_surrseq_interp = {}
    part = 'radians'
    truthvals = (df[part + '_likelihood'] <= thresh).to_numpy()
    seqs, surrs = find_true_sequences_indices(truthvals)
    parts_surrseq[part] = surrs
    print(part + ' total number of outlier sequences: ' + str(len(surrs)))
    xy = np.copy(df['radians'].to_numpy())
    num_fixed = 0
    interp_surrs = []
    for idx in surrs:
        outlier_len = idx[1] - idx[0] - 1
        print(outlier_len)
        if outlier_len > len_thresh:
            print("babe what?")
            continue
        interp_surrs.append(idx)
        num_fixed += 1
        if idx[0] < non_nan_index:
            print("EDGE CASE 1")
            meanval = xy[idx[1]]  # if starts off as outlier, we just use the second index
            # IMPORTANT - this is why we don't interpolate with light_on data because it will interpolate everything
        elif idx[1] == len(truthvals):
            print("EDGE CASE 2")
            meanval = xy[idx[0]]  # if outlier goes to the end, we just use the first index
        else:
            meanval = (xy[idx[0]] + xy[idx[1]]) / 2
        xy[idx[0] + 1:idx[1]] = meanval
    parts_surrseq_interp[part] = interp_surrs
    df[part + '_interp'] = xy
    print(part + ' interpolated ' + str(num_fixed))


    return parts_surrseq, parts_surrseq_interp


# edge cases have been coded in
def interpolate(df, thresh, bodyparts, len_thresh=5):
    parts_surrseq = {}
    parts_surrseq_interp = {}
    for part in bodyparts:
        truthvals = (df[part + '_likelihood'] <= thresh).to_numpy()
        seqs, surrs = find_true_sequences_indices(truthvals)
        parts_surrseq[part] = surrs
        print(part + ' total number of outlier sequences: ' + str(len(surrs)))
        xy = df[[part + '_x', part + '_y']].to_numpy()
        num_fixed = 0
        interp_surrs = []
        for idx in surrs:
            outlier_len = idx[1] - idx[0] - 1
            if outlier_len > len_thresh:
                continue
            interp_surrs.append(idx)
            num_fixed += 1
            if idx[0] == 0:
                print("EDGE CASE 1")
                meanval = xy[idx[1], :]  # if starts off as outlier, we just use the second index
                # IMPORTANT - this is why we don't interpolate with light_on data because it will interpolate everything
            elif idx[1] == len(truthvals):
                print("EDGE CASE 2")
                meanval = xy[idx[0], :]  # if outlier goes to the end, we just use the first index
            else:
                meanval = (xy[idx[0], :] + xy[idx[1], :]) / 2
            xy[idx[0] + 1:idx[1], :] = meanval
        parts_surrseq_interp[part] = interp_surrs
        df[part + '_x_interp'] = xy[:, 0]
        df[part + '_y_interp'] = xy[:, 1]
        print(part + ' interpolated ' + str(num_fixed))
    return parts_surrseq, parts_surrseq_interp


def bounds_convert(manip_inds, manip_data):
    dlc_inds = []
    for mi in manip_inds:
        dlc_inds.append(int(manip_data[mi, 11]))
    return dlc_inds

def determine_manip_trans(frame_count, frame_addon = 7): #new - if change frame rate for 2p, then this needs to change
    trans = np.where(np.diff(frame_count) != 0)[0]
    frame_trans = (trans[0] + 1, trans[-1] + 1 + frame_addon)
    #Note 8/23 - the average length of each step in framecounter 1 is 6.67 so we round up the inc
    return frame_trans

def determine_light_trans(df):
    light_binary = df['light_on_likelihood']>0.9
    trans_points = np.where(np.absolute(np.diff(light_binary)) == 1)[0]
    trans_points[0] += 1
    plt.figure()
    plt.plot(df['light_on_likelihood'] > 0.9)
    for t in trans_points:
        plt.axvline(x=t, c='r')
    return trans_points

#xdat is manip frame boundaries
#ydat is camera frame boundaries
def manip_to_cam(xi, xdat, ydat):
    if xi < xdat[0] or xi > xdat[1]:
        return np.nan
    mc_ratio = (ydat[1]-ydat[0])/(xdat[1]-xdat[0])
    yj = ((xi-xdat[0])*mc_ratio) + ydat[0]
    return int(yj)

##video visualization

def visualize_push(video_path, poi, col_dic, wcube_all, trial, wave_dic_gen, zero_point = 1000):
    video = load_video(video_path)
    push_dat = wave_dic_gen[trial]
    print(push_dat)
    load_frame_data(video,int(push_dat[0]), title = push_dat[-1])
    plt.plot(wcube_all[zero_point:push_dat[1], col_dic[poi+'_x_interp'], trial],wcube_all[zero_point:push_dat[1], col_dic[poi+'_y_interp'], trial] ,  'o')


def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    return video

def load_multiple_frames(video, frame_list, df, grand_title = '', keypoints_connect = None):
    ncols = 3
    nrows = math.ceil(len(frame_list)/ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15,5*nrows), squeeze=False)
    fig.suptitle(grand_title)
    for i,frame in enumerate(frame_list):
        row = math.floor(i/ncols)
        col = i % ncols
        load_frame_data_subplot_ver(ax[row][col], video, frame, df, keypoints_connect=keypoints_connect)

#keypoints_connect = [['mirror_r_hand', 'mirror_l_hand'], [['mirror_r_foot', 'mirror_l_foot']]
def load_frame_data_subplot_ver(ax, video, frame_num, df, keypoints_connect = None, bp_lists = None, bpoi = None, metrics = ['zscore_post', 'diff_zscore', 'likelihood']):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()

    ax.set_title('frame: ' +str(frame_num))
    if bp_lists != None:
        for bp_arr in bp_lists:
            ax.plot(bp_arr[:,0], bp_arr[:,1], '-o', c='r', markersize=20)
    ax.imshow(frame, origin='lower')
    if keypoints_connect is None:
        return
    for pair in keypoints_connect:
        points_mat = np.zeros([2, 2])
        for i, keypoint in enumerate(pair):
            keypoint_xy = [keypoint + add_on for add_on in ['_x_interp', '_y_interp']]
            point = df[[keypoint_xy[0], keypoint_xy[1]]].loc[frame_num].to_numpy()
            points_mat[i, :] = point
            ax.scatter(point[0], point[1], label=keypoint, s = 50)
        # print(points_mat)
        ax.plot(points_mat[:, 0], points_mat[:, 1], '-')
    # ax.legend(loc='upper right') 1/6/24 gotta add htis back baby

#keypoints_connect = [['mirror_r_hand', 'mirror_l_hand'], [['mirror_r_foot', 'mirror_l_foot']]
def load_frame_data(video, frame_num, df, keypoints_connect = None, title = '', bp_lists = None, bpoi = None, metrics = ['zscore_post', 'diff_zscore', 'likelihood']):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    plt.figure(figsize=(10,10))
    plt.title(title + ' fnum: '+str(frame_num))
    if bp_lists != None:
        for bp_arr in bp_lists:
            plt.plot(bp_arr[:,0], bp_arr[:,1], '-o', c='r', markersize=20)
    plt.imshow(frame, origin='lower')
    if keypoints_connect is None:
        return
    for pair in keypoints_connect:
        points_mat = np.zeros([2, 2])
        for i, keypoint in enumerate(pair):
            keypoint_xy = [keypoint + add_on for add_on in ['_x_interp', '_y_interp']]
            point = df[[keypoint_xy[0], keypoint_xy[1]]].loc[frame_num].to_numpy()
            points_mat[i, :] = point
            plt.scatter(point[0], point[1], label=keypoint, s = 100)
        # print(points_mat)
        plt.plot(points_mat[:, 0], points_mat[:, 1], '-')
    plt.legend(loc='upper right')


def gen_dlc_cube(manip_cube, dlc_df):

    frames = manip_cube[:,11,:]
    print(frames.shape)
    # for i in :
    cam_bounds = frames[[0,-1],:] # dimensions: 2 x Trials
    diffs = np.diff(cam_bounds, axis=0).tolist()[0]
    max_diff = max(diffs)
    print("max diff ", max_diff, " mode: ", max(set(diffs), key=diffs.count))
    count = 0
    df_cols = list(dlc_df.columns)
    numbas = np.arange(len(df_cols))
    col_dic = {col:i for col, i in zip(df_cols,numbas)}


    arrs = []
    for i in range(cam_bounds.shape[1]):
        if cam_bounds[1,i]-cam_bounds[0,i] < max_diff:
            print("adjusting from "+ str(cam_bounds[1,i] ) + " to " + str(cam_bounds[0,i]+max_diff))
            cam_bounds[1,i] = cam_bounds[0,i]+max_diff
        print(cam_bounds[0,i], cam_bounds[1,i])
        df_subset = dlc_df.iloc[int(cam_bounds[0,i]):int(cam_bounds[1,i])][list(dlc_df.columns)]
        arrs.append(df_subset.to_numpy())
    return np.dstack(arrs), col_dic

