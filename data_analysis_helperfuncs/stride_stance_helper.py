import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import scipy

FORWARD_WINDOW = 1000


def interpolate_nan(signal):
    """
    Linearly interpolates NaN values in a 1D numpy array.

    Parameters:
    - signal: numpy 1D array

    Returns:
    - interpolated_signal: numpy 1D array with NaN values replaced by interpolated values
    """
    # Find indices of NaN values
    nan_indices = np.isnan(signal)
    # nancount = np.sum(nan_indices)
    # Create an array of indices for non-NaN values
    non_nan_indices = np.arange(len(signal))[~nan_indices]
    # Interpolate NaN values using the non-NaN values
    interpolated_signal = np.interp(np.arange(len(signal)), non_nan_indices, signal[~nan_indices])
    return interpolated_signal, np.where(nan_indices)[0]

def take_mean_norm_subset(sig, bounds):
    temp = sig[bounds[0]:bounds[1]]
    return temp


def find_ppt_tpt_alt(day_dic, plot=False):  # for assessing nan rates
    dlc_df = day_dic['combin_df']
    cbounds = day_dic['waves']

    rhand, rwrist = dlc_df['r_hand_tip_side_x_interp'].to_numpy(), dlc_df['r_hand_wrist_side_x_interp'].to_numpy()
    rfoot, rankle = dlc_df['r_foot_tip_side_x_interp'].to_numpy(), dlc_df['r_foot_ankle_side_x_interp'].to_numpy()

    prom_val = 15
    cum = []
    for i, cbound in enumerate(cbounds):
        # print(i, cbound)
        mcbound = [cbound[0], cbound[0] + FORWARD_WINDOW]
        rhand_s, rwrist_s = take_mean_norm_subset(rhand, mcbound), take_mean_norm_subset(rwrist, mcbound)
        rfoot_s, rankle_s = take_mean_norm_subset(rfoot, mcbound), take_mean_norm_subset(rankle, mcbound)

        nan_assess = [np.sum(np.isnan(elem)) for elem in [rhand_s, rwrist_s, rfoot_s, rankle_s]]
        print(nan_assess)
        nan_assess = nan_assess / np.array([FORWARD_WINDOW] * 4)
        cum.append(nan_assess)
    return cum


def find_ppt_tpt(day_dic, plot=False):
    print('asdf')
    dlc_df = day_dic['combin_df']
    cbounds = day_dic['waves']

    rhand, rwrist = dlc_df['r_hand_tip_side_x_interp'].to_numpy(), dlc_df['r_hand_wrist_side_x_interp'].to_numpy()
    rfoot, rankle = dlc_df['r_foot_tip_side_x_interp'].to_numpy(), dlc_df['r_foot_ankle_side_x_interp'].to_numpy()

    # bound_oi = [71048, 71098]
    # plt.figure()
    # plt.plot(rfoot[bound_oi[0]: bound_oi[1]])
    # plt.figure()
    # plt.title('rankle')
    # plt.plot(rankle[bound_oi[0]: bound_oi[1]])

    prom_val = 15
    peaks_per_trial_h, troughs_per_trial_h = [], []
    peaks_per_trial_f, troughs_per_trial_f = [], []
    for i, cbound in enumerate(cbounds):
        # print(i, cbound)
        mcbound = [cbound[0], cbound[0] + FORWARD_WINDOW]
        rhand_s, rwrist_s = take_mean_norm_subset(rhand, mcbound), take_mean_norm_subset(rwrist, mcbound)
        rfoot_s, rankle_s = take_mean_norm_subset(rfoot, mcbound), take_mean_norm_subset(rankle, mcbound)

        nan_assess = np.array([np.isnan(elem).all() for elem in [rhand_s, rwrist_s, rfoot_s, rankle_s]])
        # print(i, nan_assess)
        if nan_assess.any():
            hand_peaks, hand_troughs = [], []
            foot_peaks, foot_troughs = [], []
            peaks_per_trial_h.append(hand_peaks)
            troughs_per_trial_h.append(hand_troughs)
            peaks_per_trial_f.append(foot_peaks)
            troughs_per_trial_f.append(foot_troughs)
            continue
        # for elem in [rhand_s, rwrist_s,rfoot_s, rankle_s]:
        #     plt.figure()
        #     plt.title(str(i) + ' ' + str(cbound))
        #     plt.plot(elem)

        nan_rates = np.array(
            [np.sum(np.isnan(elem)) for elem in [rhand_s, rwrist_s, rfoot_s, rankle_s]]) / FORWARD_WINDOW
        if (nan_rates >= 0.7).any():
            print(nan_rates)
            peaks_per_trial_h.append([])
            troughs_per_trial_h.append([])
            peaks_per_trial_f.append([])
            troughs_per_trial_f.append([])
            continue

        hand_avg, hnanc = interpolate_nan((rhand_s + rwrist_s) / 2)
        foot_avg, fnanc = interpolate_nan((rfoot_s + rankle_s) / 2)
        hand_avg = hand_avg - np.mean(hand_avg)
        foot_avg = foot_avg - np.mean(foot_avg)

        hand_peaks = scipy.signal.find_peaks(hand_avg, prominence=prom_val)[0]
        hand_troughs = find_troughs_strides_stances(hand_avg, hand_peaks)
        peaks_per_trial_h.append(hand_peaks)
        troughs_per_trial_h.append(hand_troughs)

        foot_peaks = scipy.signal.find_peaks(foot_avg, prominence=prom_val)[0]
        foot_troughs = find_troughs_strides_stances(foot_avg, foot_peaks)
        peaks_per_trial_f.append(foot_peaks)
        troughs_per_trial_f.append(foot_troughs)

    # return peaks_per_trial, troughs_per_trial
    hand_peaks_troughs = {'hand_peaks': peaks_per_trial_h, 'hand_troughs': troughs_per_trial_h}
    foot_peaks_troughs = {'foot_peaks': peaks_per_trial_f, 'foot_troughs': troughs_per_trial_f}
    return hand_peaks_troughs, foot_peaks_troughs


def find_troughs_strides_stances(signal, peaks):
    troughs_locs = []
    if len(peaks) == 1:
        return troughs_locs
    for i in range(len(peaks) - 1):
        troughs_locs.append(np.argmin(signal[peaks[i]:peaks[i + 1]]) + peaks[i])
    return np.array(troughs_locs)


def stride_or_stance_bounds(ppt, tpt, mode):
    assert len(tpt) == len(ppt)
    bounds_per_trial = []
    bad_trials = []
    for trial_ind in range(len(ppt)):  # TRIALS
        peaks = ppt[trial_ind]
        troughs = tpt[trial_ind]
        if len(peaks) == 0 or len(troughs) == 0:  # either no hand or no foot peaks
            bad_trials.append(True)
            bounds_per_trial.append([])
            continue
        assert len(peaks) - 1 == len(troughs)
        bounds = None
        if mode == 'stride':
            bounds = np.array([peaks[:-1], troughs])
        elif mode == 'stance':
            bounds = np.array([troughs, peaks[1:]])
        bounds_per_trial.append(bounds)
        bad_trials.append(False)
    return bounds_per_trial, bad_trials  # trial X 2 (hand/feet)


def compute_time(bounds, skip_trials):
    times_per_trial = []
    avgs_per_trial = []
    for trial_ind in range(len(bounds)):  # TRIALS
        if skip_trials[trial_ind] == True:
            times_per_trial.append([])
            avgs_per_trial.append(np.nan)
            continue
        times = np.diff(bounds[trial_ind], axis=0)
        times_per_trial.append(times)
        if len(times) == 0:
            print("HERE! 1 ")
        avgs_per_trial.append(np.mean(times))

    return times_per_trial, np.array(avgs_per_trial)


# mode chosen among ['hand','foot','rads']
# displacement of keypoint every stride or stance
def compute_keypoint_displacement(bounds, day_dic, skip_trials, mode='rads'):
    dlc_df = day_dic['combin_df']
    cbounds = day_dic['waves']
    rhand, rwrist = dlc_df['r_hand_tip_side_x_interp'].to_numpy(), dlc_df['r_hand_wrist_side_x_interp'].to_numpy()
    rfoot, rankle = dlc_df['r_foot_tip_side_x_interp'].to_numpy(), dlc_df['r_foot_ankle_side_x_interp'].to_numpy()
    radians = dlc_df['radians_interp'].to_numpy()

    disps_per_trial = []
    sigs_per_trial = []
    avgs_per_trial = []
    for trial_ind in range(len(bounds)):  # TRIALS
        mcbound = [cbounds[trial_ind][0], cbounds[trial_ind][0] + FORWARD_WINDOW]
        if skip_trials[trial_ind] == True:
            disps_per_trial.append([])
            sigs_per_trial.append([])
            avgs_per_trial.append(np.nan)
            continue
        if mode == 'hand':
            rhand_s, rwrist_s = take_mean_norm_subset(rhand, mcbound), take_mean_norm_subset(rwrist, mcbound)
            hand_avg, hnanc = interpolate_nan((rhand_s + rwrist_s) / 2)
            signal = hand_avg - np.mean(hand_avg)
        elif mode == 'foot':
            rfoot_s, rankle_s = take_mean_norm_subset(rfoot, mcbound), take_mean_norm_subset(rankle, mcbound)
            foot_avg, fnanc = interpolate_nan((rfoot_s + rankle_s) / 2)
            signal = foot_avg - np.mean(foot_avg)
        elif mode == 'rads':
            signal = radians[mcbound[0]:mcbound[1]]
        else:
            print('incorrect mode bitch')

        signal_low = signal[bounds[trial_ind][0, :]]
        signal_high = signal[bounds[trial_ind][1, :]]
        # print(signal.shape)
        # print('bound low: ', bounds[trial_ind][0,:], bounds[trial_ind][1,:])
        # print('bound high: ', signal_low, signal_high)
        disp = signal_high - signal_low

        sigs_per_trial.append(signal)
        disps_per_trial.append(disp)
        if len(disp) == 0:
            print("HERE! 2")
        avgs_per_trial.append(np.mean(disp))

    return disps_per_trial, np.array(avgs_per_trial), sigs_per_trial


def compute_rate(disps_per_trial, times_per_trial, skip_trials):
    assert len(times_per_trial) == len(disps_per_trial)  # same num of trials
    rates_per_trial = []
    avgs_per_trial = []
    sums_per_trial = []
    for trial_ind in range(len(times_per_trial)):
        if skip_trials[trial_ind] == True:
            rates_per_trial.append([])
            avgs_per_trial.append(np.nan)
            continue
        disps = disps_per_trial[trial_ind]
        times = times_per_trial[trial_ind]
        rates = disps / times
        rates_per_trial.append(rates)
        if len(rates) == 0:
            print("HERE! 3")
        avgs_per_trial.append(np.mean(rates))
        sums_per_trial.append(np.sum(rates))
    return rates_per_trial, np.array(avgs_per_trial), np.array(sums_per_trial)

def stride_stance_pipeline(sess_cage, ordered_sessions):
    for sessname in ordered_sessions:
        print(sessname)
        session = sess_cage.sessions[sessname]
        day_dic = session['day_dic']
        hand_dic, foot_dic = find_ppt_tpt(day_dic, plot=False)

        sess_cage.sessions[sessname]['day_dic']['hand_peaks_troughs'] = hand_dic
        sess_cage.sessions[sessname]['day_dic']['foot_peaks_troughs'] = foot_dic
        stride_stance_dic = {}
        for dic, keypoint in zip([hand_dic, foot_dic], ['hand', 'foot']):
            # dic = foot_dic
            # keypoint = 'foot'

            ppt = dic[keypoint + '_peaks']
            tpt = dic[keypoint + '_troughs']
            # MOD MARKER
            stride_bounds, skip_trials = stride_or_stance_bounds(ppt, tpt, 'stride')
            stance_bounds, skip_trials = stride_or_stance_bounds(ppt, tpt, 'stance')

            stride_time, stride_time_avgs = compute_time(stride_bounds, skip_trials)
            stance_time, stance_time_avgs = compute_time(stance_bounds, skip_trials)

            stride_disps, stride_disps_avgs, sd_sigs = compute_keypoint_displacement(stride_bounds, day_dic,
                                                                                     skip_trials, mode=keypoint)
            stride_disps_rads, stride_disps_rads_avgs, sd_sigs_rads = compute_keypoint_displacement(stride_bounds,
                                                                                                    day_dic,
                                                                                                    skip_trials,
                                                                                                    mode='rads')
            stance_disps, stance_disps_avgs, st_sigs = compute_keypoint_displacement(stance_bounds, day_dic,
                                                                                     skip_trials, mode=keypoint)
            stance_disps_rads, stance_disps_rads_avgs, st_sigs_rads = compute_keypoint_displacement(stance_bounds,
                                                                                                    day_dic,
                                                                                                    skip_trials,
                                                                                                    mode='rads')

            stride_vel, stride_vel_avgs, stride_vel_sums = compute_rate(stride_disps, stride_time, skip_trials)
            stride_vel_rads, stride_vel_rads_avgs, stride_vel_rads_sums = compute_rate(stride_disps_rads, stride_time,
                                                                                       skip_trials)
            stance_vel, stance_vel_avgs, stance_vel_sums = compute_rate(stance_disps, stance_time, skip_trials)
            stance_vel_rads, stance_vel_rads_avgs, stance_vel_rads_sums = compute_rate(stance_disps_rads, stance_time,
                                                                                       skip_trials)

            stride_params = {'bounds': stride_bounds, 'time': stride_time, 'disps': stride_disps,
                             'disps_rads': stride_disps_rads,
                             'time_avgs': stride_time_avgs, 'disps_avgs': stride_disps_avgs,
                             'disps_rads_avgs': stride_disps_rads_avgs,
                             'vel': stride_vel, 'vel_rads': stride_vel_rads, 'vel_avgs': stride_vel_avgs,
                             'vel_rads_avgs': stride_vel_rads_avgs,
                             'vel_sums': stride_vel_sums, 'vel_rads_sums': stride_vel_rads_sums,
                             'signal_list_keypoint': sd_sigs, 'signal_list_rads': sd_sigs_rads,
                             'skip_trials': skip_trials}
            stance_params = {'bounds': stance_bounds, 'time': stance_time, 'disps': stance_disps,
                             'disps_rads': stance_disps_rads,
                             'time_avgs': stance_time_avgs, 'disps_avgs': stance_disps_avgs,
                             'disps_rads_avgs': stance_disps_rads_avgs,
                             'vel': stance_vel, 'vel_rads': stance_vel_rads, 'vel_avgs': stance_vel_avgs,
                             'vel_rads_avgs': stance_vel_rads_avgs,
                             'vel_sums': stance_vel_sums, 'vel_rads_sums': stance_vel_rads_sums,
                             'signal_list_keypoint': st_sigs, 'signal_list_rads': st_sigs_rads,
                             'skip_trials': skip_trials}
            stride_stance_dic[keypoint + '_stride'] = stride_params
            stride_stance_dic[keypoint + '_stance'] = stance_params

            sess_cage.sessions[sessname]['day_dic']['stride_stance_dic'] = stride_stance_dic

def trial_type_pipeline(sess_cage, ordered_sessions):
    HAND_THRESH = 0.00299508
    FOOT_THRESH = 0.00374798

    for i, sess in enumerate(ordered_sessions):
        session = sess_cage.sessions[sess]
        hand_stance = session['day_dic']['stride_stance_dic']['hand_stance']
        foot_stance = session['day_dic']['stride_stance_dic']['foot_stance']
        hand_vals = hand_stance['vel_rads_avgs']
        foot_vals = foot_stance['vel_rads_avgs']
        trial_defs = []
        assert len(hand_vals) == len(foot_vals)
        assert len(hand_vals) == len(session['day_dic']['waves'])

        for hval, fval in zip(hand_vals, foot_vals):
            if np.isnan(hval) or np.isnan(fval):
                trial_defs.append(0)
            elif hval >= HAND_THRESH and fval >= FOOT_THRESH:
                trial_defs.append(1)
            else:
                trial_defs.append(0)
        trial_defs = np.array(trial_defs)
        sess_cage.sessions[sess]['day_dic']['trial_defs'] = trial_defs
        sess_cage.sessions[sess]['day_dic']['wave_dic'] = {'good': np.where(trial_defs == 1)[0],
                                                           'bad': np.where(trial_defs == 0)[0]}
