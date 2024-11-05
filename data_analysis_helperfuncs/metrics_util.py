import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statistics_helper as sh
import copy
import photom_helper as ph
from sklearn import metrics
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression

"""
Peaks Characterization
"""
def characterize_peaks(cube, start_point, end_point, min_height = 2, plot_trials=False):
    """
    Metric Function
    -----
    For a cube (Time x region x Trial)
    - characterizes peaks for each trial (generating a vector per trial, thus a matrix for the session)
    :return:
    - output_dic: contains one key one value - (the matrix: - peaks_mat (Trial x region))

    """
    peaks_mat = []
    for trial in range(cube.shape[2]):  # looping across trials
        peaks_per_reg = []
        for i in range(cube.shape[1]):  # looping across channels
            assert cube.shape[1] == 4
            vector = cube[start_point:end_point, i, trial]
            peaks, _ = scipy.signal.find_peaks(vector, height=min_height)
            if len(peaks) > 0:
                best_peak = np.argmax(peaks)
                peaks_per_reg.append(peaks[best_peak])
            else:
                peaks_per_reg.append(np.nan)
        peaks_mat.append(peaks_per_reg)
    peaks_mat = np.vstack(peaks_mat)
    print(peaks_mat.shape)
    output_dic = {'peaks': peaks_mat}
    return output_dic

def gen_peaks_dic(start_point, end_point, parameter_dic, session_list, sess_cage, photom_cube_type='default'):
    """
    Dictionary generating function that calls "characterize_peaks" on all sessions in session_list, accessing
    sessions within sess_cage

    :return:
    bigdic - dictionary mapping session name to metric dictionary (output_dic from characterize_peaks)
    """
    bigdic = {}
    assert photom_cube_type in ['default', 'rew', 'control']
    for sessname in session_list:
        session = sess_cage.sessions[sessname]
        addon = parameter_dic['name']
        if photom_cube_type == 'default':
            photom_cube = session['cube_dic_lowp_' + addon]['zscores']
        elif photom_cube_type == 'rew':
            photom_cube = session['rew_cube_dic_lowp_' + addon]['zscores']
        elif photom_cube_type == 'control':
            photom_cube = session['rand_cube_dic_lowp_' + addon]['zscores']
        output_dic = characterize_peaks(photom_cube, start_point, end_point)
        bigdic[sessname] = output_dic
    return bigdic

"""
Mutual information
"""

def compute_mutual_information(x,y):
    """
    Computes mutual info between x and y (both are 1 dimensional numpy arrays)
    """
    assert len(x.shape) == 1
    x_expand = np.expand_dims(x, axis=1)
    mi = mutual_info_regression(x_expand, y)
    assert len(mi) == 1
    return mi[0]

def compute_mi_cube(cube, start_point, end_point):
    """
    Metric Function
    -----
    Runs compute_mutual_info per trial using vectors from start_point to end_point
    :return:
    output_dic:
    - 'dcn_mi' - mutual info of dcn to thal
    - 'snr_mi' - mutual info of snr to mi
    - feel free to add dcn-snr mutual info
    """
    dcn_mi_list, snr_mi_list = [],[]
    for trial in range(cube.shape[2]):
        slice = cube[start_point:end_point,:,trial]
        assert slice.shape[1] == 4 #assert 4 channels
        dcn = slice[:,1] #the 0th indexed slice is ch1 fyi
        thal = slice[:, 2]
        snr = slice[:, 3]
        dcn_mi = compute_mutual_information(dcn, thal)
        snr_mi = compute_mutual_information(snr, thal)
        dcn_mi_list.append(dcn_mi)
        snr_mi_list.append(snr_mi)
    output_dic = {'dcn_mi': dcn_mi_list, 'snr_mi': snr_mi_list}
    return output_dic

def gen_mi_dic(start_point, end_point, parameter_dic, session_list, sess_cage, photom_cube_type='default'):
    """
    Dictionary generating function that calls "compute_mi_cube" on all sessions in session_list, accessing
    sessions within sess_cage

    :return:
    bigdic - dictionary mapping session name to metric dictionary (output_dic from compute_mi_cube)
    """
    bigdic = {}
    assert photom_cube_type in ['default', 'rew', 'control']
    for sessname in session_list:
        session = sess_cage.sessions[sessname]
        addon = parameter_dic['name']
        if photom_cube_type == 'default':
            photom_cube = session['cube_dic_lowp_' + addon]['zscores']
        elif photom_cube_type == 'rew':
            photom_cube = session['rew_cube_dic_lowp_' + addon]['zscores']
        elif photom_cube_type == 'control':
            photom_cube = session['rand_cube_dic_lowp_' + addon]['zscores']
        output_dic = compute_mi_cube(photom_cube, start_point, end_point)
        bigdic[sessname] = output_dic
    return bigdic

"""
Linear Regression functions
"""

def gen_shifted_vectors(cube_slice, reg_ind, start_point, end_point, stepsize, num_vecs):
    """
    if num_vecs is 1, this function does not do anything special other than return a 1D vector from cube slice
    in which the column is indexed by reg_ind, and the start and end frame are from start_point and end_point
    - will return a list of length 1, containing a vector

    if num_vecs is >1, - we are running lin regression with time-shifted copies of signal
    -   (ie: SNr - 0, SNr - 5, SNr - 10, etc.)
    then it will return a list of multiple vectors, each subsequent vector being shifted
    back in time (frames) by stepsize. the list will be of length num_vecs.
    """
    vectors = []
    for i in range(num_vecs):
        new_start_point = start_point - (i * stepsize)
        new_end_point = end_point - (i * stepsize)
        vec = cube_slice[new_start_point:new_end_point, reg_ind]
        vectors.append(vec)
    return vectors

def linregress_cube(photom_cube, start_point, end_point, num_vecs=1, step_size=2,
                    random_adjusts=False):
    """
    Metric Function
    -----
    :return:
    two lists, one containinng r^2 scores per trial, the other being a matrix (TRIAL x REGION) containing the
    coefficients for linear regression (a*SNr + b*DCN + c*1) = Thal
    """


    r2_scores = []
    coeffs = []
    rand_range = [-20, 20]

    for trial in range(photom_cube.shape[2]):
        slice = photom_cube[:, :, trial]
        zscore_coldic = {'CH1_zscore': 0, 'DCN_zscore': 1, 'Thal_zscore': 2, 'SNr_zscore': 3}
        thal = slice[start_point:end_point, zscore_coldic['Thal_zscore']]
        adjust_dcn, adjust_snr = 0, 0
        if random_adjusts:
            adjust_dcn = np.random.randint(rand_range[0], rand_range[1])
            adjust_snr = np.random.randint(rand_range[0], rand_range[1])

        dcn_shifts = gen_shifted_vectors(slice, zscore_coldic['DCN_zscore'], start_point + adjust_dcn,
                                         end_point + adjust_dcn, step_size, num_vecs)
        dcn_submat = np.vstack(dcn_shifts).T
        snr_shifts = gen_shifted_vectors(slice, zscore_coldic['SNr_zscore'], start_point + adjust_snr,
                                         end_point + adjust_snr, step_size, num_vecs)
        snr_submat = np.vstack(snr_shifts).T

        ones = np.array([np.ones(end_point - start_point)]).T


        input_mat = np.hstack([dcn_submat, snr_submat, ones])
        coefficients, _, _, sing_vals = np.linalg.lstsq(input_mat, thal, rcond=None)
        thal_pred_3 = input_mat @ coefficients
        r2_scores.append(metrics.r2_score(thal, thal_pred_3))
        coeffs.append(coefficients)
    return r2_scores, np.array(coeffs)


def corrcoefs_to_thal(photom_cube, start_point, end_point, random_adjusts=False):
    """
    Metric Function for simple region to region correlation
    -----
    :return:
    matrix (TRIAL x REGION) in which columns are ['DCN-thal','SNr-thal','SNr-DCN'] correlation coefficients
    """
    r_scores = []
    rand_range = [-20, 20]
    for trial in range(photom_cube.shape[2]):
        slice = photom_cube[:, :, trial]
        zscore_coldic = {'CH1_zscore': 0, 'DCN_zscore': 1, 'Thal_zscore': 2, 'SNr_zscore': 3}
        adjust_dcn, adjust_snr = 0, 0
        if random_adjusts:
            adjust_dcn = np.random.randint(rand_range[0], rand_range[1])
            adjust_snr = np.random.randint(rand_range[0], rand_range[1])
        thal = slice[start_point:end_point, zscore_coldic['Thal_zscore']]
        dcn = slice[start_point + adjust_dcn:end_point + adjust_dcn, zscore_coldic['DCN_zscore']]
        snr = slice[start_point + adjust_snr:end_point + adjust_snr, zscore_coldic['SNr_zscore']]
        dcn_2_thal = np.corrcoef(dcn, thal)
        snr_2_thal = np.corrcoef(snr, thal)
        snr_2_dcn = np.corrcoef(snr, dcn)
        r_scores.append([dcn_2_thal[1, 0], snr_2_thal[1, 0], snr_2_dcn[1, 0]])
    return np.vstack(r_scores)


def regression_wrapper(startpoint, endpoint, parameter_dic, session_list, sess_cage, stpt_label='', plot=True, save_flag=False, num_vecs=1,
                     step_size=2, photom_cube_type='default', random_adjusts=False, shuffle_trials=False, trial_type=None):
    """
    Wrapper running dictionary generating function
    - runs compute_singlereg_multireg on each session

    :return:
    bigdic mapping sessname to output_dic. Refer to function below for elements of
    """

    bigdic = {}
    # assert photom_cube_type in ['default', 'rew', 'ratime_ratrial', 'ratime_notrial', 'notime_ratrial']
    assert photom_cube_type in ['default','rew','control']
    for sessname in session_list:
        if photom_cube_type == 'rew':
            session = sess_cage.sessions[sessname]
            assert len(session['day_dic']['wave_dic']['rewarded']) > 0
            # if len(session['day_dic']['rew_waves']) == 0:
            #     continue
        output_dic = compute_singlereg_multireg(sessname, startpoint, endpoint, parameter_dic, sess_cage, stpt_label=stpt_label,
                                           plot=plot, save_flag=save_flag, num_vecs=num_vecs,
                                           step_size=step_size, photom_cube_type=photom_cube_type,
                                           random_adjusts=random_adjusts, shuffle_trials=shuffle_trials,
                                                trial_type=trial_type)
        bigdic[sessname] = output_dic
    return bigdic

def compute_singlereg_multireg(sessname, startpoint, endpoint, parameter_dic, sess_cage, stpt_label='', plot=True,
                          save_flag=False, num_vecs=1, step_size=2, plot_conglom=False, photom_cube_type='default',
                          random_adjusts=False, shuffle_trials=False, trial_type=None):
    """
    Dictionary generating function for BOTH multiregion linear regression and single region correlations

    :return:
    output_dic: lr = linear regression, sr = single regression (single region)
    - 'lr_r2_scores': r2_scores, 'lr_coeffs': coeffs, 'lr_high_r2_scores': high_r2_scores, 'lr_coeffs_labels' (only really useful if doing
        time shifted stuff)
    - 'sr_r2_scores': corr_coefs, 'sr_ref': ['DCN-Thal', 'SNr-Thal', 'SNr-DCN']
    """

    session = sess_cage.sessions[sessname]
    # print(sessname)
    addon = parameter_dic['name']
    if photom_cube_type == 'default':
        photom_cube = session['cube_dic_lowp_' + addon]['zscores']
    elif photom_cube_type == 'rew':
        photom_cube = session['rew_cube_dic_lowp_' + addon]['zscores']
    elif photom_cube_type == 'control':
        photom_cube = session['rand_cube_dic_lowp_' + addon]['zscores']

    if trial_type != None:

        trials_oi = session['day_dic']['wave_dic'][trial_type]
        photom_cube = photom_cube[:,:,trials_oi]


    if shuffle_trials:
        # print("SHUFFLING HAPPENING")
        photom_cube = ph.shuffle_cube_trials(photom_cube)

    output_dic = {}

    r2_scores, coeffs = linregress_cube(photom_cube, startpoint, endpoint, num_vecs=num_vecs,
                                        step_size=step_size, random_adjusts=random_adjusts)
    r2_scores = np.array(r2_scores)
    high_r2_scores = np.where(r2_scores >= 0.5)[0]
    low_r2_scores = np.where(r2_scores < 0.5)[0]
    # correlation coefficients
    corr_coefs = corrcoefs_to_thal(photom_cube, start_point=startpoint, end_point=endpoint,
                                   random_adjusts=random_adjusts) ** 2
    # save both 
    output_dic.update({'lr_r2_scores': r2_scores, 'lr_coeffs': coeffs, 'lr_high_r2_scores': high_r2_scores})
    output_dic.update({'sr_r2_scores': corr_coefs, 'sr_ref': ['DCN-Thal', 'SNr-Thal', 'SNr-DCN']})

    # making the labels for coeffs
    dup_num = int((coeffs.shape[1] - 1) / 2)
    assert dup_num == num_vecs
    dcn_labels, snr_labels = ['DCN'] * dup_num, ['SNr'] * dup_num
    labels_cum = []
    for i in range(coeffs.shape[1] - 1):
        if i < dup_num:
            lab = dcn_labels[i] + ' -' + str(i * step_size)
            labels_cum.append(lab)
        else:
            lab = snr_labels[i - dup_num] + ' -' + str((i - dup_num) * step_size)
            labels_cum.append(lab)
    output_dic['lr_coeffs_labels'] = labels_cum  # save coeffs labels ([dcn -5, dcn-10] etc.)
    # sess_cage.sessions[sessname]['day_dic']['corr_dic'] = output_dic #save into sess cage
    return output_dic