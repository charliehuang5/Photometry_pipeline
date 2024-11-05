import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

import photom_helper as ph
import statistics_helper as sh
import copy
from scipy.signal import butter, filtfilt
from scipy.stats import linregress

"""
Copied over from photom_wheel_official
"""

def apply_butter_lowpass(photom, thresh, sampling_rate=30):
    b, a = butter(4, thresh, btype='low', fs=sampling_rate)
    filtered_photom = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=0, arr=photom)
    return filtered_photom

#not used
def apply_butter_highpass(photom, thresh, sampling_rate=30):
    b, a = butter(2, thresh, btype='high', fs=sampling_rate)
    filtered_photom = np.apply_along_axis(lambda x: filtfilt(b, a, x, padtype='even'), axis=0, arr=photom)
    return filtered_photom


def preprocess_compact(photom, refpoint_framecount, phot_coldic, parameter_dic, p_BACK_WINDOW=150, p_FORWARD_WINDOW=150):
    # STEP -1: Determine photometry frame boundaries
    phot_bounds = [refpoint_framecount - p_BACK_WINDOW, refpoint_framecount + p_FORWARD_WINDOW]
    # print(phot_bounds)

    # STEP 0: Extracting raw signal (raw signal - control background)
    raw_sig = photom.loc[phot_bounds[0]:phot_bounds[1]].to_numpy()
    raw_sig_keys = list(photom.keys())

    # STEP 1: Low Pass Filtering - noise correction
    if parameter_dic['lowpass_threshold_2'] == None:
        lowpass_photom = apply_butter_lowpass(raw_sig,parameter_dic['lowpass_threshold'])  # 4th order butterworth lowpass
        lowpass_photom_keys = list(photom.keys())
    else:
        raw_470 = raw_sig[:, [0, 2, 4, 6]]
        raw_410 = raw_sig[:, [1, 3, 5, 7]]
        arr_470 = apply_butter_lowpass(raw_470, parameter_dic['lowpass_threshold'])  # 4th order butterworth lowpass
        arr_410 = apply_butter_lowpass(raw_410, parameter_dic['lowpass_threshold_2'])  # 4th order butterworth lowpass
        lowpass_photom = np.array(
            [arr_470[:, 0], arr_410[:, 0], arr_470[:, 1], arr_410[:, 1], arr_470[:, 2], arr_410[:, 2], arr_470[:, 3],
             arr_410[:, 3]]).T
        lowpass_photom_keys = list(photom.keys())

        print('lowpass photom shape', lowpass_photom.shape)

    # STEP 2: Motion correction, plot per region
    CH470_movcor = {}  # (470-410)/410 half number of channels
    regions = ['CH1', 'DCN', 'Thal', 'SNr']
    for i, reg in enumerate(regions):
        chan_470 = lowpass_photom[:, phot_coldic[reg + '-470']]
        chan_410 = lowpass_photom[:, phot_coldic[reg + '-410']]
        slope, intercept, r_value, p_value, std_err = linregress(x=chan_410, y=chan_470)  # from scipy.stats
        chan_410_fitted = intercept + slope * chan_410

        # shows the delta f/f
        CH470_movcor[reg + '-470_movcorr'] = (chan_470 - chan_410_fitted) / chan_410_fitted

    # STEP 2.2: Save movement corrected CH470:  (470-410_a)/410_a
    CH470_movcor_df = pd.DataFrame.from_dict(CH470_movcor)
    CH470_movcor_keys = list(CH470_movcor_df.keys())
    CH470_movcor_np = CH470_movcor_df.to_numpy()  # richard comment - call it chan_470_move_cor

    # STEP 3: Normalization to premovement period - zscoring
    # uses a premovement window for mean and std
    p_start = parameter_dic['norm_window'][0]
    p_end = parameter_dic['norm_window'][1]

    # STEP 3.1: zscores on CH470_movcor
    means = np.mean(CH470_movcor_np[p_start:p_end, :],
                    axis=0)  # find means -pre movement window to 0 (where 0 is back_window_p up)
    stds = np.std(CH470_movcor_np[p_start:p_end, :], axis=0)
    zscores = (CH470_movcor_np - means) / stds  # now F/F0
    zscores_keys = ['CH1_zscore', 'DCN_zscore', 'Thal_zscore', 'SNr_zscore']

    # output_dic elements: numpy arrays
    output_dic = {'raw_sig': raw_sig, 'lowpass_photom': lowpass_photom,
                  'CH470_movcor_np': CH470_movcor_np,'zscores': zscores}
    output_dic_keys = {'raw_sig': raw_sig_keys, 'lowpass_photom': lowpass_photom_keys,
                        'CH470_movcor_np': CH470_movcor_keys, 'zscores': zscores_keys}
    return output_dic, output_dic_keys
    # 'raw_sig', 'lowpass_photom' 'deltaf_im_np' 'CH470_movcor_np' 'CH470_410_ratio_np' 'zscores' 'zscores_ratio' 'f_f0', CH470_410_uratio_np, zscores_uratio', 'f_f0_u'


def within_oreg(refpoint_framecount, oreg_list, p_BACK_WINDOW = 120, p_FORWARD_WINDOW = 120):
    """
    Just returns if a photom frame is inside an outlier region (oreg)
    - used by photom_cube_generate
    """
    for pair in oreg_list:
        # if inside pair's range expanded by forward and backward window
        if pair[0] - p_FORWARD_WINDOW <= refpoint_framecount <= pair[1] + p_BACK_WINDOW:
            return True
    return False

def random_cube_generate(photom_df, day_dic, oreg_list, parameter_dic, phot_coldic_override=None, p_BACK_WINDOW = 150, p_FORWARD_WINDOW = 150, shuffle_trials=False):
    """
    Main wrapper for generating a photom cube from photom_df
    """
    phot_coldic = {key: i for i, key in enumerate(photom_df.keys())}
    if phot_coldic_override != None:
        print("USING OVERRIDE on photcoldic!")
        phot_coldic = phot_coldic_override
        print(phot_coldic)
    waves = day_dic['waves']
    upper_phot_frame = (photom_df.shape[0]) - p_FORWARD_WINDOW
    lower_phot_frame = p_BACK_WINDOW
    random_waves = []
    while len(random_waves) < len(waves):
        random_refpoint_framecount = np.random.randint(lower_phot_frame, upper_phot_frame)
        if not within_oreg(random_refpoint_framecount, oreg_list):
            random_waves.append(random_refpoint_framecount)
    mats_dic = {}
    for trial in range(len(random_waves)):
        # print('TRIAL: ' + str(trial))
        rand_wave = random_waves[trial]
        output_dic, output_dic_keys = preprocess_compact(photom_df, rand_wave, phot_coldic, parameter_dic)
        for key in output_dic.keys():
            if key in mats_dic.keys():
                mats_dic[key].append(output_dic[key])
            else:
                mats_dic[key] = [output_dic[key]]
    cube_dic = {}
    for key in mats_dic.keys():
        cube_dic[key] = np.dstack(mats_dic[key])
        if shuffle_trials:
            cube_dic[key] = ph.shuffle_cube_trials(cube_dic[key])
    return cube_dic, output_dic_keys


