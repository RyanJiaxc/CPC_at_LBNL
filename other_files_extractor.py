# -*- coding: utf-8 -*-
"""
    Created on Mon Jan 15 00:08:08 2018

    @author: MMartchevskii
    """
from __future__ import division
import numpy as np
from nptdms import TdmsFile
import pycwt as wavelet
from pycwt.helpers import find
from matplotlib import pyplot as plt
import peakutils
import h5py
import random

# import matplotlib.colors as colors

#######################################SETUP###################################
# TDMS file to load (replace with the actual path)
myfile = '/data0/stephey/SORTED/8_18_2017_11_29_52_AM_A97_Q76_15163A.tdms'
file_name = '8_18_2017_11_29_52_AM_A97_Q76_15163A'
name_to_use = 'Q076'
q076_q_t = 176925635
loc = q076_q_t

# ch_to_load1 = 'Sensor'  #Channel to load from TDMS
ch_to_load1 = 'S_top'
ch_to_load2 = 'S_bot'

Fs = 1000000  # Sampling frequency, in Hz

f_high = 250000  # filter upper frequency limit for peak detection
f_low = 25000  # filter lower frequency limit for peak detection

noise_slength = 100000  # Initial length of data representing noise
chunksize = 1000000  # Process the entire dataset in chunks of 1s duration

mywindow = 1000  # Half-width of the window taken around each found event
outputdir = '/data1/ryan/Spectrograms/'  # Output directory for spectrograms

###########################Define CWT wavelet##################################
mother = wavelet.Morlet(6)
s0 = 2 / Fs  # Starting scale
dj = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves
h5f_training_top = h5py.File('/data1/ryan/dataset/training_q076_top.h5', 'w')
h5f_training_bot = h5py.File('/data1/ryan/dataset/training_q076_bot.h5', 'w')
q076_t_bot = []
q076_t_top = []


# h5f_validation = h5py.File('/data1/ryan/dataset/validation_001_bot.h5', 'a')

###########Function to retrieve a data channel as numpy array##################
def getchdata(dframe, ch_to_load, resample):
    data = []
    for x in dframe.columns:
        if x.find(ch_to_load) != -1:
            data = dframe.loc[:, x].values[0:loc:resample]
            if len(data) == 0:
                print('channel not found')
            else:
                print(ch_to_load, len(data))

    return data


##########Function to estimate noise level in terms of scale-averaged value####
def noise_estimate(indata, f_h, f_l):
    nt = indata.size
    tt = np.arange(0, nt) / Fs
    pf = np.polyfit(tt, indata, 1)
    indata_norm = indata - np.polyval(pf, tt)
    i_wave, i_scales, i_freqs, i_coi, i_fft, i_fftfreqs = wavelet.cwt(indata_norm, 1 / Fs, dj, s0, J, mother)
    i_power = (np.abs(i_wave)) ** 2
    i_period = 1 / i_freqs
    i_sel = find((i_period >= 1 / f_h) & (i_period < 1 / f_l))  # select frequency band for averaging
    i_Cdelta = mother.cdelta
    i_scale_avg = (i_scales * np.ones((nt, 1))).transpose()
    i_scale_avg = i_power / i_scale_avg  # As in Torrence and Compo (1998) equation 24
    i_scale_avg = dj / Fs * i_Cdelta * i_scale_avg[i_sel, :].sum(axis=0)
    i_max = max(i_scale_avg)
    return i_max


###################################Load datafile###############################
print('Loading ', myfile)
tdms_file = TdmsFile(myfile, memmap_dir='/data1/ryan/')  # set buffer directory
tddf = tdms_file.as_dataframe()  # Load TDMS file AS DATAFRAME

##################optional: cut out the ramp-up only based on IMAG channel############
# tddf = tddf.loc[np.array(mf.getchdata(tddf,'IMAG',1)) > 0.95]

########################Processing#############################################
plt.ioff()
logf = open(outputdir + 'eventlog.csv', 'ab')
# h5f_ch1 = h5py.File('/data1/ryan/training_top_ch1.h5', 'w')


# Evaluate noise level in terms of scale-averaged value and set the threshold for peak detection
noisesample = getchdata(tddf, ch_to_load2, 1)[0:noise_slength]
threshold = 5 * noise_estimate(noisesample, f_high, f_low)  # 5 times noise level threshold

# Process the data
datalength = len(tddf) // chunksize
datalength = loc
print(datalength)
count1 = 0
count2 = 0
for I in range(round(loc / chunksize)):

    print('Processing ' + str(I * chunksize / Fs) + '-' + str((I + 1) * chunksize / Fs) + ' s')

    '''
    #######################################Select data channel and interval for processing#####
        mydata0 = getchdata(tddf,ch_to_load1,1) [I*chunksize:(I+1)*chunksize]

        #########################################De-trend data###########################
        N = mydata0.size
        t = np.arange(0, N) / Fs
        p = np.polyfit(t, mydata0, 1)
        dat_norm = mydata0 - np.polyval(p, t)

        #####################Perform CWT####################################
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
        power = (np.abs(wave)) ** 2

        ############Plot 2D wavelet spectrum of the entire data chunck######################
        #plt.pcolormesh(t, freqs, power)
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')

        #######################Calculate spectrum scale average and significance#############
        period = 1 / freqs
        sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
        Cdelta = mother.cdelta
        scale_avg = (scales * np.ones((N, 1))).transpose()
        scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
        scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
        #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
        #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
        #plt.show()

        #####################Locate peaks in the scale-averaged spectrum############################################

        indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
        np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')

        #####################Take a window around each found peak and calcualte 2D wavelet transform##################

        indexes = indexes[(indexes>mywindow)]
        indexes = indexes[(indexes+mywindow)<len(mydata0)]
        print('Found ' + str(len(indexes)) + ' peaks')

        for indx in indexes:

        smdat = mydata0[indx-mywindow: indx+mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        if ()
        name = file_name + '_' + str(count1)
        h5f_ch1.create_dataset(name, data=smdat)
        with open('/data1/ryan/training_top_ch1.txt', 'a') as f:
            f.write(name)
            f.write('\n')
        count1 += 1
        '''

    #######################################Select data channel and interval for processing#####
    mydata0 = getchdata(tddf, ch_to_load2, 1)[I * chunksize:(I + 1) * chunksize]

    #########################################De-trend data###########################
    N = mydata0.size
    t = np.arange(0, N) / Fs
    p = np.polyfit(t, mydata0, 1)
    dat_norm = mydata0 - np.polyval(p, t)

    #####################Perform CWT####################################
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1 / Fs, dj, s0, J, mother)
    power = (np.abs(wave)) ** 2

    ############Plot 2D wavelet spectrum of the entire data chunck######################
    # plt.pcolormesh(t, freqs, power)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    #######################Calculate spectrum scale average and significance#############
    period = 1 / freqs
    sel = find((period >= 1 / f_high) & (period < 1 / f_low))  # select frequency band for averaging
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = dj / Fs * Cdelta * scale_avg[sel, :].sum(axis=0)
    # scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
    # plt.plot(t, scale_avg, 'k-', linewidth=1.5)
    # plt.show()

    #####################Locate peaks in the scale-averaged spectrum############################################

    indexes = peakutils.indexes(scale_avg, thres=threshold / max(scale_avg), min_dist=5000)
    np.savetxt(logf, [(indexes + I * chunksize)], fmt='%d', delimiter=',', newline=',')

    #####################Take a window around each found peak and calcualte 2D wavelet transform##################

    indexes = indexes[(indexes > mywindow)]
    indexes = indexes[(indexes + mywindow) < len(mydata0)]
    print('Found ' + str(len(indexes)) + ' peaks')

    for indx in indexes:
        smdat = mydata0[indx - mywindow: indx + mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1 / Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        name = name_to_use + '_' + str(indx + I * chunksize)

        h5f_training_bot.create_dataset(name, data=smdat)
        q076_t_bot += [indx + I * chunksize]
        # with open('/data1/ryan/dataset/training_new.txt', 'a') as f:
        #     f.write(name)
        #     f.write('\n')
        # count2 += 1

# with open('/data1/ryan/dataset/validation_new_num.txt', 'a') as f:
#     f.write(file_name)
#     f.write('\t')
#     f.write(str(count1))
#     f.write('\n')
#
# with open('/data1/ryan/dataset/training_new_num.txt', 'a') as f:
#     f.write(file_name)
#     f.write('\t')
#     f.write(str(count2))
#     f.write('\n')
h5f_training_bot.close()

# Now for Q001_top
noisesample = getchdata(tddf, ch_to_load1, 1)[0:noise_slength]
threshold = 5 * noise_estimate(noisesample, f_high, f_low)  # 5 times noise level threshold

# Process the data
datalength = len(tddf) // chunksize
datalength = loc
print(datalength)
count1 = 0
count2 = 0
for I in range(round(loc / chunksize)):

    print('Processing ' + str(I * chunksize / Fs) + '-' + str((I + 1) * chunksize / Fs) + ' s')

    '''
    #######################################Select data channel and interval for processing#####
        mydata0 = getchdata(tddf,ch_to_load1,1) [I*chunksize:(I+1)*chunksize]

        #########################################De-trend data###########################
        N = mydata0.size
        t = np.arange(0, N) / Fs
        p = np.polyfit(t, mydata0, 1)
        dat_norm = mydata0 - np.polyval(p, t)

        #####################Perform CWT####################################
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
        power = (np.abs(wave)) ** 2

        ############Plot 2D wavelet spectrum of the entire data chunck######################
        #plt.pcolormesh(t, freqs, power)
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')

        #######################Calculate spectrum scale average and significance#############
        period = 1 / freqs
        sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
        Cdelta = mother.cdelta
        scale_avg = (scales * np.ones((N, 1))).transpose()
        scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
        scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
        #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
        #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
        #plt.show()

        #####################Locate peaks in the scale-averaged spectrum############################################

        indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
        np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')

        #####################Take a window around each found peak and calcualte 2D wavelet transform##################

        indexes = indexes[(indexes>mywindow)]
        indexes = indexes[(indexes+mywindow)<len(mydata0)]
        print('Found ' + str(len(indexes)) + ' peaks')

        for indx in indexes:

        smdat = mydata0[indx-mywindow: indx+mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        if ()
        name = file_name + '_' + str(count1)
        h5f_ch1.create_dataset(name, data=smdat)
        with open('/data1/ryan/training_top_ch1.txt', 'a') as f:
            f.write(name)
            f.write('\n')
        count1 += 1
        '''

    #######################################Select data channel and interval for processing#####
    mydata0 = getchdata(tddf, ch_to_load1, 1)[I * chunksize:(I + 1) * chunksize]

    #########################################De-trend data###########################
    N = mydata0.size
    t = np.arange(0, N) / Fs
    p = np.polyfit(t, mydata0, 1)
    dat_norm = mydata0 - np.polyval(p, t)

    #####################Perform CWT####################################
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1 / Fs, dj, s0, J, mother)
    power = (np.abs(wave)) ** 2

    ############Plot 2D wavelet spectrum of the entire data chunck######################
    # plt.pcolormesh(t, freqs, power)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    #######################Calculate spectrum scale average and significance#############
    period = 1 / freqs
    sel = find((period >= 1 / f_high) & (period < 1 / f_low))  # select frequency band for averaging
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = dj / Fs * Cdelta * scale_avg[sel, :].sum(axis=0)
    # scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
    # plt.plot(t, scale_avg, 'k-', linewidth=1.5)
    # plt.show()

    #####################Locate peaks in the scale-averaged spectrum############################################

    indexes = peakutils.indexes(scale_avg, thres=threshold / max(scale_avg), min_dist=5000)
    np.savetxt(logf, [(indexes + I * chunksize)], fmt='%d', delimiter=',', newline=',')

    #####################Take a window around each found peak and calcualte 2D wavelet transform##################

    indexes = indexes[(indexes > mywindow)]
    indexes = indexes[(indexes + mywindow) < len(mydata0)]
    print('Found ' + str(len(indexes)) + ' peaks')

    for indx in indexes:
        smdat = mydata0[indx - mywindow: indx + mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1 / Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        name = name_to_use + '_' + str(indx + I * chunksize)

        h5f_training_top.create_dataset(name, data=smdat)
        # with open('/data1/ryan/dataset/training_new.txt', 'a') as f:
        #     f.write(name)
        #     f.write('\n')
        # count2 += 1
        q076_t_top += [indx + I * chunksize]

# with open('/data1/ryan/dataset/validation_new_num.txt', 'a') as f:
#     f.write(file_name)
#     f.write('\t')
#     f.write(str(count1))
#     f.write('\n')
#
# with open('/data1/ryan/dataset/training_new_num.txt', 'a') as f:
#     f.write(file_name)
#     f.write('\t')
#     f.write(str(count2))
#     f.write('\n')
h5f_training_top.close()

t_bot = np.array(q076_t_bot)
t_top = np.array(q076_t_top)
np.save('/data1/ryan/dataset/time_q076_top.npy', t_top)
np.save('/data1/ryan/dataset/time_q076_bot.npy', t_bot)

'''
###### NEW ONE ######
#######################################SETUP###################################
#TDMS file to load (replace with the actual path)
myfile = '/data0/stephey/SORTED/8_18_2017_11_29_52_AM_A97_Q76_15163A.tdms'
file_name = '8_18_2017_11_29_52_AM_A97_Q76_15163A'
name_to_use = 'Q76'
loc = 180000000
loc_length = 180

#ch_to_load1 = 'Sensor'  #Channel to load from TDMS
ch_to_load1 = 'S_top'
ch_to_load2 = 'S_bot'

Fs = 1000000  #Sampling frequency, in Hz

f_high = 250000  #filter upper frequency limit for peak detection
f_low = 25000    #filter lower frequency limit for peak detection

noise_slength = 100000  # Initial length of data representing noise
chunksize=1000000       # Process the entire dataset in chunks of 1s duration

mywindow = 1000   #Half-width of the window taken around each found event
outputdir = '/data1/ryan/Spectrograms/' #Output directory for spectrograms

###########################Define CWT wavelet##################################
mother = wavelet.Morlet(6)
s0 = 2 / Fs # Starting scale
dj = 1 / 12 # Twelve sub-octaves per octaves
J = 7 / dj # Seven powers of two with dj sub-octaves

###################################Load datafile###############################
print('Loading ',myfile)
tdms_file = TdmsFile(myfile, memmap_dir='/data1/ryan/')  #set buffer directory
tddf = tdms_file.as_dataframe()  #Load TDMS file AS DATAFRAME

##################optional: cut out the ramp-up only based on IMAG channel############
#tddf = tddf.loc[np.array(mf.getchdata(tddf,'IMAG',1)) > 0.95]

########################Processing#############################################
plt.ioff()
logf = open(outputdir+'eventlog.csv', 'ab')
# h5f_ch1 = h5py.File('/data1/ryan/training_top_ch1.h5', 'w')


#Evaluate noise level in terms of scale-averaged value and set the threshold for peak detection
noisesample = getchdata(tddf,ch_to_load2,1) [0:noise_slength]
threshold = 5*noise_estimate(noisesample, f_high,f_low)   # 5 times noise level threshold

#Process the data
datalength = len(tddf) //chunksize
datalength = loc
print(datalength)
count1 = 0
count2 = 0
for I in range(loc_length) :

    print('Processing ' + str(I*chunksize/Fs) + '-' + str((I+1)*chunksize/Fs) + ' s')


    """#######################################Select data channel and interval for processing#####
        mydata0 = getchdata(tddf,ch_to_load1,1) [I*chunksize:(I+1)*chunksize]

        #########################################De-trend data###########################
        N = mydata0.size
        t = np.arange(0, N) / Fs
        p = np.polyfit(t, mydata0, 1)
        dat_norm = mydata0 - np.polyval(p, t)

        #####################Perform CWT####################################
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
        power = (np.abs(wave)) ** 2

        ############Plot 2D wavelet spectrum of the entire data chunck######################
        #plt.pcolormesh(t, freqs, power)
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')

        #######################Calculate spectrum scale average and significance#############
        period = 1 / freqs
        sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
        Cdelta = mother.cdelta
        scale_avg = (scales * np.ones((N, 1))).transpose()
        scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
        scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
        #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
        #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
        #plt.show()

        #####################Locate peaks in the scale-averaged spectrum############################################

        indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
        np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')

        #####################Take a window around each found peak and calcualte 2D wavelet transform##################

        indexes = indexes[(indexes>mywindow)]
        indexes = indexes[(indexes+mywindow)<len(mydata0)]
        print('Found ' + str(len(indexes)) + ' peaks')

        for indx in indexes:

        smdat = mydata0[indx-mywindow: indx+mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        if ()
        name = file_name + '_' + str(count1)
        h5f_ch1.create_dataset(name, data=smdat)
        with open('/data1/ryan/training_top_ch1.txt', 'a') as f:
        f.write(name)
        f.write('\n')
        count1 += 1"""


    #######################################Select data channel and interval for processing#####
    mydata0 = getchdata(tddf,ch_to_load2,1) [I*chunksize:(I+1)*chunksize]

    #########################################De-trend data###########################
    N = mydata0.size
    t = np.arange(0, N) / Fs
    p = np.polyfit(t, mydata0, 1)
    dat_norm = mydata0 - np.polyval(p, t)

    #####################Perform CWT####################################
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
    power = (np.abs(wave)) ** 2

    ############Plot 2D wavelet spectrum of the entire data chunck######################
    #plt.pcolormesh(t, freqs, power)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')

    #######################Calculate spectrum scale average and significance#############
    period = 1 / freqs
    sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
    scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
    #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
    #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
    #plt.show()

    #####################Locate peaks in the scale-averaged spectrum############################################

    indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
    np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')

    #####################Take a window around each found peak and calcualte 2D wavelet transform##################

    indexes = indexes[(indexes>mywindow)]
    indexes = indexes[(indexes+mywindow)<len(mydata0)]
    print('Found ' + str(len(indexes)) + ' peaks')

    for indx in indexes:

        smdat = mydata0[indx-mywindow: indx+mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        name = name_to_use + '_' + str(indx + I * chunksize)
        if random.random() < 0.2:
            h5f_validation.create_dataset(name, data=smdat)
            with open('/data1/ryan/dataset/validation_new.txt', 'a') as f:
                f.write(name)
                f.write('\n')
            count1 += 1
        else:
            h5f_training.create_dataset(name, data=smdat)
            with open('/data1/ryan/dataset/training_new.txt', 'a') as f:
                f.write(name)
                f.write('\n')
            count2 += 1

with open('/data1/ryan/dataset/validation_new_num.txt', 'a') as f:
    f.write(file_name)
    f.write('\t')
    f.write(str(count1))
    f.write('\n')

with open('/data1/ryan/dataset/training_new_num.txt', 'a') as f:
    f.write(file_name)
    f.write('\t')
    f.write(str(count2))
    f.write('\n')


###### NEW ONE ######
#######################################SETUP###################################
#TDMS file to load (replace with the actual path)
myfile = '/data0/stephey/SORTED/8_24_2017_2_46_30_PM_A129_Q103_15986A.tdms'
file_name = '8_24_2017_2_46_30_PM_A129_Q103_15986A'
name_to_use = 'Q103'
loc = 210000000
loc_length = 210

#ch_to_load1 = 'Sensor'  #Channel to load from TDMS
ch_to_load1 = 'S_top'
ch_to_load2 = 'S_bot'

Fs = 1000000  #Sampling frequency, in Hz

f_high = 250000  #filter upper frequency limit for peak detection
f_low = 25000    #filter lower frequency limit for peak detection

noise_slength = 100000  # Initial length of data representing noise
chunksize=1000000       # Process the entire dataset in chunks of 1s duration

mywindow = 1000   #Half-width of the window taken around each found event
outputdir = '/data1/ryan/Spectrograms/' #Output directory for spectrograms

###################################Load datafile###############################
print('Loading ',myfile)
tdms_file = TdmsFile(myfile, memmap_dir='/data1/ryan/')  #set buffer directory
tddf = tdms_file.as_dataframe()  #Load TDMS file AS DATAFRAME

##################optional: cut out the ramp-up only based on IMAG channel############
#tddf = tddf.loc[np.array(mf.getchdata(tddf,'IMAG',1)) > 0.95]

########################Processing#############################################
plt.ioff()
logf = open(outputdir+'eventlog.csv', 'ab')
# h5f_ch1 = h5py.File('/data1/ryan/training_top_ch1.h5', 'w')


#Evaluate noise level in terms of scale-averaged value and set the threshold for peak detection
noisesample = getchdata(tddf,ch_to_load2,1) [0:noise_slength]
threshold = 5*noise_estimate(noisesample, f_high,f_low)   # 5 times noise level threshold

#Process the data
datalength = len(tddf) //chunksize
datalength = loc
print(datalength)
count1 = 0
count2 = 0
for I in range(loc_length) :

    print('Processing ' + str(I*chunksize/Fs) + '-' + str((I+1)*chunksize/Fs) + ' s')


    """#######################################Select data channel and interval for processing#####
        mydata0 = getchdata(tddf,ch_to_load1,1) [I*chunksize:(I+1)*chunksize]

        #########################################De-trend data###########################
        N = mydata0.size
        t = np.arange(0, N) / Fs
        p = np.polyfit(t, mydata0, 1)
        dat_norm = mydata0 - np.polyval(p, t)

        #####################Perform CWT####################################
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
        power = (np.abs(wave)) ** 2

        ############Plot 2D wavelet spectrum of the entire data chunck######################
        #plt.pcolormesh(t, freqs, power)
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')

        #######################Calculate spectrum scale average and significance#############
        period = 1 / freqs
        sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
        Cdelta = mother.cdelta
        scale_avg = (scales * np.ones((N, 1))).transpose()
        scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
        scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
        #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
        #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
        #plt.show()

        #####################Locate peaks in the scale-averaged spectrum############################################

        indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
        np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')

        #####################Take a window around each found peak and calcualte 2D wavelet transform##################

        indexes = indexes[(indexes>mywindow)]
        indexes = indexes[(indexes+mywindow)<len(mydata0)]
        print('Found ' + str(len(indexes)) + ' peaks')

        for indx in indexes:

        smdat = mydata0[indx-mywindow: indx+mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        if ()
        name = file_name + '_' + str(count1)
        h5f_ch1.create_dataset(name, data=smdat)
        with open('/data1/ryan/training_top_ch1.txt', 'a') as f:
        f.write(name)
        f.write('\n')
        count1 += 1"""


    #######################################Select data channel and interval for processing#####
    mydata0 = getchdata(tddf,ch_to_load2,1) [I*chunksize:(I+1)*chunksize]

    #########################################De-trend data###########################
    N = mydata0.size
    t = np.arange(0, N) / Fs
    p = np.polyfit(t, mydata0, 1)
    dat_norm = mydata0 - np.polyval(p, t)

    #####################Perform CWT####################################
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
    power = (np.abs(wave)) ** 2

    ############Plot 2D wavelet spectrum of the entire data chunck######################
    #plt.pcolormesh(t, freqs, power)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')

    #######################Calculate spectrum scale average and significance#############
    period = 1 / freqs
    sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
    scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
    #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
    #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
    #plt.show()

    #####################Locate peaks in the scale-averaged spectrum############################################

    indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
    np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')

    #####################Take a window around each found peak and calcualte 2D wavelet transform##################

    indexes = indexes[(indexes>mywindow)]
    indexes = indexes[(indexes+mywindow)<len(mydata0)]
    print('Found ' + str(len(indexes)) + ' peaks')

    for indx in indexes:

        smdat = mydata0[indx-mywindow: indx+mywindow]
        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
        sm_power = (np.abs(sm_wave)) ** 2
        sm_t = np.arange(0, smdat.size) / Fs

        name = name_to_use + '_' + str(indx + I * chunksize)
        if random.random() < 0.2:
            h5f_validation.create_dataset(name, data=smdat)
            with open('/data1/ryan/dataset/validation_new.txt', 'a') as f:
                f.write(name)
                f.write('\n')
            count1 += 1
        else:
            h5f_training.create_dataset(name, data=smdat)
            with open('/data1/ryan/dataset/training_new.txt', 'a') as f:
                f.write(name)
                f.write('\n')
            count2 += 1

with open('/data1/ryan/dataset/validation_new_num.txt', 'a') as f:
    f.write(file_name)
    f.write('\t')
    f.write(str(count1))
    f.write('\n')

with open('/data1/ryan/dataset/training_new_num.txt', 'a') as f:
    f.write(file_name)
    f.write('\t')
    f.write(str(count2))
    f.write('\n')

##################Plot and save the results################################

"""fig1 = plt.figure()
    plt.plot(smdat)
    plt.ylabel('Amplitude')
    plt.xlabel('Time [sec]')
    imname = 'dat'+str(indx+I*chunksize)
    plt.savefig(outputdir+imname+'.png')  #save an individual event plot######
    plt.close(fig1)

    fig2 = plt.figure()
    plt.pcolormesh(sm_t, sm_freqs, sm_power)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    imname = 'spec'+str(indx+I*chunksize)
    plt.savefig(outputdir+imname+'.png') #save an individual spectrogram plot######
    plt.close(fig2)

    fig3 = plt.figure(figsize = (1,1), dpi= 64)
    tsize = sm_t.max() - sm_t.min()
    plt.axis([sm_t.min() + 0.25*tsize , sm_t.max() -0.25*tsize, sm_freqs.min(), 0.75*sm_freqs.max()]) # select central region of the image
    plt.pcolormesh(sm_t, sm_freqs, sm_power, cmap=plt.get_cmap('gray'))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    imname = str(indx+I*chunksize)
    plt.savefig(outputdir+'images/'+imname+'.png', pad_inches = 0)  #save spectrogram as square b/w image######
    plt.close(fig3)"""


## Loading 103 now
#myfile = '/data1/stephey/SORTED/8_24_2017_2_46_30_PM_A129_Q103_15986A.tdms'
#file_name = '8_24_2017_2_46_30_PM_A129_Q103_15986A'
#
####################################Load datafile###############################
#print('Loading ',myfile)
#tdms_file = TdmsFile(myfile, memmap_dir='/data1/ryan/')  #set buffer directory
#tddf = tdms_file.as_dataframe()  #Load TDMS file AS DATAFRAME
#
##Evaluate noise level in terms of scale-averaged value and set the threshold for peak detection
#noisesample = getchdata(tddf,ch_to_load2,1) [0:noise_slength]
#threshold = 5*noise_estimate(noisesample, f_high,f_low)   # 5 times noise level threshold
#
##Process the data
#datalength = len(tddf) //chunksize
#print(datalength)
#count1 = 0
#count2 = 0
#for I in range(datalength) :
#
#    print('Processing ' + str(I*chunksize/Fs) + '-' + str((I+1)*chunksize/Fs) + ' s')
#
#    #######################################Select data channel and interval for processing#####
#    mydata0 = getchdata(tddf,ch_to_load2,1) [I*chunksize:(I+1)*chunksize]
#
#    #########################################De-trend data###########################
#    N = mydata0.size
#    t = np.arange(0, N) / Fs
#    p = np.polyfit(t, mydata0, 1)
#    dat_norm = mydata0 - np.polyval(p, t)
#
#    #####################Perform CWT####################################
#    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, 1/Fs, dj, s0, J, mother)
#    power = (np.abs(wave)) ** 2
#
#    ############Plot 2D wavelet spectrum of the entire data chunck######################
#    #plt.pcolormesh(t, freqs, power)
#    #plt.ylabel('Frequency [Hz]')
#    #plt.xlabel('Time [sec]')
#
#    #######################Calculate spectrum scale average and significance#############
#    period = 1 / freqs
#    sel = find((period >= 1/f_high) & (period < 1/f_low)) #select frequency band for averaging
#    Cdelta = mother.cdelta
#    scale_avg = (scales * np.ones((N, 1))).transpose()
#    scale_avg = power / scale_avg # As in Torrence and Compo (1998) equation 24
#    scale_avg = dj  / Fs* Cdelta * scale_avg[sel, :].sum(axis=0)
#    #scale_avg_signif, tmp = wavelet.significance(var, 1/Fs, scales, 2, alpha,significance_level=0.95,dof=[scales[sel[0]],scales[sel[-1]]],wavelet=mother)
#    #plt.plot(t, scale_avg, 'k-', linewidth=1.5)
#    #plt.show()
#
#    #####################Locate peaks in the scale-averaged spectrum############################################
#
#    indexes = peakutils.indexes(scale_avg, thres= threshold/max(scale_avg), min_dist=5000)
#    np.savetxt(logf,[(indexes+I*chunksize)],  fmt='%d', delimiter=',', newline=',')
#
#    #####################Take a window around each found peak and calcualte 2D wavelet transform##################
#
#    indexes = indexes[(indexes>mywindow)]
#    indexes = indexes[(indexes+mywindow)<len(mydata0)]
#    print('Found ' + str(len(indexes)) + ' peaks')
#
#    for indx in indexes:
#
#        smdat = mydata0[indx-mywindow: indx+mywindow]
#        sm_wave, sm_scales, sm_freqs, sm_coi, sm_fft, sm_fftfreqs = wavelet.cwt(smdat, 1/Fs, dj, s0, J, mother)
#        sm_power = (np.abs(sm_wave)) ** 2
#        sm_t = np.arange(0, smdat.size) / Fs
#
#        name = file_name + '_' + str(indx + I * chunksize)
#        if random.random() < 0.2:
#            h5f_validation.create_dataset(name, data=smdat)
#            with open('/data1/ryan/dataset/validation_bot_ch2.txt', 'a') as f:
#                f.write(name)
#                f.write('\n')
#            count1 += 1
#        else:
#            h5f_training.create_dataset(name, data=smdat)
#            with open('/data1/ryan/dataset/training_bot_ch2.txt', 'a') as f:
#                f.write(name)
#                f.write('\n')
#            count2 += 1
#
#with open('/data1/ryan/dataset/validation_peaks_num.txt', 'a') as f:
#    f.write(str(count1))
#
#with open('/data1/ryan/dataset/training_peaks_num.txt', 'a') as f:
#    f.write(str(count2))
#
'''
# Close the files
logf.close()
plt.ion()

########################STOP here################################################
