import numpy as np
import pandas as pd 
from sklearn import metrics
import datetime 
import matplotlib.pyplot as plt

### Open Files 

#stim_path = 'organized/stim_times/stimTimes_835HLS1left_0.csv'
stim_path = 'organized/835HLS1right_1/stimTimes_835HLS1right_1.csv'

#file_path = 'organized/835HLS1right_1/dff_835HLS1right_1.csv'

df = pd.read_csv('scaled_835leftHLS1_sensory0.csv')

#'organized/833HLS1right_1/raw_833HLS1right_1.csv'

#'organized/833HLS1left_0/raw_833HLS1left_0.csv'
#'organized/835HLS1left_0/raw_835HLS1left_0.csv'
#'organized/835HLS1leftfootdrop/raw_835HLS1leftfootdrop.csv'
#'organized/833HLS1right_footdrop/raw_833HLS1right_footdrop.csv'


#df=pd.read_csv(file_path)
print(df.head())
print(df.dtypes)

df2 = df.copy(deep=True)
df3 = df.copy(deep=True)

### Initialize variables
timestamp = 'Time' 
dff       = 'dFF'


### Set video allignment and get trials 


### 835leftHLS1_sensory0.csv'
# cal_start = 8.0644
# vid_start = 5.647 
# stims     = [24.541, 35.193, 45.461, 63.371, 71.554, 95.954, 108.658, 134.592, 147.914, 163.671]

### 835HLS1right_1.csv 
cal_start = 15.3132
vid_start = 12.679
#stims     = [24.541, 35.193, 45.461, 63.371, 71.554, 95.954, 108.658, 134.592, 147.914, 163.671]

### Plot dff 

# fig = plt.figure(figsize=(14, 6))
# ax1 = fig.add_subplot(111)
# ax1.set_ylim([-5,10])
# ax1.plot(df[dff],'black',linewidth=1)
# plt.show()

###Get stim times in seconds from csv file 

stimsDF = pd.read_csv(stim_path, names=['file', 'index', 'timestamps', 'limb', 'stim_type'])
time_list = stimsDF['timestamps'].tolist()
stims = []
for t in range(len(time_list)):
    try:
        date_time = datetime.datetime.strptime(time_list[t], "%M:%S.%f")
    except:
        stims.append(float(time_list[t]))
        continue
    a_timedelta = date_time - datetime.datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    stims.append(seconds)
print("Stims:", stims)

#get synced trial times 
trials = []
for count in range(len(stims)):
    trial= stims[count] - vid_start + cal_start
    trials.append(trial)
#print("Trials:", trials)


### Whole Trace calculations
median = round(float(np.median(df['dFF'])),4)
sd     = round(float(np.std(df['dFF'])),4)
#print("Median:", median, " Standard deviation:", sd)

  
### Calculate mean and standard deviation for the (X) seconds before stimulation 
baselines = []
sds       = []
for stim in trials:
     begin = stim - 3.0
     end   = stim

     in_between = df['Time'].between(begin, end, inclusive=True)

     baselines.append(np.mean(df['dFF'].loc[in_between]))
     sds.append(np.std(df['dFF'].loc[in_between]))

     #print("Mean from {} to {}: {}".format(begin, end,np.mean(df['dFF'].loc[in_between])))


### Get thresholds for response 
thresholds = np.array(baselines) + 3*np.array(sds)
# for i in range(len(thresholds)):
#     thresholds[i] = round(thresholds[i],4)

thresholds = [round(num, 4) for num in thresholds]
print("Thresholds of Response:", thresholds)

### Calculate latency of response and maxium dFF value during stimulation
latencies = []
maximum   = []
no_cross_lat = 0
for i in range(len(trials)-1):
    stim1 = trials[i]
    stim2 = trials[i+1]

    # Find maximum dFF value between stimulations (Amplitude?)
    in_between = df['Time'].between(stim1, stim2, inclusive=True)
    maximum.append(np.max(df['dFF'].loc[in_between]))

    # Find latency of response
    lat_indx = df.index[(df['Time'].between(stim1,stim2)) & (df['dFF'] >= thresholds[i])].tolist()
    try:
        cross_threshold = df['Time'].iloc[lat_indx[0]] 
    except:
        no_cross_lat += 1 
        continue
    latencies.append(cross_threshold - trials[i])

##Find AUC 
aucBeforeStim = []
aucAfterStim  = []
thresh = median *1.65
print(thresh)
for i in range(len(trials)-1):
    auc1start = trials[i] - 3 
    auc1end   = trials[i]
    auc2start = trials[i]
    auc2end   = trials[i] + 5

    threshTest = np.where(df['dFF'] > )

    in_between1 = df['Time'].between(auc1start, auc1end, inclusive=True)
    in_between2 = df['Time'].between(auc2start, auc2end, inclusive=True)

    auc1 = metrics.auc(df['Time'].loc[in_between1], df['dFF'].loc[in_between1])
    aucBeforeStim.append(auc1)
    auc2 = metrics.auc(df['Time'].loc[in_between2], df['dFF'].loc[in_between2])
    aucAfterStim.append(auc2)
    #print(auc1, auc2)
    #print(auc1start, auc1end, auc2start, auc2end)


# from below find first True
# np.where 
#print(metrics.auc(df['Time'].loc[in_between], df['dFF'].loc[in_between]))

### Rounding to make it pretty 
baselines = [round(num, 4) for num in baselines]
peaks     = [round(num, 4) for num in peaks]
latencies = [round(num, 4) for num in latencies]
aucBeforeStim = [round(num, 4) for num in aucBeforeStim]
aucAfterStim  = [round(num, 4) for num in aucAfterStim]


print('\n---------"Outputs"---------\n')

print("Whole trace median:", median)
print("Whole trace STD:", sd)
print("Mean of all baselines", round(np.mean(baselines),4))
print("Baslines:", baselines)
print("Peaks:", peaks)
#rint("Max:", maximum)
print("Mean of peaks:", round(np.mean(peaks),4))
print("Latencies:", latencies)
print("Mean of latencies:", round(np.mean(latencies),4))
print("AUC Before stim:", aucBeforeStim)
print("Mean AUC Before stim",round(np.mean(aucBeforeStim),4))
print("AUC After stim:", aucAfterStim)
print("Mean AUC After stim",round(np.mean(aucAfterStim),4))

print('\n---------"Method 2"---------\n')
from scipy import stats
MAD = df3[dff].mad()
# MAD2 = stats.median_absolute_deviation(df3[dff])
MAD3 = stats.median_abs_deviation(df3[dff])
print("MAD:", MAD)
# print("Mad2:", MAD2)
print("MAD3:", MAD3)


#df2.to_csv('calc_test.csv', index=False)


### Extra code 

### Find maximum dFF value for each stimulation 
'''
peaks     = []
for i in range(len(trials)-1):
    stim1 = trials[i]
    stim2 = trials[i+1]
    in_between = df['Time'].between(stim1, stim2, inclusive=True)

    #Find peak dFF value of response
    peak_indx  = df['dFF'].loc[in_between].idxmax()
    peaks.append(df['dFF'].iloc[peak_indx])
    #another method to find peak dFF  
    maximum.append(np.max(df['dFF'].loc[in_between]))
'''
#print(len(baselines),len(peaks),len(latencies), len(maximum))

# for i in range(len(peaks)):
#     peaks[i] = round(peaks[i],4)
#     latencies[i] = round(latencies[i],4)


# for row in df['time']:
#     df['time'] = int(df['time']) * 1000

#trial_time= stims[0] - vid_start + cal_start 
#print(trial_time)

# print('---------******---------')
# print(df.head())

# index = df.index[df['dFF'] >= thresholds[0]].tolist()
# print(index)


# print(in_between)
# lat_indx = np.where(in_between == True)
# latency = df.index[df['dFF'].loc[in_between] >= thresholds[i]]
# print(latency)
# indices = np.where(latency == True)


# ceiling= np.percentile(df[dff], 10)
# print(ceiling)

##################################


    

    #duation of response
    #print(df['dFF'].loc[in_between.loc[in_between.iloc[5]:]] <= baselines[i])

    #latency of response
    #print(df['dFF'].loc[in_between] >= thresholds[i])

##############
# print('---------"Method 1"---------')
# print(df2.head())
# print(df2.dtypes)

# max_value = np.max(df2['dFF'])
# print(max_value)
# max_10p= max_value * .1
# print(max_10p)
# df2.loc[df2[dff] >= max_10p] = max_10p
# max2 = np.max(df2[dff])
# print(max2)
# median2 = round(float(np.median(df2['dFF'])),4)
# mean2   = round(float(np.mean(df2['dFF'])),4)
# sd2     = round(float(np.std(df2['dFF'])),4)
# print("Median:", median2)
# print("Mean:", mean2)
# print("Std deviation:", sd2)

# fig = plt.figure(figsize=(14, 6))
# ax1 = fig.add_subplot(111)
# ax1.set_ylim([-5,10])
# ax1.plot(df2[dff],'black',linewidth=1)
# plt.show()

###################
'''
Files
### 835leftHLS1_sensory0.csv'
cal_start = 8.0644
vid_start = 5.647 
stims     = [24.541, 35.193, 45.461, 63.371, 71.554, 95.954, 108.658, 134.592, 147.914, 163.671]

### 835HLS1right_1.csv 
cal_start = 15.3132
vid_start = 12.679
stims     = [28.086, 92.054, 102.400, 117.719, 144.021, 95.954, 108.658, 134.592, 147.914, 163.671]

Video name	            Start time	Calcium signal name	    Start time
# 833_leftHLS1_Footdrop	32.824s		833HLS1leftfootdrop.csv	12.5160s
833_leftHLS1_Sensory	13.655s		833HLS1left_0.csv	    19.0264s
833_rightHLS1_Footdrop	4.996s		833HLS1right_footdrop.csv	7.3397s
833_rightHLS1_sensory	4.779s		833HLS1right_1	        6.7913s
835_leftHLS1_Footdrop	11.678s		835HLS1leftfootdrop	    10.9412s
835_leftHLS1_Sensory	5.647s		835HLS1left_0	        8.0644s
835_rightHLS1_Sensory	12.679s		835HLS1right_1	        15.3132s

### 833HLS1leftfootdrop.csv
cal_start = 12.5160
vid_start = 32.824s
stims     = []

'''
