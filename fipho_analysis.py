import numpy as np
import pandas as pd 
from sklearn import metrics
import datetime 

### Open Files 
# df = pd.read_csv('835leftHLS1_sensory0.csv')

df=pd.read_csv('zDFF_test.csv')
print(df.head())
print(df.dtypes)

df2= df.copy(deep=True)

### Initialize variables
timestamp = 'Time' 
dff       = 'dFF'

### Set video allignment and get trials 


### 835leftHLS1_sensory0.csv'
cal_start = 8.0644
vid_start = 5.647 
stims     = [24.541, 35.193, 45.461, 63.371, 71.554, 95.954, 108.658, 134.592, 147.914, 163.671]

# cal_start = 15.3132
# vid_start = 12.679
# #stims     = [24.541, 35.193, 45.461, 63.371, 71.554, 95.954, 108.658, 134.592, 147.914, 163.671]

#Get stim times in seconds from csv file 
# stimsDF = pd.read_csv('stim_test.csv', names=['file', 'index', 'timestamps', 'limb', 'stim_type'])
# time_list = stimsDF['timestamps'].tolist()
# stims = []
# for t in range(len(time_list)):
#     try:
#         date_time = datetime.datetime.strptime(time_list[t], "%M:%S.%f")
#     except:
#         stims.append(float(time_list[t]))
#         continue
#     a_timedelta = date_time - datetime.datetime(1900, 1, 1)
#     seconds = a_timedelta.total_seconds()
#     stims.append(seconds)
# print(stims)

#get synced trial times 
trials = []
for count in range(len(stims)):
    trial= stims[count] - vid_start + cal_start
    trials.append(trial)
#print("Trials:", trials)


### Whole Trace calculations
median = round(float(np.median(df['dFF'])),4)
sd     = float(np.std(df['dFF']))
print("Median:", median, " Standard deviation:", sd)


  
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

# print("Baslines:", baselines)
# print("Mean of all baselines", np.mean(baselines))
#print(sds)

### Get thresholds for response 
thresholds = np.array(baselines) + 3*np.array(sds)
# print("Thresholds of Response:", thresholds)

### Calculate latency of response and peak of response for each stimulation 
latencies = []
peaks     = []
# maximum = [] 
for i in range(len(trials)-1):
    stim1 = trials[i]
    stim2 = trials[i+1]

    #Find latency of response
    lat_indx = df.index[(df['Time'].between(stim1,stim2)) & (df['dFF'] >= thresholds[i])].tolist()
    #print(lat_indx[i])
    cross_threshold = df['Time'].iloc[lat_indx[4]] 
    latencies.append(cross_threshold - trials[i])

    #Find peak dFF value of response
    in_between = df['Time'].between(stim1, stim2, inclusive=True)
    peak_indx  = df['dFF'].loc[in_between].idxmax()
    peaks.append(df['dFF'].iloc[peak_indx])

    #another method to find peak dFF  
    #maximum.append(np.max(df['dFF'].loc[in_between]))

    # from below find first True
    # np.argwhere , np.where 

    #duation of response
    #print(df['dFF'].loc[in_between.loc[in_between.iloc[5]:]] <= baselines[i])

    #latency of response
    #print(df['dFF'].loc[in_between] >= thresholds[i])

    #print(metrics.auc(df['Time'].loc[in_between], df['dFF'].loc[in_between]))
    pass


#print(len(baselines),len(peaks),len(latencies))

for i in range(len(baselines)):
    baselines[i] = round(baselines[i],4)

for i in range(len(peaks)):
    peaks[i] = round(peaks[i],4)
    latencies[i] = round(latencies[i],4)

print("Baslines:", baselines)
print("Mean of all baselines", np.mean(baselines))

print("Peaks:", peaks)
#rint("Max:", maximum)
print("Mean of peaks:", np.mean(peaks))
print("Latencies:", latencies)
print("Mean of latencies:", np.mean(latencies))

print('---------******---------')
print(df2.head())
print(df2.dtypes)

max_value = np.max(df2['dFF'])
max_10p= max_value * .1
# print(max_value)
# print(max_10p)

from scipy import stats
df2.loc[df2[dff] >= max_10p] = max_10p
MAD = df2[dff].mad()
MAD2 = stats.median_absolute_deviation(df2[dff])
max2 = np.max(df2[dff])
# print(max2)
median2 = round(float(np.median(df2['dFF'])),4)
mean2   = round(float(np.mean(df2['dFF'])),4)
sd2     = round(float(np.std(df2['dFF'])),4)
print("Median:", median2)
print("Mean:", mean2)
print("Std deviation:", sd2)
print("MAD:", MAD)
print("MAD2:", MAD2)




#df2.to_csv('calc_test.csv', index=False)


# 1:03.371
# 1:11.554
# 1:35.954
# 1:48.658
# 2:14.592
# 2:27.914
# 2:43.671



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
