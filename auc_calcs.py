import numpy as np
import pandas as pd 
from sklearn import metrics
import datetime 

stim_path = 'organized/stim_times/stimTimes_835HLS1left_0.csv'
#file_path = 'organized/835HLS1right_1/dff_835HLS1right_1.csv'

df = pd.read_csv('scaled_835leftHLS1_sensory0.csv')
#df2 = df.copy(deep=True)

print(df.head())
print(df.dtypes)

###Initialize Variables 

dFF = 'dFF'
time = 'Time'
zdFF = 'zDFF'


### Get stimulation timepoints and sync them 
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
cal_start = 8.0644
vid_start = 5.647 
trials = []
for count in range(len(stims)):
    trial= stims[count] - vid_start + cal_start
    trials.append(trial)

### Whole Trace calculations
median = round(float(np.median(df['dFF'])),4)
sd     = round(float(np.std(df['dFF'])),4)
mean   = round(float(np.mean(df['dFF'])),4)

prc50 = np.percentile(df['dFF'],50)
#print(prc50)

meanLT50 = np.mean(df['dFF'].loc[df['dFF'] <= prc50])
medLT50 = np.median(df['dFF'].loc[df['dFF'] <= prc50])
sdLT50 = np.std(df['dFF'].loc[df['dFF'] <= prc50])

thresholdMed = median + (2*sd)

### Calculate z-score dFF 
df['zDFF'] = (df['dFF'] - meanLT50) / sdLT50
sdZDFF= np.std(df['zDFF'])

prc50Z = np.percentile(df['zDFF'],50)
sdLT50Z = np.std(df['zDFF'].loc[df['zDFF'] <= prc50Z])


### Calculate mean and standard deviation for the (X) seconds before stimulation 
baselines = []
sds       = []
for stim in trials:
     begin = stim - 3.0
     end   = stim

     in_between = df['Time'].between(begin, end, inclusive=True)

     baselines.append(np.mean(df['dFF'].loc[in_between]))
     sds.append(np.std(df['dFF'].loc[in_between]))

### Calculate AUC while above threshold 

##Find AUC 
aucBeforeStim = []
aucAfterStim  = []
aucDuringPeak = []
aucDuringPeakZ = []
aucBeforeStimZ = []
aucDuringPeakMean =[]

durationsMed = []
durationsMean = []
durationsZ = []

# Get thresholds for latency of response 

latencyMed = []
latencyMean = []
latencyZ = []

no_cross=0 
for i in range(len(trials)-1):
    auc1start = trials[i] - 3 
    auc1end   = trials[i]
    auc2start = trials[i]
    auc2end   = trials[i] + 5
    stim1 = trials[i]
    stim2 = trials[i+1]

    in_between1 = df['Time'].between(auc1start, auc1end, inclusive=True)
    in_between2 = df['Time'].between(auc2start, auc2end, inclusive=True)
    peak_indx = df.index[(df['Time'].between(stim1,stim2)) & (df['dFF'] >= medLT50)].tolist()
    peak_indxMean = df.index[(df['Time'].between(stim1,stim2)) & (df['dFF'] >= baselines[i])].tolist()
    peak_indxZ = df.index[(df['Time'].between(stim1,stim2)) & (df['zDFF'] >= 1.65)].tolist()
    try:
        peak_start = peak_indx[0]
        peak_end   = peak_indx[-1]
        peak_startMean= peak_indxMean[0]
        peak_endMean = peak_indxMean[-1]
        peak_startZ = peak_indxZ[0]
        peak_endZ   = peak_indxZ[-1]
    except:
        print(i)
        no_cross+=1
        continue 
    
    ##Get duration while response was above threshold 
    duration = df['Time'].iloc[peak_end] - df['Time'].iloc[peak_start]
    durationsMed.append(duration)
    duration2 = df['Time'].iloc[peak_endMean] - df['Time'].iloc[peak_startMean]
    durationsMean.append(duration2)
    duration3 = df['Time'].iloc[peak_endZ] - df['Time'].iloc[peak_startZ]
    durationsZ.append(duration3)

    ## Get latency of response 

    #AUC 3 sec before Stim
    auc1 = metrics.auc(df['Time'].loc[in_between1], df['dFF'].loc[in_between1])
    aucBeforeStim.append(auc1)
    #AUC 5 sec after stim
    auc2 = metrics.auc(df['Time'].loc[in_between2], df['dFF'].loc[in_between2])
    aucAfterStim.append(auc2)
    #AUC during peak Median
    auc3= metrics.auc(df['Time'].iloc[peak_start:peak_end], df['dFF'].iloc[peak_start:peak_end])
    aucDuringPeak.append(auc3)
    #AUC during peak Mean
    auc4= metrics.auc(df['Time'].iloc[peak_startMean:peak_endMean], df['dFF'].iloc[peak_startMean:peak_endMean])
    aucDuringPeakMean.append(auc4)
    ##zDFF calculations
    auc5= metrics.auc(df['Time'].iloc[peak_startZ:peak_endZ], df['zDFF'].iloc[peak_startZ:peak_endZ])
    aucDuringPeakZ.append(auc5)
    auc6= auc1 = metrics.auc(df['Time'].loc[in_between1], df['zDFF'].loc[in_between1])
    aucBeforeStimZ.append(auc6)


# roundList = [baselines, durationsMed, aucBeforeStim, aucAfterStim, aucDuringPeak, 
#             aucDuringPeakZ, aucBeforeStimZ]

# for variable in roundList:
#     variable = [round(num, 4) for num in variable]
#     print(variable)

### And make it pretty (Like Emily <3) :)) 

#Auc Rounds
baselines      = [round(num, 4) for num in baselines]
aucBeforeStim  = [round(num, 4) for num in aucBeforeStim]
aucAfterStim   = [round(num, 4) for num in aucAfterStim]
aucDuringPeak  = [round(num, 4) for num in aucDuringPeak]
aucDuringPeakZ = [round(num, 4) for num in aucDuringPeakZ]
aucBeforeStimZ = [round(num, 4) for num in aucBeforeStimZ]
aucDuringPeakMean = [round(num, 4) for num in aucDuringPeakMean]

#duration rounds 
durationsMed   = [round(num, 4) for num in durationsMed]
durationsMean  =  [round(num, 4) for num in durationsMean]
durationsZ     = [round(num, 4) for num in durationsZ]

### OUTPUTS 

print("Number of times didn't cross threshold:", no_cross)
print("\nAUC Before stim:", aucBeforeStim)
print("Mean AUC Before stim:",round(np.mean(aucBeforeStim),4))
print("AUC After stim:", aucAfterStim)
print("Mean AUC After stim:",round(np.mean(aucAfterStim),4))
print("AUC During Peak:", aucDuringPeak)
print("Mean AUC During Peak:", round(np.mean(aucDuringPeak),4))



#### Method 1 calculate AUC with threshold as median 
print('\n---------"Method 1"---------\n')
print("Method 1: calculate AUC with threshold as median\n")
print("Median = whole trace without firing (dFF below 50 percentile)")
print("Median:",round(medLT50,4))
print("AUC Before stim:", aucBeforeStim)
print("Mean AUC Before stim:",round(np.mean(aucBeforeStim),4))
print("AUC During Peak:", aucDuringPeak)
print("Mean AUC During Peak:", round(np.mean(aucDuringPeak),4))
print("Duration of response:", durationsMed)
print("Mean of response duration:", round(np.mean(durationsMed),4))


print('\n---------"Method 2"---------\n')
print("Method 2: calculate AUC with threshold as mean 3 seconds before stim")
print("Baselines = mean 3 sec before stim\n")
print("Mean whole trace:", mean)
print("Mean whole trace below 50p:", round(meanLT50,4))
print("Baslines:", baselines)
print("Mean of all baselines", round(np.mean(baselines),4))
print("AUC During Peak:", aucDuringPeakMean)
print("Mean AUC During Peak:", round(np.mean(aucDuringPeakMean),4))
print("Duration of response:", durationsMean)
print("Mean of response duration:", round(np.mean(durationsMean),4))


print('\n---------"Method 3"---------\n')
print("Method 3: Do calculations based on z-scored dFF")
print("Threshold = 1.65 SD (SD = 1 in z-scored dFF\n")

print("SD LT50Z:", round(sdLT50Z,4))
print("AUC Before stim Z:", aucBeforeStimZ)
print("Mean AUC Before stim Z:",round(np.mean(aucBeforeStimZ),4))
print("AUC During Peak Z:", aucDuringPeakZ)
print("Mean AUC During Peak Z:", round(np.mean(aucDuringPeakZ),4))
print("Duration of response:", durationsZ)
print("Mean of response duration:", round(np.mean(durationsZ),4))

print('\n---------"Summary"---------\n')

print("Mean AUC:")
print("Method 1:", round(np.mean(aucDuringPeak),4))
print("Method 2:", round(np.mean(aucDuringPeakMean),4))
print("Method 3:", round(np.mean(aucDuringPeakZ),4))

print("\nMean durations:")
print("Method 1:", round(np.mean(durationsMed),4))
print("Method 2:", round(np.mean(durationsMean),4))
print("Method 3:", round(np.mean(durationsZ),4))





# dfexp.to_csv('thresh_outputTest.csv', index=False)

    # print(df['Time'].iloc[peak_start])
    # print(df['Time'].iloc[peak_end])