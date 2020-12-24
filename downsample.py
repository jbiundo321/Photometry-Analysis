import numpy as np
import pandas as pd 
from photometry_functions import *
import datetime

# df = pd.read_csv('test.csv')
# df.drop(df.index[:500],axis=0, inplace= True)
# df.drop('raw', axis=1,  inplace= True)
# df.drop('trigger', axis=1, inplace = True)

# 3:12.102 
# 3:22.619
# 3:30.927
# 3:54.570
# 4:23.638
# 4:43.962
stimsDF = pd.read_csv('stim_test.csv', names=['file', 'index', 'timestamps', 'limb', 'stim_type'])
time_list = stimsDF['timestamps'].tolist()
print(stimsDF.head())
print(stimsDF.dtypes)
print(time_list)


seconds_list = []
for t in range(len(time_list)):
    try:
        date_time = datetime.datetime.strptime(time_list[t], "%M:%S.%f")
    except:
        seconds_list.append(float(time_list[t]))
        continue
    a_timedelta = date_time - datetime.datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    seconds_list.append(seconds)

print(seconds_list)





'''

df = pd.read_csv('835HLS1left_0.csv', names= ['time', '405', '465', 'combo','trigger', 'extra1', 'extra2'], low_memory=False)
df.drop(df.index[:6000],axis=0, inplace= True)
df.drop('combo', axis=1,  inplace= True)
df.drop('trigger', axis=1, inplace = True)
df.drop('extra1', axis=1,  inplace= True)
df.drop('extra2', axis=1, inplace = True)
df = df.astype('float64')

print(df.head())
print(df.dtypes)

raw405 = df['405'].to_numpy()
raw465 = df['465'].to_numpy()
times  = df['time'].to_numpy()
size = np.size(raw405)

#print(raw405)

N = 200
time = [] 
f405 = []
f465 = [] 

# downsample time
for i in range(0, size, N ):
    time.append(np.mean(times[i:i+N-1]))

# downsample 405
for i in range(0, size, N ):
    f405.append(np.mean(raw405[i:i+N-1]))

# downsample 465
for i in range(0, size, N ):
    f465.append(np.mean(raw465[i:i+N-1]))


time = np.array(time)
f405 = np.array(f405)
f465 = np.array(f465)

# time = time.transpose()
# f405 = f405.transpose()
#f465 = f465.transpose()

print(np.shape(raw405))
print("f405:")#f405)
print(np.shape(f405))
print("size:", size)
print(len(f405))
print('------')
print("f465:") #f465)
print(np.shape(f465))
print("size:", size)
print(len(f465))
print('------')
print("time:") #time)
print("size:", size)
print(len(time))

print(f405)

columns = {'time': time, 'f405': f405, 'f465': f465}
testDF = pd.DataFrame(columns)
pd.DataFrame()
print(testDF.head())
print(testDF.dtypes)

#testDF.to_csv('test_output.csv', index=False)

testDF.to_csv('downsampled.csv', index=False)

# zdFF = get_zdFF(f405,f465)

#zdFF = get_zdFF(raw405,raw465)
# print("zdff size:", np.size(zdFF))
# print(size)

#print(zdFF)

#zDF  = pd.DataFrame({'time': times, 'zDFF': zdFF})
# print(zDF.head())
# print(zDF.dtypes)

    #duation of response

    # duration_start = trials[i] + latencies[i]
    # between= df[timestamp].between(duration_start,stim2, inclusive=True)
    # df[dff].loc[in_between] <= baselines[i]
    # print(df['dFF'].loc[in_between.loc[in_between.iloc[5]:]] <= baselines[i])

'''

