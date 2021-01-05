import numpy as np
import pandas as pd 
from sklearn import metrics
import datetime 
import matplotlib.pyplot as plt

### Open Files 

stim_path = 'organized/stim_times/stimTimes_835HLS1left_0.csv'
#stim_path = 'organized/835HLS1right_1/stimTimes_835HLS1right_1.csv'

#file_path = 'organized/835HLS1right_1/dff_835HLS1right_1.csv'

df = pd.read_csv('scaled_835leftHLS1_sensory0.csv')

#df=pd.read_csv(file_path)
print(df.head())
print(df.dtypes)

df2 = df.copy(deep=True)
df3 = df.copy(deep=True)

### Initialize variables
timestamp = 'Time' 
dff       = 'dFF'

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



### Plot dff 
def plot_subsets(axes, x, y, color="black", xlabel="Time (seconds)", ylabel="dFF"):
    # Set y-limit to center dFF values
    axes.set_ylim([-5,10])
   
    # Plot the inputs x,y in the provided color
    axes.plot(x, y, color=color)
    # Set the x-axis label
    axes.set_xlabel(xlabel)
    # Set the y-axis label
    axes.set_ylabel(ylabel, color=color)
    # Set the colors tick params for y-axis
    axes.tick_params('y', colors=color)
    # x_ticks= list(range(-4,6))
    # axes.set_xticklabels(x_ticks)
   
###########

#plot whole trace
fig, ax0 = plt.subplots()
plot_subsets(ax0,df['Time'],df[dff])
for stim in trials:
    ax0.axvline(x=stim)
plt.show()

fig,ax1 = plt.subplots()
#Plot trials
for stim in trials:
     begin = stim - 3.0
     end   = stim + 5
     in_between = df['Time'].between(begin, end, inclusive=True)
     plot_subsets(ax1, test, df[dff].loc[in_between])
print(len(in_between))
plt.show()


print(test)





