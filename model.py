import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Define the catalog directory and file path
cat_directory = './data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
figures_directory = './saved_figures/'

# Load the CSV file into a DataFrame
cat = pd.read_csv(cat_file)

# Drop the first row (index 0)
cat = cat.drop(index=0)
detection_times = []
fnames = []
# Make the characteristic function for each dataset
for i in range(len(cat)):
    catalogcur = cat.iloc[i]
    file_name = catalogcur['filename']
    arrivaltime = datetime.strptime(catalogcur['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
    arrival_time_rel = catalogcur['time_rel(sec)']
    
    # Construct the file paths
    intervaldirectory = './data/lunar/training/data/S12_GradeA/'
    intervalfile = f'{intervaldirectory}{file_name}.csv'
    
    if not os.path.exists(intervalfile):
        print(f"File not found: {intervalfile}")
        continue  
    
    intervaldata = pd.read_csv(intervalfile)
    
    mseed_file = f'{intervaldirectory}{file_name}.mseed'  
    if not os.path.exists(mseed_file):
        print(f"MiniSEED file not found: {mseed_file}")
        continue 

    st = read(mseed_file)  
    

    tr = st[0]
    starttime = tr.stats.starttime.datetime
    # Check the sampling rate
    df = tr.stats.sampling_rate
    # Filter the data
    tr.detrend('linear')  # Remove linear trends
    tr.filter('bandpass', freqmin=0.5, freqmax=2.0)  # Adjust bandpass filter to focus on relevant frequencies

    # Compute STA/LTA
    sta_len = 120  # Short-term average window in seconds
    lta_len = 600  # Long-term average window in seconds
    cft = classic_sta_lta(tr.data, int(sta_len * df), int(lta_len * df))  # Calculate characteristic function
    # Increase STA/LTA thresholds to filter out background noise
    thr_on = 4  # Higher threshold for event detection (trigger on)
    thr_off = 2  # Lower threshold to turn off the trigger
    on_off = trigger_onset(cft, thr_on, thr_off)  # Detect trigger onset and offset times
    # Plot the characteristic function and triggers
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))

    detection_times = []
    fnames = [] 
    # Plot trigger onset and offset
    for i in np.arange(0,len(on_off)):
        triggers = on_off[i]
        on_time = starttime +timedelta(seconds = tr.times[triggers[0]])
        on_time_str = datetime.strftime(on_time,'%Y-%m-%dT%H:%M:%S.%f')
        detection_times.append(on_time_str)
        fnames.append(file_name)
        ax.axvline(x=tr.times[triggers[0]], color='red', label='Trig. On' )
        ax.axvline(x=tr.times[triggers[1]], color='purple', label='Trig. Off')

ax.plot(tr.times,tr.data)
ax.set_xlim([min(tr.times),max(tr.times)])
ax.legend()



    # Add a title and save the plot
plt.title(f"Seismogram with Triggers for {file_name}, type: {catalogcur['mq_type']}", fontweight='bold')
plt.savefig(os.path.join(figures_directory, f'characteristic_function_{file_name}.png'))  # Save plot
plt.close(fig)  # Close the figure to save memory
detect_df = pd.DataFrame(data = {'filename':fnames, 'time_abs(%Y-%m-%dT%H:%M:%S.%f)':detection_times, 'time_rel(sec)':tr_times[triggers[0]]})

# detect_df.head()
detect_df.to_csv('output/path/catalog.csv', index=False)


