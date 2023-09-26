# OcDa-Files
Repo to hold files for my work with the [Ocean Data Lab](https://sites.uw.edu/abadi/)


## NB_x
Majority of work done by me will be in Jupyter Notebooks titled 'NB_x.ipynb'.

### NB_1
This notebook includes inital work done by me in my early months of being part of the lab. Lots of plotting. Worked on familiarizing myself with the usage of MSEED/PSD pickle files. Also adapted SPDF plotting code found in (https://github.com/Ocean-Data-Lab/Website-backend/blob/master/SpecGraph/SPDF.py) to be used for my own purposes. Has later work on transmission/receive/source level which is refactored into NB_2 and 3. Also started work on Cepstrum Plotting in this.

#### Cepstrum.ipynb
Created to organize the cepstrum code. Calculated using 1-minute windows of 10-minute acoustic time series. Putting it in these windows allows us to get the minutes for the spectrogram or the x-axis.

Also has code to see if we can recreate a spectrogram by using sine waves to make our own signal. We also try to recreate a spectrogram using parts from the cepstrum since we were investigating whether we could filter out noise using the cepstrum.

#### Histograms_by_type.ipynb
Notebook to plot histograms by speed or distance to see distribution. Back then, used Khirod's (former lab member) dataset which might have had differences in what features were named/included. File is defunct, basically, since it uses outdated dataset.

### NB_2
A change was made to the dataset we use (originally in a folder labelled avg_time=10). We swapped to avg_time=1 for better frequency content. I used this change as an opportunity to refactor/reorganize code and made this notebook.

As written in the notebook, the focus was on comparing plots between ships. Three MMSIs were chosen for examination. If I recall, I planned this out by looking at the 3 different locations primarily and then looking into the 3 different MMSIs at each location. Plots were examined to see where differences occur and whether certain factors influenced the plots.

### NB_3 
This notebook takes the idea from NB_2 but repositions it to be specific MMSI-focused. Instead of taking a look at the different locations and the MMSIs, we instead look at the behavior of the MMSIs at different locations and also plot source levels versus frequency and distance (this connects back to literature models for source levels).