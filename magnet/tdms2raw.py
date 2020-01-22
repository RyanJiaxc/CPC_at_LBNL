from nptdms import TdmsFile
import h5py


# Function to retrieve a data channel as numpy array
def getchdata(dframe, ch_to_load, resample):
    data = []
    for x in dframe.columns :   
        if x.find(ch_to_load) != -1 :
            data = dframe.loc[:,x].values [0::resample]       
            if len(data) == 0 : print('channel not found')
            else: print(ch_to_load, len(data))

    return data


# Filename and channels to load
myfile = '/data1/ryan/8_17_2017_4_58_50_PM_A81_Q61_15461A.tdms'
chunk_size = 30600
file_name = '8_17_2017_4_58_50_PM_A81_Q61_15461A'
path = '/data1/ryan/'

ch_to_load1 = 'S_top'  # acoustic channel 1
ch_to_load2 = 'S_bot'  # acoustic channel 2
ch_to_load3 = 'IMAG'  # magnet current channel, multiply by 1920

# This loads the entire file as dataframe
print('Loading ', myfile)
tdms_file = TdmsFile(myfile, memmap_dir=path)
tddf = tdms_file.as_dataframe()

finalData1 = getchdata(tddf, ch_to_load1, 1)
finalData2 = getchdata(tddf, ch_to_load2, 1)
finalData3 = getchdata(tddf, ch_to_load3, 1)

# Save data as .h5 files
h5_ch1 = h5py.File(path + 'S_top_ch1.h5', 'w')
h5_ch2 = h5py.File(path + 'S_bot_ch2.h5', 'w')
h5_ch3 = h5py.File(path + 'IMAG_ch3.h5', 'w')

i = 0
count = 0
while i+chunk_size < len(finalData1):
    name = file_name + '_' + str(count)
    h5_ch1.create_dataset(name, data=finalData1[i:i + chunk_size])
    with open(path + 'S_top_ch1.txt', 'a') as f:
        f.write(name)
        f.write('\n')
    count += 1
    i += chunk_size
h5_ch1.close()

i = 0
count = 0
while i + chunk_size < len(finalData2):
    name = file_name + '_' + str(count)
    h5_ch2.create_dataset(name, data=finalData2[i:i + chunk_size])
    with open(path + 'S_bot_ch2.txt', 'a') as f:
        f.write(name)
        f.write('\n')
    count += 1
    i += chunk_size
h5_ch2.close()

i = 0
count = 0
while i + chunk_size < len(finalData3):
    name = file_name + '_' + str(count)
    h5_ch3.create_dataset(name, data=finalData3[i:i + chunk_size])
    with open(path + 'IMAG_ch3.txt', 'a') as f:
        f.write(name)
        f.write('\n')
    count += 1
    i += chunk_size
h5_ch3.close()
