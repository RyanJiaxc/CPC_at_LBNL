from nptdms import TdmsFile
from matplotlib import pyplot as plt

###########Function to retrieve a data channel as numpy array#####################################
def getchdata(dframe, ch_to_load, resample):
    data =[]
    for x in dframe.columns :   
        if x.find(ch_to_load) != -1 :
            data = dframe.loc[:,x].values [0::resample]       
            if len(data) == 0 : print('channel not found')
            else: print(ch_to_load, len(data))

    return data
##########################Filename and channels to load###########################################
myfile ='G:/Data/CCT4/8_24_2017_2_46_30_PM_A129_Q103_15986A.tdms' # filename; replace as needed

ch_to_load1 = 'S_top' #acoustic channel 1
ch_to_load2 ='S_bot' #acoustic channel 2
ch_to_load3 = 'IMAG' # magnet current channel, value needs to be multiplied by 1920 to 
                   #get current in Amperes

###########################Index location of the event ############################################

location = 12345678 # index of the sample event. Sinse sampling rate is 1 MHz, this is equal to  
#################### microseconds from the start of the ramp

############################This loads the entire file as dataframe################################
print('Loading ',myfile)
tdms_file = TdmsFile(myfile, memmap_dir='c:/Temp')  #set buffer directory
tddf = tdms_file.as_dataframe()  #LOADS FILE myfile AS DATAFRAME

#############################Load single event - 5 ms duration window ###############################

ch0_sample = getchdata(tddf,ch_to_load1,1)[location-1000: location+4000]
ch1_sample = getchdata(tddf,ch_to_load2,1)[location-1000: location+4000]
imag_sample = getchdata(tddf,ch_to_load3,1)[location-1000: location+4000]

##################################################Plot the event#####################################
plt.figure(1)
plt.plot(ch0_sample)
plt.plot(ch1_sample)
plt.show()
################################################Print magnret current value##########################
imag_value = 1920*imag_sample.mean()
print('Magnet current is ',imag_value, ' A')