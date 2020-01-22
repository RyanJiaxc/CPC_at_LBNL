import os
import h5py
import soundfile as sf

trainroot = ['/data1/ryan/train-clean-100']
devroot = ['/data1/ryan/dev-clean']
testroot = ['/data1/ryan/test-clean']

"""Convert flac files to raw wave form and list txts to store
"""

# store training set
h5f = h5py.File('/data1/ryan/train-Librispeech.h5', 'w')
for rootdir in trainroot:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.flac'):
                fullpath = os.path.join(subdir, file)
                data, samplerate = sf.read(fullpath)
                # fs, data = wavfile.read(fullpath)
                h5f.create_dataset(file[:-5], data=data)
                with open('/data1/ryan/train.txt', 'a') as f:
                    f.write(file[:-5])
                    f.write('\n')

                print(file[:-5])
h5f.close()

# store dev set
h5f = h5py.File('/data1/ryan/validation-Librispeech.h5', 'w')
for rootdir in devroot:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.flac'):
                fullpath = os.path.join(subdir, file)
                data, samplerate = sf.read(fullpath)
                # fs, data = wavfile.read(fullpath)
                h5f.create_dataset(file[:-5], data=data)
                with open('/data1/ryan/validation.txt', 'a') as f:
                    f.write(file[:-5])
                    f.write('\n')
                
                print(file[:-5])
h5f.close()

# store test set
h5f = h5py.File('/data1/ryan/eval-Librispeech.h5', 'w')
for rootdir in testroot:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.flac'):
                fullpath = os.path.join(subdir, file)
                data, samplerate = sf.read(fullpath)
                # fs, data = wavfile.read(fullpath)
                h5f.create_dataset(file[:-5], data=data)
                with open('/data1/ryan/eval.txt', 'a') as f:
                    f.write(file[:-5])
                    f.write('\n')
                
                print(file[:-5])
h5f.close()
