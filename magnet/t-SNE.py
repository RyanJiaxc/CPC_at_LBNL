import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

result = np.load('/data1/ryan/context_0.npy')
for i in range(1, 83):
    temp = np.load('/data1/ryan/context_' + str(i) + '.npy')
    result = np.concatenate((result, temp))

embedded = TSNE(n_components=2, perplexity=45).fit_transform(result)

np.save('/data1/ryan/embedded1', embedded)

print('finished!')
