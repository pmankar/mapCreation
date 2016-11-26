from __future__ import division
import numpy as np
import pandas as pd
import custKmeans as ck
import matplotlib.pyplot as plt

f = pd.read_csv("combined_2_51-61.csv")
f = f[f['sid']==108]
x = np.column_stack ([f.lat, f.lon, f.accuracy])
pd.options.display.precision = 11
np.set_printoptions(precision=11)

def remOutlier(X, k):
    check = True
    initX = X
    maxPer2remove = 0.6 # ? can be configured
    clusterSizeThreashold = 0.35 # ? 0.15 hardCoded for now. this should be computed dynamically
    while check == True:
        if len(X) < len(initX) * maxPer2remove :
            return initX
        p1 = ck.getInitCenters(X, k)
        c = ck.kmeans(X, p1, metric=ck.weightedK)
        freq = np.zeros(k)
        check = False
        for i in range(k):
            freq[i] = list(c[1]).count(i) / len(c[1])
        valsToRemove = np.array([])
        for i in range(k):
            if (freq[i] < clusterSizeThreashold):
                tmp = np.where(c[1]==i)[0]
                valsToRemove = np.append(valsToRemove, tmp, axis = 0)
        if len(valsToRemove) > 0:
            X = np.delete(X, valsToRemove, axis = 0)
            check = True

    return X
ol = remOutlier(x, 2)
r = ck.kmeans(ol, ck.getInitCenters(ol,2),metric=ck.weightedK)
print r[0]


plt.scatter(ol[:,0], ol[:,1],c=2+r[1],marker='D')
plt.scatter(r[0][:,0], r[0][:,1], marker='o' , s=200, color='red', alpha=0.4)
plt.show()
