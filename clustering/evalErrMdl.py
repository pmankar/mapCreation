from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import random
from datetime import datetime
from datetime import timedelta


def realGPSAccuracyCurve():
    e = pd.read_csv('combinedAcc20.csv')
    acc = 15
    e = e[e['accuracy']<=acc]
    e['sts'] = pd.to_datetime(e.sts ,format="%Y-%m-%d %H:%M:%S.%f")

    startDate = datetime.strptime('2016-07-01', '%Y-%m-%d')    # YYYY-MM-DD format
    endDate   = datetime.strptime('2016-08-01', '%Y-%m-%d')    # YYYY-MM-DD format
    e = e.ix[( e['sts'] >= startDate) & ( e['sts'] <= endDate)]

    vids = e.modelname.unique()
    #vids = random.sample(vids, 5)
    fig, ax = plt.subplots()
    for v in vids:
        if pd.notnull(v):
            s = e[e['modelname']==v]
            if (s.accuracy.max() == acc):
                y=plt.hist(s.accuracy,bins=range(acc), alpha=0.0, normed=True)
                bincenters = 0.5*(y[1][1:]+y[1][:-1])
                #plt.plot(bincenters,y[0],'-')

                x_smooth = np.linspace(bincenters.min(),bincenters.max(),150)
                y_smooth = spline(bincenters,y[0],x_smooth)
                ax.plot(x_smooth, y_smooth, label=str(v))

    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel("Accuracy (in meters)")
    plt.ylabel("Readings (normalized)")
    plt.show()

def simAccuracy():
    f =pd.read_csv("..\..\gpserr\gpserr\output\sN_0.csv")
    acc =20
    f = f[f['accuracy']<=acc]
    b = f.accuracy.max()
    fig, ax = plt.subplots()
    plt.hist(f.accuracy, bins=b, label="Simulated Device")
    legend = ax.legend(loc='upper right', shadow=True)
    plt.show()
realGPSAccuracyCurve()
#simAccuracy()