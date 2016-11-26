import mdlcalc as mdl
import numpy as np
import random
import matplotlib.pyplot as plt
import gpxpy.geo as mdl_geo


# try fitting configuration in the model
def fitModel(candidates, target):
    '''
    candidates for tuning
    target to reach
    '''
    # prepare ds in case not present
    try:
        candidates[0].keys().index('decision')
        # do nothing, all calculations below after init
    except ValueError:
        # initialize the structure
        for e in candidates:
            e['decision'] = 0
            e['amount'] = errPercentage
            e['delta'] = errPercentage
            e['iter'] = 0
    currErrSum = np.array([i['effect']*i['amount'] for i in candidates]).sum()
    #print currErrSum
    sel = random.choice(candidates)
    sel['decision'] = 1
    sel['iter'] = sel['iter'] + 1
    sel['delta'] = sel['delta'] / 2
    if (currErrSum < (expectedError + stdDist)/2):
        sel['amount'] = sel['amount'] + sel['delta']
    else:
        sel['amount'] = sel['amount'] - sel['delta']    


def chk():
    mdl.setDevice("n1")
    sampleSize = 1000
    # enter coordinates here
    initLoc = [40, 8]
    initLoc = np.zeros((sampleSize, 2)) + initLoc
    errCords, errDist = [], []
    for i in range(0, sampleSize):
        e = mdl.addErr(initLoc[i][0], initLoc[i][1])
        errCords.append(e)
        d = mdl_geo.distance(initLoc[i][0], initLoc[i][1],None, e[0], e[1], None)
        errDist.append(d)
    plt.hist(errDist)
    plt.show()

jdata = mdl.load()
expectedError = 9 # meters
stdDist = 2 # meters

errDist = jdata['etype']

# error types which can be selected
candidates, errAmt, ttlAmt = [], [], []
for ed in errDist:
    ttlAmt.append(ed['effect'])
    if ed['effect'] <= expectedError:
        candidates.append(ed)
        errAmt.append(ed['effect'])

# assign distribution setup
distribution = []
errPercentage = np.array(errAmt).sum() / np.array(ttlAmt).sum()

iterToFit = 15 # above 10 preferred
for i in range(0, iterToFit):
    fitModel(candidates,None)

dev = [[i['id'],round(i['amount'], 3)] for i in candidates]
print dev
chk()

