import random
import gpxpy.geo as mod_geo
import json
import threading
import pandas as pd
import matplotlib.pyplot as plt

import logger as log

jdata = None
selDev = None
tls = threading.local() #thread specific local storage

class retainErr():
    arr = []
    errType = None
    def __init__(self, id, eD):
        self.errType = [jdata['etype'][i[0]] for i  in eD]
        errCode = [{'errIndx': i[0], 'ctr': 0, 'oldVal': None} for i in eD]
        self.arr.append({'id': id, 'dist':errCode})
    def reset(self, id, errIndx):
        for indx, a in enumerate(self.arr):
            if (a['id'] == id):
                i = indx
                break
        for item in self.arr[i]['dist']:
            if (item['errIndx'] == errIndx):
                item['ctr'] = 0
                item['oldVal'] = None
    def storeErr(self, id, errIndx, err):
        for indx, a in enumerate(self.arr):
            if (a['id'] == id):
                i = indx
                break
        for item in self.arr[i]['dist']:
            if (item['errIndx'] == errIndx):
                tmp = [i for i, x in enumerate(self.errType) if x['id'] == errIndx]
                if (item['ctr'] >= self.errType[tmp[0]]['duration']):
                    self.reset(id, errIndx)
                else:
                    item['oldVal'] = err
                    item['ctr'] += 1

# load error model and devices, returns parsed file
def load(file = None):
    global jdata
    if file == None:    file = 'conf.json'    
    with open(file) as data_file:
        jdata = json.load(data_file)
        return jdata

# returns lat lon blur difference values for distance
def coordDiff(lat, lon, dist=None):
    if dist == None:    dist = blur
    loc1 = mod_geo.Location(lat, lon)
    xblur = mod_geo.point_from_distance_bearing(loc1, dist, 90)
    yblur = mod_geo.point_from_distance_bearing(loc1, dist, 0)
    return abs(xblur.longitude - lon), abs(yblur.latitude - lat)

# sample Gaussian disturbance
def gauss(devx, devy):
    if devx == None:    return 0, 0
    if devy == None:    devy = devx
    return random.gauss(0, devx), random.gauss(0, devy)

# random error in both directions
def rnd(devx, devy):
    if devx == None:    return x, y
    if devy == None:    devy = devx
    return (random.random() - 0.5 ) * devx, (random.random() - 0.5 ) * devy

def setDevice(deviceId):
    global jdata
    global selDev
    global tls
    devList = jdata['devices']
    for i in devList:
        if(i["id"] == deviceId):    
            selDev = i
            log.write("device selected: " + selDev["id"])
    if selDev == None:
        log.write("No device with device name: " + deviceId)
        return
    errDist = selDev["errDist"]
    tls.retain = retainErr(selDev["id"], errDist)
    return selDev

def addErr(x, y):
    global selDev
    global tls

    errList = jdata['etype']
    delx, dely = 0, 0
    pox, posy = 0, 0
    contribution = None
    contributionFactor = 0.8
    for i in selDev['errDist']:
        posx, posy = coordDiff(x, y, errList[i[0]]['effect'] * i[1])
        if (errList[i[0]]['type'] == 'random'):
            tmp = rnd(posx, posy)
        if (errList[i[0]]['type'] == 'gauss'):
            tmp = gauss(posx, posy)
        
        for a in tls.retain.arr:
            if (a['id'] == selDev["id"]):
                for d in a['dist']:
                    if (d['errIndx'] == errList[i[0]]['id']):
                        contribution = d['oldVal']
        if (contribution == None):
            delx += tmp[0]
            dely += tmp[1]
        else:
            delx += (contribution[0] * contributionFactor) + (tmp[0] * (1 - contributionFactor))
            dely += (contribution[1] * contributionFactor) + (tmp[1] * (1 - contributionFactor))
        x, y = x + delx, y + dely
        tls.retain.storeErr(selDev["id"], int(errList[i[0]]['id']), tmp)
    return x, y

def chkDist(id):
    load()
    setDevice(id)
    t = []
    for i in range(0,10000):
        x,y = addErr(10,10)
        dist = mod_geo.distance(10,10,None, x,y,None)
        dt = {'x':x,'y':y,'dist':dist}
        t.append(dt)
    nf = pd.DataFrame(t)
    plt.hist(nf.dist, bins=20, normed=True)
    plt.show()
