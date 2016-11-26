from __future__ import division # for floating point divisions
import gpxpy
import gpxpy.gpx
import gpxpy.geo as mod_geo
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import random
import time
from datetime import datetime
from datetime import timedelta
import sklearn.cluster as s
import simplekml
import warnings
import custKmeans as ck

pd.options.display.precision = 11 # to handle more precision in lat lon pairs
np.set_printoptions(precision=11)

cxWidth = 5 # cross section distance in meters
cxInt = 40 # cross section interval in meters
cxWidthBaseLine = [25,25] # cross section distance left right pair
nlanes = 2 # must be read from map details

# custom fence:
fence = [49.873027729999997, 8.5060440889999995, 49.92141393, 8.5780801350000004]
accThreashold = 10

class gpxPtDS(): 
    def __init__(self, lat, lon, direction=None):
        self.lat = lat
        self.lon = lon
        self.dir = direction
    def inc(self):
        self.pos = self.pos + 1


class gpxTraceDS():
    pos = 0
    len = 0
    trace = [] # hold gpxPtDS
    def __init__(self, gpxPtDSList):
        self.trace = gpxPtDSList
        self.len = len(gpxPtDSList)
    def inc(self):
        self.pos = self.pos + 1
    def setPos(self, p):
        self.pos = p


class Point:
    '''
    for simple cartesian calcualtions
    // src: http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    '''
    def __init__(self,x,y):
        self.x = x
        self.y = y


def angleDiff(x, y):
    '''return x - y difference in complete circular result'''
    return (x - y + 180 + 360) % 360 - 180


def cartDist(x1, y1, x2, y2):
    '''returns cartesian distance'''
    return math.sqrt((x1 -x2)**2 + (y1-y2)**2)


def ccw(A,B,C):
    '''checks counter clockwise direction'''
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)


def intersectionPoint(A,B,C,D):
    '''
    returns (x,y) if intersected
    '''
    if (ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)):
        x1, y1 = A.x, A.y
        x2, y2 = B.x, B.y
        x3, y3 = C.x, C.y
        x4, y4 = D.x, D.y
        deno = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
        xneu = (((x1 * y2) - (y1 * x2))*(x3 - x4)) - ((x1 - x2)*((x3 * y4) - (y3 * x4)))
        yneu = (((x1 * y2) - (y1 * x2))*(y3 - y4)) - ((y1 - y2)*((x3 * y4) - (y3 * x4)))
        x, y = xneu / deno, yneu / deno
        return float(x), float(y)
    else:
        return None, None   # for easy iteration


def addCx(cp, direction, gpx_track):
    '''
    cp:     center point
    direction: direction
    gpxT:   gpx_track
    '''
    lp, rp = mod_geo.point_from_distance_bearing(cp, cxWidth, direction + 90),  mod_geo.point_from_distance_bearing(cp, cxWidth, direction - 90)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lp.latitude, lp.longitude))
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(rp.latitude, rp.longitude))


def crossSection():
    '''
    performs a cross section along a given track
    at minimum distance of every cxInt meters
    width of section is cxWidth
    points output are in right to left direction
    '''
    gpx_file = open('..\..\samples\A67lt.gpx', "r")
    gpx = gpxpy.parse(gpx_file)
    gpxOpTmp = gpxpy.gpx.GPX()
    for tracks in gpx.tracks:
        gpx_track = gpxpy.gpx.GPXTrack()
        gpxOpTmp.tracks.append(gpx_track)
        for segment in tracks.segments:
            oldPoint = []
            for point in segment.points:
                currPt = [point.latitude, point.longitude]
                if (oldPoint == []):
                    oldPoint = currPt
                else:
                    dist = mod_geo.distance(oldPoint[0], oldPoint[1], None, currPt[0], currPt[1], None, True)
                    direction = mod_geo.get_bearing(mod_geo.Location(oldPoint[0], oldPoint[1]), mod_geo.Location(currPt[0], currPt[1]))      
                    # code to take cross section at regular distance [global]
                    if (dist < cxInt):                
                        midpt = mod_geo.midpoint_cartesian(oldPoint, currPt)
                        addCx(midpt, direction, gpx_track)
                    else:
                        ctr = 1
                        while True:
                            if (ctr * cxInt > dist):    break # breaks the while loop
                            else:
                                tgtPt = mod_geo.point_from_distance_bearing(mod_geo.Location(oldPoint[0], oldPoint[1]), ctr * cxInt, direction)
                                addCx(tgtPt, direction, gpx_track)
                                ctr = ctr + 1
                oldPoint = currPt
    gpx_file.close()
    outfile = open('crossSection.gpx', 'w')
    outfile.write(gpxOpTmp.to_xml())


tracePoints = []
def samplingCX():
    '''
    uses crossSection.gpx to find intersection with traces
    writes to WP.gpx
    and gpxwpt.csv
    '''
    # prepare crossSection file
    gpx_file = open('crossSection.gpx', "r")
    gpx = gpxpy.parse(gpx_file)
    # prepare cxList for cross sections
    cxSegTuple = []
    for track in gpx.tracks:
        for seg in track.segments:
            cxList = []
            for pt in seg.points:
                cxList.append(gpxPtDS(pt.latitude, pt.longitude))
            cxSegTuple.append(cxList) # set of two gpxPtDS
    gpx_file.close()

    # prepare datastructure
    txTraces = []
    gpx_file = open('..\..\samples\A67.gpx', "r")
    gpx = gpxpy.parse(gpx_file)
    for track in gpx.tracks:
        for seg in track.segments:
            txPt = []
            prev = None
            for pt in seg.points:
                if (prev == None):
                    txPt.append(gpxPtDS(pt.latitude, pt.longitude))
                else:
                    txPt.append(gpxPtDS(pt.latitude, pt.longitude, mod_geo.get_bearing(prev, pt)))
                prev = pt
            tmp = gpxTraceDS(txPt)
            txTraces.append(tmp)
    
    # find points and write
    gpxOpTmp = gpxpy.gpx.GPX()

    # prepare the gpx wpt
    # and a csv for the same
    wptcsv = open('gpxwpt.csv','w')
    print >> wptcsv, "segID,lat,lon"
    for i, cxp in enumerate(cxSegTuple):
        # below are 2 points of cx line segment
        A = Point(cxp[0].lat, cxp[0].lon)
        B = Point(cxp[1].lat, cxp[1].lon)
        for gpxTrace in txTraces:
            C = None
            for gpxSegment in gpxTrace.trace:
                D = Point(gpxSegment.lat , gpxSegment.lon)
                if (C != None):
                    x,y = intersectionPoint(A, B, C, D)
                    if ((x != None) and (y != None)):
                        print >> wptcsv,str(i) + ',' + str(x) + ',' + str(y)
                        gpxwp = gpxpy.gpx.GPXWaypoint(x, y,description="cxSeg_"+str(i))
                        gpxOpTmp.waypoints.append(gpxwp)
                C = D

    wptcsv.close()
    outfile = open('WP.gpx', 'w')
    outfile.write(gpxOpTmp.to_xml())
    outfile.close()


def csv2gpx(pathToCSV,outFile):
    '''
    pathToCSV: input csvFile
    outFile: output gpx file name
    input file must have lat and lon columns
    Converts simulation csv to gpx file for easy viewing
    '''
    gpxOpTmp = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpxOpTmp.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    f = pd.read_csv(pathToCSV)

    # custom fence:
    f = f[f['lat'] > fence[0]]
    f = f[f['lat'] < fence[2]]
    f = f[f['lon'] > fence[1]]
    f = f[f['lon'] < fence[3]]
    f = f[f['accuracy'] <= accThreashold ]
    #f = f.dropna()

    f['sts'] = pd.to_datetime(f['date'] + " " + f['hour'],format="%d.%m.%Y %H:%M:%S:%f")
    
    startDate = datetime.strptime('2016-08-01', '%Y-%m-%d')    # YYYY-MM-DD format
    endDate   = datetime.strptime('2016-09-01', '%Y-%m-%d')    # YYYY-MM-DD format
    f = f.ix[( f['sts'] >= startDate) & ( f['sts'] <= endDate)]
    f = f.sort_values(['vid','sts'])
    f = f.reset_index(drop=True)
    veh = f.vid.unique()
    print "no. of individual vehicles : ", len(veh)
    print "no. of day(s) there was data : ", len(f.date.unique())
    print "frames: ", len(f)
    oldVeh = f.iloc[0]
    for i, r in f.iterrows():
        if ((oldVeh.vid == r.vid) and (mod_geo.distance(oldVeh.lat, oldVeh.lon, None, r.lat, r.lon, None) < 1200)):
                gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(r.lat, r.lon))
        else:
            gpx_track = gpxpy.gpx.GPXTrack()
            gpxOpTmp.tracks.append(gpx_track)
            gpx_segment = gpxpy.gpx.GPXTrackSegment()
            gpx_track.segments.append(gpx_segment)
        oldVeh = r
    outFile = outFile.split('.')[0] + str(startDate.month) + str(startDate.day) + "-" + str(endDate.month) + str(endDate.day) + ".gpx"
    outfile = open(outFile, 'w')
    outfile.write(gpxOpTmp.to_xml())
    print "Written : " + outFile


def csv2wpt(pathToCSV,outFile, description):
    '''
    pathToCSV: input csvFile
    outFile: output gpx file name
    input file must have lat and lon columns
    Converts simulation csv to gpx waypoint file for easy viewing
    '''
    gpxOpTmp = gpxpy.gpx.GPX()
    f = pd.read_csv(pathToCSV)
    if (description):
        for i, r in f.iterrows():
            gpxwp = gpxpy.gpx.GPXWaypoint(r.lat, r.lon, description = 'cxPoint'+str(r.sid))
            gpxOpTmp.waypoints.append(gpxwp)
    else:
        for i, r in f.iterrows():
            gpxwp = gpxpy.gpx.GPXWaypoint(r.lat, r.lon, description = None)
            gpxOpTmp.waypoints.append(gpxwp)
    outfile = open(outFile, 'w')
    outfile.write(gpxOpTmp.to_xml())
    print "written in file " + outFile
    

def gpx2csv(path2gpx,outFile):
    path2gpx = open(path2gpx)
    gpxFile = gpxpy.parse(path2gpx)
    rows = []
    for track in gpxFile.tracks:
        for seg in track.segments:
            for pt in seg.points:
                rows.append({'vid':0,'sts':0,'lat':pt.latitude, 'lon':pt.longitude})
    res = pd.DataFrame(rows)
    res.to_csv(outFile)
    print "Written: " + outFile


# step 1
def makeReadable(path2csv, outFile):
    '''takes the collected data 
    and makes it into readable format for processing further'''
    f = pd.read_csv(path2csv, index_col = None)
    try:
        f['sts'] = pd.to_datetime(f['date'] + " " + f["hour"],format="%d.%m.%Y %H:%M:%S:%f")
    except KeyError:
        # simulation result file
        n = datetime.now()
        hr = [(n+ timedelta(0, int(i))).time().strftime("%H:%M:%S:%f") for i in f['sts']]
        f['hour'] = hr
        f['sts'] = pd.to_datetime(f['date'] + " " + f["hour"],format="%d.%m.%Y %H:%M:%S:%f")
    l = f.columns.tolist()
    try:
        l[l.index('imei')] = 'vid'
        l[l.index('longitude')] = 'lon'
        l[l.index('latitude')] = 'lat'
        l[l.index('angle')] = 'dir'
    except ValueError:
        "Not sure if converted, check for vid, lat and lon in the file: " + outFile
    f.columns = l
    f.to_csv(outFile, index=False)
    print "done"
    return outFile

# step 2
timeThreshold = 30 # seconds
def segments(pathToCSV, outFileName):
    '''
    takes simulation converted csv
    finds and writes crossSections into csv
    '''
    print "generating segments"
    maxDist = 50 # meters
    f = pd.read_csv(pathToCSV)
    f['sts'] = pd.to_datetime( f.sts)
    f = f.sort_values(['vid','sts'])
    f = f.reset_index(drop=True)  # to reindex the sorted values]
    f = f[f['lat'] > fence[0]]
    f = f[f['lat'] < fence[2]]
    f = f[f['lon'] > fence[1]]
    f = f[f['lon'] < fence[3]]
    # fence attached
    f = f.reset_index(drop=True)

    flen = len(f)
    traces = []
    traceVal = 0
    prevRow = None
    print "\nassigning trip ids\n"
    
    for i, r in f.iterrows():
        sys.stdout.write("\r%d of %d %d%%" % (i, flen, ((i+1)*100/flen)))
        sys.stdout.flush()
        if prevRow is not None:   
            if(prevRow.vid == r.vid and not np.isnan(r.vid)):
                diff = r.sts - prevRow.sts
                if (diff.days != 0 or diff.seconds > timeThreshold):
                    traceVal = traceVal + 1
            elif (np.isnan(r.vid)):
                traces.append(-1)
            else:
                traceVal = traceVal + 1
            if (not np.isnan(r.vid)):
                traces.append(traceVal)
        else:
            traces.append(traceVal)
        prevRow = r
    print '\n'
    f['tripID'] = traces
    #f = f.dropna()
    f.to_csv("int.csv", index=False)
    tr = (f.tripID.unique()).tolist()
    try:    tr.remove(-1)
    except ValueError: "All valid traces"

    # use 0 for RU to DA
    # use 5 for DA to RU
    initVidTrace = f[f['tripID']== 0]#random.choice(tr)]
    oldPt = None
    cx = pd.DataFrame()
    
    for i, r in initVidTrace.iterrows():
        if oldPt is not None:
            direction = mod_geo.get_bearing(mod_geo.Location(oldPt.lat, oldPt.lon), mod_geo.Location(r.lat, r.lon))
            if (direction !=None):
                if ((r.lat > fence[0]) and (r.lat < fence[2]) and (r.lon > fence[1]) and (r.lon < fence[3])):
                    dist = mod_geo.distance(oldPt.lat, oldPt.lon, None, r.lat, r.lon, None)
                    if dist > maxDist:
                        for t in range(0, int(dist/maxDist) + 1):
                            midPt = mod_geo.point_from_distance_bearing(mod_geo.Location(oldPt.lat, oldPt.lon), maxDist *  t, direction)
                            lp, rp = mod_geo.point_from_distance_bearing(midPt, cxWidthBaseLine[0], direction + 90),  mod_geo.point_from_distance_bearing(midPt, cxWidthBaseLine[1], direction - 90)
                            cx = cx.append({'llat':lp.latitude,'llon':lp.longitude,'rlat':rp.latitude,'rlon':rp.longitude, 'dir': direction, 'sid': len(cx)}, ignore_index=True)
                    else:
                        midPt = [(oldPt.lat + r.lat)/2, (oldPt.lon+r.lon)/2] #simple cartesian midpoint
                        lp, rp = mod_geo.point_from_distance_bearing(mod_geo.Location(midPt[0],midPt[1]), cxWidthBaseLine[0], direction + 90),  mod_geo.point_from_distance_bearing(mod_geo.Location(midPt[0],midPt[1]), cxWidthBaseLine[1], direction - 90)
                        cx = cx.append({'llat':lp.latitude,'llon':lp.longitude,'rlat':rp.latitude,'rlon':rp.longitude, 'dir': direction, 'sid': len(cx)}, ignore_index=True)
        oldPt = r
    cx.to_csv(outFileName)
    print "written : " + outFileName
    return outFileName
    
# step 3   
def segTraceInt(segFile, traceFile, outFileName):
    '''
    segFile: segment file generated from segments
    traceFile: input csv trace file from simulation
    outFile: output file

    generates points of intersection using segments from segFile and traces from traceFile
    '''
    print "segment and trace intersection"
    cxInter = pd.DataFrame()
    cxSeg2skip = 0 # data filtering
    # thDist = 55  # 55m/s = 198 km/hr
    thDir = 30 #degrees
    startTime = time.time()
    cx = pd.read_csv(segFile) 
    # geofence
    #fence = [cx.llat.min(), cx.llon.min(), cx.rlat.max(), cx.rlon.max()]
    
    # add fence
    f = pd.read_csv(traceFile)
    f = f[f['lat'] > fence[0]]
    f = f[f['lat'] < fence[2]]
    f = f[f['lon'] > fence[1]]
    f = f[f['lon'] < fence[3]]
    f = f[f['accuracy'] <= accThreashold]
    f = f.reset_index(drop=True)

    # careful to drop, might not required in case annotations are being used.
    #f['annotation'] = f['annotation'].fillna('-')
    #f = f.dropna()

    print 'frames found\t: ',len(f)
    
    # date filter
    f['sts'] = pd.to_datetime(f['date'] + " " + f['hour'],format="%d.%m.%Y %H:%M:%S:%f")
    startDate = datetime.strptime('2016-08-01', '%Y-%m-%d')    # YYYY-MM-DD format
    
    if outFileName[-5] == 'a':
        endDate   = datetime.strptime('2016-08-04', '%Y-%m-%d')
    elif outFileName[-5] == 'b':
        endDate   = datetime.strptime('2016-08-08', '%Y-%m-%d')
    elif outFileName[-5] == 'c':
        endDate   = datetime.strptime('2016-09-01', '%Y-%m-%d')
    else:   # default setting
        endDate   = datetime.strptime('2016-09-01', '%Y-%m-%d')

    #endDate   = datetime.strptime('2016-08-02', '%Y-%m-%d')    # YYYY-MM-DD format
    f = f.ix[( f['sts'] >= startDate) & ( f['sts'] <= endDate)]
    print 'frames filtered by date\t: ', len(f)
    print 'Day(s) of data\t: ', len(f.date.unique())
    f = f.sort_values(['vid','sts'])
    f = f.reset_index(drop=True)
    
    #f.to_csv(outFileName.split('.')[0] + "__r2_IntDropped.csv")
    C = None, None
    dirList = []
    dir = None
    for i,r in f.iterrows():
        D = Point(r.lat, r.lon)
        if (C != (None, None)):
            dir = mod_geo.get_bearing(mod_geo.Location(C.x, C.y), mod_geo.Location(D.x, D.y))
        dirList.append(dir)    
        C = D
    f['dir'] = dirList
    #f.to_csv(outFileName.split('.')[0] + "__r2_IntBefore.csv")

    ### fast but risky options
    ### angular rounding check required
    f = f.ix[(f['dir'] < cx.dir.max()) & (f['dir'] > cx.dir.min())]
    print 'frames filtered by direction\t: ', len(f)
    if len(f) == 0:
        print "Nothing to process.. Exiting..."
        sys.exit()
    f = f.reset_index(drop=True)
    ### reduces the number of rows to process
    ### ends here

    #f.to_csv(outFileName.split('.')[0] + "__r2_Int.csv")
    
    print 'finding segment intersection points'
    for i, r in cx.iterrows():
        if i > cxSeg2skip:
            A = Point(r.llat, r.llon)
            B = Point(r.rlat, r.rlon)
            midPt = Point((A.x + B.x) / 2, (A.y + B.y) / 2)
            C = None, None
            sys.stdout.write("\rcxSeg # %s%s%s ==> %d%%" % (i+1,'/',len(cx), (100*(i+1)/len(cx))))
            sys.stdout.flush()
            for vi, v in f.iterrows():
                D = Point(v.lat, v.lon)
                if (C != (None, None)):
                    #traceDir = mod_geo.get_bearing(mod_geo.Location(C.x, C.y), mod_geo.Location(D.x, D.y))
                    traceDir = v.dir
                    timediff = v.sts - prev.sts
                    if (timediff.days == 0 and timediff.seconds < timeThreshold and v.vid == prev.vid):
                        sameTrace = True
                    else:
                        sameTrace = False
                    if (traceDir !=None and sameTrace):
                        deltaDir = abs(angleDiff(traceDir, r.dir))
                        if( deltaDir < thDir):
                            poi = intersectionPoint(A, B, C, D)
                            if (poi != (None, None)):
                                cxInter = cxInter.append({'sid':r.sid, 'vid': v.vid, 'lat':poi[0], 'lon':poi[1], 
                                        'lanes': 2, 'accuracy': (prev.accuracy + v.accuracy)/2, 
                                        'nos': (prev.nos + v.nos)/2}, ignore_index=True)
                                C = None, None
                C = D
                prev = v

    # custom outfilenames
    #outFileName = outFileName.split('.')[0] + '_' + str(startDate.month) + str(startDate.day) + "-" + str(endDate.month) + str(endDate.day) + ".csv"

    cxInter.to_csv(outFileName)
    print "\nIntersection segments saved in " + outFileName
    endTime = time.time()
    print "Completed in : " + str(endTime - startTime) + " seconds"
    return outFileName


def cluster(csv2read, outFileName):
    print "generating culster information"
    ptscsv = pd.read_csv(csv2read, index_col=0)
    unqSegIDs = ptscsv.sid.unique()
    result = pd.DataFrame()
    oldgrp = None
    for unqSID in unqSegIDs:
        meanCol = []
        grp = ptscsv[ptscsv['sid']==unqSID]
        grp.is_copy = False  # supress copy warning while assignment
        mean = grp.mean().lat, grp.mean().lon
        grp['mlat'] = mean[0]
        grp['mlon'] = mean[1]
        if oldgrp is not None:
            grp['dir'] = mod_geo.get_bearing(mod_geo.Location(oldgrp[0], oldgrp[1]), mod_geo.Location(mean[0], mean[1]))
        else:
            grp['dir'] = None
        result = result.append(grp, ignore_index=True)
        oldgrp = mean[0], mean[1]
    devList = []
    distList = []
    oldID = None
    oldRow = None
    for i, r in result.iterrows():
        if not pd.isnull(r.dir):
            dist = mod_geo.distance(r.mlat, r.mlon, None, r.lat, r.lon, None)
            dirTgt = mod_geo.get_bearing(mod_geo.Location(oldID.mlat, oldID.mlon), mod_geo.Location( r.lat, r.lon))
            dir90 =  mod_geo.get_bearing(mod_geo.Location(r.mlat, r.mlon), mod_geo.Location( r.lat, r.lon))
            if (r.dir > dirTgt):
                dist = dist * (-1)
            distList.append(dist)
            devList.append(cartDist(r.lat, r.lon, r.mlat, r.mlon))
        else:
            distList.append(mod_geo.distance(r.mlat, r.mlon, None, r.lat, r.lon, None))
            devList.append(cartDist(r.lat, r.lon, r.mlat, r.mlon))
        
        if oldID is None:        
            oldID = r
        else:
            if (oldRow.dir != r.dir) and (not pd.isnull(r.dir)):
                oldID = oldRow
        oldRow = r

    result['dev'] = devList
    result['dist'] = distList
    result.to_csv(outFileName)
    print "written : " + outFileName
    return outFileName


def centerLine(csv2read, outFileName):
    '''
    generates the centerline for the route
    csv2read must contain cloumns 'sid','mlat','mlon'
    '''
    print "generating center line"
    ptscsv = pd.read_csv(csv2read, index_col=0)
    ptscsv = ptscsv[['sid','mlat','mlon']]
    ptscsv = ptscsv.drop_duplicates(['sid','mlat','mlon'])
    ptscsv = ptscsv[['mlat','mlon', 'sid']]
    # rename columns
    ptscsv.columns = [c.replace('mlat','clat') for c in ptscsv.columns]
    ptscsv.columns = [c.replace('mlon','clon') for c in ptscsv.columns]

    ptscsv.to_csv(outFileName)
    print "written : " + outFileName
    return outFileName


def compareCenter(baseCL, inpCL, outFileName):
    base = pd.read_csv(baseCL)
    inpd = pd.read_csv(inpCL)
    op = pd.DataFrame()
    cxWidthBaseLine = [30, 30]
    oldR = None

    for ii, ir in inpd.iterrows():
        if oldR is not None:
            midPt = mod_geo.Location((oldR.clat + ir.clat)/2, (oldR.clon + ir.clon)/2)
            direction = mod_geo.get_bearing( mod_geo.Location(oldR.clat, oldR.clon), mod_geo.Location(ir.clat, ir.clon))
            lp , rp = mod_geo.point_from_distance_bearing(midPt, cxWidthBaseLine[0], direction - 90), mod_geo.point_from_distance_bearing(midPt, cxWidthBaseLine[1], direction + 90)
            A, B = Point(lp.latitude, lp.longitude), Point(rp.latitude, rp.longitude)
            C = None, None
            for bi, br in base.iterrows():
                D = Point(br.clat, br.clon)
                if (C != (None, None)):
                    poi = intersectionPoint(A, B, C, D)
                    if poi != (None,None):
                        dirMid = mod_geo.get_bearing(mod_geo.Location(oldR.clat, oldR.clon), mod_geo.Location(midPt.latitude, midPt.longitude))
                        dirTgt = mod_geo.get_bearing(mod_geo.Location(oldR.clat, oldR.clon), mod_geo.Location(poi[0], poi[1]))
                        deltaDist = mod_geo.distance(midPt.latitude, midPt.longitude, None, poi[0], poi[1], None)
                        if ((dirMid < dirTgt) or (abs(dirMid - dirTgt) > 180)):
                            deltaDist = (-1) * deltaDist
                        rowData = {'sid': ir.sid, 'deltaDist': deltaDist,
                                   'blat': br.clat, 'blon': br.clon,
                                   'tlat': ir.clat, 'tlon': ir.clon,
                                   'llat':lp.latitude, 'llon':lp.longitude,
                                   'rlat':rp.latitude,'rlon':rp.longitude,
                                   'mlat': midPt.latitude, 'mlon': midPt.longitude,
                                   'plat': poi[0], 'plon':poi[1]}

                        op = op.append( rowData, ignore_index=True)
                C = D
        oldR = ir
    op.to_csv(outFileName)
    print "Written : " + outFileName
    plt.hist(op.deltaDist, bins=20)
    plt.savefig("__"+outFileName.replace("csv","jpg"))
    plt.show()
    return outFileName


def addRandDev(inFile, outFileName):
    inFile = pd.read_csv(inFile, index_col=0)
    cx = pd.DataFrame()
    res = []
    mpt = []
    d = [2,5,-1,-3,4,-2]
    for i, r in inFile.iterrows():
        mp = ((r.llat + r.rlat) / 2), ((r.llon + r.rlon) / 2)
        dir = mod_geo.get_bearing(mod_geo.Location(r.llat, r.llon), mod_geo.Location(r.rlat, r.rlon))
        pt = mod_geo.point_from_distance_bearing(mod_geo.Location(mp[0],mp[1]), random.choice(d), dir)
        res.append({'lat':pt.latitude, 'lon':pt.longitude})
        mpt.append({'lat':mp[0], 'lon':mp[1]})
    inFile['nlat'] = [ i['lat'] for i in res]
    inFile['nlon'] = [ i['lon'] for i in res]
    inFile['lat'] = [ i['lat'] for i in mpt]
    inFile['lon'] = [ i['lon'] for i in mpt]
    inFile.to_csv(outFileName)
    print "Written : " + outFileName
    return outFileName

# step 4
def clusterTypes(inFile, outFile, clusteringMethod):
    '''
    clusteringMethod : in (kmeans, mbKMeans, meanshift, wkmeans)
    '''
    f = pd.read_csv(inFile)
    # custom filter
    #f = f[f['sid']==108]
    f = f.dropna()
    sids = f.sid.unique()
    laneCenters = []
    nlanes = 2

    for sid in sids:
        sel = f[f['sid']==sid]
        # change paramters for weighting...
#        dt = np.column_stack((sel.lat, sel.lon, sel.accuracy))
        dt = np.column_stack((sel.lat, sel.lon, sel.accuracy ** 2))
#        dt = np.column_stack((sel.lat, sel.lon))
        # perform mandatory check to avoid unnecessary errors
        if len(dt) >= nlanes:
            if clusteringMethod.lower() == "kmeans":
                c = s.KMeans(n_clusters=nlanes)
                c.fit(dt)
            if clusteringMethod.lower() == "mbkmeans":
                c = s.MiniBatchKMeans(n_clusters=nlanes)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c.fit_predict(dt)
            if clusteringMethod.lower() == "meanshift":
                c = s.MeanShift()
                c.fit(dt)
            if clusteringMethod.lower() == "wkmeans":
                c = ck.Kmeans(dt, nlanes, metric=ck.weightedK, iterno=sid)
            centroids = c.cluster_centers_[:,[0,1]]
            laneCenters.append([centroids,sid])
        else:
            print "Not enough data yet across all sections!!!"
            sys.exit()

    oldCenterRd = f.lat.mean(), f.lon.mean()
    tmp = []

    offsetList = []
    
    for i in laneCenters:
        sel = f[f['sid']==i[1]] # since its a pair centroids, sid
        currCenter = sel.lat.mean(), sel.lon.mean()
        dir = mod_geo.get_bearing(mod_geo.Location(oldCenterRd[0], oldCenterRd[1]), mod_geo.Location(currCenter[0],currCenter[1]))
        dt = i[0].ravel()
        
        for indx, irow in sel.iterrows():
            offset = 0
            if oldCenterRd is not None:
                ndir = mod_geo.get_bearing(mod_geo.Location(oldCenterRd[0], oldCenterRd[1]), mod_geo.Location(irow.lat, irow.lon))
                offset = mod_geo.distance(currCenter[0], currCenter[1],None,irow.lat, irow.lon,None)
                if ndir < dir:
                    offset = offset * (-1)
            offsetList.append(offset)

        for j in range(0,nlanes):
            ll =  dt[j*2], dt[(j*2)+1]
            dirPt = mod_geo.get_bearing(mod_geo.Location(oldCenterRd[0], oldCenterRd[1]), mod_geo.Location(ll[0],ll[1]))
            dist = mod_geo.distance(ll[0],ll[1],None, currCenter[0], currCenter[1], None)
            direction = dirPt - dir
            tmp.append({'lat': ll[0], 'lon': ll[1], 'lane': direction, 'sid' : sel.sid.unique()[0]})

        oldCenterRd = currCenter

    f['offset'] = offsetList
    #f.to_csv("offsets.csv",index=False)

    k = pd.DataFrame(tmp)
    k = k.sort_values(['sid'])
    #k.to_csv(inFile.split('.')[0]+"__k.csv")
    sids = k.sid.unique()
    result = pd.DataFrame()

    tmpDict = {}
    for sid in sids:
        sel = k[k['sid']==sid]
        sel = sel.sort_values(['lane'])
        sel = sel.reset_index(drop=True)
        prevLane = None
        for i, r in sel.iterrows():
            tmpDict['c'+str(i)+'_lat'] = r.lat
            tmpDict['c'+str(i)+'_lon'] = r.lon
            if prevLane is not None:
                delta = mod_geo.distance(prevLane.lat, prevLane.lon, None, r.lat, r.lon, None)
                tmpDict['del_'+str(i-1)+str(i)] = delta
            prevLane = r
        selDet = f[f['sid']==sid]
        tmpDict['_sid'] = sid
        tmpDict['var'] = selDet.offset.var()
        tmpDict['std'] = selDet.offset.std()
        tmpDict['mean'] = np.abs(selDet).offset.mean()

        result = result.append(tmpDict, ignore_index=True)
        
    result.to_csv(outFile, index=False)
    print "Written : " + outFile
    return outFile


def details(intfile, clusterFile, outfile):
    f = pd.read_csv(intfile)
    c = pd.read_csv(clusterFile)
    sids = f.sid.unique()

    var, width, dev = [], [], []


    print "Written : " + outfile
    return outfile


def csvToKML(inFile, outFile):
    kml = simplekml.Kml()
    f = pd.read_csv(inFile)
    
    # only a part of file
    #f = f.head(int( len(f) * 10/ 100) )
    
    cols = f.columns.tolist()
    elanes = {}
    kml = simplekml.Kml()

    for i in cols:
        if (i.endswith("_lat") or i.endswith("_lon")):
            key = i.replace("_lat","").replace("_lon","")
            elanes[key] = {}

    for i in elanes.keys():
        elanes[i] = {'lat': i + '_lat', 'lon': i + '_lon'}
        ls = kml.newlinestring(name=i)
        c2add = []
        for ii, r in f.iterrows():
            ############ order changed due to KML specifications, add altitude for good display
            pt = r[i+'_lon'], r[i+'_lat'], 2 
            c2add.append(pt)
        ls.coords = c2add
        ls.extrude = 1
        ls.altitudemode = simplekml.AltitudeMode.relativetoground
        ls.style.linestyle.width = 2
        ls.style.linestyle.color = simplekml.Color.blue
        
    kml.save(outFile)
    print "Written : " + outFile
    return outFile


def kcsvToKML(inFile, outFile):
    kml = simplekml.Kml()
    f = pd.read_csv(inFile)
    f['ts'] = pd.to_datetime(f.ts,unit='s')
    kml = simplekml.Kml()
    c2add = []
    oldr = None
    for ii, r in f.iterrows():
        if oldr is not None:
            dist = mod_geo.distance(r.lat, r.lon, None, oldr.lat,oldr.lon, None)
            '''
            tdel = (oldr.ts - r.ts).seconds
            diffTrip = False
            if dist > 10 or tdel > 90:
                diffTrip = True
            '''
            if (ii % 40000 == 0 or dist > 10) :
                ls = kml.newlinestring(name=str(ii))
                ls.coords = c2add
                ls.extrude = 1
                ls.altitudemode = simplekml.AltitudeMode.relativetoground
                ls.style.linestyle.width = 1
                ls.style.linestyle.color = simplekml.Color.yellow
                c2add = []
        
            ############ order changed due to KML specifications, add altitude for good display
            pt = r['lon'], r['lat'], 1
            c2add.append(pt) 
        oldr = r

    kml.save(outFile)
    print "Written : " + outFile
    return outFile


def dgpsInt(inFile, traceFile, outFileName):
    cx = pd.read_csv(inFile)
    f = pd.read_csv(traceFile)
    C = None, None
    dirList = []
    dir = None
    cxInter = pd.DataFrame()
    thDir = 30 #degrees
    startTime = time.time()
    for i,r in f.iterrows():
        D = Point(r.lat, r.lon)
        if (C != (None, None)):
            dir = mod_geo.get_bearing(mod_geo.Location(C.x, C.y), mod_geo.Location(D.x, D.y))
        dirList.append(dir)    
        C = D
    f['dir'] = dirList
    #f.to_csv(outFileName.split('.')[0] + "__r2_IntBefore.csv")

    ### fast but risky options
    ### angular rounding check required
    f = f.ix[(f['dir'] < cx.dir.max()) & (f['dir'] > cx.dir.min())]
    print 'frames filtered by direction\t: ', len(f)
    if len(f) == 0:
        print "Nothing to process.. Exiting..."
        sys.exit()
    f = f.reset_index(drop=True)
    print 'finding segment intersection points'
    for i, r in cx.iterrows():
        A = Point(r.llat, r.llon)
        B = Point(r.rlat, r.rlon)
        midPt = Point((A.x + B.x) / 2, (A.y + B.y) / 2)
        C = None, None
        sys.stdout.write("\rcxSeg # %s%s%s ==> %d%%" % (i+1,'/',len(cx), (100*(i+1)/len(cx))))
        sys.stdout.flush()
        for vi, v in f.iterrows():
            D = Point(v.lat, v.lon)
            if (C != (None, None)):
                #traceDir = mod_geo.get_bearing(mod_geo.Location(C.x, C.y), mod_geo.Location(D.x, D.y))
                traceDir = v.dir
                timediff = v.ts - prev.ts
                if (timediff < timeThreshold):
                    sameTrace = True
                else:
                    sameTrace = False
                if (traceDir !=None and sameTrace):
                    deltaDir = abs(angleDiff(traceDir, r.dir))
                    if( deltaDir < thDir):
                        poi = intersectionPoint(A, B, C, D)
                        if (poi != (None, None)):
                            cxInter = cxInter.append({'sid':r.sid, 'lat':poi[0], 'lon':poi[1], 
                                    'lanes': 2}, ignore_index=True)
                            C = None, None
            C = D
            prev = v
   
    cxInter.to_csv(outFileName)
    print "\nIntersection segments saved in " + outFileName
    endTime = time.time()
    print "Completed in : " + str(endTime - startTime) + " seconds"
    return outFileName



def plotConstructionSites(inFile, outFile):
    f = pd.read_csv(inFile)
    kml = simplekml.Kml()
    f = f.dropna()
    f['sts'] = pd.to_datetime(f['date'] + " " + f["hour"],format="%d.%m.%Y %H:%M:%S:%f")
    f = f.sort_values(['imei','sts'])
    strtCtr, endCtr = 0,0
    for i, r in f.iterrows():
        if r.annotation == "Construction Side Start":
            pnt = kml.newpoint(name="S_"+str(strtCtr), coords=[(r.longitude, r.latitude)])
            pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png"
            strtCtr = strtCtr + 1
        if r.annotation == "Construction Side End":
            pnt = kml.newpoint(name="E_"+str(endCtr), coords=[(r.longitude, r.latitude)])
            pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png"
            endCtr = endCtr + 1
    kml.save(outFile)
    print "Written : " + outFile
    return outFile


def compareMeta(dt, m1, m2, m1Label, m2Label):
    l1, = plt.plot(dt[m1], label = m1Label)
    l2, = plt.plot(dt[m2], label = m2Label)
    plt.legend([l1, l2], [m1Label, m2Label])
    plt.show()

def runs(inFile):
    f = pd.read_csv(inFile)
    res = pd.DataFrame()
    for i in range(2, len(f) + 1):
        sel = f.head(i)
        tdf = pd.DataFrame([[sel.Kmeans0.mean(), sel.Kmeans1.mean(), sel.MiniBatch0.mean(), sel.MiniBatch1.mean(), sel.MeanShift0.mean(), sel.MeanShift1.mean(), sel.WeightedKmeans0.mean(), sel.WeightedKmeans1.mean(), sel.SquareWeighted0.mean(), sel.SquareWeighted1.mean(), sel.ShiftingPoints0.mean(), sel.ShiftingPoints1.mean(), sel.InitialCenters0.mean(), sel.InitialCenters1.mean()]],
                           columns=['Kmeans0', 'Kmeans1','MiniBatch0', 'MiniBatch1','MeanShift0', 'MeanShift1','WeightedKmeans0', 'WeightedKmeans1','SquareWeighted0', 'SquareWeighted1','ShiftingPoints0', 'ShiftingPoints1','InitialCenters0', 'InitialCenters1'])
        res = res.append(tdf)
    res.to_csv("runningRes.csv",index=False)

def runPlots(inFile):
    f = pd.read_csv(inFile)
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.plot(f.Kmeans0, label = 'Kmeans0')
    #plt.plot(f.MiniBatch0, label = 'MiniBatch0')
    #plt.plot(f.MeanShift0, label = 'MeanShift0')
    #plt.plot(f.WeightedKmeans0, label = 'WeightedKmeans0')
    #plt.plot(f.SquareWeighted0, label = 'SquareWeighted0')
    plt.plot(f.ShiftingPoints0, label = 'ShiftingPoints0')
    plt.plot(f.InitialCenters0, label = 'InitialCenters0')

    #plt.plot(f.Kmeans1, label = 'Kmeans1')
    #plt.plot(f.MiniBatch1, label = 'MiniBatch1')
    #plt.plot(f.MeanShift1, label = 'MeanShift1')
    #plt.plot(f.WeightedKmeans1, label = 'WeightedKmeans1')
    #plt.plot(f.SquareWeighted1, label = 'SquareWeighted1')
    #plt.plot(f.ShiftingPoints1, label = 'ShiftingPoints1')
    #plt.plot(f.InitialCenters1, label = 'InitialCenters1')

    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.legend()
    plt.show()
