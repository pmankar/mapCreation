from __future__ import division
from collections import Counter
import sys
import gpxpy
import gpxpy.gpx
import gpxpy.geo as mdl_geo
import getopt
import threading
import time
import random
import pandas as pd

# load custom modules
import mdlcalc as mdl
import logger as log

# global configuration variables
blur = 3 # in meters. this is expect 3x variance in gaussian distribution
noopfiles = 1 #no of output files
log.verboseMode = True
exeStartTime = time.time()

def num_gen(num_probs):
    '''
    return elements with custom probability
    num_prob: an array of (elment, probability)

    src: http://stackoverflow.com/questions/4265988/generate-random-numbers-with-a-given-numerical-distribution
    '''
    # calculate minimum probability to normalize
    min_prob = min(prob for num, prob in num_probs)
    
    lstx = []
    for num, prob in num_probs:
        # keep appending num to lst, proportional to its probability in the distribution
        for _ in range(int(prob/min_prob)):
            lstx.append(num)
	# all elems in lst occur proportional to their distribution probablities
    while True:
        # pick a random index from lst
        ind = random.randint(0, len(lstx)-1)
        yield lstx[ind]

class threadPx (threading.Thread):
    def __init__(self, threadID, gpx, fno, fname, optype):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.gpx = gpx
        self.fno = fno
        self.fname = fname
        self.optype = optype
    def run(self):
        writeTrace(self.gpx, self.fno, self.fname, self.optype)

def writeTrace(gpx, fno, fname, optype):
    '''
    writes a blurred gpx output(s)
    input
    gpx : parsed gpx file
    fno : file number
    fname : file name
    optype: gpx|csv
    output blurred gpx output(s)
    '''
    # device distribution here
    gen = num_gen([('d1', 0.01),('n1', 0.99)])
    devName = gen.next()
    
    # set local device model here
    d = mdl.load()
    mdl.setDevice(devName)

    if (optype == "gpx"):
        # output file
        gpxOp = gpxpy.gpx.GPX()

        recTrack,recSeg,recPt = 0,0,0
        log.write("reading trace")
    
        for track in gpx.tracks:
            gpx_track = gpxpy.gpx.GPXTrack()
            gpx_track.description = devName
            gpxOp.tracks.append(gpx_track)
            recTrack+=1
            log.write("completed "+ str(round(recTrack * 100/gpx.tracks.__len__(), 2)) + "% of tracks\tfile: " + str(fno+1))
            for segment in track.segments:
                gpx_segment = gpxpy.gpx.GPXTrackSegment()
                gpx_track.segments.append(gpx_segment)
                recSeg+=1
                xo, yo = 0, 0
                for point in segment.points:
                    x, y = mdl.addErr(point.latitude, point.longitude)
                    xe, ye = x - point.latitude, y - point.longitude
                    xe, ye = ( xo * oec ) + (xe * (1 - oec)), ( yo * oec ) + (ye * (1 - oec))
                    x, y = xe + point.latitude, ye + point.longitude
                    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(x, y))
                    recPt+=1
                    xo, yo = xe, ye

        log.write("trace reading done")
        log.write("Tracks: " + str(recTrack) + "\tSegments: " + str(recSeg) + "\tPoints: " + str(recPt))
        # write output
        # this can be threaded to increase speed
        # approx 1 - 1.5 sec requried to write depending on the size
        log.write("writing new trace")
        outfile = open('output/output_' + fname.replace('.gpx','') + '_' + str(fno) +'.gpx', 'w')
        outfile.write(gpxOp.to_xml())
        outfile.close()
        log.write("new trace written")
    if (optype == "csv"):
        # get unique vehicles
        vehicles = gpx.vid.unique()
        withGPXFile = True
        gpxOp = gpxpy.gpx.GPX()
        for veh in vehicles:
            trace = gpx[gpx['vid']==veh]
            # sort according to sts
            trace = trace.sort_values(['sts'])
            xo, yo = 0, 0
            log.write('vehicle: '+str(veh))
            if withGPXFile:
                gpx_track = gpxpy.gpx.GPXTrack()
                gpx_track.description = 'v'+ str(veh)
                gpxOp.tracks.append(gpx_track)
                gpx_segment = gpxpy.gpx.GPXTrackSegment()
                gpx_track.segments.append(gpx_segment)
            for i, r in trace.iterrows():
                lat, lon = trace.loc[i, "lat"], trace.loc[i, "lon"]
                x, y = mdl.addErr(lat, lon)
                xe, ye = x - lat, y - lon
                xe, ye = ( xo * oec ) + (xe * (1 - oec)), ( yo * oec ) + (ye * (1 - oec))
                x, y = xe + lat, ye + lon
                acc = round(mdl_geo.distance(x, y, None, lat, lon, None) * 1.33)
                acc = 3 if acc <=3 else acc
                gpx.loc[i, "lat"] = x
                gpx.loc[i, "lon"] = y
                gpx.loc[i, "accuracy"] = acc
                if withGPXFile:
                    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(x, y))
                xo, yo = xe, ye
                
        log.write("writing new trace")
        gpx.to_csv('output/'  + fname.replace('.csv','') + '_' + str(fno) +'.csv', sep=',', index=False)
        if withGPXFile:
            outfile = open('output/output_' + fname.replace('.csv','') + '_' + str(fno) +'.gpx', 'w')
            outfile.write(gpxOp.to_xml())
            outfile.close()
        log.write("new trace written")
    exeEndTime = time.time()
    exeTime = exeEndTime - exeStartTime
    log.write("Program execution time "+ "{0:.3f}".format(exeTime) + " sec")

#log.write("Number of arguments:" + str(len(sys.argv)) + "\n\tArgument List:" + str(sys.argv))
'''
-d:     device id
-i:     interations i.e. number of output files
-f:     input file(s)
-s:		smoothing factor
-t:     filetype

-F:     input folder
-dd:    device distribution
'''

devName = "d3"
filesList = []
oec = 0.75

# check?set if command line invoke
# python gpserr.py -d d3 -i 4 -f geoa67.gpx
if (len(sys.argv) > 1):
    try:
        argList, args = getopt.getopt(sys.argv[1:], "i:d:f:p:s:")
    except getopt.GetoptError as err:
        print "Something wrong with argument list"
    
    for k,v in argList:
        if k == '-d':            devName = v
        if k == '-i':            noopfiles = int(v)
        if k == '-f':            filesList = v.split(',')
        if k == '-p':            print v
        if k == '-s':            oec = float(v)
else:
    # set manually
    noopfiles = 1
    filesList = 'gpsop.gpx'

for cFile in filesList:
    opType = cFile.split('.')[-1]
    if (opType == 'gpx'):
        log.write("opening trace file")
        gpx_file = open(cFile, 'r')
        log.write("parsing trace file")
        gpx = gpxpy.parse(gpx_file)
        log.write("trace file parsed")
        for i in range(0, noopfiles):
            threadPx(str(i) + cFile.replace('.gpx',''), gpx, i, cFile, opType).start()
    if (opType == 'csv'):
        log.write("opening csv file")
        csv_file = pd.read_csv(cFile)
        log.write("csv file read done")
        for i in range(0, noopfiles):
            threadPx(str(i) + cFile.replace('.csv',''), csv_file, i, cFile, opType).start()