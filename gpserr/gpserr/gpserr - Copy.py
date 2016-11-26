from __future__ import division
from collections import Counter
import sys
import gpxpy
import gpxpy.gpx
import getopt
import threading
import time
import random


# load custom modules
import mdlcalc as mdl
import logger as log

# global configuration variables
blur = 3 # in meters. this is expect 3x variance in gaussian distribution
noopfiles = 1 #no of output files
log.verboseMode = True

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
    def __init__(self, threadID, gpx, fno, fname):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.gpx = gpx
        self.fno = fno
        self.fname = fname
    def run(self):
        writeTrace(self.gpx, self.fno, self.fname)

def writeTrace(gpx, fno, fname):
    '''
    writes a blurred gpx output(s)
    input
    gpx : parsed gpx file
    fno : file number
    fname : file name
    output blurred gpx output(s)
    '''
    # device distribution here
    gen = num_gen([('d1', 0.3),('d3', 0.7)])
    devName = gen.next()
    
    # set local device model here
    d = mdl.load()
    mdl.setDevice(devName)

    # output file
    gpxOp = gpxpy.gpx.GPX()

    recTrack,recSeg,recPt = 0,0,0
    log.write("reading trace")
    
    #oec = 0.9  #oldErrorContribution 
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
            #xo, yo = 0, 0
            for point in segment.points:
                x, y = mdl.addErr(point.latitude, point.longitude)
                #xe, ye = x - point.latitude, y - point.longitude
                #xe, ye = ( xo * oec ) + (xe * (1 - oec)), ( yo * oec ) + (ye * (1 - oec))
                #x, y = xe + point.latitude, ye + point.longitude
                gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(x, y))
                recPt+=1
                #xo, yo = xe, ye

    log.write("trace reading done")
    log.write("Tracks: " + str(recTrack) + "\tSegments: " + str(recSeg) + "\tPoints: " + str(recPt))
    # write output
    # this can be threaded to increase speed
    # approx 1 - 1.5 sec requried to write
    log.write("writing new trace")
    outfile = open('output/output_' + fname.replace('.gpx','') + '_' + str(fno) +'.gpx', 'w')
    outfile.write(gpxOp.to_xml())
    log.write("new trace written")

#log.write("Number of arguments:" + str(len(sys.argv)) + "\n\tArgument List:" + str(sys.argv))
'''
-d:     device id
-i:     interations i.e. number of output files
-f:     input file(s)

-F:     input folder
-dd:    device distribution
'''

devName = "d3"
filesList = []

# check?set if command line invoke
# python gpserr.py -d d3 -i 4 -f geoa67.gpx
if (len(sys.argv) > 1):
    try:
        argList, args = getopt.getopt(sys.argv[1:], "i:d:f:p:")
    except getopt.GetoptError as err:
        print "Something wrong with argument list"
    
    for k,v in argList:
        if k == '-d':            devName = v
        if k == '-i':            noopfiles = int(v)
        if k == '-f':            filesList = v.split(',')
        if k == '-p':
            print v

else:
    # set manually
    noopfiles = 1
    filesList = 'gpsop.gpx'

for cFile in filesList:
    log.write("opening trace file")
    gpx_file = open(cFile, 'r')
    log.write("parsing trace file")
    gpx = gpxpy.parse(gpx_file)
    log.write("trace file parsed")
    for i in range(0, noopfiles):
        threadPx(str(i) + cFile.replace('.gpx',''), gpx, i, cFile).start()