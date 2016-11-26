import gpxpy
import gpxpy.gpx
import gpxpy.geo as mod_geo
import getopt
import sys

def dist(lat1, lon1, lat2, lon2):
    # returns cartesian distance between 2 points
    dist = (((lat1 - lat2) ** 2) + ((lon1 - lon2) ** 2)) ** 0.5
    return dist

latlon = [49.88354465397444, 8.560190649686984]
radius = 0.0005 # in latlon distance, for easy calculations
nextPtDist = 0.005 # next point
filesList = ["C:/Users/usr/Desktop/A67filtered2.gpx"]
direction = 20

# python gpxFilter.py -p 49.91259382559233,8.510661062464754 -r 0.004 -n 0.004 -d 20 -f A67.gpx
if (len(sys.argv) > 1):
    try:
        argList, args = getopt.getopt(sys.argv[1:], "p:r:n:f:d:")
    except getopt.GetoptError as err:
        print "Something wrong with argument list"
    
    for k,v in argList:
        if k == '-p':            latlon = map(float, v.split(','))
        if k == '-r':            radius = float(v)
        if k == '-n':            nextPtDist = float(v)
        if k == '-f':            filesList = v.split(',')
        if k == '-d':            direction = float(v)

for fname in filesList:
    print "opening trace"
    gpx_file = open(fname, "r")
    gpx = gpxpy.parse(gpx_file);
    print "parsed"
    gpxOp = gpxpy.gpx.GPX()
    gpxOpTmp = gpxpy.gpx.GPX()

    print "cleaning traces..."
    for track in gpx.tracks:
        gpx_track = gpxpy.gpx.GPXTrack()
        gpxOpTmp.tracks.append(gpx_track)
        for segment in track.segments:
            gpx_segment = gpxpy.gpx.GPXTrackSegment()
            gpx_track.segments.append(gpx_segment)
            prev = [segment.points[0].latitude, segment.points[0].longitude]
            bearingOld = None
            bearingDelta = None
            for point in segment.points:
                neighCond = dist(prev[0], prev[1], point.latitude, point.longitude) < nextPtDist
                bearing = mod_geo.get_bearing(mod_geo.Location(prev[0], prev[1]), point)
                bearingCond = False
                if(bearingDelta == None):                    bearingCond = True
                else:                    bearingCond = bearingDelta < direction
                if (neighCond and bearingCond):
                    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point.latitude, point.longitude))
                else:
                    gpx_track = gpxpy.gpx.GPXTrack()
                    gpxOpTmp.tracks.append(gpx_track)
                    gpx_segment = gpxpy.gpx.GPXTrackSegment()
                    gpx_track.segments.append(gpx_segment)

                prev = [point.latitude, point.longitude]
                if (bearingOld != None and bearing != None):
                    bearingDelta = bearingOld - bearing
                    bearingDelta = (bearingDelta + 180) % 360 - 180 # for angles closer to 0 and 360 degrees
                if (bearing !=None):
                    bearingOld = bearing
                

    print "processing traces..."
    recTrack,recSeg,recPt = 0,0,0
    for track in gpxOpTmp.tracks:
        recTrack+=1
        for segment in track.segments:
            incSeg = False
            for point in segment.points:
                if (dist(latlon[0], latlon[1], point.latitude, point.longitude) < radius):
                    incSeg = True
                    recSeg+=1
                    break
            if(incSeg):
                gpx_track = gpxpy.gpx.GPXTrack()
                gpxOp.tracks.append(gpx_track)
                gpx_segment = gpxpy.gpx.GPXTrackSegment()
                gpx_track.segments.append(gpx_segment)
                for point in segment.points:
                    recPt+=1
                    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point.latitude, point.longitude))

    gpx_file.close()
    print "Tracks: ", recTrack, "\trecSeg: ", recSeg, "\trecPoints: ", recPt
    print "writing new trace"
    outfile = open('zop.gpx', 'w')
    outfile.write(gpxOp.to_xml())
    print "written successfully..."
