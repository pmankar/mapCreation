import gpxpy

gpx = gpxpy.parse(open('C:\\Users\\usr\\Desktop\\tstA67.gpx', 'r'))
print("{} track(s)".format(len(gpx.tracks)))
track = gpx.tracks[1]

print("{} segment(s)".format(len(track.segments)))
segment = track.segments[0]

print("{} point(s)".format(len(segment.points)))

data = []
segment_length = segment.length_3d()
for point_idx, point in enumerate(segment.points):
    data.append([point.longitude, point.latitude])
from pandas import DataFrame
columns = ['Longitude', 'Latitude']
df = DataFrame(data, columns=columns)
df.head()
print df.__len__()

import mplleaflet
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(df['Longitude'], df['Latitude'], 'rs', linewidth=2, alpha=0.5)
ax.plot(df['Longitude'], df['Latitude'], 'b', linewidth=1, alpha=0.5)
sub = 10
#ax.quiver(df['Longitude'][::sub], df['Latitude'][::sub], df['u'][::sub], df['v'][::sub], color='deepskyblue', alpha=0.8, scale=10)
mplleaflet.show()