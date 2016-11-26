from xml.etree import ElementTree
import sys
import getopt
import pandas as pd
import datetime

if (len(sys.argv) > 1):
    try:
        argList, args = getopt.getopt(sys.argv[1:], "i:o:")
    except getopt.GetoptError as err:
        print "Something wrong with argument list"
    
    for k,v in argList:
        if k == '-i':            ipFile = v
        if k == '-o':            opFile = v
else:
    # set manually
    ipFile = 'fcd.xml'
    opFile = 'traces.csv'

tree = ElementTree.parse(ipFile)
root = tree.getroot()

op=open(opFile, 'w+')
print >>op,'sts,date,vid,lat,lon,speed,angle,pos,lane,slope,type'

dt = '03.08.2016' #%d.%m.%Y
st = datetime.datetime.strptime('12:00:00:000100', '%H:%M:%S:%f') #%H:%M:%S:%f

for c in root.getchildren():
    for v in c.getchildren():
        print >>op,c.attrib['time']+','+dt+','+v.attrib['id']+','+v.attrib['y']+','+v.attrib['x']+','+v.attrib['speed']+','+v.attrib['angle']+','+v.attrib['pos']+','+v.attrib['lane']+','+v.attrib['slope']+','+v.attrib['type']
op.close()
# sort the file for readability
tf = pd.read_csv(opFile)
sorted = tf.sort_values(['vid','sts'], ascending=[1, 1])

dtList=[]
for i, r in sorted.iterrows():
    dtList.append((st + datetime.timedelta(0, r.sts)).strftime("%H:%M:%S:%f"))
sorted['hour'] = dtList
sorted.to_csv(opFile.split('.')[0]+'_s.csv',sep=',', index=False)