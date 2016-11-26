import clustering as c
import time
startTime = time.time()
print "Started..."

c.runPlots('runningRes.csv')
#fileName = "combined"
#method = "wkmeans"
#m= ["_41-51", "_51-61", "_61-71", "_71-81", "_81-91"]
#m= ["_81-91"]
#pattern = "_afbc"
#c.makeReadable(fileName+".csv", fileName+"_read.csv")
#c.segments(fileName+"_read.csv", fileName+"_segs.csv")
#c.segments("sN_0_read.csv", fileName+"DA2RA_segs.csv")

#c.segTraceInt("sN_0_segs_RU2DA.csv", fileName+"_read.csv",fileName+"_2" + ".csv") #use same file cxFile for all scenarios
#c.segTraceInt(fileName+"_segs.csv",fileName+"_read.csv",fileName+"_2.csv") #use same file cxFile for all scenarios
#c.dgpsInt("sN_0_segs_DA2RU.csv",fileName+".csv",fileName+"_2.csv")


#c.segTraceInt("sN_0_segs_RU2DA.csv", fileName+"_read.csv",fileName+"_2" + pattern + ".csv") #use same file cxFile for all scenarios
#c.segTraceInt(fileName+"_segs.csv", fileName+"_read.csv",fileName+"_2" + pattern + ".csv") #use same file cxFile for all scenarios

#for mnth in m:
    #c.clusterTypes(fileName + "_2" + mnth + pattern  + ".csv", fileName + "_" + method[:2] + "_3" + mnth + pattern + ".csv", method)
    #c.csvToKML(fileName + "_" + method[:2] + "_3" + mnth + pattern + ".csv", fileName+ "_" + method +  mnth + pattern + ".kml")

#c.clusterTypes(fileName + "_2_ada" + pattern[-1] +".csv", fileName + "_" + method[:2] + "_3" + pattern + ".csv", method)
#c.csvToKML(fileName + "_" + method[:2] + "_3" + pattern + ".csv", fileName+ "_" + method + pattern + ".kml")

#c.csv2gpx(fileName+"_read.csv",fileName+".gpx")

#c.details(fileName+"_2.csv", fileName+"_3.csv", fileName+"_details.csv")
#c.plotConstructionSites(fileName+".csv",fileName+".kml")


endTime = time.time()
print "Time taken: " + str(endTime - startTime) + " seconds"