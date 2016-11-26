# source: http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

import gpxpy.geo as mod_geo
import sklearn.cluster as sk
import pandas as pd
import sys


pd.options.display.precision = 11
np.set_printoptions(precision=11)

__date__ = "2011-11-17 Nov denis"
    # X sparse, any cdist metric: real app ?
    # centers get dense rapidly, metrics in high dim hit distance whiteout
    # vs unsupervised / semi-supervised svm

#...............................................................................
def kmeans( X, centers, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centers, Xtocentre, distances = kmeans( X, initial centers ... )
    in:
        X N x dim  may be sparse
        centers k x dim: initial centers, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centers
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centers, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centers = centers.todense() if issparse(centers) else centers.copy()
   
    N, dim = X.shape
    k, cdim = centers.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centers %s must have the same number of columns" % (X.shape, centers.shape ))
    #if verbose:
    #    print "kmeans: X %s  centers %s  delta=%.2g  maxiter=%d  metric=%s" % (X.shape, centers.shape, delta, maxiter, metric)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centers, metric=metric, p=p)  # |X| x |centers|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx, xtoc]
        avdist = distances.mean()  # median ?
        #if verbose >= 2:
        #    print "kmeans: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                if type(metric)==str:
                    centers[jc] = X[c].mean( axis=0 )
                else:
                    if metric.func_name == 'weightedK':
                        # change here to use different center determination functions
                        #centers[jc] = estCenters(X[c])
                        centers[jc] = weightedCenters(X[c], inverse = False, square = False)

    return centers, xtoc, distances


#...............................................................................
def kmeanssample( X, k, nsample=0, iterno=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centers
    """
    Xsample = remOutlier(X, k)
    #print len(Xsample), len(X), iterno
    pass1centers = getInitCenters(Xsample, k)

    '''#comment below to not use initial centers
    ctr = pd.read_csv("gTruthRU2DA.csv")
    rc = ctr[ctr['sid']==iterno]
    rc = np.array([[rc.c0_lat.values[0], rc.c0_lon.values[0], 1], [rc.c1_lat.values[0], rc.c1_lon.values[0], 1]])
    pass1centers = rc
    until here'''

    samplecenters = kmeans( Xsample, pass1centers, **kwargs)[0]
 
    return kmeans( Xsample, samplecenters, **kwargs )


def remOutlier(X, k):
    check = True
    initX = X
    maxPer2remove = 0.6 # ? can be configured
    clusterSizeThreashold = 0.33 #? 0.15 hardCoded for now. this should be computed dynamically
    while check == True:
        if len(X) < len(initX) * maxPer2remove :
            return initX
        p1 = getInitCenters(X, k)
        c = kmeans(X, p1, metric = weightedK)
        freq = np.zeros(k)
        check = False
        for i in range(k):
            freq[i] = list(c[1]).count(i) / len(c[1])
        valsToRemove = np.array([])
        for i in range(k):
            if (freq[i] < clusterSizeThreashold):
                tmp = np.where(c[1]==i)[0]
                valsToRemove = np.append(valsToRemove, tmp, axis = 0)
        if len(valsToRemove) > 0:
            X = np.delete(X, valsToRemove, axis = 0)
            check = True
    return X
        

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d


def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    sampleix = X[:,2].argsort()[:n]
    return X[sampleix]


def nearestcenters( X, centers, metric="euclidean", p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centers, metric=metric, p=p )  # |X| x |centers|
    return D.argmin(axis=1)


def Lqmetric( x, y=None, q=.5, **kwargs ):
    # yes a metric, may increase weight of near matches; see ...
    wk(x,y, **kwargs)
    return (np.abs(x - y) ** q).mean() if y is not None else (np.abs(x) ** q) .mean()


def weightedK(x, y, **kwargs):
    '''
    x and y expect data in the following format
    -------------------------------------------
    lat     lon     acc
    '''
  
    # check point's accuracy includes the center
    if (mod_geo.distance(x[0], x[1], None, y[0], y[1], None) <= x[2]):
        return 0
    else:
        return (mod_geo.distance(x[0], x[1], None, y[0], y[1], None) - x[2])


def estCenters(c):
    # c is an array of points on the sampling segment.
    mc = c.mean( axis=0 )
    percentToMove = 1.5
    for i in range(len(c)):
        dir = mod_geo.get_bearing(mod_geo.Location(c[i][0],c[i][1]), mod_geo.Location(mc[0], mc[1]))
        distBet = mod_geo.distance(c[i][0],c[i][1],None, mc[0], mc[1], None)
        if (distBet >= c[i][2]):
            mDist = c[i][2] * percentToMove
            updatedPoint = mod_geo.point_from_distance_bearing(mod_geo.Location(c[i][0], c[i][1]), mDist, dir)
        else:
            updatedPoint = mod_geo.Location(mc[0], mc[1])
        c[i][0], c[i][1] = updatedPoint.latitude, updatedPoint.longitude
    return c.mean( axis=0 )


def weightedCenters(c, wIndex=2, inverse=False, square=False):
    '''
    c is an array of points on the sampling segment.
    wIndex = weights index to be used.
    '''
    m = c[:,2:].mean(axis=0)
    wts = c[:,wIndex]
    # to get square weights
    if (square == True):
        wts = wts ** 2
    # to inverse the weights (for accuracy, not for nos)
    if (inverse == True and len(wts) > 1) :
        wts = sum(wts) - wts
    latlon = c[:,0:2]
    retvar = np.average(latlon, weights=wts, axis=0)
    return np.append(retvar, wts.mean(axis=0))



distMin = 35 # minimum % points in one cluster
def _removeOutliers(X, k):
    chkAgain = True

    while chkAgain == True:
        #c = s.KMeans(n_clusters=k).fit_predict(X)
        d = sk.k_means(X, 2)
        c = d[1]
        #plt.scatter(X[:,0],X[:,1],c=c)
        uniq, dist = np.unique(c,return_counts=True)
        dist = [i*100/dist.sum() for i in dist]
        chkAgain = False
        indexes = []
        for i in range(0,len(dist)):
            if dist[i] < distMin:
                chkAgain = True
                #print "loop # ", i, dist[i], distMin
                val = uniq[i]
                for i in range(0, len(c)):
                    if c[i] == val:
                        indexes.append(i)
        X = np.delete(X, indexes, axis=0)
    return (X, d[0])


def getInitCenters(f, nop):
    '''
    get Initial centers with nop points to start weighted k means
    '''
    #X = np.column_stack([f.lat, f.lon])
    X = f.mean(axis = 0)
    std = f.std(axis = 0)
    
    if nop == 1:
        pts = np.column_stack(X)
    if nop > 1:
        diff = np.column_stack(std) * 2
        pts = np.column_stack(X - std)
        delta = diff / (nop - 1)
        for i in range(1, nop ):
            pts = np.append(pts, (delta * i) + pts[0], axis=0)
    
    return pts

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centers=, ... )
        in: either initial centers= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centers, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centers[jcentre]
                J indexes e.g. X[J], classes[J]
    """

    def __init__( self, X, k=0, centers=None, nsample=0, **kwargs ):
        self.X = X
        if centers is None:
            self.centers, self.Xtocentre, self.distances = kmeanssample(X, k=k, nsample=nsample, **kwargs )
        else:
            self.centers, self.Xtocentre, self.distances = kmeans(X, centers, **kwargs )
        self.cluster_centers_ = self.centers
    def __iter__(self):
        for jc in range(len(self.centers)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................
if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10
    dim = 4
    ncluster = 2
    kmsample = 10  # 0: random centers, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    metric = "euclidean"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    metric = weightedK
    seed = 18

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True)
    np.random.seed(seed)
    random.seed(seed)

    print "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
        N, dim, ncluster, kmsample, metric)
    X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/

    t0 = time()
    if kmsample > 0:
        centers, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample, delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcenters = randomsample( X, ncluster )
        centers, xtoc, dist = kmeans( X, randomcenters, delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print "%.0f msec" % ((time() - t0) * 1000)
    m = Kmeans(X,2)
    print m.cluster_centers_
    # also ~/py/np/kmeans/test-kmeans.py