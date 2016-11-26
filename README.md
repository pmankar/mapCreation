# Introduction

This project creates highly accurate maps using mobile sensors information only. Currently using only GPS information, but it can easiliy be modified to use other sensor information as well.

# Contents
* **clustering** : Uses several clustering algorithm to cluster the traces.
* **fcd2csv** : A simple convertor for .fcd files to .csv files.
* **gpserr** : A simple GPS Error Model using a simple error configuration and virtual GPS device generator.
* **GPSRawTestData** : Result of 3 hours of data collected from mobile.
* **gpxFilter** : Tool to filter the sensor information collected from the mobile devices.
* **gpxMap** : Tool to draw map from the generated map data or a .gpx file.
* **gpxpy** : Modified version of the original gpxpy https://github.com/tkrajina/gpxpy.
* **README.md** : This file.

# Requirements:
* **matplotlib**
* **scikit-learn**
* **gpxpy**