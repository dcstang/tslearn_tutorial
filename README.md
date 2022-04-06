# Time Series Clustering: tslearn tutorial

Looking at various time series clustering methods.
There are various packages of interest among which are `tslearn`, `sktime` 


### Time Series K Means
Here the unique difference is instead of using euclidean distance as a differential distance metric between cluster centroids, we use `dynamic time wraping` instead.
(see code in `timeSeriesKMeans.py`)

![initial][./figures/initial_powerDemand.png]
![clustered][./figures/timeSeriesKM.png]