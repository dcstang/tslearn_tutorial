#%%

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import matplotlib.pyplot as plt

firstTS = [1,2,3,5,7]
secondTS = [0,5,np.nan,10,12]
thirdTS = [10,np.nan,2,np.nan]
fourthTS = [10,np.nan,5,1]
fifthTS = [2,4,np.nan,12]


X = to_time_series_dataset([
    firstTS, secondTS, thirdTS, fourthTS, fifthTS
    ])

#%% after interpolation
from scipy.interpolate import interp1d

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B

tempList = []
for n in range(0,5):
    tempList.append(fill_nan(np.squeeze(X[n])))

X2 = to_time_series_dataset(tempList)

#%% before and after interpolation
fig, ax = plt.subplots(
    nrows=3,
    figsize=(8,15),
    gridspec_kw={"hspace":0.15}
)
for ts in X:
    ax[0].plot(ts, marker='o', alpha = 0.7)

for ts in X2:
    ax[1].plot(ts, marker='o', alpha = 0.7)

for ts1, ts2 in zip(X, X2):
    print(np.ma.masked_array(np.squeeze(ts2), mask=[ts1 == ts2]))
    ax[2].plot(ts2, alpha=0.5, color="gray")
    ax[2].scatter(
        np.arange(0, len(ts2)),
        np.ma.masked_array(np.squeeze(ts2), mask=[ts1 == ts2]),
        color = "red", alpha=0.5)

for title, i in zip(
    ["Raw Data", "Linear Interpolation", "Highlight Interpolated"],
    fig.axes
    ):
    i.set_title(title)
    i.yaxis.set_visible(False)
    i.xaxis.set_visible(False)

plt.show()
plt.savefig("figures/timeseries_linear_interpolation.png")

#%% unable to handle missingness without interpolation 

km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
# pred = km.fit_predict(X)
pred2 = km.fit_predict(X2)
