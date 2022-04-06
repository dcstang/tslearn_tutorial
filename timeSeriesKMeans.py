#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sktime.datasets import load_italy_power_demand
from sktime.datasets import load_arrow_head

from tslearn.datasets import UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans

from tslearn.utils import to_time_series_dataset

#%% Italy Power Demand
X, y = load_italy_power_demand()
df = X
df["label"] = y.tolist()

#%% Initial time series 
fig, ax = plt.subplots(figsize=(15, 2.5 * 1.618))

for series in X["dim_0"][0:500]:
    ax.plot(series, alpha = 0.05, color="gray")

#%% Show two classes
fig, ax = plt.subplots(
    nrows=2, ncols=1,
    sharex=True,
    figsize=(15, 2.5 * 1.618*2),
    gridspec_kw={"hspace":0})

colorList = ["red", "skyblue"]

for classType in range(2):
    for series in df[df["label"] == f"{classType + 1}"]["dim_0"]:
        ax[classType].plot(series, alpha = 0.02, color=colorList[classType])

for idx, classType in enumerate([2, 1]):
    for series in df[df["label"] == f"{classType}"]["dim_0"]:
        ax[idx].plot(series, alpha = 0.01, color="gray")
#%% Data manipulation
X_train = to_time_series_dataset(X["dim_0"].to_numpy())

#%% Perform tsKMeans

italy_km = TimeSeriesKMeans(n_clusters=2, n_jobs=-1, metric="dtw", verbose=1)
y_pred = italy_km.fit_predict(X_train)

#%%
predicted = X
predicted["y_pred"] = (y_pred+1).tolist()
predicted["label"] = y.tolist()
predicted["label"] = predicted["label"].astype(int) 
predicted["y_pred"].replace({1:2, 2:1}, inplace=True)
predicted["correct"] = np.where(predicted["y_pred"] == predicted["label"], True, False)

#%% plot predictions  

fig, ax = plt.subplots(
    nrows=2, ncols=1,
    sharex=True,
    figsize=(15, 2.5 * 1.618*2),
    gridspec_kw={"hspace":0})

colorList = ["orange", "purple"]

for classType in range(2):
    for series in predicted[predicted["y_pred"] == classType + 1]["dim_0"]:
        ax[classType].plot(series, alpha = 0.02, color=colorList[classType])

for idx, classType in enumerate([2, 1]):
    for series in predicted[predicted["y_pred"] == classType]["dim_0"]:
        ax[idx].plot(series, alpha = 0.01, color="gray")

#%% plot accuracy 

fig, ax = plt.subplots(
    nrows=2, ncols=1,
    sharex=True,
    figsize=(15, 2.5 * 1.618*2),
    gridspec_kw={"hspace":0})

colorList = ["yellow", "purple"]

for classType in range(2):
    for series in predicted[
        (predicted["y_pred"] == classType + 1) & (predicted["correct"] == True)]["dim_0"]:
        ax[classType].plot(series, alpha = 0.02, color="green")

for idx, classType in enumerate([2, 1]):
    for series in predicted[predicted["y_pred"] == classType]["dim_0"]:
        ax[idx].plot(series, alpha = 0.01, color="gray")

#%%

len(predicted[predicted["correct"] == True])/len(predicted)

#%%
"""
X_train, X_test, y_train, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")

k_means = TimeSeriesKMeans(n_clusters=3, metric="dtw")
y_pred = k_means.fit_predict(X_train)


#%%
fig, ax = plt.subplots(
    nrows=3, ncols=1,
    figsize=(15, 3.5 * 1.618))

colorList = ["red", "green", "blue"]

for yi in range(3):
    for xx in np.squeeze(X_train[y_pred == yi][0:10]):
        ax[yi].plot(xx, alpha=0.1, color=colorList[yi])
        #ax.plot(np.squeeze(X_train)[0:5].T, alpha = 0.3, color="gray")

# %%
# print((np.squeeze(X_train)[0:2].T).shape)
# print(np.squeeze(X_train)[0:2])
print(X_train[y_pred == 1][0])
print(X_train[y_pred == 1][0:10].shape)
print(np.squeeze(X_train[y_pred == 1][0:10]).T.shape)
print(X_train[y_pred == 0].shape)
"""