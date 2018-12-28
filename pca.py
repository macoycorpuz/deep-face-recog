import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ev_file = 'embedded-vectors/labeled_ev.csv'
df = pd.read_csv(ev_file, delimiter=' ', quotechar='|', header=None)
df.dropna(how="all", inplace=True)
data = df.values

pca = PCA(n_components=128)
pca.fit(data)
pca_data = pca.transform(data)
pca_inv_data = pca.inverse_transform(pca_data)


print("original shape:   ", data.shape)
print("transformed shape:", pca_data.shape)
print("inverse transform shape:", pca_inv_data.shape)
print(data)
print(pca_inv_data)
# plt.scatter(pca_data[:, 0], pca_data[:, 1])
# plt.axis('equal')
# plt.show()
