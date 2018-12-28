import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

evfile = 'embedded.csv'
df = pd.read_csv(evfile, delimiter=' ', quotechar='|', header=None)
df.dropna(how="all", inplace=True)
data = df.ix[:,0:8].values

z_scaler = StandardScaler()
z_data = z_scaler.fit_transform(data)
pca_trafo = PCA().fit(z_data);

pca_trafo = PCA(n_components=2)
pca_data = pca_trafo.fit_transform(z_data)
pca_inv_data = pca_trafo.inverse_transform(np.eye(2))

get_distance(0,1)
print(distance(pca_data[0], pca_data[1]))
