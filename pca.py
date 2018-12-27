import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os.path

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def get_distance(idx1, idx2):
    ev1 = []
    ev2 = []
    with open(evfile, 'r') as csvfile:
        embedded_vectors = csv.reader(csvfile, delimiter=' ', quotechar='|')
        ctr = 0
        for row in embedded_vectors:
            if ctr == idx1:
                for vector in row:
                    ev1.append(float(vector))
            if ctr == idx2:
                for vector in row:
                    ev2.append(float(vector))
            ctr += 1
        ev1 = np.array(ev1)
        ev2 = np.array(ev2)
    print(distance(ev1,ev2))

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

###################

from sklearn.metrics import f1_score, accuracy_score

metadata = load_metadata('images')
distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(1, num):
        distances.append(distance(pca_data[i], pca_data[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)
        
distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score');
plt.plot(thresholds, acc_scores, label='Accuracy');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
plt.xlabel('Distance threshold')
plt.legend()
plt.show()