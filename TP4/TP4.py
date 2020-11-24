import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import cophenet, inconsistent, maxRstat
from scipy.spatial.distance import pdist


df_iris = pd.read_csv('iris.csv')

print(df_iris.head())

X = df_iris.drop(['Class'],axis=1)
Y = df_iris['Class']

pca = PCA(n_components=2)
PCA_val = pca.fit_transform(X)	
df_iris_PCA = pd.DataFrame(data = PCA_val,columns = ['PC1', 'PC2'])
df_iris_PCA_class = pd.concat([df_iris_PCA, Y], axis = 1)


print(df_iris_PCA.head())


colors = {'setosa':'red', 'virginica':'green', 'versicolor':'blue'}
fig1, ax1 = plt.subplots()
ax1.scatter(df_iris_PCA['PC1'],df_iris_PCA['PC2'],c=Y.map(colors))




#Kmeans code, change X by your dataset
kmeans = KMeans(n_clusters=3, n_init=5, max_iter=300).fit(X)
kmeans.score(X)
prediction = kmeans.predict(X)

print(prediction)

fig2, ax2 = plt.subplots()
ax2.scatter(df_iris_PCA['PC1'],df_iris_PCA['PC2'],c=prediction)

tab=contingency_matrix(Y,prediction)
print(tab)

def silhouette_kmeans(X,range_n):
  
  range_n_clusters = range_n

  for n_clusters in range_n_clusters:
      # Create a subplot with 1 row and 2 columns
      fig, (ax1, ax2) = plt.subplots(1, 2)
      fig.set_size_inches(18, 7)

      # The 1st subplot is the silhouette plot
      # The silhouette coefficient can range from -1, 1 but in this example all
      # lie within [-0.1, 1]
      ax1.set_xlim([-0.1, 1])
      # The (n_clusters+1)*10 is for inserting blank space between silhouette
      # plots of individual clusters, to demarcate them clearly.
      ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

      # Initialize the clusterer with n_clusters value and a random generator
      # seed of 10 for reproducibility.
      clusterer = KMeans(n_clusters=n_clusters, random_state=10)
      cluster_labels = clusterer.fit_predict(X)

      # The silhouette_score gives the average value for all the samples.
      # This gives a perspective into the density and separation of the formed
      # clusters
      silhouette_avg = silhouette_score(X, cluster_labels)
      print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

      # Compute the silhouette scores for each sample
      sample_silhouette_values = silhouette_samples(X, cluster_labels)



      y_lower = 10


      for i in range(n_clusters):
          # Aggregate the silhouette scores for samples belonging to
          # cluster i, and sort them
          ith_cluster_silhouette_values = \
              sample_silhouette_values[cluster_labels == i]

          ith_cluster_silhouette_values.sort()

          size_cluster_i = ith_cluster_silhouette_values.shape[0]
          y_upper = y_lower + size_cluster_i

          color = cm.nipy_spectral(float(i) / n_clusters)
          ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

          # Label the silhouette plots with their cluster numbers at the middle
          ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

          # Compute the new y_lower for next plot
          y_lower = y_upper + 10  # 10 for the 0 samples

      ax1.set_title("The silhouette plot for the various clusters.")
      ax1.set_xlabel("The silhouette coefficient values")
      ax1.set_ylabel("Cluster label")

      # The vertical line for average silhouette score of all the values
      ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

      ax1.set_yticks([])  # Clear the yaxis labels / ticks
      ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

      # 2nd Plot showing the actual clusters formed
      colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
      ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=60, lw=0, alpha=0.7,
                  c=colors, edgecolor='k')

      # Labeling the clusters
      centers = clusterer.cluster_centers_
      # Draw white circles at cluster centers
      ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                  c="white", alpha=1, s=200, edgecolor='k')

      for i, c in enumerate(centers):
          ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                      s=50, edgecolor='k')

      ax2.set_title("The visualization of the clustered data.")
      ax2.set_xlabel("Feature space for the 1st feature")
      ax2.set_ylabel("Feature space for the 2nd feature")
      
      return 
      
silhouette_kmeans(X,[3])


######

# Exercice C
#  needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet, inconsistent, maxRstat
from scipy.spatial.distance import pdist


np.random.seed(42)
a=np.random.multivariate_normal( [10, 0], [[3, 1], [1, 4]], size=[100, ])
b=np.random.multivariate_normal( [0, 20], [[3, 1], [1, 4]], size=[50, ])
X = np.concatenate((a, b),)
plt.scatter(X[:, 0], X[:, 1])
plt.title('My data distribution')
plt.show()


Z = linkage(X, 'ward', optimal_ordering=True)
c, coph_dists = cophenet(Z, pdist(X))
print ('Cophenetic Correlation : %1.2f' % c)
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (full)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
print(Z)

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

max_d = 14
clusters = fcluster(Z, max_d, criterion='distance')
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')


k=4
clusters = fcluster(Z, k, criterion='distance')
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')


# Exercice D
df_exo4_atm_extr = pd.read_csv('exo4_atm_extr.csv', sep=";")

X = df_exo4_atm_extr.drop(['Type'], axis=1)
Y = df_exo4_atm_extr['Type']
Y.columns = ['Type']
print(X.head())


from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

history_DB=[]
history_CH=[]
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42).fit(X)
    labels = kmeans.labels_
    value_DB = davies_bouldin_score(X, labels)
    value_CH = calinski_harabasz_score(X, labels)
    history_DB.append(value_DB)
    history_CH.append(value_CH)
x_value = np.arange(2,11)



fig, ax = plt.subplots()
ax.scatter(range(2, 11), history_DB)
ax.axvline(x=np.argmin(history_DB)+2, color="red", linestyle="--", label='the best partition')
ax.set_xlabel("Nb of clusters")
ax.set_ylabel("Davies-Bouldin score")
plt.show()

fig, ax = plt.subplots()
ax.scatter(range(2, 11), history_CH)
ax.axvline(x=np.argmax(history_CH)+2, color="red", linestyle="--", label='the best partition')
ax.set_xlabel("Nb of clusters")
ax.set_ylabel("Calinski-Harabasz score")
plt.show()
 Even if 10 clusters give the biggest CH score in absolute,
 we should consider that the most optimal nb of clusters is 4 as the gap between 3 and 4 is the biggest


# Question 4 for DB
#Visualize data with PCA
colors = "bgrcmykw"
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(X)
data_pca = pca.transform(X)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(Y)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][Y==cl], data_pca[:,1][Y==cl], c=colors[i], label=cl, alpha=0.5) #we generate new colors iteratively
ax.legend()
plt.title('PCA visualization')
plt.show()

# We do k-means for 4 clusters (minimum score of DB)
prediction = KMeans(n_clusters=4, random_state=42, n_init=5, max_iter=300).fit_predict(X)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(prediction)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][prediction==cl], data_pca[:,1][prediction==cl], c=colors[i], label=cl, alpha=0.5) #we generate new colors iteratively
ax.legend()
plt.title('Kmeans visualization')
plt.show()

#even better with data normalization
scaler = StandardScaler().fit(X.values) 
X_norm = scaler.transform(X.values)

colors = "bgrcmykw"
n_components = 2
pca2 = PCA(n_components=n_components)
pca2.fit(X_norm)
data_pca2 = pca2.transform(X_norm)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(Y)): #we do the loop to be able to show labels
    ax.scatter(data_pca2[:,0][Y==cl], data_pca2[:,1][Y==cl], c=colors[i], label=cl, alpha=0.5) #we generate new colors iteratively
ax.legend()
plt.title('PCA visualization (normalized)')
plt.show()


prediction = KMeans(n_clusters=4, random_state=42, n_init=5, max_iter=300).fit_predict(X_norm)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(prediction)): #we do the loop to be able to show labels
    ax.scatter(data_pca2[:,0][prediction==cl], data_pca2[:,1][prediction==cl], c=colors[i], label=cl, alpha=0.5) #we generate new colors iteratively
ax.legend()
plt.title('Kmeans visualization (normalized)')
plt.show()

#given the 4 outliers, it could be interesting to remove them and try again.
