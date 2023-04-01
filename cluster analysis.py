# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:41:19 2022

@author: harigran
"""

import pandas as pd
df= pd.read_csv(r"C:\Users\my pc\Desktop\MBA - BA II\lab/usarrest.csv")
# Using set_index() method on 'unnamed' column
df = df.set_index('City_Name')
df.head()

#Creating dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward
result=ward(df)
dendrogram(result,leaf_rotation=90, leaf_font_size=8, labels=df.index)
plt.title("DENDROGRAM")
plt.xlabel('Observations')
plt.ylabel('Distances')
#plt.axhline(300, color='r')
plt.show()

#Perform Clustering
from sklearn.cluster import AgglomerativeClustering

agg=AgglomerativeClustering(n_clusters=3,affinity='euclidean', linkage='ward')
agg.fit_predict(df)
x=agg.labels_
print(x)

plt.figure(figsize=(10, 7))
plt.scatter(df['Murder'], df['Assault'], c=agg.labels_ , cmap='rainbow')
# annotate points in axis
for idx, row in enumerate(df.index):
    plt.annotate(row, (df['Murder'][idx],df['Assault'][idx]) )
# force matplotlib to draw the graph
plt.show()

