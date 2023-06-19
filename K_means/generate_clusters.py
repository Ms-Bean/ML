import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

centroids = [[-1,1,-1,1],[0,0,0,0],[1,1,1,-0.5]]
dat = [[],[],[],[]]

for i in range(150):
    for j in range(4):
        for k in range(3):
            dat[j].append(np.random.normal(centroids[k][j], 0.4))



f = open("data.csv", "w")
writer = csv.writer(f)
writer.writerow(['w', 'x', 'y', 'z'])
for i in range(150):
    writer.writerow([dat[0][i], dat[1][i], dat[2][i], dat[3][i]])
f.close()

os.system("gcc K_means.c -lm")
os.system("./a.out data.csv")


cluster_1 = [[],[],[]]
f = open("cluster_1.csv", "r")
reader = csv.DictReader(f)
for row in reader:
    cluster_1[0].append(float(row['x']))
    cluster_1[1].append(float(row['y']))
    cluster_1[2].append(float(row['z']))
f.close()

cluster_2 = [[],[],[]]
f = open("cluster_2.csv", "r")
reader = csv.DictReader(f)
for row in reader:
    cluster_2[0].append(float(row['x']))
    cluster_2[1].append(float(row['y']))
    cluster_2[2].append(float(row['z']))
f.close()

cluster_3 = [[],[],[]]
f = open("cluster_3.csv", "r")
reader = csv.DictReader(f)
for row in reader:
    cluster_3[0].append(float(row['x']))
    cluster_3[1].append(float(row['y']))
    cluster_3[2].append(float(row['z']))
f.close()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cluster_1[0], cluster_1[1], cluster_1[2], c='r',)
ax.scatter(cluster_2[0], cluster_2[1], cluster_2[2], c='b')
ax.scatter(cluster_3[0], cluster_3[1], cluster_3[2], c='g',)

plt.show()