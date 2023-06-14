import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import subprocess
import numpy as np
import os

x_axis = []
y_axis = []
z_axis = []
cov = [[11.50, 50.0, 34.75],
       [50.0, 1250.0, 205.0],
       [34.75, 205.0, 110.0]
       ]
mean = [0,0,0]

for i in range(10):
    x, y, z = np.random.multivariate_normal(cov=cov, mean=mean)
    x_axis.append(x)
    y_axis.append(y)
    z_axis.append(z)


f = plt.figure(1)
ax = plt.axes(projection="3d")
ax.scatter3D(x_axis, y_axis, z_axis, c=z_axis, s=40)
f.show()
f.show()

f = open("trivariate.csv", "w")
writer = csv.writer(f)
writer.writerow(['x', 'y', 'z'])
for i in range(10):
    writer.writerow([x_axis[i], y_axis[i], z_axis[i]])
f.close()

os.system("gcc PCA2.c -lm")
os.system("./a.out trivariate.csv > trivariate_reduced.csv")

f = open("trivariate_reduced.csv", "r")
reader = csv.DictReader(f)
a_axis = []
b_axis = []
for row in reader:
    a_axis.append(float(row['a']))
    b_axis.append(float(row['b']))

g = plt.figure(2)
plt.scatter(a_axis, b_axis, s=40)
g.show()

plt.show()
