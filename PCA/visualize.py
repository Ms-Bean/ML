import csv
import matplotlib.pyplot as plt
import subprocess
import numpy as np
csv_file = open("data.csv")
csv_reader = csv.DictReader(csv_file)
x_axis = []
y_axis = []
for row in csv_reader:
    x_axis.append(float(row['x']))
    y_axis.append(float(row['y']))

subprocess.run(['gcc', 'PCA.c', '-lm'])
text = subprocess.check_output(['./a.out', 'data.csv']).decode('UTF-8')[:-1]
eigenvectors = text.split('\n')
for i in range(len(eigenvectors)):
    eigenvectors[i] = eigenvectors[i].split(',')
    for j in range(len(eigenvectors[i])):
        eigenvectors[i][j] = float(eigenvectors[i][j])


fig, ax = plt.subplots()

ax.axis('equal')
ax.scatter(x_axis, y_axis)

print(eigenvectors)
ax.quiver(0, 0, eigenvectors[0][0], eigenvectors[0][1], scale=1)
ax.quiver(0, 0, eigenvectors[1][0], eigenvectors[1][1], scale=1)
plt.show()