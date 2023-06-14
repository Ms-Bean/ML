import numpy as np
import matplotlib.pyplot as plt
x_axis = []
y_axis = []
cov = [[0.2,-0.15],[-0.15,0.2]]
mean = [0,0]

print("x,y")
for i in range(100):
    x, y = np.random.multivariate_normal(cov=cov, mean=mean)
    x_axis.append(x)
    y_axis.append(y)
    print("%lf,%lf" % (x, y))

plt.scatter(x_axis, y_axis)
plt.show()