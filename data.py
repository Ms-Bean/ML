import numpy as np
import matplotlib.pyplot as plt
x_axis = []
y_axis = []
cov = [[1,3],[0.5,1]]
mean = [0,0]
for i in range(100):
    x, y = np.random.multivariate_normal(cov=cov, mean=mean)
    print("%lf,%lf" % (x, y))
