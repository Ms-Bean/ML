import matplotlib.pyplot as plt
import random
x_axis = []
y_axis = []
cov = [[0.2,-0.15],[-0.15,0.2]]
mean = [0,0]

print("x,y")
for i in range(100):
    x = random.uniform(-1, 1)
    y = 0.3*random.uniform(-1, 1) 
    x_axis.append(x)
    y_axis.append(y)
    print("%lf,%lf" % (x, y))

plt.axis('equal')
plt.scatter(x_axis, y_axis)
plt.show()