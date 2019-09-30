import numpy as np
import matplotlib.pyplot as plot
xi = []
for i in np.arange(-1, 10, .01):
    xi.append(i)
d = 0  # This one is for plotting 1D and 2D
h = [.1, 1, 5, 10]  #bandwidth
#Algortihm
# by referring to the algorithm in the ppt
'''Part 1:
A function [p, x] = mykde(X,h) that performs kernel density estimation on X with
bandwidth h. It should return the estimated density p(x) and its domain x where you estimated the
p(x) for X in 1-D and 2-D.'''
def mykde(X, x, h):
    # X is the our data set
    # x is the current point passed as argument to the function
    # h is the bin size
    sum_k = 0.0
    if (d == 1):  # if 1D like part 2 and 3
        N = len(X)  # the number of point in the data set
        for i in range(len(X)):  # for each point in the data set,
            # calculate the current point - that point and then divide by the bin size
            u = (x - X[i]) / h
            if (abs(u) <= 0.5):  # if that value is within the hypercube
                k = 1
            else:
                k = 0
            sum_k = sum_k + k  # find the summation
        p = float(sum_k / (N * h))  # find the kernal density
        return p, x

    if (d == 2):  # if 2D like part 4
        # for 2D we will use h*h
        N = len(X)
        for i in range(len(X)):
            u1 = (x - X[i][0]) / h
            u2 = (x - X[i][1]) / h
            if (abs(u1) <= 0.5 and abs(u2) <= 0.5):
                k = 1
            else:
                k = 0
            sum_k = sum_k + k
        p = float(sum_k / (N * h * h))
        return p, x


'''part 3: 
Generate N = 1000 Gaussian random data with mu1 = 5 and sigma1 = 1 and another Gaussian
random data with mu2 = 0 and sigma2 = 0:2. Test your function mykde on this data with h = {0.1, 1, 5, 10}.
''' 
mu1 = 5
mu2 = 0
sigma1 = 1
sigma2 = 0.2
N1 = 500
N2 = 500
X = np.concatenate((np.random.normal(mu1, sigma1, N1), np.random.normal(mu2, sigma2, N2)), 0)
d = 1
p1 = [0.0] * (len(xi))
p2 = [0.0] * (len(xi))
p3 = [0.0] * (len(xi))
p4 = [0.0] * (len(xi))

for i in range(len(xi)):  # for every point we calculate the kernel density using all 4 bin values
    p1[i], xi[i] = mykde(X, xi[i], h[0])
    p2[i], xi[i] = mykde(X, xi[i], h[1])
    p3[i], xi[i] = mykde(X, xi[i], h[2])
    p4[i], xi[i] = mykde(X, xi[i], h[3])

figure2, axes = plot.subplots(5, 1, constrained_layout=True)
figure2.canvas.set_window_title('Problem 2 - Part 3')

axes[0].hist(X, 100, density=True, color='cyan')
axes[1].plot(xi, p1, c='cyan')
axes[2].plot(xi, p2, c='cyan')
axes[3].plot(xi, p3, c='cyan')
axes[4].plot(xi, p4, c='cyan')

axes[1].set_title('h=.1 q2-part3')
axes[2].set_title('h=1 q2-part3')
axes[3].set_title('h=5 q2-part3')
axes[4].set_title('h=10 q2-part3')




'''part 2: 
Generate N = 1000 Gaussian random data with mu1 = 5 and sigma1 = 1. Test your function mykde
on this data with h = {0.1, 1, 5, 10}. 
'''
m = 5
s = 1
N = 1000

X = np.random.normal(m, s, N)  # generate the 1000 data points

d = 1

p1 = [0.0] * (len(xi))
p2 = [0.0] * (len(xi))
p3 = [0.0] * (len(xi))
p4 = [0.0] * (len(xi))

for i in range(len(xi)):  # for every point we calculate the kernel density using all 4 bin values
    p1[i], xi[i] = mykde(X, xi[i], h[0])
    p2[i], xi[i] = mykde(X, xi[i], h[1])
    p3[i], xi[i] = mykde(X, xi[i], h[2])
    p4[i], xi[i] = mykde(X, xi[i], h[3])

figure1, axes = plot.subplots(nrows=5, ncols=1, constrained_layout=True)
figure1.canvas.set_window_title('Problem 2 - Part 2')

axes[0].hist(X, 100, density=True, color='g')
axes[1].plot(xi, p1, c='g')
axes[2].plot(xi, p2, c='g')
axes[3].plot(xi, p3, c='g')
axes[4].plot(xi, p4, c='g')

axes[1].set_title('h=.1 q2-part2')
axes[2].set_title('h=1 q2-part2')
axes[3].set_title('h=5 q2-part2')
axes[4].set_title('h=10 q2-part2')
######################################################################################3

'''part 4:
'''
mu1 = [1, 0]
mu2 = [0, 1.5]
sigma1 = [[0.9, 0.4], [0.4, 0.9]]
sigma2 = [[0.9, 0.4], [0.4, 0.9]]
N = 500

# generate the data, 500 for eac set
data = np.concatenate((np.random.multivariate_normal(mu1, sigma1, N), np.random.multivariate_normal(mu2, sigma2, N)), 0)

d = 2  # this question asks for 2-D gaussian data

p1 = [0.0] * (len(xi))
p2 = [0.0] * (len(xi))
p3 = [0.0] * (len(xi))
p4 = [0.0] * (len(xi))

for i in range(len(xi)):  # for every point we calculate the kernel density using all 4 bin values
    p1[i], xi[i] = mykde(data, xi[i], h[0])
    p2[i], xi[i] = mykde(data, xi[i], h[1])
    p3[i], xi[i] = mykde(data, xi[i], h[2])
    p4[i], xi[i] = mykde(data, xi[i], h[3])

figure3, axes = plot.subplots(5, 1, constrained_layout=True)
figure3.canvas.set_window_title('Problem 2 - Part 4')

axes[0].hist(data, 100, density=True)
axes[1].plot(xi, p1, c='blue')
axes[2].plot(xi, p2, c='blue')
axes[3].plot(xi, p3, c='blue')
axes[4].plot(xi, p4, c='blue')

axes[1].set_title('h=.1 q2-part4')
axes[2].set_title('h=1 q2-part4')
axes[3].set_title('h=5 q2-part4')
axes[4].set_title('h=10 q2-part4')

# plot the histograms
plot.show()