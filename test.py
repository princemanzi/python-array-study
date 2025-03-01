import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

# Test NumPy
A = np.array([[1, 2], [3, 4]])
print("NumPy Matrix A:\n", A)

# Test scikit-learn
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression()
model.fit(X, y)
print("Scikit-learn Linear Regression Test: Prediction for 5 →", model.predict([[5]])[0])

# Test the time module
start_time = time.time()
time.sleep(1)
end_time = time.time()
print(f"Time module test: Slept for {end_time - start_time:.2f} seconds")
 

print("Hello , world")
 
x = [-1.2, 0.4, 4, 2.5, 2.4]
len(x)


 
 # 2D vector
v = np.array([3, 4])
print ("2D vector v:", v)

# 3D vector 
w = np.array([4, 1, 6])
print("3D vector w:", w)

# Addition of vectors in python

a = np.array([2, 5])
b = np.array([4, 1])

add_result = a + b
print("a + b =", add_result)

# subtraction of vectors in python

subtract_results = a - b
print("a - b =", subtract_results)

# scalar multiplication

c = 3 * np.array([1, 2])
print("c = 3 * [1, 2]", c)

#  Dot Product (Inner Product)

a = np.array([1, 2])
b = np.array([3, 4])

dot_product = np.dot(a, b)
print ("a*b =", dot_product)

x = np.array([2.4, 2, 4.6, 1.3, 4.6, -7.2, 3.5])
# print(len(x))
# print(x[3])
 
# x[3] = 1.5
# print(x)

# x[-1]
# print(x)

y = x
x[3] = 20
print(y)

# vector equality

x = np.array([2.4, 3.1, 3.6, -7.2])
y = x.copy()
x[2] = 20.1
print(y)

y = x.copy()
x == y
print(x == y)

y = x.copy()
x[2] = 3.7
y == x
print(y == x)


#Scalars versus 1-vectors.

x = 2.4
y = [2.4]
x == y
print(x == y)

x = 2.4
y = [2.4]
print(y[0] == 2.4)

#block and stocked vectors

x = np.array([2, 1])
y = np.array([1, 2, 3])
w = np.concatenate((x,y))
print(w)

x = np.array([3, 5, 8, 2, 7, 9])
y = x[1:4]
print(y)

x = [1, 2]
y = [3, 4]
z = [5, 6]
list_of_vectors = [x, y, z]
print (list_of_vectors[1] [0])


#zero vector
print(np.zeros(3))

i = 2
n = 4
x = np.zeros(n)
x[i] = 1

print(x)

print(np.ones(3))

# random vectors

print(np.random.random(2))

# Plotting

plt.ion()
temps = [ 71, 71, 68, 69, 68, 69, 68, 74, 77, 82, 85, 86, 88, 86,
85, 86, 84, 79, 77, 75, 73, 71, 70, 70, 69, 69, 69, 69, 67,
68, 68, 73, 76, 77, 82, 84, 84, 81, 80, 78, 79, 78, 73, 72,
70, 70, 68, 67 ]
plt.plot(temps, '-bo')
plt.savefig('temperature.pdf', format = 'pdf')

# vector addition and subtraction
import numpy as np # type: ignore
x = np.array([1, 2, 3])
y = np.array([100, 200, 300])
result_addition = x + y
result_subtraction = x - y
print("sum of vectors = ", result_addition)
print("Difference of vectors =", result_subtraction)

p_initial = np.array([22.15, 89.32, 56.77])
p_final = np.array([23.05, 87.32, 53.13])
r = (p_final - p_initial) / p_initial
print(r)


#Linear functions

#functions in python
#f(x) = x1 + x2 - x3^(2) can be defined as follows

f = lambda x: x[0] + x[1] - x[3]**2
print(f([-1, 0, 1, 2]))



# Angle

# define angle function which defines radians

ang = lambda x, y : np.arccos(x @ y / (np.linalg.norm(x)*(np.linalg.norm(y))))
a = np.array([1, 2, -1])
b = np.array([2, 0, -3])
print(ang(a,b))

# get angle in degrees

print(ang(a,b) * (360/(2*np.pi)))




ang = lambda x, y : np.arccos( x @ y / (np.linalg.norm(x) * (np.linalg.norm(y))))
a = np.array([2, 1, -1])
b = np.array([3, 2, -3])
print(ang(a,b))

print(ang(a,b) * (360 / (2*np.pi)))

# Permutations
data = np.array([1, 2, 3, 4])
random_permutation = np.random.permutation(data)
print(random_permutation)