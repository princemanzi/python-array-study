# import numpy as np
# from sklearn.linear_model import LinearRegression
# import time

# # Test NumPy
# A = np.array([[1, 2], [3, 4]])
# print("NumPy Matrix A:\n", A)

# # Test scikit-learn
# X = np.array([[1], [2], [3], [4]])
# y = np.array([2, 4, 6, 8])
# model = LinearRegression()
# model.fit(X, y)
# print("Scikit-learn Linear Regression Test: Prediction for 5 →", model.predict([[5]])[0])

# # Test the time module
# start_time = time.time()
# time.sleep(1)
# end_time = time.time()
# print(f"Time module test: Slept for {end_time - start_time:.2f} seconds")
 

# print("Hello , world")
 
# x = [-1.2, 0.4, 4, 2.5, 2.4]
# len(x)

import numpy as np
 
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


x = np.array([2.4, 3.1, 3.6, -7.2])
y = x.copy
x[2] = 20.1
