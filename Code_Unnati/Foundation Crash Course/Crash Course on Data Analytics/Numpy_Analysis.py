   #### Data Analysis Using NumPy Library  ####

### Using NumPy

import numpy as np

### NumPy Datatypes

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr.dtype)

### Numpy Arrays

## Creating NumPy Arrays

import numpy as np

arr = np.array((1, 2, 3, 4, 5))

print(arr)

## 0-D Arrays

import numpy as np

arr = np.array(42)

print(arr)

## 1-D Arrays

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)

## 2-D Arrays

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)

## 3-D arrays

import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr)

## Check Number of Dimensions?

import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

### I/O with NumPy

## numpy.save()

import numpy as np 
a = np.array([1,2,3,4,5]) 
np.save('outfile',a)

import numpy as np 
b = np.load('outfile.npy') 
print(b) 

## savetxt()

import numpy as np 

a = np.array([1,2,3,4,5]) 
np.savetxt('out.txt',a) 
b = np.loadtxt('out.txt') 
print(b) 

### Indexing

## Indexing Using Index Arrays

import numpy as np
arr=np.arange(1,10,2) 
print("Elements of array: ",arr)
arr1=arr[np.array([4,0,2,-1,-2])]
print("Indexed Elements of array arr: ",arr1)

## Indexing in 1 dimension

import numpy as np 
arr1=np.arange(4)
print("Array arr11:",arr1)
print("Element at index 0 of arri is: ", arr1[0])
print("Element at index 1 of arr1 is: ", arr1[1])

## Indexing in 2 Dimensions

import numpy as np
arr=np.arange(12)
arr1=arr.reshape(3,4)
print("Array arr1:\n",arr1)
print("Element at eth row and eth column of arr1 is:",arr1[0,0]) 
print("Element at 1st row and 2nd column of arr1 is:", arr1[1,2])

## Indexing in 3 Dimensions

import numpy as np 
arr=np.arange(12) 
arr1=arr.reshape(2,2,3) 
print("Array arr1:\n", arr1) 
print("Element:", arr1[1,0,2])

###  Slicing an Array

## Slicing 1D NumPy Arrays

import numpy as np 
arr = np.arange(6) 
print("array arr:",arr)
print("sliced element of array: ", arr[1:5])

## Slicing a 2D Array

import numpy as np
arr=np.arange(12)
arr1=arr.reshape(3,4)
print("Array arr1: \n",arr1)
print("\n")
print("elements of 1st row and 1st column upto last column \n", arr1[1:,1:4])

### Broadcasting

import numpy as np 
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  
   
print('First array:') 
print(a) 
print('\n')  
   
print('Second array:') 
print(b) 
print('\n')  
   
print('First Array + Second Array' )
print( a + b)

### Structured arrays

#Python program to demonstrate
# Structured array
import numpy as np
a = np.array([('Sana', 2, 21.0), ('Mansi', 7, 29.0)], 
             dtype=[('name', (np. str_, 10)), ('age', np.int32), ('weight', np.float64)])
print(a)

### Statistical Functions

import numpy as np
arr = np.arange(0,10)

arr + arr

arr * arr

arr - arr

# Warning on division by zero, but not an error!
# Just replaced with nan
arr/arr

# Also warning, but not an error instead infinity
1/arr

arr**3








