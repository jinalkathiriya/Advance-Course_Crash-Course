   #### Data Visualization Using Matplotlib Library  ####


### Importing the required module

import matplotlib.pyplot as plt
# x axis values
x = [1,2,3]
# corresponding y axis values
y = [2,4,1]
# plotting the points
plt.plot(x, y)
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('My first graph!')
# function to show the plot
plt.show()

import numpy as np
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# figures in matplotlib using figure()
plt.figure(figsize=(10, 8))
plt.plot(x, y)
plt.show()

### subplots()¶

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 1, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 1, 2)
plt.plot(x,y)

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, linestyle = 'dotted')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, color = 'r')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
plt.show()

### Line plot

# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7]

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot
plt.plot(x, y)

# function to show the plot
plt.show()

### Bar Plot

# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7]

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot
plt.bar(x, y)

# function to show the plot
plt.show()

### Scatter Plot

# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7, 3, 8, 10, 1]

# Y-axis values
y = [10, 5, 8, 4, 2, 1, 6, 3, 9]

# Function to plot scatter
plt.scatter(x, y)

# function to show the plot
plt.show()

## Legends, labels and titles

# Ticks are the markers denoting data points on axes.
# importing libraries
import matplotlib.pyplot as plt
import numpy as np

# values of x and y axes
x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
y = [1, 4, 3, 2, 7, 6, 9, 8, 10, 5]

plt.plot(x, y, 'g')
plt.xlabel('x')
plt.ylabel('y')
# here we set the size for ticks, rotation and color value

plt.tick_params(axis="x", labelsize=18, labelrotation=-60, labelcolor="blue")
plt.tick_params(axis="y", labelsize=12, labelrotation=20, labelcolor="black")

plt.show()

## Adding title and Labelling the Axes in the graph

# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5, 2, 9, 4, 7]

# Y-axis values
y = [10, 5, 8, 4, 2]

# Function to plot
plt.bar(x, y)

# Adding Title
plt.title("Bar graph ")

# Labeling the axes
plt.xlabel("Time (hr)")
plt.ylabel("Position (Km)")

# function to show the plot
plt.show()

## Adding Legend in the graph

# importing modules
import numpy as np
import matplotlib.pyplot as plt

# Y-axis values
y1 = [2, 3, 4.5]

# Y-axis values
y2 = [1, 1.5, 5]

# Function to plot
plt.plot(y1)
plt.plot(y2)

# Function add a legend
plt.legend(["blue", "green"], loc ="lower right")

# function to show the plot
plt.show()

#subplots
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)

plt.show()

### Histogram

import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(170, 10, 250)
#The hist() function will read the array and produce a histogram:
#set the bin value 30
plt.hist(data);

plt.show() 
plt.hist(data, bins=30);

plt.show() 

#import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

#draw plot as fig
fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
#text annotation to the plot where it indicate maximum value of the curv
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
#text annotation to the plot where it indicate minimum value of the curv
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));


# angleA : starting angle of the path

# angleB : ending angle of the path

# armA : length of the starting arm

# armB : length of the ending arm

# rad : rounding radius of the edges

# connect(posA, posB)

### 3D-Plot

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()    #USE TO CREATE A NEW FIG
ax = plt.axes(projection ='3d')

# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# defining all 3 axes
z = np.linspace(0, 1, 1000)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)

# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot')
plt.show()

### Pie Charts

# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = ['AUDI', 'BMW', 'FORD',
		'TESLA', 'JAGUAR', 'MERCEDES']

data = [23, 17, 35, 29, 12, 41]

# Creating plot
fig = plt.figure(figsize =(10, 7))
plt.pie(data, labels = cars)

# show plot
plt.show()

#import numpy 
import numpy as np 
from matplotlib import pyplot as plt 

x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()

import numpy as np 
import matplotlib.pyplot as plt  

# Compute the x and y coordinates for points on a sine curve 
x = np.arange(0, 3 * np.pi, 0.1) 
print(x)
y = np.sin(x) 
plt.title("sine wave form") 

# Plot the points using matplotlib 
plt.plot(x, y) 
plt.show()

### Logarithmic Axes in Matplotlib

import matplotlib.pyplot as plt

# exponential function y = 10^x
data = [10**i for i in range(5)]

plt.plot(data)

import matplotlib.pyplot as plt

# exponential function y = 10^x
data = [10**i for i in range(5)]

# convert y-axis to Logarithmic scale
plt.yscale("log")

plt.plot(data)

import matplotlib.pyplot as plt

# exponential function x = 10^y
datax = [ 10**i for i in range(5)]
datay = [ i for i in range(5)]

#convert x-axis to Logarithmic scale
plt.xscale("log")

plt.plot(datax,datay)




