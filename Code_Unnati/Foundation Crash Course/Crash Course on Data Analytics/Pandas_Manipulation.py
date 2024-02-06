   #### Data Manipulation Using Pandas Library  ####

## pip install pandas
import pandas as pd

### Introducing Pandas Objects​

### Creating a Series

import pandas as pd
import numpy as np

# Creating empty series. 
ser = pd.Series()
print(ser)
# simple array
data = np.array(['g', 'e', 'e', 'k', 's'])
ser = pd.Series(data) 
print(ser)

## Creating a series from Lists:

import pandas as pd
# a simple list
list = ['g', 'e', 'e', 'k', 's']
# create series form a list 
ser = pd.Series(list) 
print(ser)

##3 Pandas Index

## Creating index

# importing pandas package 
import pandas as pd 
data = pd.read_csv("airlines.csv")
data

### Pandas DataFrame

## Creating a Pandas DataFrame

#import pandas as pd import pandas as pd
# list of strings
lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks']
# Calling DataFrame constructor on list 
df = pd.DataFrame(lst)
print(df)

# Python code demonstrate creating
# DataFrame from dict narray / lists #By default addresses.
import pandas as pd
# intialise data of lists.
data = { 'Name': ['Tom', 'nick', 'krish', 'jack'], 
        'Age': [20, 21, 19, 18]}
# Create DataFrame
df = pd.DataFrame(data)
# Print the output.
print(df)

## Reindexing

import pandas as pd
# Create dataframe
info = pd.DataFrame({"P":[4, 7, 1, 8, 9], 
                     "Q":[6, 8, 10, 15, 11], 
                     "R":[17, 13, 12, 16, 14], 
                     "S":[15, 19, 7, 21, 9]}, 
                    index =["Parker", "William", "Smith", "Terry", "Phill"])
#Print dataframe
info

# reindexing with new index values 
info.reindex(["A", "B", "C", "D", "E"])

# reindexing with new index values 
info.reindex(["A", "B", "C", "D", "E"])

# filling the missing values by 100 
info.reindex (["A", "B", "C", "D", "E"], fill_value =100)

### Pandas Sort

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn (10,2), index= [1,4,6,2,3,5,9,8,8,7], columns = ['co12', 'col1'])
sorted_df = unsorted_df.sort_index() 
print (sorted_df)

## Order of Sorting

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn (10,2), index= [1,4,6,2,3,5,9,8,8,7], columns = ['col2', 'col1'])
sorted_df = unsorted_df.sort_index(ascending=False)
print (sorted_df)

## Sort the Columns

import pandas as pd
import numpy as np

unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[1,4,6,2,3,5,9,8,8,7], columns = ['col2', 'col1'])

sorted_df=unsorted_df.sort_index(axis=1)

print (sorted_df)

### By Value

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame({'col1': [2,1,1,1], 'col2': [1,3,2,4]}) 
sorted_df = unsorted_df.sort_values (by='col1')
print (sorted_df)

### Working with Text Data

import pandas as pd
import numpy as np
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234', 'SteveSmith'])
print (s. str.lower())

import pandas as pd
import numpy as np
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234', 'SteveSmith'])
print (s. str.upper())

### Statistical Functions

## Pandas sum() method

import pandas as pd
# Dataset
data = {
'Maths' :[90, 85, 98, 80, 55, 78],
'Science': [92, 87, 59, 64, 87, 96], 'English': [95, 94, 84, 75, 67, 65]
}
# DataFrame
df = pd.DataFrame(data)
# Display the DataFrame 
print("DataFrame = \n",df)
# Display the Sum of Marks in each column 
print("\nSum = \n",df.sum())

## Pandas count() method

import pandas as pd
# Dataset
data = {
'Maths': [90, 85, 98, None, 55, 78],
'Science': [92, 87, 59, None, None, 96],
'English': [95, None, 84, 75, 67, None]
}
# DataFrame
df = pd.DataFrame(data)
# Display the DataFrame 
print("DataFrame = \n", df)
# Display the Count of non-empty values in each column 
print("\nCount of non-empty values = \n", df.count())

## Pandas max() method

import pandas as pd
# Dataset
data = { 'Maths': [90, 85, 98, 80, 55, 78],
'Science': [92, 87, 59, 64, 87, 96],
'English': [95, 94, 84, 75, 67, 65]
}
#DataFrame
df = pd.DataFrame(data)
# Display the DataFrame 
print("DataFrame = \n",df)
# Display the Maximum of Marks in each column 
print("\nMaximum Marks = \n", df.max())

## Pandas min() method

import pandas as pd
# Dataset
data = {
'Maths' : [90, 85, 98, 80, 55, 78], 'Science': [92, 87, 59, 64, 87, 96], 'English': [95, 94, 84, 75, 67, 65]
}
# DataFrame
df = pd.DataFrame(data)
# Display the DataFrame
print("DataFrame = \n", df)
# Display the Minimum of Marks in each column 
print("\nMinimum Marks = \n", df.min())

## Pandas median() method

import pandas as pd
# Dataset
data = {
'Maths': [90, 85, 98, 80, 55, 78],
'Science': [92, 87, 59, 64, 87, 96], 'English': [95, 94, 84, 75, 67, 65]
}
# DataFrame
df = pd.DataFrame(data)
# Display the DataFrame
print("DataFrame = \n", df)
# Display the Median of Marks in each column 
print("\nMedian = \n",df.median())

### Indexing and Selecting Data

## Indexing a Data frame using indexing operator [] :

# importing pandas package
import pandas as pd

# making data frame from csv file
data = pd.read_csv("nba.csv", index_col ="Name")

# retrieving columns by indexing operator
first = data["Age"]

print(first)

## Indexing a DataFrame using .loc[ ] :

# importing pandas package
import pandas as pd

# making data frame from csv file
data = pd.read_csv("nba.csv", index_col ="Name")

# retrieving row by loc method
first = data.loc["Avery Bradley"]
second = data.loc["R.J. Hunter"]

print(first, "\n\n\n", second)

## Indexing a DataFrame using .iloc[ ] :

import pandas as pd

# making data frame from csv file
data = pd.read_csv("nba.csv", index_col ="Name")

# retrieving rows by iloc method
row2 = data.iloc[3]

print(row2)


