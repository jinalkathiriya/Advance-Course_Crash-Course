######  Python Accelerated: A Crash Course for Beginners  ######


from platform import python_version
print(python_version())

print("Hello Jupyter")

### Comments in python

# This is a comment
print("Hello World")

### Python Variables

myvar = "Hello"
my_var = "Hello"
_my_var = "Hello"
myVar = "Hello"
print(myvar)
print(my_var)
print(_my_var)
print(myVar)

### Types of variables   

## 1. Local Variables
def hello():
    a = 10          # Here a is local variable
    print(a) 
    return
hello()

## 2. Global Variables  
a = "Welcome to Code Unnati 2.0" # Here a is global variable
def hello():
    a = 20
    print(a)  # Here a is local variable
    return
print(a)

### Python built-in data types

## 1.Python Numbers
#This is integer datatype
a = 10
print(a)
print(type(a))

#Multiple assignment
b,c,d = 11, 12, 13

print(a,b,c,d)

## Operators

## Arithematic operator

# addition, sub, mul, div: +, -, *, / etc
x = 5
y = 3

print(x + y)

## Assignment operator

x = 5
x += 3

print(x)

x = 5

x *= 3

print(x)

## Comaprison operator

x = 5
y = 3

print(x == y)

# returns False because 5 is not equal to 3

x = 5
y = 3

print(x != y)

# returns True because 5 is not equal to 3

## Logical operator

x = 5

print(x > 3 and x < 10)

# returns True because 5 is greater than 3 AND 5 is less than 10

x = 5

print(x > 3 or x < 4)

# returns True because one of the conditions are true (5 is greater than 3, but 5 is not less than 4)

x = 5

print(not(x > 3 and x < 10))

# returns False because not is used to reverse the result

#use input keyword to take input from user

#aks user to enter the value and store it in varialbe x

x = int(input("Enter the value of x : "))

#aks user to enter the value and store it in varialbe y

y = int(input("Enter the value of y : "))

#perform multiplication of x and y and store answer in variable Z

z = x*y

#Print answer

print(z)

## 2.Python Strings

statement = "      Python is a programming language      "
print(statement)

print(type(statement)) #type is used to check the datatype

## Methods in python strings
## Lower Method

statement.lower() #to convert text into lowercase

## Upper Method

statement.upper()#to convert text into uppercase

## Title Method

statement.title()#to convert text into snakewriting

## Strips Methods

statement.lstrip() #to remove the space from left side

statement.rstrip()#to remove the space from right side

statement.strip()#to remove the space from both side

## String Formatting

string = "{} {} {}".format('Python','Programming','Language')
string

name = 'Priyanka'
course = 'Code Unnti Program 2.0'
print(f"Hello, My name is {name} and I registered for {course}")

## String Slicing

str1 = "FACE"

str1[0]

str1[2]

str1[0:2]

str1[::-1]

## 3.Python Lists

#create list

list1 = ["parul", 22, "Bharat", 45.5, "vadodara", 99]
print(list1)

print(type(list1))

list1[0] #access 1st position item from list

list1[3] #access 4th position item from list

## Methods in python lists

## Append Method

fruits = ['apple', 'banana', 'cherry']
fruits.append("orange")
print(fruits)

## Extend Method

fruits = ['apple', 'banana', 'cherry']
cars = ['Ford', 'BMW', 'Volvo']

fruits.extend(cars)
print(fruits)

## Insert Method

fruits = ['apple', 'banana', 'cherry']

fruits.insert(1, "orange") # insert orange at index 1
print(fruits)

## Pop Method

fruits = ['apple', 'banana', 'cherry']

fruits.pop(1) # if no argument pass in pop by default it will remove last item and retun it.
print(fruits)

## Remove Method

fruits = ['apple', 'banana', 'cherry']

fruits.remove("banana")
print(fruits)

## Sort Method

cars = ['Ford', 'BMW', 'Volvo']

cars.sort()  # The words will be arrange in aplphabetical order
print(cars)

## Split Method

# write a program to take input a sentence and split it into word.

text = input("Enter the word: ")  # To take the input from the user "input" keyword is used.

text.split()

## 4.Python Tuples

#create tuples -

tup1 = ("parul", 22, "Bharat", 45.5, "vadodara", 99)

print(tup1)
print(type(tup1))

tup1[4] # To access 5th item in tuple

tup1[2:4]

tup2=("python", 87)

# tup3 = tup1 + tup2
# print(tup3)

# #immutable 

# del tup3[3]    #del is keyword used for deleting the items.
# print(tup3)

## Methods in python tuples

## Count Method

numbers = (1, 3, 7, 8, 7, 5, 4, 6, 8, 5)

x = numbers.count(5)  # 5 is the element in tuple

print(x)

## Index Method

numbers = (1, 3, 7, 8, 7, 5, 4, 6, 8, 5)

x = numbers.index(8) # 8 is the element in tuple

print(x)

## 5.Python Dictionary

cars = {
  "brand": "Mahindra",
  "model": "Thar",
  "year": 2010
}

print(cars)
print(type(cars))

print(cars["model"]) #Accessing Items

## Methods in python Dictionaries

## Accessing Method

#print only key values
print(cars.keys())      # To access all keys

#print only values
print(cars.values())   # To access all values

cars.items()          # To access all keys,values   

## Update Method

cars.update({"year": 2012})
cars

## Adding Items

cars["color"] = "black"  # To add key-value pair 
print(cars)

## Removing Items`

cars.pop("model")
cars

## 6.Python Sets

subjects1 = {'Physics', 'Hindi', 'Chemistry', 'maths', 'Hindi'}
print(subjects1)
print(type(subjects1))

subjects2 = {'History', 'Social Science', 'Biology','Hindi'}
print(subjects2)

subjects1|subjects2  # This pipe symbol used to make union of two sets

subjects1.union(subjects2) # This is another method to make union of two sets

subjects1.intersection(subjects2)  # To find the common iterm in both the sets

### Python Built in Functions in data types

## Len Function

fruits = ['apple', 'banana', 'cherry']
len(fruits)

name = 'Shivaji Maharaj'  #Counts the spaces as well
len(name)

## Sorted Function

theater = {
  "Movie": "Pushpa",
  "Actor": "Allu Arjun",
  "year": 2021
}
sorted(theater)

## Max Function

numbers = [1, 3, 7, 8, 7, 5, 4, 6, 8, 5]
max(numbers)

## Sum Function

numbers = [1, 3, 7, 8, 7, 5, 4, 6, 8, 5]
sum(numbers)

## Dir Function

dir(list)

dir(str)


## Type casting

n = 10  # Integer
print(type(n))

n = float(n)  # Converted to float
print(type(n))

n = 10 # Integer
n = str(n) # Converted to string
print(type(n))

fruits = ['apple', 'banana', 'cherry'] # List
print(type(fruits))

fruits = tuple(fruits) # Converted to tuple
print(type(fruits))


### Python Functions

def add_numbers(num1, num2):
    result = num1 + num2
    return result

# Using our magical recipe "add_numbers"
sum_result = add_numbers(5, 3)
print("The sum is:", sum_result)

### Python Conditional Statements

## Python if statement

number = 10

# check if number is greater than 0
if number > 0:
    print('Number is positive.')

print('The if statement is easy')

## Python if...else Statement

number = 10

if number > 0:
    print('Positive number')

else:
    print('Negative number')

print('This statement is always executed')

## Python if...elif...else Statement

number = int(input("Enter Number  " ))

if number > 0:
    print("Positive number")

elif number == 0:
    print('Zero')
else:
    print('Negative number')

print('This statement is always executed')

## Python Loops

## 1. While Loops

counter = 0

while counter < 3:
    print('Hello world')
    counter = counter + 1

## 2. For Loops

# use of range() to define a range of values- range(start, stop, step)

for i in range(10):
    print(i)

for letter in "Python":
    print(letter)

fruits = ['apple', 'banana', 'cherry']

for fruit in fruits:
    print(fruit)


### Python expression statement

## Pass statement

n = 10

# use pass inside if statement
if n > 10:
    pass

## Del statement

x = ["physics", "chemistry", "maths"]

del x[0]

print(x)

## Break statement

#Python break Statement
i = 1
while i < 9:
  print(i)
  if i == 3:
    break
  i += 1

## Continue statement

#Python continue Statement
i = 0
while i < 10:
  i += 1
  if i == 3:
    continue
  print(i)


### Embark on a Code Adventure: Let's Start Programming!

def calculate_total_bill(items, prices):
    if len(items) != len(prices):
        return "Error: The number of items and prices should be the same."

    total_bill = 0
    for price in prices:
        total_bill += price
    return total_bill

# Example usage:
items_list = ["Apple", "Banana", "Milk", "Bread"]
prices_list = [1.2, 0.6, 2.3, 1.5]
total_bill_amount = calculate_total_bill(items_list, prices_list)
print("Total bill amount:", total_bill_amount)


def calculate_discounted_bill(original_prices, discount_items):
    discounted_bill = 0
    for i in range(len(original_prices)):
        if discount_items[i]:
            discounted_bill += original_prices[i] * 0.9
        else:
            discounted_bill += original_prices[i]
    return discounted_bill

# Example usage:
original_prices_list = [10, 5, 8, 12]
discount_items_list = [True, False, True, False]
discounted_bill_amount = calculate_discounted_bill(original_prices_list, discount_items_list)
print("Discounted bill amount:", discounted_bill_amount)

### Python Lambda Function

x = lambda a: a + 10
print(x(5))

x = lambda a, b : a * b
print(x(5, 6))

### Exception Handling in python

## 1. try-except block

try:
    numerator = int(input("Enter the numerator: "))
    denominator = int(input("Enter the denominator: "))
    result = numerator / denominator
    print("Result:", result)
except ZeroDivisionError:
    print("Error: Cannot divide by zero.")
except ValueError:
    print("Error: Please enter valid integers for numerator and denominator.")

## 2. try-except-else block

try:
    number = int(input("Enter a number: "))
except ValueError:
    print("Error: Please enter a valid integer.")
else:
    square = number ** 2
    print("Square of the number:", square)

## 3. try-except-finally block

try:
    file = open("data.txt", "r")
    content = file.read()
    print(content)
except FileNotFoundError:
    print("Error: File not found.")
finally:
    if 'file' in locals():
        file.close()



