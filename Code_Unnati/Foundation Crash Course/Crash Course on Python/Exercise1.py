### Exercise_1_Solution


# Create an empty list called fruits_list.
fruits_list = []

# Add "apple", "banana", "orange", "grape", and "mango" to the fruits_list.
fruits_list.extend(["apple", "banana", "orange", "grape", "mango"])

# Print the first, last, and second & third fruits in the list using slicing.
print("First fruit:", fruits_list[0])
print("Last fruit:", fruits_list[-1])
print("Second and third fruits:", fruits_list[1:3])

# Print the number of fruits in the list.   
print("Number of fruits:", len(fruits_list))

# Replace the third fruit with "pear" in the list.
fruits_list[2] = "pear"

# Check if "apple" is present in the fruits_list and print the result.
print("\"apple\" is present in the list:", "apple" in fruits_list)

# Remove "banana" and the last fruit from the list.
fruits_list.remove("banana")
fruits_list.pop()

# Create another list additional_fruits with "watermelon" and "kiwi", then concatenate it with fruits_list.
additional_fruits = ["watermelon", "kiwi"]
fruits_list += additional_fruits

# Sort the fruits_list in alphabetical order.
fruits_list.sort()

# Count the number of times "kiwi" appears in the fruits_list.
kiwi_count = fruits_list.count("kiwi")

# Print the final fruits_list and the count of "kiwi".
print("Final fruits_list:", fruits_list)
print("Number of kiwis:", kiwi_count)