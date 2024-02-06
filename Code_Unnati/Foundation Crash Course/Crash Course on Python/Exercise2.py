### Exercise2_Solution:Counting Even Numbers

def count_even_numbers(numbers):
    count = 0
    for num in numbers:
        if num % 2 == 0:
            count += 1
    return count

# Example usage:
numbers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_count = count_even_numbers(numbers_list)
print("Number of even numbers:", even_count)