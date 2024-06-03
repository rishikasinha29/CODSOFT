calculator_records = []

def add(a, b):
    result = a + b
    calculator_records.append(result)
    print_records()

def subtract(a, b):
    result = a - b
    calculator_records.append(result)
    print_records()

def multiply(a, b):
    result = a * b
    calculator_records.append(result)
    print_records()

def divide(a, b):
    if b == 0:
        print("Error: Division by zero is not allowed.")
        return
    result = a / b
    calculator_records.append(result)
    print_records()

def power(a, b):
    result = a ** b
    calculator_records.append(result)
    print_records()

def print_records():
    print("Calculator Records:", calculator_records)

def main():
    try:
        a = float(input("Enter the first number: "))
        b = float(input("Enter the second number: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    print("Choose an operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Exponentiate")

    try:
        ch = int(input("Choose an operation (1/2/3/4/5): "))
    except ValueError:
        print("Invalid choice. Please enter a number between 1 and 5.")
        return

    if ch == 1:
        add(a, b)
    elif ch == 2:
        subtract(a, b)
    elif ch == 3:
        multiply(a, b)
    elif ch == 4:
        divide(a, b)
    elif ch == 5:
        power(a, b)
    else:
        print("Wrong choice, please recheck")

main()
