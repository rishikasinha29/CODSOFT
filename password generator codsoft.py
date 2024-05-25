import random
import string

def generate_password(length):
    
    characters = string.ascii_letters + string.digits + string.punctuation   #defining the list of characters to be included 
   
    password = ''.join(random.choice(characters) for i in range(length))     #generating a random password
    return password

def check_password_strength(password):
    # Initialize criteria flags
    has_upper = False
    has_lower = False
    has_digit = False
    has_special = False

    
    for char in password:         # Check each character in the password        
        if char.isupper():
            has_upper = True
        elif char.islower():
            has_lower = True
        elif char.isdigit():
            has_digit = True
        elif char in string.punctuation:
            has_special = True
    
    # Calculate the score based on the criteria
    score = sum([has_upper, has_lower, has_digit, has_special])
    
    # Determine the strength
    if len(password) >= 12 and score == 4:
        return "Strong"
    elif len(password) >= 8 and score >= 3:
        return "Moderate"
    else:
        return "Weak"

def main():
    
    while True:
        try:
            length = int(input("Enter the desired length for the password: "))  #input the length of password
            if length <= 0:
                print("invalid length")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    # Generate and display the password
    password = generate_password(length)
    print("password generated: ",password)


    #checking the strength of password
    strength = check_password_strength(password)
    print("Password Strength: ",strength)

main()
