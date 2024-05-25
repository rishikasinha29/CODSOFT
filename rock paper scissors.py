import random

def computer_choice():
    options=["rock","paper","scissors"]
    return random.choice(options)

def determine_winners(user,computer):
    if user==computer:
        return("tie")
    elif (user == "rock" and computer == "scissors") or \
         (user== "scissors" and computer== "paper") or \
         (user == "paper" and computer == "rock"):
        return "user"
    else:
        return "computer"


def display_results(user,computer,winner):
    print("your choose : ",user)
    print("computer choose : ",computer)
    if winner=="tie":
        print("It is a tie")
    elif winner=="user":
        print("Congrats! U win!!")
    else:
        print("Sorry! You lost the game ")


def play_again():
    choice=input("Do you want to play again(yes/no): ").lower()
    return choice=="yes"

def main():
    user_score=0
    computer_score=0
    rounds=0

    print("Rock !!!  Paper !!!  Scissors")

    while True:
        user = input("\nEnter your choice (rock, paper, or scissors): ").lower()
        if user not in ["rock", "paper", "scissors"]:
            print("Invalid choice. Please try again.")
            continue
        
        computer = computer_choice()
        winner = determine_winners(user, computer)
        
        display_results(user, computer, winner)
        
        if winner == "user":
            user_score += 1
        elif winner == "computer":
            computer_score += 1
        
        rounds += 1
        print(f"\nScores after {rounds} rounds:")
        print(f"User: {user_score}")
        print(f"Computer: {computer_score}")
        
        if not play_again():
            break

    print("\nThank you for playing! Final scores:")
    print(f"User: {user_score}")
    print(f"Computer: {computer_score}")

main()

