import random 

word_list = ["kiwi", "pomegranite", "mango", "nectarine", "apple"]
word = random.choice(word_list)

# %%
guess = input("please enter a letter")

if len(guess) == 1 and guess in "abcdefghijklmnopqrstuvwxyz":
    print("Good guess!")

else: 
    print("Oops! That is not a valid input.")
# %%