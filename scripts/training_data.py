import random
import pandas as pd
import numpy as np

"""
This is an rough example of training data that can
be used to understand noisy data and trying to find
matches between two dataframes with close, but not
identical data. While only one random issue I could
think of, this is a typo error called the transpose 
effect, where two letters are swapped among a word. 
I had one completely clean dataframe and another with
 data simulating swapping the second and third letter
 of each individual's forename.
"""


# the number of people
n = 200

first_names = ["Alyssa", "Bethany", "Caroline", "Daniella", "Evelyn", "Francesca", "Gianna", "Helena", "Isabeau", "Jenna"]
last_names = ["Zambrano", "Young", "Ximenez", "Wallace", "Verona", "Underwood", "Tacconi", "Stigliano", "Rigatoni", "Queen"]
street_names = ["Palm Circle", "Latte Lane", "Shadow Street", "Daytona Drive", "Playa Avenue", "Lakefield Lane", "Dove Street", "Maple Boulevard", "Stormy Circle", "Marina Street"]
cities = ["Chicago", "New York City", "Las Vegas", "Boca Raton", "Rome"]
states = ["HI", "MA", "VA", "CA", "PA"]

# randomize a phone number
def random_phone():
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

# randomize a house number to fill in to address
def random_address():
    return f"{random.randint(50,9999)} {random.choice(street_names)}, {random.choice(cities), {random.choice(states)}}"

# generate a clean phonebook
clean_data = []

for entry in range(n):
    clean_data.append({"clean_id" : entry,
        "forename" : random.choice(first_names),
        "surname" : random.choice(last_names),
        "address" : random_address(),
        "phone_number" : random_phone()})
    
clean_df = pd.DataFrame(clean_data)

# create phonebook B ( noisy )

def swap_third_fourth(name):
    chars = list(name)
    chars[2], chars[3], = chars[3], chars[2]
    
    return "".join(chars)

# assemble noisy dataframe with the function appleid
noisy_df = clean_df.copy()

noisy_df["forename"] = noisy_df["forename"].apply(swap_third_fourth)

clean_df.to_csv("clean_phonebook", index = False)
noisy_df.to_csv("noisy_phonebook", index = False)