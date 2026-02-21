import random
import pandas as pd
import numpy as np

"""

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

    
