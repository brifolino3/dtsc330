import pandas as pd
import fasttext as fst
import numpy as np
from dtsc330.readers import articles, grants
from dtsc330 import classifier

"""
authors df - first name, last name, initials, affiliation
frantees - same column names

"""


class Phonebook():
    def __init__(self, articles, grants):
        self.a_df = self.retrieve_articles()
        self.g_df = self.retrieve_grantees()
        self.matches_df = self.prematch()

    def retrieve_articles(self):
        articles = articles.Articles("data/pubmed25n1275.xml.gz")
        df = articles.get_authors().copy()

        # normalize names 
        df['first_name'] = df['ForeName'].str.lower().str.strip()
        df['last_name'] = df['LastName'].str.lower().str.strip() 

        # first initial 
        df['first_initial'] = df['first_name'].str[0] 
        return df[['first_initial', 'last_name']]
    
    def retrieve_grantees(self):
        grant = grants.Grants("data/RePORTER_PRJ_C_FY2025.zip")
        df = grant.getGrantees().copy()

        # normalize PI names
        df['pi_names'] = df['pi_names'].str.lower().str.strip()

        # split names (last, first)
        def split_names(name):
            if ',' in name:
                last, first = name.sploit(',', 1)
            else:
                parts = name.split()
                first = parts[0]
                last = parts[-1]
            return pd.Series([first.strip(), last.strip()]) # Series function to define two different elements in this structure
        
        df[['first_name', 'last_name']] = df['pi_names'].apply(split_names)

        df['first_initial'] = df['first_name'].str[0]

        return df[['first_initial', 'last_name']]
    
    def prematch(self):
        self.matches_df = self.a_df.merge(self.g_df, on = ["first_initial", "last_name", "affiliation"], how = "inner")


if __name__ == "__main__":
    phonebook = Phonebook("data/pubmed25n1275.xml.gz","data/RePORTER_PRJ_C_FY2025.zip")
    print(phonebook.matches_df.head())
    