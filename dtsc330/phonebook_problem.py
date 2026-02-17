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
    def __init__(self, path: str):
        self.a_df = self.retrieve_articles()
        self.g_df = self.retrieve_grantees()
        self.matches_df = self.prematch()

    def retrieve_articles(self):
        articles = articles.Articles()
        return articles.get_authors()
    
    def retrieve_grantees(self):
        grant = grants.Grants()
        return grant.getGrantees()
    
    def prematch(self):
        self.matches_df = self.a_df.merge(self.g_df, on = ["first_name", "last_name", "affiliation"], how = "inner")





def main():
    phonebook = Phonebook()


if __name__ == "__main__":
    main()
    