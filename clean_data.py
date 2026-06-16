"""
Clean the raw TRBC Datastream export into the tidy CSV the loaders read.

Expects DATA/Empirical/TRBC_Business_Sectors.csv (the raw Datastream export) and
writes DATA/Empirical/TRBC_Business_Sectors_clean.csv.

    python clean_data.py
"""
from garch.data.cleaning import clean_trbc


if __name__ == "__main__":
    clean_trbc()
