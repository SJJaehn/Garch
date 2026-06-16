"""
Clean the raw TRBC Datastream export.

The Datastream export is German-locale: ";" separator, literal "NULL" for
missing values, and the first column holds Excel serial date numbers. This turns
it into the tidy ``TRBC_Business_Sectors_clean.csv`` the loaders read.
"""
import pandas as pd

from garch import config

INPUT  = f"{config.EMPIRICAL_DIR}/TRBC_Business_Sectors.csv"
OUTPUT = f"{config.EMPIRICAL_DIR}/TRBC_Business_Sectors_clean.csv"


def clean_trbc(input_path: str = INPUT, output_path: str = OUTPUT) -> None:
    """Read the raw export, clean it, and write the tidy CSV."""
    df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig", na_values=["NULL"], dtype=str)

    # 1) The first column is the date (its header is junk like "Paused at ..."). Rename it.
    df.columns = ["Date", *df.columns[1:]]

    # 2) Each instrument is exported several times. pandas suffixes the repeats with
    #    ".1", ".2", ...; drop those so only the first occurrence of each header stays.
    df = df.loc[:, ~df.columns.str.contains(r"\.\d+$", regex=True)]

    # 3) Drop columns that are at least 95% empty (were "NULL").
    keep = df.isna().mean() < 0.95
    keep["Date"] = True  # never drop the date column
    df = df.loc[:, keep]

    # 4) Reformat the date: Excel serial number -> real date (Excel epoch is 1899-12-30).
    df["Date"] = pd.to_datetime(df["Date"].astype(float), unit="D", origin="1899-12-30")

    # 5) Order chronologically (oldest first).
    df = df.sort_values("Date").reset_index(drop=True)

    # Write back with the same ";" separator so the German decimal commas stay intact.
    df.to_csv(output_path, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"Cleaned {df.shape[0]} rows x {df.shape[1]} columns -> {output_path}")


if __name__ == "__main__":
    clean_trbc()
