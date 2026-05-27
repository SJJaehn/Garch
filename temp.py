import lseg.data as ld
import pandas as pd

# -------------------------------------------------------------------
# Open LSEG session
# -------------------------------------------------------------------
ld.open_session()

# -------------------------------------------------------------------
# TRBC Business Sector RIC -> Sector Name mapping
# Split into batches to respect per-request limits
# -------------------------------------------------------------------
ric_names = {
    ".TRXFLDUSPA1": "Academic & Educational Services",
    ".TRXFLDUSPC1": "Telecommunications Services",
    ".TRXFLDUSPE1": "Energy - Fossil Fuels",
    ".TRXFLDUSPE2": "Renewable Energy",
    ".TRXFLDUSPF1": "Banking & Investment Services",
    ".TRXFLDUSPF3": "Insurance",
    ".TRXFLDUSPF4": "Real Estate",
    ".TRXFLDUSPF6": "Investment Holding Companies",
    ".TRXFLDUSPH1": "Healthcare Services & Equipment",
    ".TRXFLDUSPH2": "Pharmaceuticals & Medical Research",
    ".TRXFLDUSPI1": "Industrial Goods",
    ".TRXFLDUSPI2": "Industrial & Commercial Services",
    ".TRXFLDUSPI3": "Consumer Goods Conglomerates",
    ".TRXFLDUSPI4": "Transportation",
    ".TRXFLDUSPM1": "Chemicals",
    ".TRXFLDUSPM2": "Mineral Resources",
    ".TRXFLDUSPM3": "Applied Resources",
    ".TRXFLDUSPN1": "Food & Beverages",
    ".TRXFLDUSPN2": "Personal & Household Products & Services",
    ".TRXFLDUSPN3": "Food & Drug Retailing",
    ".TRXFLDUSPT1": "Technology Equipment",
    ".TRXFLDUSPT2": "Software & IT Services",
    ".TRXFLDUSPT3": "Financial Technology (Fintech) & Infrastructure",
    ".TRXFLDUSPU1": "Utilities",
    ".TRXFLDUSPY1": "Automobiles & Auto Parts",
    ".TRXFLDUSPY2": "Cyclical Consumer Products",
    ".TRXFLDUSPY3": "Cyclical Consumer Services",
    ".TRXFLDUSPY4": "Retailers",
}

BATCH_SIZE = 5
START_DATE = "2006-06-01"
END_DATE   = "2026-05-21"

# -------------------------------------------------------------------
# Download in batches and collect into a single wide DataFrame
# -------------------------------------------------------------------
ric_list = list(ric_names.keys())
batches  = [ric_list[i:i + BATCH_SIZE] for i in range(0, len(ric_list), BATCH_SIZE)]

results = pd.DataFrame()

for batch_num, batch in enumerate(batches, 1):
    print(f"\nBatch {batch_num}/{len(batches)}: {batch}")
    for ric in batch:
        sector = ric_names[ric]
        try:
            df = ld.get_history(
                universe=ric,
                fields=["TR.CLOSEPRICE"],
                interval="1D",
                start=START_DATE,
                end=END_DATE,
            )

            if df is not None and not df.empty:
                df = df.sort_index()
                if "Close Price" in df.columns:
                    df = df.rename(columns={"Close Price": sector})
                    results = pd.concat([results, df[[sector]]], axis=1)
                    print(f"  ✓ {sector} ({len(df)} rows)")
                else:
                    print(f"  ✗ {sector} — unexpected columns: {df.columns.tolist()}")
            else:
                print(f"  ✗ {sector} — empty response")

        except Exception as e:
            print(f"  ✗ {sector} — error: {e}")

# -------------------------------------------------------------------
# Save combined CSV (Date as index, one column per business sector)
# -------------------------------------------------------------------
results.index.name = "Date"
results.sort_index(inplace=True)
output_file = "TRBC_Business_Sectors_Price_History.csv"
results.to_csv(output_file)
print(f"\nSaved {results.shape[1]} sectors × {results.shape[0]} dates → {output_file}")

# -------------------------------------------------------------------
# Close session
# -------------------------------------------------------------------
ld.close_session()