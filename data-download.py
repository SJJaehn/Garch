import lseg.data as ld
import pandas as pd
import time

# -------------------------------------------------------------------
# Open LSEG session
# -------------------------------------------------------------------

ld.open_session()

# -------------------------------------------------------------------
# 1. Get current S&P 500 constituents
# -------------------------------------------------------------------

print("Fetching S&P 500 constituents...")

members = ld.get_data(
    universe=".SPX",
    fields=["TR.IndexConstituentRIC"]
)

print(members.columns)

rics = (
    members["Constituent RIC"]
    .dropna()
    .unique()
    .tolist()
)

print(f"Found {len(rics)} constituents")

# -------------------------------------------------------------------
# 2. Define date chunks
# -------------------------------------------------------------------

# Keep each request safely under LSEG historical row limits
PERIODS = [
    ("2006-06-01", "2015-05-31"),
    ("2015-06-01", "2026-05-21"),
]

# -------------------------------------------------------------------
# 3. Download adjusted close history
# -------------------------------------------------------------------

prices = pd.DataFrame()
failed = []

for i, ric in enumerate(rics):

    print(f"[{i+1}/{len(rics)}] {ric}")

    ric_frames = []

    for start, end in PERIODS:

        try:

            
            df = ld.get_history(
                universe=ric,
                fields=["TR.CLOSEPRICE"],
                interval="1D",
                start=start,
                end=end,
                parameters={"Adjusted": 1}
            )

            if df is not None and not df.empty:
                series = df[["Close Price"]].rename(columns={"Close Price": ric})
                ric_frames.append(series)

            else:
                print(f"  No data: {start} → {end}")

        except Exception as e:

            print(f"  Failed {start} → {end}: {e}")

            failed.append({
                "RIC": ric,
                "Start": start,
                "End": end,
                "Error": str(e)
            })

        # Stay under API rate limits
        time.sleep(0.25)

    # ----------------------------------------------------------------
    # Merge chunks for this ticker
    # ----------------------------------------------------------------

    if ric_frames:

        ric_df = pd.concat(ric_frames)

        # Remove duplicate dates if chunk boundaries overlap
        ric_df = ric_df[~ric_df.index.duplicated(keep="first")]

        # Join into master dataframe
        if prices.empty:
            prices = ric_df
        else:
            prices = prices.join(ric_df, how="outer")

    output_file = "sp500_adjusted_close.csv"

    prices.to_csv(output_file)

# -------------------------------------------------------------------
# 4. Final cleanup
# -------------------------------------------------------------------

prices = prices.sort_index()
prices.index.name = "Date"

# -------------------------------------------------------------------
# 5. Save CSV
# -------------------------------------------------------------------

output_file = "sp500_adjusted_close.csv"

prices.to_csv(output_file)

print("\nDone.")
print(f"Saved: {output_file}")
print(f"Shape: {prices.shape[0]} rows × {prices.shape[1]} columns")

# -------------------------------------------------------------------
# 6. Report failures
# -------------------------------------------------------------------

if failed:

    failed_df = pd.DataFrame(failed)

    failed_file = "failed_requests.csv"

    failed_df.to_csv(failed_file, index=False)

    print(f"\n{len(failed)} failed requests")
    print(f"Saved failure log: {failed_file}")

# -------------------------------------------------------------------
# Close session
# -------------------------------------------------------------------

ld.close_session()