# Raw Data

This folder intentionally does not include the full raw file (~51 MB, 398K+ rows) to keep
the repository lightweight. Source and download instructions below.

**Source:** U.S. Bureau of Transportation Statistics — Airline Service Quality Performance
(On-Time Performance) data, "Airline Delay Cause" extract, mirrored on Kaggle as
*Airline On-Time Statistics and Delay Causes*.

**To reproduce locally:**
1. Download `Airline_Delay_Cause.csv` from the BTS/Kaggle source above.
2. Place it in this folder (`data/raw/Airline_Delay_Cause.csv`).
3. Run `scripts/01_clean_data.py` to regenerate everything in `data/cleaned/`.

**Schema (21 columns):** year, month, carrier, carrier_name, airport, airport_name,
arr_flights, arr_del15, carrier_ct, weather_ct, nas_ct, security_ct, late_aircraft_ct,
arr_cancelled, arr_diverted, arr_delay, carrier_delay, weather_delay, nas_delay,
security_delay, late_aircraft_delay.
