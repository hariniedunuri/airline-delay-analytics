# Data Dictionary — Airline Delay Cause

| Column | Type | Description |
|---|---|---|
| year | int | Calendar year |
| month | int | Calendar month (1-12) |
| carrier | str | IATA carrier code |
| carrier_name | str | Full airline name |
| airport | str | Airport IATA code |
| airport_name | str | Full airport name, city, state |
| arr_flights | float | Total arrivals scheduled for this carrier/airport/month |
| arr_del15 | float | Arrivals delayed 15+ minutes |
| carrier_ct | float | Number of delays attributed to the carrier |
| weather_ct | float | Number of delays attributed to weather |
| nas_ct | float | Number of delays attributed to the National Airspace System |
| security_ct | float | Number of delays attributed to security |
| late_aircraft_ct | float | Number of delays attributed to a late incoming aircraft |
| arr_cancelled | float | Cancelled arrivals |
| arr_diverted | float | Diverted arrivals |
| arr_delay | float | Total arrival delay, all causes (minutes) |
| carrier_delay | float | Delay minutes attributed to the carrier |
| weather_delay | float | Delay minutes attributed to weather |
| nas_delay | float | Delay minutes attributed to NAS |
| security_delay | float | Delay minutes attributed to security |
| late_aircraft_delay | float | Delay minutes attributed to late incoming aircraft |

## Engineered Fields (added during cleaning)

| Column | Formula |
|---|---|
| effective_arrivals | arr_flights - arr_cancelled - arr_diverted |
| delay_rate | arr_del15 / arr_flights |
| cancellation_rate | arr_cancelled / arr_flights |
| total_delay_minutes | arr_delay (capped at 99th percentile) |
| delay_hours | total_delay_minutes / 60 |
| avg_delay_per_delayed_flight | total_delay_minutes / arr_del15 |
| month_ts | datetime built from year + month |
