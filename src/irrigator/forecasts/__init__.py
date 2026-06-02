"""Block 5 - Forecast Integration.

It extends the water balance forward in time using NWP forecasts to 
anticipate irrigation needs.

It combines short term (ST) and mid term (MT) as follows:

ST -- AROME (0-48h, 1.3km resolution) + ARPEGE (48-96h, 10km) from
      MeteoFrance

MT -- SEAS5 (0-6 months, 32km) ensemble monthly anomalies for temperature and 
      precipitation

Both are accessed through APIs and downscaled/bias corrected based on Quantile
Mapping.      
"""