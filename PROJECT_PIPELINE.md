# IrriSol — Irrigation Decision Support for Maize in Dordogne

## Architecture Overview

```
[Block 0: Data Ingestion]
        │
        ▼
[Block 1: Static Layers]──────────────────┐
  Soil (EU-SoilHydroGrids)                │
  Topography (IGN RGE ALTI)               │
  Geology (BRGM InfoTerre)                │
        │                                  │
        ▼                                  │
[Block 2: Atmospheric Forcing]             │
  ERA5-Land reanalysis                     │
  COMEPHORE bias correction                │
        │                                  │
        ▼                                  │
[Block 3: Reference ET + Soil Water Balance]
  Penman-Monteith ET0                      │
  FAO-56 bucket model                      │
        │                                  │
        ▼                                  │
[Block 4: Crop Module]                     │
  GDD accumulation                         │
  Kc curve → ETc                           │
  Stress feedback                          │
        │                                  │
        ▼                                  │
[Block 5: Forecast Integration]
  AROME/ARPEGE (short-term)
  SEAS5 (seasonal)
        │
        ▼
[Block 6: Irrigation Decision Layer]
  Rule-based triggers
  Forward water balance under forecast
  Recommendation output
```

---

## Block 0 — Data Ingestion & Storage

**Goal:** Fetch, store, and standardize all external datasets into a common spatiotemporal grid.

### Data sources

| Dataset | Variables | Resolution | Access |
|---|---|---|---|
| ERA5-Land | 2m temp, dewpoint, u/v wind, surface pressure, total precip, surface solar rad | ~9 km, hourly | CDS API (`cdsapi`) |
| COMEPHORE | Radar-gauge merged precipitation | 1 km, hourly | Météo-France Public API |
| EU-SoilHydroGrids | θ_FC, θ_WP, K_sat, available water capacity | 250 m, static | ESDAC (registration, free download) |
| IGN RGE ALTI | Digital elevation model | 5 m, static | IGN Géoservices (free) |
| BRGM InfoTerre | Bedrock geology, borehole logs | varies, static | InfoTerre WFS/WMS (free) |
| Sentinel-2 (THEIA) | NDVI, LAI surface reflectance | 10 m, ~5-day revisit | theia.cnes.fr (free) |
| AROME | T, precip, wind, humidity forecasts | 1.3 km, up to 48h | Météo-France Public API |
| ARPEGE | Same, coarser | ~10 km, up to 4 days | Météo-France Public API |
| SEAS5 | Monthly T and precip anomalies (ensemble) | ~36 km, 7 months ahead | CDS API |

### Implementation

```
src/
  ingestion/
    __init__.py
    cds_client.py          # ERA5-Land + SEAS5 download via cdsapi
    meteofrance_client.py  # COMEPHORE + AROME/ARPEGE via public API
    esdac_loader.py        # EU-SoilHydroGrids raster loading
    ign_loader.py          # DEM tiles from Géoservices
    brgm_loader.py         # Geology WFS queries
    theia_client.py        # Sentinel-2 L2A/L3A download
    grid.py                # Common grid definition for Dordogne (EPSG:2154)
```

**Key decisions:**
- Work in Lambert-93 (EPSG:2154) throughout — all French public data is natively in this CRS.
- Define a target grid at 250 m to match EU-SoilHydroGrids as the finest static layer.
- Store intermediate data as NetCDF (xarray) or GeoTIFF (rasterio). Use `xarray` + `dask` for lazy loading if memory becomes an issue.

**Python packages:** `cdsapi`, `requests`, `rasterio`, `xarray`, `geopandas`, `owslib` (for WFS/WMS), `pystac-client` (for THEIA STAC catalog)

---

## Block 1 — Static Layers (Soil, Topography, Geology)

**Goal:** Build a parcel-level parameter set: θ_FC(z), θ_WP(z), K_sat(z), rooting depth constraint, slope, aspect.

### Steps

1. **Load EU-SoilHydroGrids** for the Dordogne bounding box.
   - Extract θ_FC, θ_WP, K_sat at available depth intervals.
   - These are pre-computed from SoilGrids texture via pedotransfer functions (Tóth et al., 2015).
   - Clip to Dordogne boundary (admin boundary from IGN ADMIN EXPRESS, free).

2. **Load DEM and derive terrain variables.**
   - Slope and aspect from RGE ALTI at 5 m, aggregate statistics to 250 m grid.
   - Compute terrain-based radiation correction factors (important for ET0 spatial variation).

3. **Load BRGM geology and classify.**
   - Map geological units to broad drainage classes: alluvium (high storage, good drainage), limestone plateau (shallow soil, karstic drainage), clay-limestone (low permeability).
   - Use as a stratification layer: within each geology class, EU-SoilHydroGrids values should be more homogeneous.

4. **Farmer override mechanism.**
   - If a farmer provides a soil analysis (texture, organic matter) or probe-derived θ_FC, override the grid value for that parcel.
   - Store overrides in a simple parcel-level config (JSON or YAML per parcel).

### Output

A `SoilProfile` object per parcel or grid cell:
```python
@dataclass
class SoilProfile:
    theta_fc: list[float]     # field capacity by depth layer [m³/m³]
    theta_wp: list[float]     # wilting point by depth layer
    k_sat: list[float]        # saturated hydraulic conductivity [mm/day]
    z_layers: list[float]     # depth boundaries [m], e.g. [0.3, 0.6, 1.0]
    total_awc: float          # total available water capacity [mm]
    geology_class: str        # alluvium / limestone / clay-limestone
    source: str               # "eu_soilhydrogrids" or "farmer_override"
```

```
src/
  static_layers/
    __init__.py
    soil.py          # EU-SoilHydroGrids loading + pedotransfer fallback
    terrain.py       # DEM processing, slope, aspect, radiation correction
    geology.py       # BRGM classification
    parcels.py       # Parcel definition, farmer overrides
```

---

## Block 2 — Atmospheric Forcing (Reanalysis + Downscaling)

**Goal:** Produce parcel-scale hourly/daily meteorological forcing: temperature, precipitation, humidity, wind, radiation.

### Steps

1. **ERA5-Land extraction.**
   - Download hourly fields for Dordogne bounding box.
   - Variables: 2m temperature, 2m dewpoint temperature, 10m u-wind, 10m v-wind, surface pressure, surface solar radiation downward, total precipitation.
   - Aggregate to daily for the water balance (daily timestep is standard for FAO-56).

2. **Temperature downscaling.**
   - Apply elevation-based lapse rate correction: T_local = T_ERA5 + Γ × (z_ERA5 − z_local).
   - Standard environmental lapse rate Γ ≈ −6.5 °C/km.
   - Use DEM from Block 1.

3. **Precipitation bias correction.**
   - Fit quantile mapping (QM) or CDF-t from ERA5-Land to COMEPHORE over a calibration period (e.g., 2010–2020).
   - Apply correction to get bias-corrected daily precipitation at 250 m.
   - CDF-t handles non-stationarity better than simple QM — prefer it if precipitation trends matter.
   - **Library:** `scikit-downscale` or implement CDF-t following Michelangeli et al. (2009).

4. **Radiation correction.**
   - Adjust surface solar radiation for slope/aspect using terrain factors from Block 1.
   - Matters for parcels on hillsides; flat valley parcels need minimal correction.

### Validation

- Compare bias-corrected precipitation against independent Météo-France station data (synoptic stations in Bergerac, Sarlat).
- Compare temperature against station records.
- Quantify residual bias and RMSE.

### Output

Daily forcing per parcel:
```python
@dataclass
class DailyForcing:
    date: datetime.date
    t_min: float       # °C
    t_max: float       # °C
    t_mean: float      # °C
    dewpoint: float    # °C
    wind_speed: float  # m/s at 2m (convert from 10m using log profile)
    pressure: float    # kPa
    rs: float          # incoming solar radiation [MJ/m²/day]
    precip: float      # precipitation [mm/day]
```

```
src/
  atmospheric/
    __init__.py
    era5_processor.py       # Daily aggregation from hourly ERA5-Land
    temperature_downscale.py
    precip_bias_correction.py  # CDF-t or quantile mapping
    radiation_correction.py
    validation.py
```

**Python packages:** `xarray`, `scipy` (for QM/CDF-t fitting), `numpy`, `pyet` (optionally for wind height conversion)

---

## Block 3 — Reference ET and Soil Water Balance

**Goal:** Compute ET0 and run a daily soil water balance to track available water.

### Models

**ET0: FAO-56 Penman-Monteith** (Allen et al., 1998, eq. 6)

Use the `pyet` library — it implements FAO-56 PM and handles the unit conversions correctly. Inputs are all in the `DailyForcing` from Block 2.

```python
import pyet
et0 = pyet.pm_fao56(tmean, wind, rs, rn, pressure, e_s, e_a)
```

**Soil water balance: single-layer FAO-56 bucket**

Daily update:
```
W(t) = W(t-1) + P(t) + I(t) − ETc(t) − D(t) − R(t)
```

Where:
- `W(t)` = root zone soil water content [mm]
- `P(t)` = precipitation [mm] (from Block 2)
- `I(t)` = irrigation [mm] (from farmer input or decision layer)
- `ETc(t)` = crop ET = Kc × ET0 (Kc from Block 4, ET0 from this block)
- `D(t)` = deep drainage: max(0, W(t-1) + P + I − ETc − θ_FC × Z_r) — water above field capacity drains
- `R(t)` = runoff. Start with a simple threshold (SCS curve number or fraction of excess rainfall). Refine later if needed.

Stress coefficient:
```
Ks = (W − θ_WP × Zr) / ((1 − p) × TAW)   when W < (1-p) × TAW + θ_WP × Zr
Ks = 1.0                                     otherwise
```
Where TAW = (θ_FC − θ_WP) × Z_r, and p ≈ 0.55 for maize (FAO-56 Table 22). Under stress, ETc_adj = Ks × Kc × ET0.

### Output

Daily time series per parcel:
```python
@dataclass
class WaterBalanceState:
    date: datetime.date
    et0: float            # reference ET [mm]
    etc: float            # crop ET (adjusted for stress) [mm]
    precip: float         # effective precipitation [mm]
    irrigation: float     # applied irrigation [mm]
    drainage: float       # deep percolation [mm]
    runoff: float         # surface runoff [mm]
    soil_water: float     # root zone water content [mm]
    depletion: float      # current depletion [mm]
    stress_coeff: float   # Ks [0-1]
    fraction_awc: float   # W as fraction of TAW [0-1]
```

```
src/
  water_balance/
    __init__.py
    et0.py              # Penman-Monteith wrapper using pyet
    bucket_model.py     # FAO-56 daily soil water balance
    runoff.py           # SCS curve number or simple threshold
    state.py            # WaterBalanceState dataclass
```

**Python packages:** `pyet`, `numpy`, `pandas`

---

## Block 4 — Crop Module

**Goal:** Track maize phenological stage via GDD and provide the time-varying Kc and rooting depth.

### Growing Degree Days (GDD)

```
GDD(t) = max(0, (T_max + T_min)/2 − T_base)
```

T_base = 10°C for maize (standard). Accumulated GDD from planting date drives stage transitions.

### Phenological stages and Kc (FAO-56, grain maize)

| Stage | Cumulative GDD (approx.) | Duration (typical, SW France) | Kc |
|---|---|---|---|
| Initial (emergence → ~6 leaves) | 0 – 300 | ~30 days | 0.30 |
| Development (6 leaves → tasseling) | 300 – 700 | ~30 days | 0.30 → 1.20 (linear) |
| Mid-season (tasseling → grain fill) | 700 – 1300 | ~40 days | 1.20 |
| Late (grain fill → maturity) | 1300 – 1700 | ~30 days | 1.20 → 0.35 (linear) |

GDD thresholds are approximate and variety-dependent. **Farmer-provided planting date** anchors day 0. If they also provide variety, adjust thresholds based on Arvalis maturity group indices (available publicly for major French varieties).

### Rooting depth

Root depth increases linearly from ~0.15 m at emergence to ~1.0–1.2 m at mid-season, then constant. This modulates the volume of soil the bucket model draws from.

### Optional: Sentinel-2 NDVI/LAI cross-check

Use THEIA Sentinel-2 NDVI or LAI as an independent check on crop development stage. If NDVI shows the canopy is behind where GDD says it should be, flag it (possible stress, late emergence, or bad planting date input).

```
src/
  crop/
    __init__.py
    gdd.py              # GDD accumulation from daily T_min, T_max
    phenology.py        # Stage determination from cumulative GDD
    kc_curve.py         # Kc interpolation by stage
    rooting_depth.py    # Root depth evolution
    sentinel_check.py   # NDVI/LAI cross-validation (optional)
```

**Python packages:** `numpy`, `pandas`, `rasterio` (for Sentinel-2)

---

## Block 5 — Forecast Integration

**Goal:** Extend the water balance forward in time using NWP forecasts to anticipate irrigation needs.

### Short-term (0–4 days): AROME + ARPEGE

1. Fetch latest AROME (48h, 1.3 km) and ARPEGE (96h, ~10 km) forecasts from Météo-France Public API.
2. Extract same variables as ERA5-Land: T, precip, wind, humidity, radiation.
3. Apply same downscaling/bias corrections as Block 2 (temperature lapse rate, precipitation QM — but note that forecast bias structure may differ from reanalysis bias; calibrate separately if possible, or apply a simple forecast-specific multiplicative correction for precipitation).
4. Run Block 3 water balance forward with forecast forcing and current soil state as initial condition.
5. Determine: does projected depletion cross the stress threshold (p = 0.55 of TAW) within the forecast window?

### Seasonal (1–6 months): SEAS5

1. Download SEAS5 ensemble (25 or 51 members) monthly anomalies for temperature and precipitation.
2. Construct scenarios: apply anomalies to ERA5-Land climatology to generate plausible daily sequences (e.g., delta method or stochastic weather generator conditioned on monthly anomalies).
3. Run water balance ensemble forward from current state.
4. Output: probability distribution of total irrigation demand for the rest of the season.
5. **Be transparent:** SEAS5 precipitation skill in western France is low. Frame outputs as "range of plausible scenarios" not as forecasts. Temperature-driven ET0 projections are more reliable than precipitation projections.

```
src/
  forecasts/
    __init__.py
    arome_arpege.py     # Short-term forecast retrieval and processing
    seas5.py            # Seasonal forecast retrieval
    forward_balance.py  # Run water balance forward under forecast
    scenarios.py        # Seasonal scenario generation from SEAS5 anomalies
```

**Python packages:** `cdsapi`, `requests`, `xarray`, `numpy`

---

## Block 6 — Irrigation Decision Layer

**Goal:** Translate water balance state + forecasts into actionable irrigation recommendations.

### Decision logic (rule-based, start here)

```python
def irrigation_recommendation(state, forecast_states, params):
    """
    state: current WaterBalanceState
    forecast_states: list of WaterBalanceState for next N days
    params: crop and management parameters
    """
    # 1. Current depletion check
    if state.fraction_awc > (1 - params.p_threshold):
        return IrrigationAdvice(irrigate=False, reason="Soil water adequate")

    # 2. Will rain refill within forecast horizon?
    forecast_precip_total = sum(f.precip for f in forecast_states)
    projected_depletion = state.depletion + sum(f.etc for f in forecast_states) - forecast_precip_total

    if projected_depletion < params.p_threshold * params.total_awc:
        return IrrigationAdvice(
            irrigate=False,
            reason=f"Forecast rain ({forecast_precip_total:.0f} mm) should be sufficient"
        )

    # 3. Irrigation needed — compute amount
    # Target: refill to field capacity minus a safety margin
    # (don't fully refill if rain is possible)
    target_refill = state.depletion - forecast_precip_total * params.rain_confidence
    dose_mm = max(0, min(target_refill, params.max_dose_mm))

    # 4. When? Find the driest window in forecast
    best_day = find_driest_window(forecast_states, window_days=1)

    return IrrigationAdvice(
        irrigate=True,
        dose_mm=dose_mm,
        recommended_date=best_day,
        reason=f"Depletion will reach {projected_depletion:.0f} mm without irrigation",
        confidence=assess_confidence(forecast_states)
    )
```

### Outputs

Per parcel, daily or on-demand:
- **Irrigate: yes/no**
- **How much: mm**
- **When: recommended date within forecast window**
- **Confidence level** (tied to forecast uncertainty)
- **Seasonal outlook** (total expected irrigation need, range)
- **Current state summary** (soil water %, crop stage, stress level)

### Farmer interface (minimal first version)

A daily output (JSON, CSV, or simple dashboard) per parcel:
```json
{
  "parcel_id": "dordogne_24_001",
  "date": "2026-07-15",
  "crop_stage": "mid-season (tasseling)",
  "gdd_accumulated": 842,
  "soil_water_pct": 38,
  "stress_coefficient": 0.85,
  "recommendation": {
    "irrigate": true,
    "dose_mm": 35,
    "recommended_date": "2026-07-16",
    "reason": "No significant rain forecast in next 4 days, depletion approaching stress threshold",
    "confidence": "high"
  },
  "seasonal_outlook": {
    "expected_remaining_irrigation_mm": [80, 140],
    "scenario": "Drier than average summer (SEAS5 ensemble median)"
  }
}
```

```
src/
  decision/
    __init__.py
    rules.py            # Rule-based irrigation logic
    confidence.py       # Forecast uncertainty → confidence label
    output.py           # JSON/CSV output formatter
```

---

## Repo Structure (complete)

```
irrisol/
├── README.md
├── LICENSE                  # Open-source license (MIT or Apache-2.0)
├── pyproject.toml           # Project metadata, dependencies
├── configs/
│   ├── dordogne.yaml        # Region config: bounding box, CRS, grid resolution
│   └── parcels/
│       └── example.yaml     # Per-parcel config: farmer inputs, soil overrides
├── src/
│   ├── ingestion/           # Block 0
│   ├── static_layers/       # Block 1
│   ├── atmospheric/         # Block 2
│   ├── water_balance/       # Block 3
│   ├── crop/                # Block 4
│   ├── forecasts/           # Block 5
│   └── decision/            # Block 6
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_soil_analysis.ipynb
│   ├── 03_et0_validation.ipynb
│   ├── 04_water_balance_test.ipynb
│   ├── 05_crop_module_test.ipynb
│   └── 06_full_pipeline.ipynb
├── tests/
│   ├── test_et0.py
│   ├── test_bucket_model.py
│   ├── test_gdd.py
│   └── test_decision.py
└── data/
    ├── raw/                 # Downloaded, untouched
    ├── processed/           # Cleaned, gridded
    └── static/              # Soil, DEM, geology (version-controlled metadata only)
```

---

## Dependencies

```
# Core scientific
numpy
pandas
xarray
scipy
dask

# Geospatial
rasterio
geopandas
pyproj
shapely
fiona

# Climate/met specific
cdsapi              # ERA5-Land, SEAS5
pyet                # FAO-56 Penman-Monteith ET0
netcdf4             # NetCDF I/O backend for xarray

# Remote sensing
pystac-client       # THEIA catalog access
rioxarray           # xarray + rasterio integration

# Utilities
requests            # Météo-France API, THEIA downloads
pyyaml              # Config files
owslib              # WFS/WMS for BRGM, IGN

# Validation / viz
matplotlib
cartopy
scikit-learn        # Optional, for QM / statistical methods
```

---

## Build Order (suggested)

| Phase | Blocks | Milestone |
|---|---|---|
| 1 | Block 0 + 1 | Static layers loaded, grid defined, soil params for Dordogne |
| 2 | Block 0 + 2 | ERA5-Land downloaded, downscaled, validated against stations |
| 3 | Block 3 | ET0 computed, water balance running on historical period |
| 4 | Block 4 | Crop module integrated, ETc and stress tracked |
| 5 | Block 3+4 validation | Compare soil water evolution against ERA5-Land soil moisture or any probe data |
| 6 | Block 5 | Forecasts integrated, forward water balance working |
| 7 | Block 6 | Decision logic producing irrigation recommendations |
| 8 | End-to-end test | Full pipeline on one real parcel with farmer input, iterate |
