"""Historical forecast interface for IrriGator backtesting.

The retrospective forecast architecture is deliberately homogeneous across
backtest years:

    ECMWF IFS HRES (deterministic)   issue day -> issue day + 1
    ECMWF TIGGE ENS (51 members)     issue day + 2 -> forecast horizon

HRES
----
Historical deterministic short-range forecasts are exact Open-Meteo Single
Runs from the native ~9 km ECMWF IFS HRES archive.  They are cached per parcel
and per initialisation by :mod:`irrigator.ingestion.hres_historical_client`.

TIGGE
-----
Historical medium-range forecasts are operational ECMWF ENS forecasts from
TIGGE, processed by :mod:`irrigator.ingestion.ifs_tigge_client`.  The processed
TIGGE cache contains control member 0 plus perturbed members 1..50.

Both ingestion modules normalize their output to the same daily meteorological
schema before this module converts them to :class:`DailyForcing`:

    t_min, t_max, t_mean, dewpoint          [degC]
    wind_speed_10m                          [m/s]
    pressure_kpa                            [kPa]
    precip_mm                               [mm/day]
    rs_mj                                   [MJ/m2/day]

This module is intentionally loading-only.  Missing HRES/TIGGE archives raise
clear errors rather than silently launching potentially long historical data
downloads during an irrigation backtest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from irrigator.atmospheric.forcing import (
    DailyForcing,
    extract_parcel_forcing,
)
from irrigator.config import ParcelConfig
from irrigator.ingestion.hres_historical_client import (
    DEFAULT_MODEL as DEFAULT_HRES_MODEL,
    load_hres_run,
    select_hres_daily_window,
)
from irrigator.ingestion.ifs_tigge_client import load_tigge_daily
from irrigator.static_layers.terrain import TerrainParams


logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")


@dataclass(frozen=True)
class HistoricalForecast:
    """Retrospective forecast bundle for one irrigation decision date.

    Parameters
    ----------
    issue_date
        Date on which the irrigation plan is re-optimized.
    hres_forcing
        Deterministic short-range HRES forcing, normally issue_date and
        issue_date + 1.
    ifs_member_forcings
        TIGGE IFS ENS medium-range forcing keyed by actual ensemble member
        number.  Member 0 is the control; members 1..50 are perturbed members.
    hres_source, ifs_source
        Human-readable provenance labels.

    Notes
    -----
    ``combined_member_forcings`` prepends the same deterministic HRES segment
    to every TIGGE member.  This is normally the representation required by an
    ensemble irrigation optimizer.
    """

    issue_date: date
    hres_forcing: DailyForcing
    ifs_member_forcings: dict[int, DailyForcing]
    hres_source: str
    ifs_source: str

    @property
    def n_members(self) -> int:
        """Number of medium-range ensemble members."""
        return len(self.ifs_member_forcings)

    @property
    def forecast_start(self) -> date | None:
        """First available forecast date."""
        if self.hres_forcing.n_days:
            return _forcing_date(self.hres_forcing.dates[0])

        if self.ifs_member_forcings:
            first = next(iter(self.ifs_member_forcings.values()))
            if first.n_days:
                return _forcing_date(first.dates[0])

        return None

    @property
    def forecast_end(self) -> date | None:
        """Last available forecast date."""
        if self.ifs_member_forcings:
            first = next(iter(self.ifs_member_forcings.values()))
            if first.n_days:
                return _forcing_date(first.dates[-1])

        if self.hres_forcing.n_days:
            return _forcing_date(self.hres_forcing.dates[-1])

        return None

    @property
    def combined_member_forcings(self) -> dict[int, DailyForcing]:
        """Return HRES + TIGGE forcing for every ensemble member.

        The deterministic HRES prefix is identical for all members.  The
        existing ``DailyForcing.concat`` implementation gives the HRES prefix
        precedence if an unexpected overlap is present at the hand-off.

        If the requested horizon contains only HRES days and therefore no
        TIGGE members, a deterministic member ``0`` is returned.
        """
        if not self.ifs_member_forcings:
            return {0: self.hres_forcing}

        return {
            member: self.hres_forcing.concat(ifs_forcing)
            for member, ifs_forcing in self.ifs_member_forcings.items()
        }


def _forcing_date(value: np.datetime64) -> date:
    """Convert a numpy datetime value to ``datetime.date``."""
    return np.datetime64(value, "D").astype(object)


def _expected_daily_dates(
    start_date: date,
    end_date: date,
) -> np.ndarray:
    """Inclusive sequence of expected daily dates."""
    if end_date < start_date:
        return np.array([], dtype="datetime64[D]")

    n_days = (end_date - start_date).days + 1
    return np.datetime64(start_date, "D") + np.arange(n_days).astype("timedelta64[D]")


def _validate_forcing_dates(
    forcing: DailyForcing,
    *,
    start_date: date,
    end_date: date,
    label: str,
) -> None:
    """Require complete, unique, contiguous daily coverage."""
    expected = _expected_daily_dates(start_date, end_date)
    actual = forcing.dates.astype("datetime64[D]")

    if len(actual) != len(np.unique(actual)):
        raise RuntimeError(f"{label} contains duplicate daily dates: {actual}")

    if not np.array_equal(actual, expected):
        missing = np.setdiff1d(expected, actual)
        extra = np.setdiff1d(actual, expected)

        raise RuntimeError(
            f"{label} does not cover the expected period "
            f"{start_date} -> {end_date}. "
            f"Missing={missing.astype(str).tolist()}, "
            f"extra={extra.astype(str).tolist()}"
        )


def load_historical_hres_forcing(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    run_hour: int = 0,
    n_days: int = 2,
    model: str = DEFAULT_HRES_MODEL,
) -> tuple[DailyForcing, str]:
    """Load deterministic HRES forcing for one historical issue date.

    The HRES ingestion cache is parcel-specific and already contains daily
    meteorological variables in IrriGator units.  ``extract_parcel_forcing``
    then applies the same parcel-level temperature/radiation/wind processing
    used elsewhere in the project.

    For the standard 00Z retrospective architecture, ``n_days=2`` returns
    issue_date and issue_date + 1.
    """
    if n_days < 1:
        raise ValueError("n_days must be >= 1")

    processed_dir = Path(processed_dir)

    try:
        ds = load_hres_run(
            parcel.id,
            issue_date,
            run_hour=run_hour,
            processed_dir=processed_dir,
            model=model,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Historical HRES run is not cached for parcel={parcel.id!r}, "
            f"date={issue_date}, {run_hour:02d}Z. "
            "Run irrigator.ingestion.hres_historical_client."
            "archive_hres_run(...) first."
        ) from exc

    try:
        selected = select_hres_daily_window(
            ds,
            start_date=issue_date,
            n_days=n_days,
        ).load()

        forcing = extract_parcel_forcing(
            selected,
            parcel,
            terrain,
            precip_correction=None,
        )
    finally:
        ds.close()

    end_date = issue_date + timedelta(days=n_days - 1)
    _validate_forcing_dates(
        forcing,
        start_date=issue_date,
        end_date=end_date,
        label="Historical HRES forcing",
    )

    source = (
        f"Open-Meteo Single Runs - ECMWF IFS HRES "
        f"({issue_date.isoformat()} {run_hour:02d}Z, model={model})"
    )

    return forcing, source


def load_historical_ifs_members(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    start_date: date,
    end_date: date,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    run_hour: int = 0,
) -> tuple[dict[int, DailyForcing], str]:
    """Load TIGGE ENS members for the medium-range historical horizon.

    Parameters
    ----------
    issue_date
        Forecast initialization date.
    start_date, end_date
        Inclusive valid-date interval to retain.  In the standard IrriGator
        retrospective setup this starts at issue_date + 2.
    parcel, terrain
        Existing parcel and terrain configuration.
    processed_dir
        IrriGator processed-data root.
    run_hour
        TIGGE initialization hour, normally 00Z.

    Returns
    -------
    dict[int, DailyForcing], str
        Member forcings keyed by actual TIGGE ensemble number and a provenance
        label.
    """
    if end_date < start_date:
        return {}, "ECMWF TIGGE historical IFS ENS (not required)"

    processed_dir = Path(processed_dir)

    try:
        ds = load_tigge_daily(
            issue_date,
            run_hour=run_hour,
            processed_dir=processed_dir,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Historical TIGGE run is not cached for "
            f"{issue_date} {run_hour:02d}Z. "
            "Run irrigator.ingestion.ifs_tigge_client."
            "process_tigge_run(...) first."
        ) from exc

    try:
        if "valid_time" not in ds.dims:
            raise RuntimeError("Processed TIGGE dataset has no valid_time dimension.")

        # Restrict once before iterating through all 51 members.
        subset = ds.sel(
            valid_time=slice(
                np.datetime64(start_date),
                np.datetime64(end_date),
            )
        ).load()

        if subset.sizes.get("valid_time", 0) == 0:
            raise RuntimeError(
                f"TIGGE run {issue_date} {run_hour:02d}Z has no data for "
                f"{start_date} -> {end_date}."
            )

        if "number" in subset.dims:
            members = [int(v) for v in subset.number.values]
        else:
            members = [0]

        result: dict[int, DailyForcing] = {}

        for member in members:
            member_ds = subset.sel(number=member) if "number" in subset.dims else subset

            forcing = extract_parcel_forcing(
                member_ds,
                parcel,
                terrain,
                precip_correction=None,
            )

            _validate_forcing_dates(
                forcing,
                start_date=start_date,
                end_date=end_date,
                label=f"TIGGE member {member}",
            )

            result[member] = forcing

    finally:
        ds.close()

    if not result:
        raise RuntimeError(f"No TIGGE ensemble members found for {issue_date} {run_hour:02d}Z.")

    # The current TIGGE ingestion combines control 0 with perturbed 1..50.
    numbers = sorted(result)
    if 0 not in numbers:
        raise RuntimeError("Processed TIGGE ensemble is missing control member 0.")

    source = (
        f"ECMWF IFS ENS historical operational forecast via TIGGE "
        f"({issue_date.isoformat()} {run_hour:02d}Z; "
        f"{len(result)} members)"
    )

    return result, source


def get_historical_forecast(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    run_hour: int = 0,
    forecast_days: int = 15,
    hres_days: int = 2,
    hres_model: str = DEFAULT_HRES_MODEL,
) -> HistoricalForecast:
    """Build the retrospective HRES -> TIGGE forecast for one decision date.

    Parameters
    ----------
    issue_date
        Day on which irrigation is re-optimized.
    parcel, terrain
        Existing IrriGator parcel and terrain objects.
    processed_dir
        Root of processed IrriGator data.
    run_hour
        Forecast initialization hour.  For the standard backtest use 00Z.
        The same initialization hour is used for HRES and TIGGE.
    forecast_days
        Total forecast horizon counted from ``issue_date``.  The default
        15-day horizon ends at ``issue_date + 14``.
    hres_days
        Number of deterministic HRES calendar days at the front of the
        forecast.  The standard architecture uses two days (D+0 and D+1),
        then TIGGE begins on D+2.
    hres_model
        Open-Meteo HRES model identifier used when locating the cached file.

    Returns
    -------
    HistoricalForecast
        Deterministic HRES prefix plus TIGGE ensemble tails.

    Notes
    -----
    This function never downloads data.  Archive HRES and TIGGE runs first.
    This makes a backtest deterministic and prevents an optimization loop from
    accidentally launching hundreds of remote requests.
    """
    if forecast_days < 1:
        raise ValueError("forecast_days must be >= 1")
    if hres_days < 1:
        raise ValueError("hres_days must be >= 1")

    # A shorter requested horizon simply truncates the deterministic prefix.
    deterministic_days = min(hres_days, forecast_days)

    hres, hres_source = load_historical_hres_forcing(
        issue_date,
        parcel,
        terrain,
        processed_dir=processed_dir,
        run_hour=run_hour,
        n_days=deterministic_days,
        model=hres_model,
    )

    forecast_end = issue_date + timedelta(days=forecast_days - 1)
    ifs_start = issue_date + timedelta(days=deterministic_days)

    if ifs_start <= forecast_end:
        members, ifs_source = load_historical_ifs_members(
            issue_date,
            parcel,
            terrain,
            start_date=ifs_start,
            end_date=forecast_end,
            processed_dir=processed_dir,
            run_hour=run_hour,
        )
    else:
        members = {}
        ifs_source = "ECMWF TIGGE historical IFS ENS (not required)"

    logger.info(
        "Historical forecast %s parcel=%s: HRES=%d day(s), TIGGE=%d member(s), horizon=%d day(s)",
        issue_date,
        parcel.id,
        hres.n_days,
        len(members),
        forecast_days,
    )

    result = HistoricalForecast(
        issue_date=issue_date,
        hres_forcing=hres,
        ifs_member_forcings=members,
        hres_source=hres_source,
        ifs_source=ifs_source,
    )

    # Final end-to-end validation after concatenating deterministic + ensemble
    # segments.  This catches any unnoticed one-day hand-off shift.
    expected_end = forecast_end
    for member, forcing in result.combined_member_forcings.items():
        _validate_forcing_dates(
            forcing,
            start_date=issue_date,
            end_date=expected_end,
            label=f"Combined historical forecast member {member}",
        )

    return result
