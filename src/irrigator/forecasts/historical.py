"""Historical forecast interface for IrriGator backtesting.

The objective is to expose one source-independent representation of the
forecast information that would have been available around a historical
irrigation decision date.

Source priority
---------------
AROME
    1. Exact IrriGator archive: ``data/processed/arome/forecast``.
    2. Parcel-level Open-Meteo Previous Runs cache (historical approximation).

IFS ENS
    1. Exact IrriGator live archive: ``data/processed/ifs_ens``.
    2. Historical ECMWF TIGGE cache: ``data/processed/ifs_tigge``.

The returned fields are already converted to :class:`DailyForcing`, so the
optimizer does not need to know which historical backend was used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from irrigator.atmospheric.forcing import DailyForcing, extract_parcel_forcing
from irrigator.config import ParcelConfig
from irrigator.ingestion.arome_historical_client import (
    DEFAULT_MODEL as DEFAULT_AROME_HISTORICAL_MODEL,
)
from irrigator.ingestion.arome_historical_client import (
    load_arome_previous_runs,
    select_fixed_lead_days,
)
from irrigator.ingestion.ifs_tigge_client import load_tigge_daily
from irrigator.static_layers.terrain import TerrainParams

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")

AromeLeadMode = Literal["current_like", "causal_next_day"]
SourcePreference = Literal["auto", "exact", "historical"]


@dataclass(frozen=True)
class HistoricalForecast:
    """Forecast bundle for one historical decision date.

    Attributes
    ----------
    issue_date
        Date on which the irrigation plan is re-optimised.
    arome_forcing
        Deterministic short-range AROME forcing.
    ifs_member_forcings
        Medium-range IFS ENS forcing keyed by member number.
    arome_source, ifs_source
        Provenance labels (exact archive or historical reconstruction).
    arome_exact, ifs_exact
        Whether the data are exact archived runs from IrriGator's live
        collection rather than a historical fallback.
    """

    issue_date: date
    arome_forcing: DailyForcing
    ifs_member_forcings: dict[int, DailyForcing]
    arome_source: str
    ifs_source: str
    arome_exact: bool
    ifs_exact: bool

    @property
    def n_members(self) -> int:
        return len(self.ifs_member_forcings)

    @property
    def forecast_end(self) -> date | None:
        if not self.ifs_member_forcings:
            if self.arome_forcing.n_days == 0:
                return None
            return np.datetime64(self.arome_forcing.dates[-1], "D").astype(object)
        first = next(iter(self.ifs_member_forcings.values()))
        if first.n_days == 0:
            return None
        return np.datetime64(first.dates[-1], "D").astype(object)


def _exact_arome_path(issue_date: date, processed_dir: Path) -> Path:
    return Path(processed_dir) / "arome" / "forecast" / f"arome_daily_{issue_date.isoformat()}.nc"


def _exact_ifs_path(issue_date: date, run_hour: int, processed_dir: Path) -> Path:
    return (
        Path(processed_dir) / "ifs_ens" / f"ifs_daily_{issue_date.isoformat()}_{run_hour:02d}z.nc"
    )


def _historical_ifs_path(issue_date: date, run_hour: int, processed_dir: Path) -> Path:
    return (
        Path(processed_dir) / "ifs_tigge" / f"ifs_daily_{issue_date.isoformat()}_{run_hour:02d}z.nc"
    )


def _load_exact_arome(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    processed_dir: Path,
) -> DailyForcing:
    path = _exact_arome_path(issue_date, processed_dir)
    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        # Load before closing the file because extract_parcel_forcing accesses
        # the values immediately but returning an in-memory copy is safer.
        forcing = extract_parcel_forcing(ds.load(), parcel, terrain)
    return forcing


def _load_historical_arome_dataset(
    issue_date: date,
    parcel: ParcelConfig,
    processed_dir: Path,
    model: str,
) -> xr.Dataset:
    """Load enough parcel-level year caches to cover issue_date + 2."""
    years = {issue_date.year, (issue_date + timedelta(days=2)).year}
    datasets = [
        load_arome_previous_runs(
            parcel.id,
            year,
            processed_dir=processed_dir,
            model=model,
        )
        for year in sorted(years)
    ]
    if len(datasets) == 1:
        return datasets[0]

    result = xr.concat(
        datasets,
        dim="valid_time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer",
    ).sortby("valid_time")
    return result


def _load_reconstructed_arome(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    processed_dir: Path,
    *,
    lead_mode: AromeLeadMode,
    model: str,
) -> DailyForcing:
    ds = _load_historical_arome_dataset(issue_date, parcel, processed_dir, model)
    try:
        if lead_mode == "current_like":
            # Closest match to the existing IrriGator 00Z forecast interface:
            # today from fixed lead 0, tomorrow from fixed lead 1.
            # Caveat: lead 0 is a fixed-lead series, not a single archived 00Z
            # run, and can therefore reflect later same-day AROME cycles.
            selections = [
                (issue_date, 0),
                (issue_date + timedelta(days=1), 1),
            ]
        elif lead_mode == "causal_next_day":
            # Strict no-look-ahead formulation for daily backtests.  Decisions
            # at issue_date use weather that was forecast at least 24 h before
            # each valid day.  This shifts the deterministic AROME window to
            # tomorrow and the day after tomorrow.
            selections = [
                (issue_date + timedelta(days=1), 1),
                (issue_date + timedelta(days=2), 2),
            ]
        else:  # pragma: no cover - Literal protects normal callers
            raise ValueError(f"Unknown AROME lead mode: {lead_mode}")

        selected = select_fixed_lead_days(ds, selections)
        forcing = extract_parcel_forcing(selected, parcel, terrain)
        return forcing
    finally:
        try:
            ds.close()
        except Exception:
            pass


def load_historical_arome_forcing(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    source: SourcePreference = "auto",
    lead_mode: AromeLeadMode = "current_like",
    historical_model: str = DEFAULT_AROME_HISTORICAL_MODEL,
) -> tuple[DailyForcing, str, bool]:
    """Load the best available AROME forcing for one historical issue date."""
    processed_dir = Path(processed_dir)
    exact_path = _exact_arome_path(issue_date, processed_dir)

    if source in {"auto", "exact"} and exact_path.exists():
        return (
            _load_exact_arome(issue_date, parcel, terrain, processed_dir),
            f"IrriGator exact AROME archive ({exact_path.name})",
            True,
        )

    if source == "exact":
        raise FileNotFoundError(f"Exact AROME historical run not found: {exact_path}")

    forcing = _load_reconstructed_arome(
        issue_date,
        parcel,
        terrain,
        processed_dir,
        lead_mode=lead_mode,
        model=historical_model,
    )
    return (
        forcing,
        f"Open-Meteo AROME Previous Runs ({lead_mode})",
        False,
    )


def _load_ifs_dataset(
    issue_date: date,
    run_hour: int,
    processed_dir: Path,
    source: SourcePreference,
) -> tuple[xr.Dataset, str, bool]:
    exact_path = _exact_ifs_path(issue_date, run_hour, processed_dir)
    historical_path = _historical_ifs_path(issue_date, run_hour, processed_dir)

    if source in {"auto", "exact"} and exact_path.exists():
        return (
            xr.open_dataset(exact_path),
            f"IrriGator exact IFS ENS archive ({exact_path.name})",
            True,
        )

    if source == "exact":
        raise FileNotFoundError(f"Exact IFS ENS run not found: {exact_path}")

    if not historical_path.exists():
        raise FileNotFoundError(
            f"Historical IFS TIGGE cache not found: {historical_path}. "
            "Run irrigator.ingestion.ifs_tigge_client.process_tigge_run() first."
        )

    return (
        load_tigge_daily(issue_date, run_hour, processed_dir),
        f"ECMWF TIGGE historical IFS ENS ({historical_path.name})",
        False,
    )


def load_historical_ifs_members(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    run_hour: int = 0,
    source: SourcePreference = "auto",
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[dict[int, DailyForcing], str, bool]:
    """Load IFS members and convert each member to parcel DailyForcing."""
    processed_dir = Path(processed_dir)
    ds, source_name, exact = _load_ifs_dataset(issue_date, run_hour, processed_dir, source)
    try:
        if "number" in ds.dims:
            members = [int(v) for v in ds.number.values]
        else:
            members = [0]

        result: dict[int, DailyForcing] = {}
        for member in members:
            member_ds = ds.sel(number=member) if "number" in ds.dims else ds
            forcing = extract_parcel_forcing(member_ds, parcel, terrain)
            if start_date is not None or end_date is not None:
                forcing = forcing.slice(start=start_date, end=end_date)
            result[member] = forcing

        return result, source_name, exact
    finally:
        try:
            ds.close()
        except Exception:
            pass


def get_historical_forecast(
    issue_date: date,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    run_hour: int = 0,
    forecast_days: int = 15,
    arome_source: SourcePreference = "auto",
    ifs_source: SourcePreference = "auto",
    arome_lead_mode: AromeLeadMode = "current_like",
    historical_arome_model: str = DEFAULT_AROME_HISTORICAL_MODEL,
) -> HistoricalForecast:
    """Return the forecast bundle available for a historical decision date.

    Parameters
    ----------
    issue_date
        Day on which irrigation is re-optimised.
    parcel, terrain
        Existing IrriGator parcel/terrain objects.
    forecast_days
        Total horizon counted from ``issue_date``.  The default 15 means the
        final valid date is ``issue_date + 14 days``.
    arome_source, ifs_source
        ``"auto"`` prefers an exact IrriGator archive and falls back to the
        historical backend; ``"exact"`` requires your own archive;
        ``"historical"`` forces the fallback source even when exact data exist.
    arome_lead_mode
        ``"current_like"`` selects issue-day lead0 + next-day lead1 and most
        closely mirrors the current two-day AROME interface.

        ``"causal_next_day"`` uses next-day lead1 + day-after lead2.  It is
        stricter for scientific backtests because no same-day fixed-lead-0
        series is used.
    """
    if forecast_days < 1:
        raise ValueError("forecast_days must be >= 1")

    arome, arome_source_name, arome_exact = load_historical_arome_forcing(
        issue_date,
        parcel,
        terrain,
        processed_dir=processed_dir,
        source=arome_source,
        lead_mode=arome_lead_mode,
        historical_model=historical_arome_model,
    )

    if arome.n_days:
        last_arome = np.datetime64(arome.dates[-1], "D").astype(object)
        ifs_start = last_arome + timedelta(days=1)
    else:
        ifs_start = issue_date

    forecast_end = issue_date + timedelta(days=forecast_days - 1)
    members, ifs_source_name, ifs_exact = load_historical_ifs_members(
        issue_date,
        parcel,
        terrain,
        processed_dir=processed_dir,
        run_hour=run_hour,
        source=ifs_source,
        start_date=ifs_start,
        end_date=forecast_end,
    )

    # Guard against a silent empty-member backtest caused by an incompatible
    # AROME/IFS handoff or missing lead times.
    empty_members = [member for member, forcing in members.items() if forcing.n_days == 0]
    if empty_members:
        raise RuntimeError(
            f"IFS members have no data for {ifs_start} -> {forecast_end}: {empty_members[:5]}"
        )

    logger.info(
        "Historical forecast %s parcel=%s: AROME=%s (%d d), IFS=%s (%d members)",
        issue_date,
        parcel.id,
        arome_source_name,
        arome.n_days,
        ifs_source_name,
        len(members),
    )

    return HistoricalForecast(
        issue_date=issue_date,
        arome_forcing=arome,
        ifs_member_forcings=members,
        arome_source=arome_source_name,
        ifs_source=ifs_source_name,
        arome_exact=arome_exact,
        ifs_exact=ifs_exact,
    )
