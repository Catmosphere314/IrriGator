"""Output formatter for irrigation recommendations.

Produces structured JSON output that can be sent to a farmer via
API, email, SMS, or dashboard.  Combines the tactical recommendation
(irrigate now?) with the seasonal outlook (how much this season?).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from irrigator.decision.rules import IrrigationAdvice
from irrigator.decision.seasonal_outlook import SeasonalOutlook

logger = logging.getLogger(__name__)


def build_parcel_report(
    parcel_id: str,
    today: date,
    advice: IrrigationAdvice,
    outlook: SeasonalOutlook | None = None,
) -> dict:
    """Build the complete recommendation report for one parcel.

    This is the final output of the pipeline — what the farmer sees.
    """
    report = {
        "parcel_id": parcel_id,
        "date": today.isoformat(),
        "recommendation": advice.to_dict(),
    }

    if outlook:
        report["seasonal_outlook"] = outlook.to_dict()

    return report


def save_report(report: dict, output_dir: Path) -> Path:
    """Save report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parcel_id = report["parcel_id"]
    report_date = report["date"]
    path = output_dir / f"recommendation_{parcel_id}_{report_date}.json"

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Report saved: %s", path)
    return path


def format_sms(advice: IrrigationAdvice) -> str:
    """Format recommendation as a short SMS-friendly message."""
    if not advice.irrigate:
        return f"[IrriGator] {advice.date}: Pas d'irrigation nécessaire. {advice.reason}"

    return (
        f"[IrriGator] {advice.date}: IRRIGUER {advice.dose_mm:.0f}mm "
        f"le {advice.recommended_date}. "
        f"Réserve sol: {advice.current_ks * 100:.0f}%. "
        f"Stade: {advice.crop_stage}."
    )
