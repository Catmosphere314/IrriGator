"""Block 6 - Irrigation decision layer.

It translates the water balance state and forecasts into actionable
irrigation recommendations. 

It is based on a rule-based decision logic (for now).
"""


from irrigator.decision.confidence import assess_confidence
from irrigator.decision.rules import irrigation_recommendation, IrrigationAdvice

__all__ = ["assess_confidence", "irrigation_recommendation", "IrrigationAdvice"]