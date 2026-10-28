"""Monitoring package (PRD FR-7).

Tracks prediction error, data drift, prediction distribution, and model version.
"""

from .monitor import CLVMonitor

__all__ = ["CLVMonitor"]
