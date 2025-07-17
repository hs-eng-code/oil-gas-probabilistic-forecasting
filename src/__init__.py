"""
Probabilistic Oil Production Forecasting Package

This package provides comprehensive tools for oil production forecasting
using Arps decline curve analysis and Bayesian regression.
"""

__version__ = "1.0.0"
__author__ = "Oil Production Forecasting Team"

# Import main modules
from .data_loader import WellProductionDataLoader, DataValidationError
from .arps_dca import AdvancedArpsDCA, ArpsDeclineError
from .bayesian_forecaster import ModernizedBayesianForecaster, AssetScaleBayesianForecaster
from .aggregator import AssetAggregator
from .revenue_calculator import RevenueCalculator
from .visualizations import SingleScenarioResultsVisualizer, AcquisitionAnalysisVisualizer

__all__ = [
    "WellProductionDataLoader",
    "DataValidationError", 
    "AdvancedArpsDCA",
    "ArpsDeclineError",
    "ModernizedBayesianForecaster",
    "AssetScaleBayesianForecaster",
    "AssetAggregator",
    "RevenueCalculator",
    "SingleScenarioResultsVisualizer",
    "AcquisitionAnalysisVisualizer"
] 