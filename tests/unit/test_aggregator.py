"""
Unit tests for aggregator module.

These tests verify basic functionality of the AssetAggregator class.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from aggregator import AssetAggregator, AssetAggregationError


class TestAssetAggregator(unittest.TestCase):
    """Test cases for AssetAggregator."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = AssetAggregator(
            forecast_months=36,  # 3 years for testing
            validation_enabled=True
        )
        
        # Create sample well forecasts
        dates = pd.date_range(start='2025-01-01', periods=36, freq='MS')
        
        # Well A forecast
        self.well_a_forecast = pd.DataFrame({
            'Date': dates,
            'P10_Production_bbl': np.linspace(1000, 800, 36),  # Declining from 1000 to 800
            'P50_Production_bbl': np.linspace(900, 700, 36),   # Declining from 900 to 700
            'P90_Production_bbl': np.linspace(800, 600, 36)    # Declining from 800 to 600
        })
        
        # Well B forecast
        self.well_b_forecast = pd.DataFrame({
            'Date': dates,
            'P10_Production_bbl': np.linspace(1200, 900, 36),  # Declining from 1200 to 900
            'P50_Production_bbl': np.linspace(1100, 800, 36),  # Declining from 1100 to 800
            'P90_Production_bbl': np.linspace(1000, 700, 36)   # Declining from 1000 to 700
        })
        
        # Combined well forecasts
        self.sample_well_forecasts = {
            'Well_A': self.well_a_forecast,
            'Well_B': self.well_b_forecast
        }

    def test_initialization(self):
        """Test AssetAggregator initialization."""
        aggregator = AssetAggregator(
            forecast_months=120,
            validation_enabled=False
        )
        
        self.assertEqual(aggregator.forecast_months, 120)
        self.assertEqual(aggregator.validation_enabled, False)
        self.assertIsNone(aggregator.asset_production_forecast)
        self.assertEqual(aggregator.well_contributions, {})
        self.assertEqual(aggregator.aggregation_metrics, {})

    def test_aggregate_well_forecasts_basic(self):
        """Test basic well forecast aggregation."""
        # Test aggregation
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Basic assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 36)
        
        # Check required columns
        required_cols = [
            'Date', 'P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl',
            'P10_Cumulative_bbl', 'P50_Cumulative_bbl', 'P90_Cumulative_bbl'
        ]
        for col in required_cols:
            self.assertIn(col, result.columns)
        
        # Check that aggregated values are sums of individual wells
        # First month values
        expected_p10 = self.well_a_forecast['P10_Production_bbl'].iloc[0] + self.well_b_forecast['P10_Production_bbl'].iloc[0]
        expected_p50 = self.well_a_forecast['P50_Production_bbl'].iloc[0] + self.well_b_forecast['P50_Production_bbl'].iloc[0]
        expected_p90 = self.well_a_forecast['P90_Production_bbl'].iloc[0] + self.well_b_forecast['P90_Production_bbl'].iloc[0]
        
        self.assertAlmostEqual(result['P10_Production_bbl'].iloc[0], expected_p10, places=1)
        self.assertAlmostEqual(result['P50_Production_bbl'].iloc[0], expected_p50, places=1)
        self.assertAlmostEqual(result['P90_Production_bbl'].iloc[0], expected_p90, places=1)
        
        # Check that asset_production_forecast is stored
        self.assertIsNotNone(self.aggregator.asset_production_forecast)

    def test_cumulative_production_calculation(self):
        """Test cumulative production calculation."""
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Check that cumulative is increasing
        self.assertTrue(np.all(np.diff(result['P10_Cumulative_bbl']) >= 0))
        self.assertTrue(np.all(np.diff(result['P50_Cumulative_bbl']) >= 0))
        self.assertTrue(np.all(np.diff(result['P90_Cumulative_bbl']) >= 0))
        
        # Check that cumulative equals sum of monthly production
        expected_p50_cumulative = result['P50_Production_bbl'].cumsum()
        pd.testing.assert_series_equal(result['P50_Cumulative_bbl'], expected_p50_cumulative, check_names=False)

    def test_industry_convention_validation(self):
        """Test industry convention validation (P10 >= P50 >= P90)."""
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Check industry convention: P10 >= P50 >= P90
        # Allow for small numerical tolerance
        tolerance = 1e-6
        
        p10_prod = result['P10_Production_bbl'].values
        p50_prod = result['P50_Production_bbl'].values
        p90_prod = result['P90_Production_bbl'].values
        
        # Check P10 >= P50
        self.assertTrue(np.all(p10_prod >= p50_prod - tolerance))
        
        # Check P50 >= P90
        self.assertTrue(np.all(p50_prod >= p90_prod - tolerance))
        
        # Check P10 >= P90
        self.assertTrue(np.all(p10_prod >= p90_prod - tolerance))

    def test_well_contributions_storage(self):
        """Test that individual well contributions are stored."""
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Check that well contributions are stored
        self.assertEqual(len(self.aggregator.well_contributions), 2)
        self.assertIn('Well_A', self.aggregator.well_contributions)
        self.assertIn('Well_B', self.aggregator.well_contributions)
        
        # Check structure of well contributions
        well_a_contribution = self.aggregator.well_contributions['Well_A']
        expected_cols = ['Date', 'P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl']
        
        for col in expected_cols:
            self.assertIn(col, well_a_contribution.columns)
        
        self.assertEqual(len(well_a_contribution), 36)

    def test_aggregation_metrics_calculation(self):
        """Test aggregation metrics calculation."""
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Check that metrics are calculated
        self.assertIsInstance(self.aggregator.aggregation_metrics, dict)
        
        # Check key metrics
        metrics = self.aggregator.aggregation_metrics
        self.assertIn('well_count', metrics)
        self.assertIn('asset_p10_eur_bbl', metrics)
        self.assertIn('asset_p50_eur_bbl', metrics)
        self.assertIn('asset_p90_eur_bbl', metrics)
        self.assertIn('uncertainty_p10_p90_ratio', metrics)
        
        # Check well count
        self.assertEqual(metrics['well_count'], 2)
        
        # Check that EUR values are positive
        self.assertGreater(metrics['asset_p10_eur_bbl'], 0)
        self.assertGreater(metrics['asset_p50_eur_bbl'], 0)
        self.assertGreater(metrics['asset_p90_eur_bbl'], 0)
        
        # Check uncertainty ratio
        self.assertGreater(metrics['uncertainty_p10_p90_ratio'], 1.0)

    def test_aggregation_summary(self):
        """Test aggregation summary generation."""
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Get summary
        summary = self.aggregator.get_aggregation_summary()
        
        # Basic assertions
        self.assertIsInstance(summary, dict)
        self.assertIn('Asset Summary', summary)
        
        asset_summary = summary['Asset Summary']
        self.assertIn('Total Wells', asset_summary)
        self.assertIn('Asset P10 EUR (bbl)', asset_summary)
        self.assertIn('Asset P50 EUR (bbl)', asset_summary)
        self.assertIn('Asset P90 EUR (bbl)', asset_summary)

    def test_empty_well_forecasts(self):
        """Test handling of empty well forecasts."""
        # Should raise error for empty input
        with self.assertRaises(AssetAggregationError):
            self.aggregator.aggregate_well_forecasts({})

    def test_invalid_well_forecast_structure(self):
        """Test handling of invalid well forecast structure."""
        # Create invalid forecast (missing required columns)
        invalid_forecast = pd.DataFrame({
            'Date': pd.date_range(start='2025-01-01', periods=12, freq='MS'),
            'P10_Production_bbl': np.linspace(1000, 800, 12),
            # Missing P50 and P90 columns
        })
        
        invalid_well_forecasts = {
            'Invalid_Well': invalid_forecast
        }
        
        # Should raise error for invalid structure
        with self.assertRaises(AssetAggregationError):
            self.aggregator.aggregate_well_forecasts(invalid_well_forecasts)

    def test_uncertainty_trend_analysis(self):
        """Test uncertainty trend analysis."""
        # First aggregate the forecasts
        result = self.aggregator.aggregate_well_forecasts(
            self.sample_well_forecasts, 
            start_date='2025-01-01'
        )
        
        # Test uncertainty analysis
        uncertainty_analysis = self.aggregator.analyze_uncertainty_trends(result)
        
        # Basic assertions
        self.assertIsInstance(uncertainty_analysis, dict)
        self.assertIn('uncertainty_metrics', uncertainty_analysis)
        self.assertIn('range_analysis', uncertainty_analysis)
        self.assertIn('cv_analysis', uncertainty_analysis)
        self.assertIn('ratio_analysis', uncertainty_analysis)
        self.assertIn('business_interpretation', uncertainty_analysis)
        
        # Check uncertainty metrics structure
        uncertainty_metrics = uncertainty_analysis['uncertainty_metrics']
        self.assertIn('uncertainty_range', uncertainty_metrics)
        self.assertIn('coefficient_of_variation', uncertainty_metrics)
        self.assertIn('p10_p90_ratio', uncertainty_metrics)
        
        # Check that arrays have correct length
        self.assertEqual(len(uncertainty_metrics['uncertainty_range']), 36)
        self.assertEqual(len(uncertainty_metrics['coefficient_of_variation']), 36)


if __name__ == '__main__':
    unittest.main() 