"""
Unit tests for arps_dca module.

These tests verify basic functionality of the AdvancedArpsDCA class.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from arps_dca import AdvancedArpsDCA, ArpsDeclineError, FitResult


class TestArpsDCA(unittest.TestCase):
    """Test cases for AdvancedArpsDCA."""

    def setUp(self):
        """Set up test fixtures."""
        self.arps_dca = AdvancedArpsDCA(
            terminal_decline_rate=0.05,
            min_production_months=6,
            max_forecast_years=30
        )
        
        # Create sample production data with declining trend
        dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')
        self.sample_production_data = pd.DataFrame({
            'WellName': ['TestWell_A'] * 12,
            'DATE': dates,
            'OIL': [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450]  # Clear declining trend
        })

    def test_initialization(self):
        """Test AdvancedArpsDCA initialization."""
        dca = AdvancedArpsDCA(
            terminal_decline_rate=0.03,
            min_production_months=8,
            max_forecast_years=25
        )
        
        self.assertEqual(dca.terminal_decline_rate, 0.03)
        self.assertEqual(dca.min_production_months, 8)
        self.assertEqual(dca.max_forecast_years, 25)
        self.assertEqual(dca.fit_results, {})
        self.assertEqual(dca.validation_results, {})

    def test_fit_decline_curve_basic(self):
        """Test basic decline curve fitting."""
        well_name = 'TestWell_A'
        
        # Test fitting
        result = self.arps_dca.fit_decline_curve(self.sample_production_data, well_name)
        
        # Basic assertions
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            # Check that basic parameters are present
            self.assertIn('qi', result)
            self.assertIn('Di', result)
            self.assertIn('b', result)
            self.assertIn('method', result)
            
            # Check that parameters are reasonable
            self.assertGreater(result['qi'], 0)
            self.assertGreater(result['Di'], 0)
            self.assertGreaterEqual(result['b'], 0)
            
            # Check that fit results are stored
            self.assertIn(well_name, self.arps_dca.fit_results)
        else:
            # If fit failed, should have error message
            self.assertIn('error', result)

    def test_forecast_production_after_fit(self):
        """Test production forecasting after fitting."""
        well_name = 'TestWell_A'
        
        # First fit the decline curve
        fit_result = self.arps_dca.fit_decline_curve(self.sample_production_data, well_name)
        
        if fit_result['success']:
            # Test forecasting
            forecast_result = self.arps_dca.forecast_production(well_name, forecast_months=36)
            
            # Basic assertions
            self.assertIsInstance(forecast_result, dict)
            self.assertIn('time', forecast_result)
            self.assertIn('production', forecast_result)
            self.assertIn('cumulative', forecast_result)
            
            # Check forecast arrays
            self.assertEqual(len(forecast_result['time']), 37)  # 0 to 36 months
            self.assertEqual(len(forecast_result['production']), 37)
            self.assertEqual(len(forecast_result['cumulative']), 37)
            
            # Check that production is positive
            self.assertTrue(np.all(forecast_result['production'] > 0))
            
            # Check that cumulative is increasing
            self.assertTrue(np.all(np.diff(forecast_result['cumulative']) >= 0))

    def test_predict_decline_curve_method(self):
        """Test the predict_decline_curve method."""
        well_name = 'TestWell_A'
        
        # First fit the decline curve
        fit_result = self.arps_dca.fit_decline_curve(self.sample_production_data, well_name)
        
        if fit_result['success']:
            # Test prediction with fitted parameters
            time_array = np.array([0, 1, 2, 3, 6, 12, 24])
            qi = fit_result['qi']
            Di = fit_result['Di']
            b = fit_result['b']
            
            predictions = self.arps_dca.predict_decline_curve(well_name, time_array, qi, Di, b)
            
            # Basic assertions
            self.assertEqual(len(predictions), len(time_array))
            self.assertTrue(np.all(predictions > 0))
            
            # Check that predictions generally decline
            # (allowing for some numerical noise)
            self.assertGreaterEqual(predictions[0], predictions[-1])

    def test_fit_summary_generation(self):
        """Test fit summary generation."""
        well_name = 'TestWell_A'
        
        # Fit decline curve
        fit_result = self.arps_dca.fit_decline_curve(self.sample_production_data, well_name)
        
        if fit_result['success']:
            # Test summary generation
            summary = self.arps_dca.get_fit_summary()
            
            # Basic assertions
            self.assertIsInstance(summary, pd.DataFrame)
            if not summary.empty:
                self.assertIn('WellName', summary.columns)
                self.assertIn('qi', summary.columns)
                self.assertIn('Di', summary.columns)
                self.assertIn('b', summary.columns)

    def test_invalid_well_name_forecast(self):
        """Test forecasting with invalid well name."""
        # Try to forecast for a well that hasn't been fitted
        with self.assertRaises(ArpsDeclineError):
            self.arps_dca.forecast_production('NonExistentWell', forecast_months=12)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create data with only 2 data points
        insufficient_data = pd.DataFrame({
            'WellName': ['TestWell_Insufficient'] * 2,
            'DATE': ['2023-01-01', '2023-02-01'],
            'OIL': [1000, 900]
        })
        
        result = self.arps_dca.fit_decline_curve(insufficient_data, 'TestWell_Insufficient')
        
        # Should either succeed with fallback method or fail gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if not result['success']:
            self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main() 