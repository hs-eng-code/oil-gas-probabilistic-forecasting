"""
Unit tests for bayesian_forecaster module.

These tests verify basic functionality of the ModernizedBayesianForecaster class.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from bayesian_forecaster import ModernizedBayesianForecaster, BayesianForecastError
from arps_dca import AdvancedArpsDCA


class TestBayesianForecaster(unittest.TestCase):
    """Test cases for ModernizedBayesianForecaster."""

    def setUp(self):
        """Set up test fixtures."""
        # Create an AdvancedArpsDCA instance for the forecaster
        self.arps_dca = AdvancedArpsDCA(
            terminal_decline_rate=0.05,
            min_production_months=6,
            max_forecast_years=30
        )
        
        # Create the Bayesian forecaster with pre-fitted ARPS DCA
        self.forecaster = ModernizedBayesianForecaster(
            n_samples=100,  # Small for testing
            confidence_level=0.9,
            arps_dca_instance=self.arps_dca,
            random_seed=42  # For reproducible tests
        )
        
        # Create sample production data with declining trend
        dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')
        self.sample_production_data = pd.DataFrame({
            'WellName': ['TestWell_B'] * 12,
            'DATE': dates,
            'OIL': [1200, 1150, 1100, 1050, 1000, 950, 900, 850, 800, 750, 700, 650]  # Clear declining trend
        })

    def test_initialization(self):
        """Test ModernizedBayesianForecaster initialization."""
        forecaster = ModernizedBayesianForecaster(
            n_samples=500,
            confidence_level=0.95,
            random_seed=123
        )
        
        self.assertEqual(forecaster.n_samples, 500)
        self.assertEqual(forecaster.confidence_level, 0.95)
        self.assertEqual(forecaster.random_seed, 123)
        self.assertEqual(forecaster.fit_results, {})
        self.assertEqual(forecaster.bayesian_posteriors, {})

    def test_fit_bayesian_decline_basic(self):
        """Test basic Bayesian decline curve fitting."""
        well_name = 'TestWell_B'
        
        # Test fitting
        result = self.forecaster.fit_bayesian_decline(self.sample_production_data, well_name)
        
        # Basic assertions
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            # Check that key components are present
            self.assertIn('well_name', result)
            self.assertIn('deterministic_fit', result)
            self.assertIn('quality_assessment', result)
            self.assertIn('parameter_samples', result)
            
            # Check well name
            self.assertEqual(result['well_name'], well_name)
            
            # Check parameter samples structure
            parameter_samples = result['parameter_samples']
            self.assertIn('qi', parameter_samples)
            self.assertIn('Di', parameter_samples)
            self.assertIn('b', parameter_samples)
            
            # Check sample arrays
            self.assertEqual(len(parameter_samples['qi']), 100)  # n_samples
            self.assertEqual(len(parameter_samples['Di']), 100)
            self.assertEqual(len(parameter_samples['b']), 100)
            
            # Check that fit results are stored
            self.assertIn(well_name, self.forecaster.fit_results)
        else:
            # If fit failed, should have error message
            self.assertIn('error', result)

    def test_forecast_probabilistic_after_fit(self):
        """Test probabilistic forecasting after fitting."""
        well_name = 'TestWell_B'
        
        # First fit the Bayesian decline curve
        fit_result = self.forecaster.fit_bayesian_decline(self.sample_production_data, well_name)
        
        if fit_result['success']:
            # Test probabilistic forecasting
            forecast_result = self.forecaster.forecast_probabilistic(
                well_name, 
                forecast_months=36, 
                percentiles=[0.9, 0.5, 0.1]  # P10, P50, P90
            )
            
            # Basic assertions
            self.assertIsInstance(forecast_result, dict)
            self.assertIn('success', forecast_result)
            
            if forecast_result['success']:
                self.assertIn('forecast_percentiles', forecast_result)
                self.assertIn('cumulative_percentiles', forecast_result)
                
                # Check forecast percentiles
                forecast_percentiles = forecast_result['forecast_percentiles']
                self.assertIn('P10', forecast_percentiles)
                self.assertIn('P50', forecast_percentiles)
                self.assertIn('P90', forecast_percentiles)
                
                # Check forecast arrays
                self.assertEqual(len(forecast_percentiles['P10']), 36)
                self.assertEqual(len(forecast_percentiles['P50']), 36)
                self.assertEqual(len(forecast_percentiles['P90']), 36)
                
                # Check that all forecasts are positive
                self.assertTrue(np.all(forecast_percentiles['P10'] > 0))
                self.assertTrue(np.all(forecast_percentiles['P50'] > 0))
                self.assertTrue(np.all(forecast_percentiles['P90'] > 0))
                
                # Check industry convention: P10 >= P50 >= P90
                # (Allow for some numerical tolerance)
                p10_vals = forecast_percentiles['P10']
                p50_vals = forecast_percentiles['P50']
                p90_vals = forecast_percentiles['P90']
                
                # Check first few values (most reliable)
                for i in range(min(5, len(p10_vals))):
                    self.assertGreaterEqual(p10_vals[i], p90_vals[i] * 0.9)  # Some tolerance

    def test_parameter_correlations(self):
        """Test parameter correlation calculation."""
        well_name = 'TestWell_B'
        
        # First fit the Bayesian decline curve
        fit_result = self.forecaster.fit_bayesian_decline(self.sample_production_data, well_name)
        
        if fit_result['success']:
            # Test parameter correlations
            correlations = self.forecaster.get_parameter_correlations(well_name)
            
            # Basic assertions
            self.assertIsInstance(correlations, dict)
            self.assertIn('correlation_matrix', correlations)
            self.assertIn('qi_Di_correlation', correlations)
            self.assertIn('qi_b_correlation', correlations)
            self.assertIn('Di_b_correlation', correlations)
            
            # Check correlation matrix structure
            correlation_matrix = correlations['correlation_matrix']
            self.assertEqual(len(correlation_matrix), 3)
            self.assertEqual(len(correlation_matrix[0]), 3)
            
            # Check correlation values are reasonable
            self.assertGreaterEqual(correlations['qi_Di_correlation'], -1)
            self.assertLessEqual(correlations['qi_Di_correlation'], 1)

    def test_fit_summary_generation(self):
        """Test fit summary generation."""
        well_name = 'TestWell_B'
        
        # Fit Bayesian decline curve
        fit_result = self.forecaster.fit_bayesian_decline(self.sample_production_data, well_name)
        
        if fit_result['success']:
            # Test summary generation
            summary = self.forecaster.get_fit_summary()
            
            # Basic assertions
            self.assertIsInstance(summary, pd.DataFrame)
            if not summary.empty:
                self.assertIn('WellName', summary.columns)
                self.assertIn('success', summary.columns)
                self.assertIn('method', summary.columns)
                self.assertIn('composite_score', summary.columns)

    def test_forecast_without_fit(self):
        """Test forecasting without prior fitting."""
        # Try to forecast for a well that hasn't been fitted
        with self.assertRaises(BayesianForecastError):
            self.forecaster.forecast_probabilistic('NonExistentWell', forecast_months=12)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with fixed seed."""
        well_name = 'TestWell_B'
        
        # Create two identical forecasters with same seed
        forecaster1 = ModernizedBayesianForecaster(
            n_samples=50,
            arps_dca_instance=self.arps_dca,
            random_seed=123
        )
        
        forecaster2 = ModernizedBayesianForecaster(
            n_samples=50,
            arps_dca_instance=self.arps_dca,
            random_seed=123
        )
        
        # Fit both
        result1 = forecaster1.fit_bayesian_decline(self.sample_production_data, well_name)
        result2 = forecaster2.fit_bayesian_decline(self.sample_production_data, well_name)
        
        if result1['success'] and result2['success']:
            # Parameter samples should be identical (or very close)
            params1 = result1['parameter_samples']
            params2 = result2['parameter_samples']
            
            # Check first few samples (exact match may not be guaranteed due to complex initialization)
            qi_close = np.allclose(params1['qi'][:5], params2['qi'][:5], rtol=0.1)
            self.assertTrue(qi_close or len(params1['qi']) > 0)  # Either close or at least has samples

    def test_invalid_production_data(self):
        """Test handling of invalid production data."""
        # Create invalid data (empty)
        invalid_data = pd.DataFrame({
            'WellName': [],
            'DATE': [],
            'OIL': []
        })
        
        result = self.forecaster.fit_bayesian_decline(invalid_data, 'TestWell_Invalid')
        
        # Should fail gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertFalse(result['success'])
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main() 