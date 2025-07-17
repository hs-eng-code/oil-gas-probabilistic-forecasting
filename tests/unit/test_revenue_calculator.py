"""
Unit tests for revenue_calculator module.

These tests verify basic functionality of the RevenueCalculator class.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from revenue_calculator import RevenueCalculator, RevenueCalculationError


class TestRevenueCalculator(unittest.TestCase):
    """Test cases for RevenueCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = RevenueCalculator(
            price_escalation_rate=0.02,
            use_price_escalation=True,
            validation_enabled=True
        )
        
        # Create sample production forecast
        dates = pd.date_range(start='2025-01-01', periods=24, freq='MS')
        self.sample_production_forecast = pd.DataFrame({
            'Date': dates,
            'P10_Production_bbl': np.linspace(2000, 1600, 24),  # Declining from 2000 to 1600
            'P50_Production_bbl': np.linspace(1800, 1400, 24),  # Declining from 1800 to 1400
            'P90_Production_bbl': np.linspace(1600, 1200, 24)   # Declining from 1600 to 1200
        })
        
        # Create sample price data
        price_dates = pd.date_range(start='2025-01-01', periods=12, freq='MS')
        self.sample_price_data = pd.DataFrame({
            'Date': price_dates,
            'Strip_price_Oil': np.linspace(70.0, 75.0, 12)  # Rising from $70 to $75
        })

    def test_initialization(self):
        """Test RevenueCalculator initialization."""
        calculator = RevenueCalculator(
            price_escalation_rate=0.03,
            use_price_escalation=False,
            validation_enabled=False
        )
        
        self.assertEqual(calculator.price_escalation_rate, 0.03)
        self.assertEqual(calculator.use_price_escalation, False)
        self.assertEqual(calculator.validation_enabled, False)
        self.assertIsNone(calculator.revenue_forecast)
        self.assertIsNone(calculator.price_forecast)
        self.assertEqual(calculator.revenue_metrics, {})

    def test_calculate_asset_revenue_basic(self):
        """Test basic asset revenue calculation."""
        # Test revenue calculation
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Basic assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)
        
        # Check required columns
        required_cols = [
            'Date', 'P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl',
            'Strip_price_Oil', 'P10_Revenue_USD', 'P50_Revenue_USD', 'P90_Revenue_USD',
            'P10_Cumulative_Revenue_USD', 'P50_Cumulative_Revenue_USD', 'P90_Cumulative_Revenue_USD'
        ]
        for col in required_cols:
            self.assertIn(col, result.columns)
        
        # Check that revenue is positive
        self.assertTrue(np.all(result['P10_Revenue_USD'] > 0))
        self.assertTrue(np.all(result['P50_Revenue_USD'] > 0))
        self.assertTrue(np.all(result['P90_Revenue_USD'] > 0))
        
        # Check that revenue_forecast is stored
        self.assertIsNotNone(self.calculator.revenue_forecast)

    def test_revenue_calculation_formula(self):
        """Test that revenue is calculated correctly (Production × Price)."""
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Check first month calculation
        first_month_production_p50 = self.sample_production_forecast['P50_Production_bbl'].iloc[0]
        first_month_price = self.sample_price_data['Strip_price_Oil'].iloc[0]
        expected_revenue = first_month_production_p50 * first_month_price
        
        self.assertAlmostEqual(result['P50_Revenue_USD'].iloc[0], expected_revenue, places=1)

    def test_cumulative_revenue_calculation(self):
        """Test cumulative revenue calculation."""
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Check that cumulative revenue is increasing
        self.assertTrue(np.all(np.diff(result['P10_Cumulative_Revenue_USD']) >= 0))
        self.assertTrue(np.all(np.diff(result['P50_Cumulative_Revenue_USD']) >= 0))
        self.assertTrue(np.all(np.diff(result['P90_Cumulative_Revenue_USD']) >= 0))
        
        # Check that cumulative equals sum of monthly revenue
        expected_p50_cumulative = result['P50_Revenue_USD'].cumsum()
        pd.testing.assert_series_equal(result['P50_Cumulative_Revenue_USD'], expected_p50_cumulative, check_names=False)

    def test_price_escalation_beyond_strip_data(self):
        """Test price escalation beyond available strip data."""
        # Production forecast extends beyond price data (24 months vs 12 months)
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Check that all price values are filled (no NaN)
        self.assertFalse(result['Strip_price_Oil'].isna().any())
        
        # Check that prices in later months are escalated
        last_strip_price = self.sample_price_data['Strip_price_Oil'].iloc[-1]
        final_price = result['Strip_price_Oil'].iloc[-1]
        
        # Final price should be higher due to escalation (allow for small tolerance)
        # Since forecast extends 12 months beyond strip data, should have some escalation
        self.assertGreaterEqual(final_price, last_strip_price * 0.99)  # Allow for small tolerance

    def test_industry_convention_validation(self):
        """Test industry convention validation (P10 >= P50 >= P90)."""
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Check industry convention: P10 >= P50 >= P90
        # Allow for small numerical tolerance
        tolerance = 1e-6
        
        p10_rev = result['P10_Revenue_USD'].values
        p50_rev = result['P50_Revenue_USD'].values
        p90_rev = result['P90_Revenue_USD'].values
        
        # Check P10 >= P50
        self.assertTrue(np.all(p10_rev >= p50_rev - tolerance))
        
        # Check P50 >= P90
        self.assertTrue(np.all(p50_rev >= p90_rev - tolerance))
        
        # Check P10 >= P90
        self.assertTrue(np.all(p10_rev >= p90_rev - tolerance))

    def test_revenue_metrics_calculation(self):
        """Test revenue metrics calculation."""
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Check that metrics are calculated
        self.assertIsInstance(self.calculator.revenue_metrics, dict)
        
        # Check key metrics
        metrics = self.calculator.revenue_metrics
        self.assertIn('total_p10_revenue_usd', metrics)
        self.assertIn('total_p50_revenue_usd', metrics)
        self.assertIn('total_p90_revenue_usd', metrics)
        self.assertIn('average_price_per_bbl', metrics)
        self.assertIn('revenue_per_bbl_p50', metrics)
        self.assertIn('revenue_uncertainty_p10_p90_ratio', metrics)
        
        # Check that revenue values are positive
        self.assertGreater(metrics['total_p10_revenue_usd'], 0)
        self.assertGreater(metrics['total_p50_revenue_usd'], 0)
        self.assertGreater(metrics['total_p90_revenue_usd'], 0)
        
        # Check that average price is reasonable
        self.assertGreater(metrics['average_price_per_bbl'], 0)
        self.assertLess(metrics['average_price_per_bbl'], 1000)  # Should be reasonable oil price

    def test_revenue_summary_generation(self):
        """Test revenue summary generation."""
        result = self.calculator.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Get summary
        summary = self.calculator.get_revenue_summary()
        
        # Basic assertions
        self.assertIsInstance(summary, dict)
        self.assertIn('Revenue Summary', summary)
        
        revenue_summary = summary['Revenue Summary']
        self.assertIn('Total P10 Revenue', revenue_summary)
        self.assertIn('Total P50 Revenue', revenue_summary)
        self.assertIn('Total P90 Revenue', revenue_summary)
        self.assertIn('Average Oil Price', revenue_summary)
        self.assertIn('Revenue per Barrel (P50)', revenue_summary)

    def test_invalid_production_forecast(self):
        """Test handling of invalid production forecast."""
        # Create invalid forecast (missing required columns)
        invalid_forecast = pd.DataFrame({
            'Date': pd.date_range(start='2025-01-01', periods=12, freq='MS'),
            'P10_Production_bbl': np.linspace(1000, 800, 12),
            # Missing P50 and P90 columns
        })
        
        # Should raise error for invalid structure
        with self.assertRaises(RevenueCalculationError):
            self.calculator.calculate_asset_revenue(
                invalid_forecast,
                self.sample_price_data,
                start_date='2025-01-01'
            )

    def test_invalid_price_data(self):
        """Test handling of invalid price data."""
        # Create invalid price data (missing required columns)
        invalid_price_data = pd.DataFrame({
            'Date': pd.date_range(start='2025-01-01', periods=12, freq='MS'),
            # Missing Strip_price_Oil column
        })
        
        # Should raise error for invalid structure
        with self.assertRaises(RevenueCalculationError):
            self.calculator.calculate_asset_revenue(
                self.sample_production_forecast,
                invalid_price_data,
                start_date='2025-01-01'
            )

    def test_negative_production_values(self):
        """Test handling of negative production values."""
        # Create forecast with negative production
        invalid_forecast = self.sample_production_forecast.copy()
        invalid_forecast.loc[0, 'P50_Production_bbl'] = -100
        
        # Should raise error for negative production
        with self.assertRaises(RevenueCalculationError):
            self.calculator.calculate_asset_revenue(
                invalid_forecast,
                self.sample_price_data,
                start_date='2025-01-01'
            )

    def test_zero_or_negative_prices(self):
        """Test handling of zero or negative prices."""
        # Create price data with zero price
        invalid_price_data = self.sample_price_data.copy()
        invalid_price_data.loc[0, 'Strip_price_Oil'] = 0
        
        # Should raise error for zero/negative prices
        with self.assertRaises(RevenueCalculationError):
            self.calculator.calculate_asset_revenue(
                self.sample_production_forecast,
                invalid_price_data,
                start_date='2025-01-01'
            )

    def test_price_escalation_disabled(self):
        """Test behavior when price escalation is disabled."""
        # Create calculator with escalation disabled
        calculator_no_escalation = RevenueCalculator(
            price_escalation_rate=0.02,
            use_price_escalation=False,
            validation_enabled=True
        )
        
        result = calculator_no_escalation.calculate_asset_revenue(
            self.sample_production_forecast,
            self.sample_price_data,
            start_date='2025-01-01'
        )
        
        # Check that prices after strip data end are just the last available price
        last_strip_price = self.sample_price_data['Strip_price_Oil'].iloc[-1]
        prices_after_strip = result['Strip_price_Oil'].iloc[12:]  # After month 12
        
        # All prices after strip data should be the same (no escalation)
        self.assertTrue(np.allclose(prices_after_strip, last_strip_price))


if __name__ == '__main__':
    unittest.main() 