"""
Integration test for the full oil production forecasting pipeline.

This test verifies that the complete workflow from data loading to revenue calculation
works correctly using actual data files and existing APIs.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Local imports
from data_loader import WellProductionDataLoader
from arps_dca import AdvancedArpsDCA
from bayesian_forecaster import ModernizedBayesianForecaster
from aggregator import AssetAggregator
from revenue_calculator import RevenueCalculator


class TestFullPipeline(unittest.TestCase):
    """Integration test for the complete forecasting pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Path to actual data files
        self.project_root = Path(__file__).parent.parent.parent
        self.well_data_path = self.project_root / "data" / "QCG_DS_Exercise_well_prod_data.csv"
        self.price_data_path = self.project_root / "data" / "QCG_DS_Exercise_price_data.csv"
        
        # Test configuration
        self.test_forecast_months = 60  # 5 years for testing (instead of 360)
        self.test_sample_wells = 3      # Test with 3 wells only (instead of 374)
        
        # Initialize pipeline components
        self.data_loader = WellProductionDataLoader(min_decline_months=6)
        self.arps_dca = AdvancedArpsDCA(
            terminal_decline_rate=0.05,
            min_production_months=6,
            max_forecast_years=30
        )
        self.bayesian_forecaster = ModernizedBayesianForecaster(
            n_samples=100,  # Small for testing
            arps_dca_instance=self.arps_dca,
            random_seed=42  # For reproducible tests
        )
        self.aggregator = AssetAggregator(
            forecast_months=self.test_forecast_months,
            validation_enabled=True
        )
        self.revenue_calculator = RevenueCalculator(
            price_escalation_rate=0.02,
            use_price_escalation=True,
            validation_enabled=True
        )

    def test_data_files_exist(self):
        """Test that required data files exist."""
        self.assertTrue(self.well_data_path.exists(), f"Well data file not found: {self.well_data_path}")
        self.assertTrue(self.price_data_path.exists(), f"Price data file not found: {self.price_data_path}")

    def test_phase_1_data_loading(self):
        """Test Phase 1: Data Loading and Validation."""
        # Load well data
        well_data = self.data_loader.load_well_data(str(self.well_data_path))
        
        # Basic assertions
        self.assertIsInstance(well_data, pd.DataFrame)
        self.assertGreater(len(well_data), 0)
        self.assertIn('WellName', well_data.columns)
        self.assertIn('DATE', well_data.columns)
        self.assertIn('OIL', well_data.columns)
        
        # Load price data
        price_data = self.data_loader.load_price_data(str(self.price_data_path))
        
        # Basic assertions
        self.assertIsInstance(price_data, pd.DataFrame)
        self.assertGreater(len(price_data), 0)
        self.assertIn('Date', price_data.columns)
        self.assertIn('Strip_price_Oil', price_data.columns)
        
        # Cross-validate datasets
        cross_val_result = self.data_loader.cross_validate_datasets()
        self.assertIsInstance(cross_val_result, dict)
        
        # Store data for next phases
        self.well_data = well_data
        self.price_data = price_data
        
        print(f"Phase 1 complete: Loaded {len(well_data)} well records and {len(price_data)} price records")

    def test_phase_2_arps_dca_processing(self):
        """Test Phase 2: Arps DCA Processing."""
        # First run Phase 1
        self.test_phase_1_data_loading()
        
        # Get sample wells for testing
        sample_wells = self.well_data['WellName'].unique()[:self.test_sample_wells]
        
        # Test DCA fitting for sample wells
        successful_fits = 0
        arps_results = {}
        
        for well_name in sample_wells:
            result = self.arps_dca.fit_decline_curve(self.well_data, well_name)
            arps_results[well_name] = result
            
            # Basic assertions
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            
            if result['success']:
                successful_fits += 1
                # Check that parameters are present
                self.assertIn('qi', result)
                self.assertIn('Di', result)
                self.assertIn('b', result)
                self.assertIn('method', result)
                
                # Check that parameters are reasonable
                self.assertGreater(result['qi'], 0)
                self.assertGreater(result['Di'], 0)
                self.assertGreaterEqual(result['b'], 0)
        
        # Store results for next phases
        self.arps_results = arps_results
        self.sample_wells = sample_wells
        
        print(f"Phase 2 complete: Successfully fitted {successful_fits}/{len(sample_wells)} wells")

    def test_phase_3_bayesian_forecasting(self):
        """Test Phase 3: Bayesian Forecasting."""
        # First run Phase 2
        self.test_phase_2_arps_dca_processing()
        
        # Test Bayesian forecasting for sample wells
        successful_forecasts = 0
        bayesian_results = {}
        
        for well_name in self.sample_wells:
            if self.arps_results[well_name]['success']:
                # Test Bayesian fitting
                result = self.bayesian_forecaster.fit_bayesian_decline(self.well_data, well_name)
                bayesian_results[well_name] = result
                
                # Basic assertions
                self.assertIsInstance(result, dict)
                self.assertIn('success', result)
                
                if result['success']:
                    successful_forecasts += 1
                    # Check that key components are present
                    self.assertIn('parameter_samples', result)
                    self.assertIn('quality_assessment', result)
                    
                    # Check parameter samples
                    parameter_samples = result['parameter_samples']
                    self.assertIn('qi', parameter_samples)
                    self.assertIn('Di', parameter_samples)
                    self.assertIn('b', parameter_samples)
                    
                    # Check sample sizes
                    self.assertEqual(len(parameter_samples['qi']), 100)
                    self.assertEqual(len(parameter_samples['Di']), 100)
                    self.assertEqual(len(parameter_samples['b']), 100)
        
        # Store results for next phases
        self.bayesian_results = bayesian_results
        
        print(f"Phase 3 complete: Successfully generated Bayesian forecasts for {successful_forecasts}/{len(self.sample_wells)} wells")

    def test_phase_4_asset_aggregation(self):
        """Test Phase 4: Asset Aggregation."""
        # First run Phase 3
        self.test_phase_3_bayesian_forecasting()
        
        # Generate probabilistic forecasts for successful wells
        well_forecasts = {}
        successful_probabilistic_forecasts = 0
        
        for well_name in self.sample_wells:
            if (well_name in self.bayesian_results and 
                self.bayesian_results[well_name]['success']):
                
                # Generate probabilistic forecast
                forecast_result = self.bayesian_forecaster.forecast_probabilistic(
                    well_name, 
                    forecast_months=self.test_forecast_months,
                    percentiles=[0.9, 0.5, 0.1]  # P10, P50, P90
                )
                
                if forecast_result['success']:
                    successful_probabilistic_forecasts += 1
                    
                    # Convert to expected format for aggregation
                    forecast_percentiles = forecast_result['forecast_percentiles']
                    dates = pd.date_range(start='2025-01-01', periods=self.test_forecast_months, freq='MS')
                    
                    well_forecast_df = pd.DataFrame({
                        'Date': dates,
                        'P10_Production_bbl': forecast_percentiles['P10'],
                        'P50_Production_bbl': forecast_percentiles['P50'],
                        'P90_Production_bbl': forecast_percentiles['P90']
                    })
                    
                    well_forecasts[well_name] = well_forecast_df
        
        # Test aggregation
        if well_forecasts:
            asset_forecast = self.aggregator.aggregate_well_forecasts(
                well_forecasts,
                start_date='2025-01-01'
            )
            
            # Basic assertions
            self.assertIsInstance(asset_forecast, pd.DataFrame)
            self.assertEqual(len(asset_forecast), self.test_forecast_months)
            
            # Check required columns
            required_cols = [
                'Date', 'P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl',
                'P10_Cumulative_bbl', 'P50_Cumulative_bbl', 'P90_Cumulative_bbl'
            ]
            for col in required_cols:
                self.assertIn(col, asset_forecast.columns)
            
            # Check that production values are positive
            self.assertTrue(np.all(asset_forecast['P10_Production_bbl'] > 0))
            self.assertTrue(np.all(asset_forecast['P50_Production_bbl'] > 0))
            self.assertTrue(np.all(asset_forecast['P90_Production_bbl'] > 0))
            
            # Store for next phase
            self.asset_forecast = asset_forecast
            
            print(f"Phase 4 complete: Aggregated {len(well_forecasts)} well forecasts into asset-level production")
        else:
            self.skipTest("No successful probabilistic forecasts available for aggregation")

    def test_phase_5_revenue_calculation(self):
        """Test Phase 5: Revenue Calculation."""
        # First run Phase 4
        self.test_phase_4_asset_aggregation()
        
        # Test revenue calculation
        revenue_forecast = self.revenue_calculator.calculate_asset_revenue(
            self.asset_forecast,
            self.price_data,
            start_date='2025-01-01'
        )
        
        # Basic assertions
        self.assertIsInstance(revenue_forecast, pd.DataFrame)
        self.assertEqual(len(revenue_forecast), self.test_forecast_months)
        
        # Check required columns
        required_cols = [
            'Date', 'P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl',
            'Strip_price_Oil', 'P10_Revenue_USD', 'P50_Revenue_USD', 'P90_Revenue_USD',
            'P10_Cumulative_Revenue_USD', 'P50_Cumulative_Revenue_USD', 'P90_Cumulative_Revenue_USD'
        ]
        for col in required_cols:
            self.assertIn(col, revenue_forecast.columns)
        
        # Check that revenue values are positive
        self.assertTrue(np.all(revenue_forecast['P10_Revenue_USD'] > 0))
        self.assertTrue(np.all(revenue_forecast['P50_Revenue_USD'] > 0))
        self.assertTrue(np.all(revenue_forecast['P90_Revenue_USD'] > 0))
        
        # Check that cumulative revenue is increasing
        self.assertTrue(np.all(np.diff(revenue_forecast['P10_Cumulative_Revenue_USD']) >= 0))
        self.assertTrue(np.all(np.diff(revenue_forecast['P50_Cumulative_Revenue_USD']) >= 0))
        self.assertTrue(np.all(np.diff(revenue_forecast['P90_Cumulative_Revenue_USD']) >= 0))
        
        # Check industry convention: P10 >= P50 >= P90
        tolerance = 1e-6
        p10_rev = revenue_forecast['P10_Revenue_USD'].values
        p50_rev = revenue_forecast['P50_Revenue_USD'].values
        p90_rev = revenue_forecast['P90_Revenue_USD'].values
        
        self.assertTrue(np.all(p10_rev >= p50_rev - tolerance))
        self.assertTrue(np.all(p50_rev >= p90_rev - tolerance))
        
        # Store final result
        self.revenue_forecast = revenue_forecast
        
        print(f"Phase 5 complete: Generated revenue forecast with P50 total revenue of ${revenue_forecast['P50_Cumulative_Revenue_USD'].iloc[-1]:,.0f}")

    def test_complete_pipeline_integration(self):
        """Test the complete pipeline integration."""
        # Run all phases in sequence
        self.test_phase_1_data_loading()
        self.test_phase_2_arps_dca_processing()
        self.test_phase_3_bayesian_forecasting()
        self.test_phase_4_asset_aggregation()
        self.test_phase_5_revenue_calculation()
        
        # Final validation
        self.assertIsNotNone(self.revenue_forecast)
        
        # Check that we have meaningful results
        final_p50_revenue = self.revenue_forecast['P50_Cumulative_Revenue_USD'].iloc[-1]
        final_p10_revenue = self.revenue_forecast['P10_Cumulative_Revenue_USD'].iloc[-1]
        final_p90_revenue = self.revenue_forecast['P90_Cumulative_Revenue_USD'].iloc[-1]
        
        # Revenue should be positive and reasonable
        self.assertGreater(final_p50_revenue, 0)
        self.assertGreater(final_p10_revenue, 0)
        self.assertGreater(final_p90_revenue, 0)
        
        # P10 should be greater than P90 (industry convention)
        self.assertGreater(final_p10_revenue, final_p90_revenue)
        
        # Check aggregation metrics
        aggregation_metrics = self.aggregator.aggregation_metrics
        self.assertIsInstance(aggregation_metrics, dict)
        self.assertIn('well_count', aggregation_metrics)
        self.assertIn('asset_p50_eur_bbl', aggregation_metrics)
        
        # Check revenue metrics
        revenue_metrics = self.revenue_calculator.revenue_metrics
        self.assertIsInstance(revenue_metrics, dict)
        self.assertIn('total_p50_revenue_usd', revenue_metrics)
        self.assertIn('average_price_per_bbl', revenue_metrics)
        
        print(f"Complete pipeline integration test passed:")
        print(f"  - Wells processed: {len(self.sample_wells)}")
        print(f"  - Forecast period: {self.test_forecast_months} months")
        print(f"  - Final P50 revenue: ${final_p50_revenue:,.0f}")
        print(f"  - Final P10 revenue: ${final_p10_revenue:,.0f}")
        print(f"  - Final P90 revenue: ${final_p90_revenue:,.0f}")
        print(f"  - P10/P90 ratio: {final_p10_revenue/final_p90_revenue:.2f}")

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid inputs."""
        # Test with invalid well data path
        with self.assertRaises(Exception):
            self.data_loader.load_well_data("nonexistent_file.csv")
        
        # Test with invalid price data path
        with self.assertRaises(Exception):
            self.data_loader.load_price_data("nonexistent_file.csv")
        
        # Test DCA with invalid well name
        self.test_phase_1_data_loading()
        result = self.arps_dca.fit_decline_curve(self.well_data, 'NonExistentWell')
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
        print("Pipeline error handling tests passed")

    def test_data_quality_validation(self):
        """Test data quality validation."""
        # Load data first
        self.test_phase_1_data_loading()
        
        # Check well data quality
        well_validation = self.data_loader.validation_report.get('well_data', {})
        self.assertIsInstance(well_validation, dict)
        
        # Check price data quality
        price_validation = self.data_loader.validation_report.get('price_data', {})
        self.assertIsInstance(price_validation, dict)
        
        # Check cross-validation
        cross_validation = self.data_loader.validation_report.get('cross_validation', {})
        self.assertIsInstance(cross_validation, dict)
        
        print("Data quality validation tests passed")


if __name__ == '__main__':
    unittest.main(verbosity=2) 