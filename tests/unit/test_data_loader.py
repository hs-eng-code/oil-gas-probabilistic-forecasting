"""
Unit tests for data_loader module.

These tests verify basic functionality of the WellProductionDataLoader class.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_loader import WellProductionDataLoader, DataValidationError


class TestDataLoader(unittest.TestCase):
    """Test cases for WellProductionDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = WellProductionDataLoader(min_decline_months=6)
        
        # Create sample well data
        self.sample_well_data = pd.DataFrame({
            'WellName': ['Well_A', 'Well_A', 'Well_A', 'Well_B', 'Well_B', 'Well_B'],
            'DATE': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-01-01', '2023-02-01', '2023-03-01'],
            'OIL': [1000, 900, 800, 1200, 1100, 1000]
        })
        
        # Create sample price data
        self.sample_price_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'Strip_price_Oil': [70.0, 72.0, 75.0]
        })

    def test_initialization(self):
        """Test WellProductionDataLoader initialization."""
        loader = WellProductionDataLoader(min_decline_months=10)
        self.assertEqual(loader.min_decline_months, 10)
        self.assertIsNone(loader.well_data)
        self.assertIsNone(loader.price_data)
        self.assertEqual(loader.validation_report, {})

    def test_load_well_data_from_dataframe(self):
        """Test loading well data from existing DataFrame."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            self.sample_well_data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        try:
            # Test loading
            result = self.loader.load_well_data(tmp_file_path)
            
            # Basic assertions
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 6)
            self.assertIn('WellName', result.columns)
            self.assertIn('DATE', result.columns)
            self.assertIn('OIL', result.columns)
            
            # Test that well_data is stored
            self.assertIsNotNone(self.loader.well_data)
            
        finally:
            # Clean up
            os.unlink(tmp_file_path)

    def test_load_price_data_from_dataframe(self):
        """Test loading price data from existing DataFrame."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            self.sample_price_data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        try:
            # Test loading
            result = self.loader.load_price_data(tmp_file_path)
            
            # Basic assertions
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 3)
            self.assertIn('Date', result.columns)
            self.assertIn('Strip_price_Oil', result.columns)
            
            # Test that price_data is stored
            self.assertIsNotNone(self.loader.price_data)
            
        finally:
            # Clean up
            os.unlink(tmp_file_path)

    def test_validation_with_invalid_data(self):
        """Test validation with invalid data structures."""
        # Test with missing columns
        invalid_well_data = pd.DataFrame({
            'WellName': ['Well_A'],
            'DATE': ['2023-01-01']
            # Missing OIL column
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            invalid_well_data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        try:
            # Should raise DataValidationError
            with self.assertRaises(DataValidationError):
                self.loader.load_well_data(tmp_file_path)
                
        finally:
            os.unlink(tmp_file_path)

    def test_cross_validation_with_both_datasets(self):
        """Test cross-validation when both datasets are loaded."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_well:
            self.sample_well_data.to_csv(tmp_well.name, index=False)
            well_path = tmp_well.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_price:
            self.sample_price_data.to_csv(tmp_price.name, index=False)
            price_path = tmp_price.name
        
        try:
            # Load both datasets
            self.loader.load_well_data(well_path)
            self.loader.load_price_data(price_path)
            
            # Test cross-validation
            cross_val_result = self.loader.cross_validate_datasets()
            
            # Basic assertions
            self.assertIsInstance(cross_val_result, dict)
            self.assertIn('overlap_days', cross_val_result)
            
        finally:
            # Clean up
            os.unlink(well_path)
            os.unlink(price_path)

    def test_validation_report_generation(self):
        """Test that validation report is generated properly."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_well:
            self.sample_well_data.to_csv(tmp_well.name, index=False)
            well_path = tmp_well.name
            
        try:
            # Load data
            self.loader.load_well_data(well_path)
            
            # Get validation summary
            summary = self.loader.get_validation_summary()
            
            # Basic assertions
            self.assertIsInstance(summary, dict)
            self.assertIn('validation_report', summary)
            self.assertIn('data_loaded', summary)
            
        finally:
            # Clean up
            os.unlink(well_path)


if __name__ == '__main__':
    unittest.main() 