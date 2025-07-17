"""
Data loading and validation module for probabilistic oil production forecasting.

This module handles:
- CSV data loading with robust error handling
- Comprehensive data validation and quality checks
- Well-specific validation (minimum decline periods)
- Data integrity and consistency checks
- Logging and diagnostic reporting
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class WellProductionDataLoader:
    """
    Comprehensive data loader for oil production forecasting with extensive validation.
    
    This class handles loading and validation of:
    - Well production data
    - Oil price strip data
    - Cross-validation between datasets
    """
    
    def __init__(self, min_decline_months: int = 6):
        """
        Initialize the data loader.
        
        Args:
            min_decline_months: Minimum months of declining production required per well
        """
        self.min_decline_months = min_decline_months
        self.well_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None
        self.validation_report: Dict = {}
        
    def load_well_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate well production data.
        
        Args:
            file_path: Path to well production CSV file
            
        Returns:
            Validated DataFrame with well production data
            
        Raises:
            DataValidationError: If data validation fails
        """
        logger.info(f"Loading well production data from {file_path}")
        
        try:
            # Load CSV with error handling
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from well production file")
            
            # Validate basic structure
            self._validate_well_data_structure(df)
            
            # Clean and standardize data
            df = self._clean_well_data(df)
            
            # Comprehensive validation
            self._validate_well_data_quality(df)
            
            # Well-specific validation
            self._validate_well_decline_requirements(df)
            
            self.well_data = df
            logger.info(f"Well data validation completed successfully")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load well production data: {str(e)}")
            raise DataValidationError(f"Well data loading failed: {str(e)}")
    
    def load_price_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate oil price strip data.
        
        Args:
            file_path: Path to price data CSV file
            
        Returns:
            Validated DataFrame with price data
            
        Raises:
            DataValidationError: If data validation fails
        """
        logger.info(f"Loading price data from {file_path}")
        
        try:
            # Load CSV with error handling
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from price file")
            
            # Validate basic structure
            self._validate_price_data_structure(df)
            
            # Clean and standardize data
            df = self._clean_price_data(df)
            
            # Comprehensive validation
            self._validate_price_data_quality(df)
            
            self.price_data = df
            logger.info(f"Price data validation completed successfully")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load price data: {str(e)}")
            raise DataValidationError(f"Price data loading failed: {str(e)}")
    
    def _validate_well_data_structure(self, df: pd.DataFrame) -> None:
        """Validate well data has required columns and structure."""
        required_columns = ['WellName', 'DATE', 'OIL']
        
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        
        logger.info("Well data structure validation passed")
    
    def _validate_price_data_structure(self, df: pd.DataFrame) -> None:
        """Validate price data has required columns and structure."""
        required_columns = ['Date', 'Strip_price_Oil']
        
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        
        logger.info("Price data structure validation passed")
    
    def _clean_well_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize well production data."""
        df = df.copy()
        
        # Convert date column to datetime
        try:
            df['DATE'] = pd.to_datetime(df['DATE'])
        except Exception as e:
            raise DataValidationError(f"Failed to parse dates in well data: {str(e)}")
        
        # Ensure production values are numeric
        try:
            df['OIL'] = pd.to_numeric(df['OIL'], errors='coerce')
        except Exception as e:
            raise DataValidationError(f"Failed to convert oil production to numeric: {str(e)}")
        
        # Clean well names
        df['WellName'] = df['WellName'].astype(str).str.strip()
        
        # Sort by well and date
        df = df.sort_values(['WellName', 'DATE'])
        
        logger.info("Well data cleaning completed")
        return df
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize price data."""
        df = df.copy()
        
        # Convert date column to datetime
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            raise DataValidationError(f"Failed to parse dates in price data: {str(e)}")
        
        # Ensure price values are numeric
        try:
            df['Strip_price_Oil'] = pd.to_numeric(df['Strip_price_Oil'], errors='coerce')
        except Exception as e:
            raise DataValidationError(f"Failed to convert prices to numeric: {str(e)}")
        
        # Sort by date
        df = df.sort_values('Date')
        
        logger.info("Price data cleaning completed")
        return df
    
    def _validate_well_data_quality(self, df: pd.DataFrame) -> None:
        """Comprehensive quality validation for well data."""
        validation_results = {}
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            validation_results['missing_data'] = missing_data.to_dict()
            logger.warning(f"Missing values found: {missing_data.to_dict()}")
        
        # Check for negative production values
        negative_production = (df['OIL'] < 0).sum()
        if negative_production > 0:
            validation_results['negative_production'] = negative_production
            logger.warning(f"Found {negative_production} negative production values")
        
        # Check for unrealistic production values (>10,000 bbl/month per well)
        high_production = (df['OIL'] > 10000).sum()
        if high_production > 0:
            validation_results['high_production'] = high_production
            logger.warning(f"Found {high_production} unusually high production values (>10,000 bbl/month)")
        
        # Check date range
        date_range = df['DATE'].max() - df['DATE'].min()
        validation_results['date_range_days'] = date_range.days
        logger.info(f"Well data spans {date_range.days} days")
        
        # Check number of unique wells
        unique_wells = df['WellName'].nunique()
        validation_results['unique_wells'] = unique_wells
        logger.info(f"Found {unique_wells} unique wells")
        
        # Check for wells with single data point
        wells_single_point = df.groupby('WellName').size()
        single_point_wells = (wells_single_point == 1).sum()
        if single_point_wells > 0:
            validation_results['single_point_wells'] = single_point_wells
            logger.warning(f"Found {single_point_wells} wells with only one data point")
        
        self.validation_report['well_data'] = validation_results
        logger.info("Well data quality validation completed")
    
    def _validate_price_data_quality(self, df: pd.DataFrame) -> None:
        """Comprehensive quality validation for price data."""
        validation_results = {}
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            validation_results['missing_data'] = missing_data.to_dict()
            logger.warning(f"Missing values found: {missing_data.to_dict()}")
        
        # Check for negative or zero prices
        invalid_prices = (df['Strip_price_Oil'] <= 0).sum()
        if invalid_prices > 0:
            validation_results['invalid_prices'] = invalid_prices
            logger.warning(f"Found {invalid_prices} invalid price values (<=0)")
        
        # Check for unrealistic price values
        high_prices = (df['Strip_price_Oil'] > 200).sum()
        if high_prices > 0:
            validation_results['high_prices'] = high_prices
            logger.warning(f"Found {high_prices} unusually high price values (>$200/bbl)")
        
        # Check date range
        date_range = df['Date'].max() - df['Date'].min()
        validation_results['date_range_days'] = date_range.days
        logger.info(f"Price data spans {date_range.days} days")
        
        # Check for duplicate dates
        duplicate_dates = df.duplicated(subset=['Date']).sum()
        if duplicate_dates > 0:
            validation_results['duplicate_dates'] = duplicate_dates
            logger.warning(f"Found {duplicate_dates} duplicate date entries")
        
        self.validation_report['price_data'] = validation_results
        logger.info("Price data quality validation completed")
    
    def _validate_well_decline_requirements(self, df: pd.DataFrame) -> None:
        """Validate wells meet minimum decline period requirements."""
        validation_results = {}
        
        wells_with_sufficient_data = []
        wells_with_insufficient_data = []
        wells_without_decline = []
        
        for well_name in df['WellName'].unique():
            well_data = df[df['WellName'] == well_name].copy()
            well_data = well_data.sort_values('DATE')
            
            # Check if well has minimum months of data
            if len(well_data) < self.min_decline_months:
                wells_with_insufficient_data.append(well_name)
                continue
            
            # Check for declining production trend
            if self._has_declining_trend(well_data):
                wells_with_sufficient_data.append(well_name)
            else:
                wells_without_decline.append(well_name)
        
        validation_results['wells_with_sufficient_data'] = len(wells_with_sufficient_data)
        validation_results['wells_with_insufficient_data'] = len(wells_with_insufficient_data)
        validation_results['wells_without_decline'] = len(wells_without_decline)
        
        logger.info(f"Wells with sufficient declining data: {len(wells_with_sufficient_data)}")
        logger.info(f"Wells with insufficient data: {len(wells_with_insufficient_data)}")
        logger.info(f"Wells without clear decline: {len(wells_without_decline)}")
        
        # Store detailed results
        self.validation_report['decline_validation'] = validation_results
        self.validation_report['wells_sufficient_data'] = wells_with_sufficient_data
        self.validation_report['wells_insufficient_data'] = wells_with_insufficient_data
        self.validation_report['wells_without_decline'] = wells_without_decline
        
        # Warning if too many wells have insufficient data
        insufficient_pct = len(wells_with_insufficient_data) / len(df['WellName'].unique()) * 100
        if insufficient_pct > 20:
            logger.warning(f"{insufficient_pct:.1f}% of wells have insufficient data for decline analysis")
    
    def _has_declining_trend(self, well_data: pd.DataFrame) -> bool:
        """Check if well shows declining production trend."""
        # Simple trend analysis using linear regression
        if len(well_data) < 3:
            return False
        
        # Create time index
        well_data = well_data.copy()
        well_data['time_index'] = range(len(well_data))
        
        # Calculate correlation between time and production
        # Negative correlation indicates decline
        correlation = well_data['time_index'].corr(well_data['OIL'])
        
        # Consider declining if correlation is negative and significant
        # Also check if the last 3 months show decline
        if len(well_data) >= 3:
            recent_trend = well_data['OIL'].iloc[-3:].is_monotonic_decreasing
            return correlation < -0.3 or recent_trend
        
        return correlation < -0.3
    
    def cross_validate_datasets(self) -> Dict:
        """Cross-validate well and price data for consistency."""
        if self.well_data is None or self.price_data is None:
            raise DataValidationError("Both datasets must be loaded before cross-validation")
        
        cross_validation_results = {}
        
        # Check date overlap
        well_date_range = (self.well_data['DATE'].min(), self.well_data['DATE'].max())
        price_date_range = (self.price_data['Date'].min(), self.price_data['Date'].max())
        
        # Check if there's sufficient overlap or if price data is for future forecasting
        overlap_start = max(well_date_range[0], price_date_range[0])
        overlap_end = min(well_date_range[1], price_date_range[1])
        
        if overlap_start >= overlap_end:
            # Check if this is a forecasting scenario (price data starts after well data ends)
            if price_date_range[0] > well_date_range[1]:
                logger.info("Forecasting scenario detected: Price data starts after well data ends")
                # This is acceptable for forecasting scenarios
                overlap_days = 0
            else:
                raise DataValidationError("No date overlap between well and price data")
        else:
            overlap_days = (overlap_end - overlap_start).days
        
        cross_validation_results['overlap_days'] = overlap_days
        if overlap_days > 0:
            cross_validation_results['overlap_start'] = overlap_start
            cross_validation_results['overlap_end'] = overlap_end
        else:
            cross_validation_results['overlap_start'] = None
            cross_validation_results['overlap_end'] = None
        
        if overlap_days > 0:
            logger.info(f"Data overlap period: {overlap_days} days ({overlap_start} to {overlap_end})")
        else:
            logger.info(f"Forecasting scenario: No historical overlap, price data starts after well data ends")
        
        # Check for future price data availability (for forecasting)
        latest_well_date = self.well_data['DATE'].max()
        future_price_data = self.price_data[self.price_data['Date'] > latest_well_date]
        
        cross_validation_results['future_price_months'] = len(future_price_data)
        logger.info(f"Future price data available: {len(future_price_data)} months")
        
        self.validation_report['cross_validation'] = cross_validation_results
        
        return cross_validation_results
    
    def get_validation_summary(self) -> Dict:
        """Get comprehensive validation summary."""
        if not self.validation_report:
            logger.warning("No validation report available. Run load methods first.")
            return {}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'validation_report': self.validation_report,
            'data_loaded': {
                'well_data': self.well_data is not None,
                'price_data': self.price_data is not None
            }
        }
        
        if self.well_data is not None:
            summary['well_data_shape'] = self.well_data.shape
            summary['unique_wells'] = self.well_data['WellName'].nunique()
        
        if self.price_data is not None:
            summary['price_data_shape'] = self.price_data.shape
        
        return summary
    
    def export_validation_report(self, output_path: str) -> None:
        """Export validation report to JSON file."""
        import json
        
        summary = self.get_validation_summary()
        
        # Convert non-serializable objects to strings
        def convert_for_json(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            return obj
        
        # Deep convert the summary
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_for_json(obj)
        
        summary_json = deep_convert(summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        logger.info(f"Validation report exported to {output_path}")


def load_and_validate_data(
    well_data_path: str,
    price_data_path: str,
    min_decline_months: int = 6,
    export_validation: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convenience function to load and validate both datasets.
    
    Args:
        well_data_path: Path to well production data
        price_data_path: Path to price data
        min_decline_months: Minimum months of decline required
        export_validation: Whether to export validation report
        
    Returns:
        Tuple of (well_data, price_data, validation_summary)
    """
    logger.info("Starting comprehensive data loading and validation")
    
    # Initialize loader
    loader = WellProductionDataLoader(min_decline_months=min_decline_months)
    
    # Load datasets
    well_data = loader.load_well_data(well_data_path)
    price_data = loader.load_price_data(price_data_path)
    
    # Cross-validate
    loader.cross_validate_datasets()
    
    # Get validation summary
    validation_summary = loader.get_validation_summary()
    
    # Export validation report if requested
    if export_validation:
        # Calculate correct path relative to project root
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "output", "data_validation_report.json")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        loader.export_validation_report(output_path)
    
    logger.info("Data loading and validation completed successfully")
    
    return well_data, price_data, validation_summary


if __name__ == "__main__":
    # Example usage
    try:
        well_data, price_data, validation = load_and_validate_data(
            "../data/QCG_DS_Exercise_well_prod_data.csv",
            "../data/QCG_DS_Exercise_price_data.csv"
        )
        
        print("Data loading completed successfully!")
        print(f"Well data shape: {well_data.shape}")
        print(f"Price data shape: {price_data.shape}")
        print(f"Unique wells: {well_data['WellName'].nunique()}")
        
    except DataValidationError as e:
        print(f"Data validation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}") 