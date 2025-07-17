"""
Revenue Calculator Module for Probabilistic Oil Production Forecasting

This module implements revenue calculation by matching production forecasts
with oil strip price data to generate scenario-based revenue projections.

Key Features:
- Match production forecasts with strip price data by date
- Calculate monthly revenue: Production(bbl) × Strip_Price($/bbl)
- Handle price forecast beyond available strip data
- Generate cumulative revenue forecasts
- Validate revenue calculations and provide summary statistics
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
import openpyxl

# Configure logging
logger = logging.getLogger(__name__)


class RevenueCalculationError(Exception):
    """Custom exception for revenue calculation errors."""
    pass


class RevenueCalculator:
    """
    Revenue calculation for oil production forecasting.
    
    This class calculates revenue by matching production forecasts with
    oil strip price data, handling price escalation and missing data.
    """
    
    def __init__(self, 
                 price_escalation_rate: float = 0.02,
                 use_price_escalation: bool = True,
                 validation_enabled: bool = True):
        """
        Initialize revenue calculator.
        
        Args:
            price_escalation_rate: Annual price escalation rate (default: 2%)
            use_price_escalation: Whether to use price escalation beyond strip data
            validation_enabled: Whether to perform validation checks
        """
        self.price_escalation_rate = price_escalation_rate
        self.use_price_escalation = use_price_escalation
        self.validation_enabled = validation_enabled
        
        # Results storage
        self.revenue_forecast: Optional[pd.DataFrame] = None
        self.price_forecast: Optional[pd.DataFrame] = None
        self.revenue_metrics: Dict = {}
        
    def calculate_asset_revenue(self, 
                              production_forecast: pd.DataFrame,
                              price_data: pd.DataFrame,
                              start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate revenue forecast by matching production with price data.
        
        Args:
            production_forecast: Asset production forecast DataFrame
            price_data: Oil strip price DataFrame
            start_date: Start date for forecast period
            
        Returns:
            DataFrame with production and revenue forecasts by scenario
            
        Raises:
            RevenueCalculationError: If revenue calculation fails
        """
        logger.info("Calculating asset revenue forecast")
        
        try:
            # Validate inputs
            self._validate_inputs(production_forecast, price_data)
            
            # Prepare price forecast
            price_forecast = self._prepare_price_forecast(
                production_forecast, price_data, start_date
            )
            
            # Calculate revenue for each scenario
            revenue_forecast = self._calculate_revenue_by_scenario(
                production_forecast, price_forecast
            )
            
            # Add cumulative revenue
            revenue_forecast = self._add_cumulative_revenue(revenue_forecast)
            
            # Validate results
            if self.validation_enabled:
                self._validate_revenue_forecast(revenue_forecast)
            
            # Calculate revenue metrics
            self._calculate_revenue_metrics(revenue_forecast)
            
            self.revenue_forecast = revenue_forecast
            self.price_forecast = price_forecast
            
            logger.info("Revenue calculation completed successfully")
            
            return revenue_forecast
            
        except Exception as e:
            logger.error(f"Revenue calculation failed: {str(e)}")
            raise RevenueCalculationError(f"Failed to calculate revenue: {str(e)}")
    
    def _validate_inputs(self, 
                        production_forecast: pd.DataFrame, 
                        price_data: pd.DataFrame) -> None:
        """Validate input data for revenue calculation."""
        
        # Check production forecast
        if production_forecast.empty:
            raise RevenueCalculationError("Production forecast is empty")
        
        required_prod_cols = ['Date', 'P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl']
        missing_prod_cols = [col for col in required_prod_cols if col not in production_forecast.columns]
        if missing_prod_cols:
            raise RevenueCalculationError(f"Missing production columns: {missing_prod_cols}")
        
        # Check price data
        if price_data.empty:
            raise RevenueCalculationError("Price data is empty")
        
        required_price_cols = ['Date', 'Strip_price_Oil']
        missing_price_cols = [col for col in required_price_cols if col not in price_data.columns]
        if missing_price_cols:
            raise RevenueCalculationError(f"Missing price columns: {missing_price_cols}")
        
        # Check for valid values
        prod_cols = ['P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl']
        for col in prod_cols:
            if (production_forecast[col] < 0).any():
                raise RevenueCalculationError(f"Negative production values found in {col}")
        
        if (price_data['Strip_price_Oil'] <= 0).any():
            raise RevenueCalculationError("Non-positive price values found")
        
        logger.info("Input validation completed")
    
    def _prepare_price_forecast(self, 
                              production_forecast: pd.DataFrame,
                              price_data: pd.DataFrame,
                              start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare price forecast by matching with production forecast dates.
        
        Args:
            production_forecast: Production forecast DataFrame
            price_data: Price data DataFrame
            start_date: Start date for forecast period
            
        Returns:
            Price forecast DataFrame aligned with production forecast
        """
        # Convert date columns to datetime
        price_data = price_data.copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        
        production_dates = pd.to_datetime(production_forecast['Date'])
        
        # Create price forecast DataFrame
        price_forecast = pd.DataFrame({
            'Date': production_dates,
            'Strip_price_Oil': np.nan
        })
        
        # Match available strip prices
        for i, date in enumerate(production_dates):
            # Find closest price date
            price_match = price_data[price_data['Date'] <= date]
            
            if not price_match.empty:
                # Use most recent available price
                latest_price = price_match.iloc[-1]['Strip_price_Oil']
                price_forecast.loc[i, 'Strip_price_Oil'] = latest_price
            else:
                # Use first available price if forecast starts before price data
                first_price = price_data.iloc[0]['Strip_price_Oil']
                price_forecast.loc[i, 'Strip_price_Oil'] = first_price
        
        # Handle missing prices beyond strip data
        if price_forecast['Strip_price_Oil'].isna().any():
            price_forecast = self._handle_missing_prices(price_forecast)
        
        logger.info(f"Price forecast prepared for {len(price_forecast)} months")
        
        return price_forecast
    
    def _handle_missing_prices(self, price_forecast: pd.DataFrame) -> pd.DataFrame:
        """Handle missing prices beyond available strip data."""
        price_forecast = price_forecast.copy()
        
        # Find last available price
        last_valid_idx = price_forecast['Strip_price_Oil'].last_valid_index()
        
        if last_valid_idx is None:
            raise RevenueCalculationError("No valid prices found in strip data")
        
        last_price = price_forecast.loc[last_valid_idx, 'Strip_price_Oil']
        last_date = price_forecast.loc[last_valid_idx, 'Date']
        
        # Fill missing prices
        for i in range(last_valid_idx + 1, len(price_forecast)):
            if pd.isna(price_forecast.loc[i, 'Strip_price_Oil']):
                current_date = price_forecast.loc[i, 'Date']
                
                if self.use_price_escalation:
                    # Apply price escalation
                    years_elapsed = (current_date - last_date).days / 365.25
                    escalated_price = last_price * (1 + self.price_escalation_rate) ** years_elapsed
                    price_forecast.loc[i, 'Strip_price_Oil'] = escalated_price
                else:
                    # Use last available price
                    price_forecast.loc[i, 'Strip_price_Oil'] = last_price
        
        # Forward fill any remaining NaN values
        price_forecast['Strip_price_Oil'] = price_forecast['Strip_price_Oil'].fillna(method='ffill')
        
        logger.info(f"Handled missing prices using {'escalation' if self.use_price_escalation else 'last price'}")
        
        return price_forecast
    
    def _calculate_revenue_by_scenario(self, 
                                     production_forecast: pd.DataFrame,
                                     price_forecast: pd.DataFrame) -> pd.DataFrame:
        """Calculate revenue for each scenario (P10, P50, P90)."""
        
        # Create revenue forecast DataFrame
        revenue_forecast = pd.DataFrame({
            'Date': production_forecast['Date'],
            'P10_Production_bbl': production_forecast['P10_Production_bbl'],
            'P50_Production_bbl': production_forecast['P50_Production_bbl'],
            'P90_Production_bbl': production_forecast['P90_Production_bbl'],
            'Strip_price_Oil': price_forecast['Strip_price_Oil']
        })
        
        # Calculate monthly revenue: Production(bbl) × Price($/bbl)
        revenue_forecast['P10_Revenue_USD'] = (
            revenue_forecast['P10_Production_bbl'] * revenue_forecast['Strip_price_Oil']
        )
        revenue_forecast['P50_Revenue_USD'] = (
            revenue_forecast['P50_Production_bbl'] * revenue_forecast['Strip_price_Oil']
        )
        revenue_forecast['P90_Revenue_USD'] = (
            revenue_forecast['P90_Production_bbl'] * revenue_forecast['Strip_price_Oil']
        )
        
        logger.info("Revenue calculation by scenario completed")
        
        return revenue_forecast
    
    def _add_cumulative_revenue(self, revenue_forecast: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative revenue columns."""
        revenue_forecast = revenue_forecast.copy()
        
        # Calculate cumulative revenue
        revenue_forecast['P10_Cumulative_Revenue_USD'] = revenue_forecast['P10_Revenue_USD'].cumsum()
        revenue_forecast['P50_Cumulative_Revenue_USD'] = revenue_forecast['P50_Revenue_USD'].cumsum()
        revenue_forecast['P90_Cumulative_Revenue_USD'] = revenue_forecast['P90_Revenue_USD'].cumsum()
        
        # Add cumulative production if not already present
        if 'P10_Cumulative_bbl' not in revenue_forecast.columns:
            revenue_forecast['P10_Cumulative_bbl'] = revenue_forecast['P10_Production_bbl'].cumsum()
            revenue_forecast['P50_Cumulative_bbl'] = revenue_forecast['P50_Production_bbl'].cumsum()
            revenue_forecast['P90_Cumulative_bbl'] = revenue_forecast['P90_Production_bbl'].cumsum()
        
        logger.info("Cumulative revenue calculation completed")
        
        return revenue_forecast
    
    def _validate_revenue_forecast(self, revenue_forecast: pd.DataFrame) -> None:
        """Validate the revenue forecast results."""
        
        # Check for negative revenues
        revenue_cols = ['P10_Revenue_USD', 'P50_Revenue_USD', 'P90_Revenue_USD']
        for col in revenue_cols:
            if (revenue_forecast[col] < 0).any():
                raise RevenueCalculationError(f"Negative revenue values found in {col}")
        
        # Check that P10 >= P50 >= P90 for revenue (should follow production)
        # INDUSTRY CONVENTION: P10 = optimistic (high), P50 = median, P90 = conservative (low)
        p10_rev = revenue_forecast['P10_Revenue_USD'].values
        p50_rev = revenue_forecast['P50_Revenue_USD'].values
        p90_rev = revenue_forecast['P90_Revenue_USD'].values
        
        # Allow for small numerical errors
        tolerance = 1e-6
        
        if not np.all(p10_rev >= p50_rev - tolerance):
            logger.warning("P10 revenue not always >= P50 revenue - INDUSTRY CONVENTION VIOLATION")
        
        if not np.all(p50_rev >= p90_rev - tolerance):
            logger.warning("P50 revenue not always >= P90 revenue - INDUSTRY CONVENTION VIOLATION")
        
        # Additional check: P10 should be >= P90 (optimistic >= conservative)
        if not np.all(p10_rev >= p90_rev - tolerance):
            logger.warning("P10 revenue not always >= P90 revenue - SERIOUS ISSUE")
        
        # Log validation results for debugging
        logger.info(f"Revenue validation: P10[0]=${p10_rev[0]:,.0f}, P50[0]=${p50_rev[0]:,.0f}, P90[0]=${p90_rev[0]:,.0f}")
        
        # Check for reasonable revenue values
        max_monthly_rev = revenue_forecast[revenue_cols].max().max()
        
        if max_monthly_rev > 1e9:  # $1 billion per month seems excessive
            logger.warning(f"Very high monthly revenue detected: ${max_monthly_rev:,.0f}")
        
        if max_monthly_rev <= 0:
            raise RevenueCalculationError("All revenue forecasts are zero or negative")
        
        # Check price consistency
        if revenue_forecast['Strip_price_Oil'].isna().any():
            raise RevenueCalculationError("NaN values found in price forecast")
        
        logger.info("Revenue forecast validation completed")
    
    def _calculate_revenue_metrics(self, revenue_forecast: pd.DataFrame) -> None:
        """Calculate revenue metrics and statistics."""
        
        # Total revenue metrics
        total_p10_rev = revenue_forecast['P10_Cumulative_Revenue_USD'].iloc[-1]
        total_p50_rev = revenue_forecast['P50_Cumulative_Revenue_USD'].iloc[-1]
        total_p90_rev = revenue_forecast['P90_Cumulative_Revenue_USD'].iloc[-1]
        
        # Total production metrics
        total_p10_prod = revenue_forecast['P10_Cumulative_bbl'].iloc[-1]
        total_p50_prod = revenue_forecast['P50_Cumulative_bbl'].iloc[-1]
        total_p90_prod = revenue_forecast['P90_Cumulative_bbl'].iloc[-1]
        
        # Average price metrics
        avg_price = revenue_forecast['Strip_price_Oil'].mean()
        min_price = revenue_forecast['Strip_price_Oil'].min()
        max_price = revenue_forecast['Strip_price_Oil'].max()
        
        # Revenue per barrel metrics
        rev_per_bbl_p10 = total_p10_rev / total_p10_prod if total_p10_prod > 0 else 0
        rev_per_bbl_p50 = total_p50_rev / total_p50_prod if total_p50_prod > 0 else 0
        rev_per_bbl_p90 = total_p90_rev / total_p90_prod if total_p90_prod > 0 else 0
        
        # Peak revenue metrics
        peak_p10_rev = revenue_forecast['P10_Revenue_USD'].max()
        peak_p50_rev = revenue_forecast['P50_Revenue_USD'].max()
        peak_p90_rev = revenue_forecast['P90_Revenue_USD'].max()
        
        # Store metrics
        self.revenue_metrics = {
            'total_p10_revenue_usd': total_p10_rev,
            'total_p50_revenue_usd': total_p50_rev,
            'total_p90_revenue_usd': total_p90_rev,
            'total_p10_production_bbl': total_p10_prod,
            'total_p50_production_bbl': total_p50_prod,
            'total_p90_production_bbl': total_p90_prod,
            'average_price_per_bbl': avg_price,
            'min_price_per_bbl': min_price,
            'max_price_per_bbl': max_price,
            'revenue_per_bbl_p10': rev_per_bbl_p10,
            'revenue_per_bbl_p50': rev_per_bbl_p50,
            'revenue_per_bbl_p90': rev_per_bbl_p90,
            'peak_monthly_revenue_p10': peak_p10_rev,
            'peak_monthly_revenue_p50': peak_p50_rev,
            'peak_monthly_revenue_p90': peak_p90_rev,
            'revenue_uncertainty_p10_p90_ratio': total_p10_rev / total_p90_rev if total_p90_rev > 0 else np.inf,
            'calculation_date': datetime.now().isoformat()
        }
        
        logger.info("Revenue metrics calculated")
    
    def get_revenue_summary(self) -> Dict:
        """Get summary of revenue calculation results."""
        if not self.revenue_metrics:
            return {}
        
        summary = {
            'Revenue Summary': {
                'Total P10 Revenue': f"${self.revenue_metrics['total_p10_revenue_usd']:,.0f}",
                'Total P50 Revenue': f"${self.revenue_metrics['total_p50_revenue_usd']:,.0f}",
                'Total P90 Revenue': f"${self.revenue_metrics['total_p90_revenue_usd']:,.0f}",
                'Average Oil Price': f"${self.revenue_metrics['average_price_per_bbl']:.2f}/bbl",
                'Price Range': f"${self.revenue_metrics['min_price_per_bbl']:.2f} - ${self.revenue_metrics['max_price_per_bbl']:.2f}/bbl",
                'Revenue per Barrel (P50)': f"${self.revenue_metrics['revenue_per_bbl_p50']:.2f}/bbl",
                'Peak Monthly Revenue (P50)': f"${self.revenue_metrics['peak_monthly_revenue_p50']:,.0f}",
                'Revenue Uncertainty Ratio (P10/P90)': f"{self.revenue_metrics['revenue_uncertainty_p10_p90_ratio']:.1f}"
            }
        }
        
        return summary
    
    def export_revenue_forecast(self, 
                              output_file: str = "output/asset_revenue_forecast.csv",
                              include_intermediate_columns: bool = False) -> None:
        """
        Export revenue forecast to CSV file in the required format.
        
        Args:
            output_file: Output file path
            include_intermediate_columns: Whether to include intermediate calculation columns
        """
        if self.revenue_forecast is None:
            raise RevenueCalculationError("No revenue forecast available to export")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate cumulative revenue for case study requirement
        revenue_forecast_with_cumulative = self.revenue_forecast.copy()
        revenue_forecast_with_cumulative['P10_Cumulative_Revenue_USD'] = revenue_forecast_with_cumulative['P10_Revenue_USD'].cumsum()
        revenue_forecast_with_cumulative['P50_Cumulative_Revenue_USD'] = revenue_forecast_with_cumulative['P50_Revenue_USD'].cumsum()
        revenue_forecast_with_cumulative['P90_Cumulative_Revenue_USD'] = revenue_forecast_with_cumulative['P90_Revenue_USD'].cumsum()
        
        # Select columns for export
        if include_intermediate_columns:
            # Include all columns
            export_df = revenue_forecast_with_cumulative.copy()
        else:
            # Include only required columns as specified
            export_df = revenue_forecast_with_cumulative[[
                'Date',
                'P10_Production_bbl',
                'P10_Revenue_USD',
                'P10_Cumulative_Revenue_USD',
                'P50_Production_bbl',
                'P50_Revenue_USD',
                'P50_Cumulative_Revenue_USD',
                'P90_Production_bbl',
                'P90_Revenue_USD',
                'P90_Cumulative_Revenue_USD'
            ]].copy()
        
        # Export to CSV (main format)
        export_df.to_csv(output_file, index=False)
        logger.info(f"Revenue forecast exported to {output_file}")
        
        # Export in Excel format with separate sheets as requested
        excel_file = output_path.with_suffix('.xlsx')
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Create separate sheets for each scenario
                for scenario in ['P10', 'P50', 'P90']:
                    scenario_df = export_df[['Date', f'{scenario}_Production_bbl', f'{scenario}_Revenue_USD', f'{scenario}_Cumulative_Revenue_USD']].copy()
                    scenario_df.columns = ['Date', 'Production_bbl', 'Revenue_USD', 'Cumulative_Revenue_USD']
                    scenario_df.to_excel(writer, sheet_name=scenario, index=False)
            
            logger.info(f"Revenue forecast with separate scenario sheets exported to {excel_file}")
        except ImportError:
            logger.warning("openpyxl not available, skipping Excel export with separate sheets")
        
        # Note: Redundant P10/P50/P90 CSV files are now handled by the reporting module cleanup 