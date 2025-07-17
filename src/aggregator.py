"""
Asset Aggregation Module for Probabilistic Oil Production Forecasting

This module implements asset-level aggregation of individual well forecasts
to create portfolio-level P10, P50, and P90 production scenarios.

Key Features:
- Simple summation approach for each scenario separately
- Handles correlation between wells through scenario-based aggregation
- Validates aggregated production trends
- Provides summary statistics and validation checks
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

# Configure logging
logger = logging.getLogger(__name__)


class AssetAggregationError(Exception):
    """Custom exception for asset aggregation errors."""
    pass


class AssetAggregator:
    """
    Asset-level aggregation of individual well probabilistic forecasts.
    
    This class implements portfolio-level aggregation using simple summation
    for each scenario (P10, P50, P90) separately, which implicitly handles
    correlation between wells by maintaining scenario consistency.
    """
    
    def __init__(self, 
                 forecast_months: int = 360,
                 validation_enabled: bool = True):
        """
        Initialize asset aggregator.
        
        Args:
            forecast_months: Number of months to forecast (default: 30 years)
            validation_enabled: Whether to perform validation checks
        """
        self.forecast_months = forecast_months
        self.validation_enabled = validation_enabled
        
        # Results storage
        self.asset_production_forecast: Optional[pd.DataFrame] = None
        self.well_contributions: Dict[str, pd.DataFrame] = {}
        self.aggregation_metrics: Dict = {}
        
    def aggregate_well_forecasts(self, 
                               well_forecasts: Dict[str, pd.DataFrame],
                               start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Aggregate individual well forecasts into asset-level production scenarios.
        
        Args:
            well_forecasts: Dictionary mapping well names to forecast DataFrames
            start_date: Start date for forecast period (default: 2025-01-01)
            
        Returns:
            DataFrame with asset-level P10, P50, P90 production forecasts
            
        Raises:
            AssetAggregationError: If aggregation fails
        """
        logger.info(f"Aggregating forecasts for {len(well_forecasts)} wells")
        
        try:
            # Validate inputs
            self._validate_well_forecasts(well_forecasts)
            
            # Create time index
            if start_date is None:
                start_date = "2025-01-01"
            
            time_index = pd.date_range(
                start=start_date,
                periods=self.forecast_months,
                freq='MS'  # Month start
            )
            
            # Initialize aggregated arrays
            p10_total = np.zeros(self.forecast_months)
            p50_total = np.zeros(self.forecast_months)
            p90_total = np.zeros(self.forecast_months)
            
            # Track individual well contributions
            self.well_contributions = {}
            
            # Aggregate using simple summation for each scenario
            for well_name, forecast_df in well_forecasts.items():
                logger.debug(f"Aggregating well: {well_name}")
                
                # Extract production arrays (handle different forecast lengths)
                well_months = min(len(forecast_df), self.forecast_months)
                
                # Get production values for each scenario
                p10_well = self._extract_production_values(forecast_df, 'P10', well_months)
                p50_well = self._extract_production_values(forecast_df, 'P50', well_months)
                p90_well = self._extract_production_values(forecast_df, 'P90', well_months)
                
                # Add to totals (simple summation)
                p10_total[:well_months] += p10_well
                p50_total[:well_months] += p50_well
                p90_total[:well_months] += p90_well
                
                # Store individual contributions
                self.well_contributions[well_name] = pd.DataFrame({
                    'Date': time_index[:well_months],
                    'P10_Production_bbl': p10_well,
                    'P50_Production_bbl': p50_well,
                    'P90_Production_bbl': p90_well
                })
            
            # Create asset-level forecast DataFrame
            asset_forecast = pd.DataFrame({
                'Date': time_index,
                'P10_Production_bbl': p10_total,
                'P50_Production_bbl': p50_total,
                'P90_Production_bbl': p90_total
            })
            
            # Add cumulative production
            asset_forecast['P10_Cumulative_bbl'] = asset_forecast['P10_Production_bbl'].cumsum()
            asset_forecast['P50_Cumulative_bbl'] = asset_forecast['P50_Production_bbl'].cumsum()
            asset_forecast['P90_Cumulative_bbl'] = asset_forecast['P90_Production_bbl'].cumsum()
            
            # Validate aggregated results
            if self.validation_enabled:
                self._validate_aggregated_forecast(asset_forecast)
            
            # Calculate aggregation metrics
            self._calculate_aggregation_metrics(asset_forecast, well_forecasts)
            
            self.asset_production_forecast = asset_forecast
            
            logger.info(f"Asset aggregation completed successfully")
            logger.info(f"Asset P50 EUR: {asset_forecast['P50_Cumulative_bbl'].iloc[-1]:,.0f} barrels")
            
            return asset_forecast
            
        except Exception as e:
            logger.error(f"Asset aggregation failed: {str(e)}")
            raise AssetAggregationError(f"Failed to aggregate well forecasts: {str(e)}")
    
    def _validate_well_forecasts(self, well_forecasts: Dict[str, pd.DataFrame]) -> None:
        """Validate individual well forecast inputs."""
        if not well_forecasts:
            raise AssetAggregationError("No well forecasts provided")
        
        required_columns = ['P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl']
        
        for well_name, forecast_df in well_forecasts.items():
            if forecast_df.empty:
                logger.warning(f"Empty forecast for well {well_name}, skipping")
                continue
            
            # Check required columns
            missing_cols = [col for col in required_columns if col not in forecast_df.columns]
            if missing_cols:
                raise AssetAggregationError(f"Well {well_name} missing columns: {missing_cols}")
            
            # Check for valid production values
            for col in required_columns:
                if forecast_df[col].isna().all():
                    raise AssetAggregationError(f"Well {well_name} has all NaN values in {col}")
                
                if (forecast_df[col] < 0).any():
                    raise AssetAggregationError(f"Well {well_name} has negative production values in {col}")
        
        logger.info(f"Validated {len(well_forecasts)} well forecasts")
    
    def _extract_production_values(self, 
                                 forecast_df: pd.DataFrame, 
                                 scenario: str, 
                                 months: int) -> np.ndarray:
        """Extract production values for a specific scenario."""
        col_name = f"{scenario}_Production_bbl"
        
        if col_name not in forecast_df.columns:
            raise AssetAggregationError(f"Missing column: {col_name}")
        
        # Get production values, fill NaN with 0, and truncate to required months
        production_values = forecast_df[col_name].fillna(0).values[:months]
        
        # Ensure we have the right length
        if len(production_values) < months:
            # Pad with zeros if forecast is shorter
            padded_values = np.zeros(months)
            padded_values[:len(production_values)] = production_values
            production_values = padded_values
        
        return production_values
    
    def _validate_aggregated_forecast(self, asset_forecast: pd.DataFrame) -> None:
        """Validate the aggregated asset forecast."""
        
        # Check that P10 >= P50 >= P90 (optimistic >= median >= conservative)
        # This is the INDUSTRY CONVENTION where:
        # P10 = 90th percentile = optimistic/high reserves
        # P50 = 50th percentile = median reserves  
        # P90 = 10th percentile = conservative/low reserves
        p10_prod = asset_forecast['P10_Production_bbl'].values
        p50_prod = asset_forecast['P50_Production_bbl'].values
        p90_prod = asset_forecast['P90_Production_bbl'].values
        
        # Allow for small numerical errors
        tolerance = 1e-6
        
        if not np.all(p10_prod >= p50_prod - tolerance):
            logger.warning("P10 production not always >= P50 production - INDUSTRY CONVENTION VIOLATION")
        
        if not np.all(p50_prod >= p90_prod - tolerance):
            logger.warning("P50 production not always >= P90 production - INDUSTRY CONVENTION VIOLATION")
        
        # Additional validation: Check that P10 > P90 (optimistic > conservative)
        if not np.all(p10_prod >= p90_prod - tolerance):
            logger.warning("P10 production not always >= P90 production - SERIOUS ISSUE")
        
        # Log the validation results for debugging
        logger.info(f"Validation check: P10[0]={p10_prod[0]:.0f}, P50[0]={p50_prod[0]:.0f}, P90[0]={p90_prod[0]:.0f}")
        logger.info(f"Industry convention check: P10 >= P50 >= P90? {np.all(p10_prod >= p50_prod - tolerance) and np.all(p50_prod >= p90_prod - tolerance)}")
        
        # Check that production generally declines over time
        for scenario in ['P10', 'P50', 'P90']:
            prod_col = f"{scenario}_Production_bbl"
            
            # Calculate trend over first 5 years (60 months)
            trend_months = min(60, len(asset_forecast))
            if trend_months > 12:
                early_avg = asset_forecast[prod_col].iloc[:12].mean()
                late_avg = asset_forecast[prod_col].iloc[trend_months-12:trend_months].mean()
                
                if late_avg > early_avg:
                    logger.warning(f"{scenario} production trend is increasing rather than declining")
        
        # Check for reasonable production values
        max_monthly_prod = asset_forecast[['P10_Production_bbl', 'P50_Production_bbl', 'P90_Production_bbl']].max().max()
        
        if max_monthly_prod > 1e7:  # 10 million barrels per month seems excessive
            logger.warning(f"Very high monthly production detected: {max_monthly_prod:,.0f} bbl/month")
        
        if max_monthly_prod <= 0:
            raise AssetAggregationError("All production forecasts are zero or negative")
        
        logger.info("Asset forecast validation completed")
    
    def _calculate_aggregation_metrics(self, 
                                     asset_forecast: pd.DataFrame, 
                                     well_forecasts: Dict[str, pd.DataFrame]) -> None:
        """Calculate aggregation metrics and statistics."""
        
        # Asset-level metrics
        asset_p10_eur = asset_forecast['P10_Cumulative_bbl'].iloc[-1]
        asset_p50_eur = asset_forecast['P50_Cumulative_bbl'].iloc[-1]
        asset_p90_eur = asset_forecast['P90_Cumulative_bbl'].iloc[-1]
        
        # Well-level metrics
        well_count = len(well_forecasts)
        well_p50_eurs = []
        
        for well_name, forecast_df in well_forecasts.items():
            if 'P50_Production_bbl' in forecast_df.columns:
                well_p50_eur = forecast_df['P50_Production_bbl'].sum()
                well_p50_eurs.append(well_p50_eur)
        
        # Calculate diversification metrics
        individual_sum_p50 = sum(well_p50_eurs) if well_p50_eurs else 0
        diversification_ratio = asset_p50_eur / individual_sum_p50 if individual_sum_p50 > 0 else 1.0
        
        # Uncertainty metrics
        # ASSET-LEVEL CUMULATIVE P10/P90 RATIO: Total optimistic EUR ÷ Total conservative EUR
        # This measures overall asset uncertainty for the entire 30-year forecast period
        # Different from time-series median ratio which measures typical monthly uncertainty
        uncertainty_p10_p90 = asset_p10_eur / asset_p90_eur if asset_p90_eur > 0 else np.inf
        uncertainty_p50_range = (asset_p10_eur - asset_p90_eur) / asset_p50_eur if asset_p50_eur > 0 else np.inf
        
        # Store metrics
        self.aggregation_metrics = {
            'well_count': well_count,
            'asset_p10_eur_bbl': asset_p10_eur,
            'asset_p50_eur_bbl': asset_p50_eur,
            'asset_p90_eur_bbl': asset_p90_eur,
            'individual_sum_p50_eur_bbl': individual_sum_p50,
            'diversification_ratio': diversification_ratio,
            'uncertainty_p10_p90_ratio': uncertainty_p10_p90,
            'uncertainty_range_ratio': uncertainty_p50_range,
            'initial_production_p50_bbl_month': asset_forecast['P50_Production_bbl'].iloc[0],
            'final_production_p50_bbl_month': asset_forecast['P50_Production_bbl'].iloc[-1],
            'aggregation_date': datetime.now().isoformat()
        }
        
        logger.info(f"Aggregation metrics calculated for {well_count} wells")
    
    def get_aggregation_summary(self) -> Dict:
        """Get summary of aggregation results."""
        if not self.aggregation_metrics:
            return {}
        
        summary = {
            'Asset Summary': {
                'Total Wells': self.aggregation_metrics['well_count'],
                'Asset P10 EUR (bbl)': f"{self.aggregation_metrics['asset_p10_eur_bbl']:,.0f}",
                'Asset P50 EUR (bbl)': f"{self.aggregation_metrics['asset_p50_eur_bbl']:,.0f}",
                'Asset P90 EUR (bbl)': f"{self.aggregation_metrics['asset_p90_eur_bbl']:,.0f}",
                'P10/P90 Uncertainty Ratio': f"{self.aggregation_metrics['uncertainty_p10_p90_ratio']:.1f} (asset-level cumulative)",
                'Initial Monthly Production (P50)': f"{self.aggregation_metrics['initial_production_p50_bbl_month']:,.0f} bbl/month",
                'Final Monthly Production (P50)': f"{self.aggregation_metrics['final_production_p50_bbl_month']:,.0f} bbl/month"
            }
        }
        
        return summary
    
    def export_aggregation_results(self, 
                                 output_dir: str = "output",
                                 include_well_contributions: bool = False) -> None:
        """
        Export aggregation results to files.
        
        Args:
            output_dir: Output directory path
            include_well_contributions: Whether to export individual well contributions
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export asset-level forecast
        if self.asset_production_forecast is not None:
            asset_file = output_path / "asset_production_forecast.csv"
            self.asset_production_forecast.to_csv(asset_file, index=False)
            logger.info(f"Asset production forecast exported to {asset_file}")
        
        # Export well contributions if requested
        if include_well_contributions and self.well_contributions:
            contributions_dir = output_path / "well_contributions"
            contributions_dir.mkdir(exist_ok=True)
            
            for well_name, contribution_df in self.well_contributions.items():
                well_file = contributions_dir / f"{well_name}_contribution.csv"
                contribution_df.to_csv(well_file, index=False)
            
            logger.info(f"Well contributions exported to {contributions_dir}")
        
        # Export aggregation metrics
        if self.aggregation_metrics:
            metrics_file = output_path / "aggregation_metrics.json"
            import json
            
            with open(metrics_file, 'w') as f:
                json.dump(self.aggregation_metrics, f, indent=2, default=str)
            
            logger.info(f"Aggregation metrics exported to {metrics_file}")

    def analyze_uncertainty_trends(self, asset_forecast: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze uncertainty trends in the asset-level production forecast.
        
        This method provides comprehensive analysis of how uncertainty evolves
        over the forecast period, including business interpretations.
        
        Args:
            asset_forecast: Asset forecast DataFrame (uses stored if None)
            
        Returns:
            Dictionary containing uncertainty trend analysis and interpretations
        """
        if asset_forecast is None:
            asset_forecast = self.asset_production_forecast
            
        if asset_forecast is None:
            raise AssetAggregationError("No asset forecast available for uncertainty analysis")
        
        logger.info("Analyzing asset-level uncertainty trends...")
        
        # Extract production scenarios
        p10_prod = asset_forecast['P10_Production_bbl'].values
        p50_prod = asset_forecast['P50_Production_bbl'].values
        p90_prod = asset_forecast['P90_Production_bbl'].values
        
        # Calculate uncertainty metrics
        uncertainty_range = p10_prod - p90_prod
        cv_production = uncertainty_range / np.maximum(p50_prod, 1.0)  # Avoid division by zero
        p10_p90_ratio = p10_prod / np.maximum(p90_prod, 1e-10)
        
        # Time arrays
        years = np.arange(len(p10_prod)) / 12
        months = np.arange(len(p10_prod))
        
        # Uncertainty range analysis
        range_analysis = self._analyze_uncertainty_range_trends(uncertainty_range, years)
        
        # Coefficient of variation analysis
        cv_analysis = self._analyze_coefficient_variation_trends(cv_production, years)
        
        # P10/P90 ratio analysis
        ratio_analysis = self._analyze_p10_p90_ratio_trends(p10_p90_ratio, years)
        
        # Distribution analysis
        distribution_analysis = self._analyze_uncertainty_distribution(uncertainty_range)
        
        # Overall assessment and business interpretation
        business_interpretation = self._generate_uncertainty_business_interpretation(
            range_analysis, cv_analysis, ratio_analysis, distribution_analysis
        )
        
        # Compile comprehensive analysis
        uncertainty_analysis = {
            'uncertainty_metrics': {
                'uncertainty_range': uncertainty_range.tolist(),
                'coefficient_of_variation': cv_production.tolist(),
                'p10_p90_ratio': p10_p90_ratio.tolist(),
                'years': years.tolist(),
                'months': months.tolist()
            },
            'range_analysis': range_analysis,
            'cv_analysis': cv_analysis,
            'ratio_analysis': ratio_analysis,
            'distribution_analysis': distribution_analysis,
            'business_interpretation': business_interpretation,
            'industry_benchmarks': self._get_industry_uncertainty_benchmarks(),
            'forecast_quality_assessment': self._assess_forecast_quality(cv_analysis, ratio_analysis)
        }
        
        logger.info("Uncertainty trend analysis completed")
        
        return uncertainty_analysis
    
    def _analyze_uncertainty_range_trends(self, uncertainty_range: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
        """Analyze how absolute uncertainty range changes over time."""
        
        # Calculate trend metrics
        initial_range = uncertainty_range[0] if len(uncertainty_range) > 0 else 0
        final_range = uncertainty_range[-1] if len(uncertainty_range) > 0 else 0
        max_range = np.max(uncertainty_range)
        min_range = np.min(uncertainty_range)
        avg_range = np.mean(uncertainty_range)
        median_range = np.median(uncertainty_range)
        
        # Trend direction analysis
        early_period = uncertainty_range[:min(36, len(uncertainty_range))]  # First 3 years
        late_period = uncertainty_range[-min(36, len(uncertainty_range)):]  # Last 3 years
        
        early_avg = np.mean(early_period)
        late_avg = np.mean(late_period)
        
        if late_avg > early_avg * 1.2:
            trend_direction = "increasing"
            trend_strength = "significant"
        elif late_avg > early_avg * 1.05:
            trend_direction = "increasing"
            trend_strength = "moderate"
        elif late_avg < early_avg * 0.8:
            trend_direction = "decreasing"
            trend_strength = "significant"
        elif late_avg < early_avg * 0.95:
            trend_direction = "decreasing"
            trend_strength = "moderate"
        else:
            trend_direction = "stable"
            trend_strength = "minimal"
        
        # Peak uncertainty analysis (keep for reference, but use avg in business interpretation)
        peak_index = np.argmax(uncertainty_range)
        peak_year = years[peak_index] if len(years) > peak_index else 0
        
        return {
            'initial_range_bbl': float(initial_range),
            'final_range_bbl': float(final_range),
            'max_range_bbl': float(max_range),
            'min_range_bbl': float(min_range),
            'avg_range_bbl': float(avg_range),
            'median_range_bbl': float(median_range),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'peak_uncertainty_year': float(peak_year),
            'peak_uncertainty_bbl': float(max_range),
            'typical_uncertainty_bbl': float(avg_range),  # Use for business interpretation
            'range_change_pct': float((final_range - initial_range) / max(initial_range, 1) * 100)
        }
    
    def _analyze_coefficient_variation_trends(self, cv_production: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
        """Analyze relative uncertainty trends using coefficient of variation."""
        
        # Remove infinite values and outliers
        cv_clean = cv_production[np.isfinite(cv_production)]
        cv_clean = cv_clean[cv_clean < 10]  # Remove extreme outliers
        
        if len(cv_clean) == 0:
            return {'error': 'No valid CV data available'}
        
        initial_cv = cv_clean[0] if len(cv_clean) > 0 else 0
        final_cv = cv_clean[-1] if len(cv_clean) > 0 else 0
        avg_cv = np.mean(cv_clean)
        median_cv = np.median(cv_clean)
        max_cv = np.max(cv_clean)
        
        # Stability analysis
        cv_std = np.std(cv_clean)
        cv_stability = "high" if cv_std < 0.1 else "moderate" if cv_std < 0.3 else "low"
        
        # Industry comparison - use average instead of max for business interpretation
        if avg_cv < 0.3:
            industry_assessment = "low_uncertainty"
        elif avg_cv < 0.6:
            industry_assessment = "moderate_uncertainty"
        elif avg_cv < 1.0:
            industry_assessment = "high_uncertainty"
        else:
            industry_assessment = "very_high_uncertainty"
        
        return {
            'initial_cv': float(initial_cv),
            'final_cv': float(final_cv),
            'avg_cv': float(avg_cv),
            'median_cv': float(median_cv),
            'max_cv': float(max_cv),
            'typical_cv': float(avg_cv),  # Use for business interpretation
            'cv_stability': cv_stability,
            'cv_std': float(cv_std),
            'industry_assessment': industry_assessment,
            'cv_trend': "increasing" if final_cv > initial_cv * 1.1 else "decreasing" if final_cv < initial_cv * 0.9 else "stable"
        }
    
    def _analyze_p10_p90_ratio_trends(self, p10_p90_ratio: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
        """
        Analyze P10/P90 ratio trends over time.
        
        This calculates TIME-SERIES MEDIAN P10/P90 RATIO, the typical monthly uncertainty ratio.
        This is different from the asset-level cumulative ratio (total P10 EUR ÷ total P90 EUR),
        which measures overall asset uncertainty for the entire forecast period.
        """
        
        # Remove infinite values
        ratio_clean = p10_p90_ratio[np.isfinite(p10_p90_ratio)]
        
        if len(ratio_clean) == 0:
            return {'error': 'No valid ratio data available'}
        
        initial_ratio = ratio_clean[0]
        final_ratio = ratio_clean[-1]
        avg_ratio = np.mean(ratio_clean)
        median_ratio = np.median(ratio_clean)
        min_ratio = np.min(ratio_clean)
        max_ratio = np.max(ratio_clean)
        
        # Ratio stability assessment
        ratio_std = np.std(ratio_clean)
        ratio_stability = "stable" if ratio_std < 0.2 else "moderate" if ratio_std < 0.5 else "volatile"
        
        # Industry benchmark comparison (typical oil & gas P10/P90 ratios: 1.5-3.0)
        # Use median for business interpretation due to extreme outliers from division by small P90 values
        if median_ratio < 1.2:
            ratio_assessment = "very_conservative"
        elif median_ratio < 1.8:
            ratio_assessment = "conservative"
        elif median_ratio < 2.5:
            ratio_assessment = "industry_typical"
        elif median_ratio < 3.5:
            ratio_assessment = "optimistic"
        else:
            ratio_assessment = "very_optimistic"
        
        return {
            'initial_ratio': float(initial_ratio),
            'final_ratio': float(final_ratio),
            'avg_ratio': float(avg_ratio),
            'median_ratio': float(median_ratio),
            'min_ratio': float(min_ratio),
            'max_ratio': float(max_ratio),
            'typical_ratio': float(median_ratio),  # Use median for business interpretation due to extreme outliers
            'ratio_stability': ratio_stability,
            'ratio_std': float(ratio_std),
            'ratio_assessment': ratio_assessment,
            'ratio_trend': "increasing" if final_ratio > initial_ratio * 1.1 else "decreasing" if final_ratio < initial_ratio * 0.9 else "stable"
        }
    
    def _analyze_uncertainty_distribution(self, uncertainty_range: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of uncertainty across time periods."""
        
        # Calculate percentiles of uncertainty range
        percentiles = [10, 25, 50, 75, 90]
        uncertainty_percentiles = [float(np.percentile(uncertainty_range, p)) for p in percentiles]
        
        # Identify periods of high and low uncertainty
        median_uncertainty = np.median(uncertainty_range)
        high_uncertainty_periods = np.sum(uncertainty_range > median_uncertainty * 1.5)
        low_uncertainty_periods = np.sum(uncertainty_range < median_uncertainty * 0.5)
        
        # Calculate uncertainty concentration
        total_periods = len(uncertainty_range)
        high_uncertainty_pct = (high_uncertainty_periods / total_periods) * 100
        low_uncertainty_pct = (low_uncertainty_periods / total_periods) * 100
        
        return {
            'uncertainty_percentiles': uncertainty_percentiles,
            'percentile_labels': [f'P{p}' for p in percentiles],
            'median_uncertainty_bbl': float(median_uncertainty),
            'high_uncertainty_periods': int(high_uncertainty_periods),
            'low_uncertainty_periods': int(low_uncertainty_periods),
            'high_uncertainty_pct': float(high_uncertainty_pct),
            'low_uncertainty_pct': float(low_uncertainty_pct),
            'uncertainty_concentration': "early_heavy" if high_uncertainty_periods > total_periods * 0.6 else "late_heavy" if high_uncertainty_periods < total_periods * 0.3 else "balanced"
        }
    
    def _generate_uncertainty_business_interpretation(self, 
                                                   range_analysis: Dict, 
                                                   cv_analysis: Dict, 
                                                   ratio_analysis: Dict,
                                                   distribution_analysis: Dict) -> Dict[str, Any]:
        """Generate business interpretation of uncertainty trends."""
        
        interpretations = []
        risk_factors = []
        recommendations = []
        
        # Range trend interpretation
        if range_analysis['trend_direction'] == 'increasing' and range_analysis['trend_strength'] == 'significant':
            interpretations.append("Uncertainty increases significantly over time, indicating growing forecast uncertainty in later years")
            risk_factors.append("High late-stage production uncertainty")
            recommendations.append("Consider phased development approach to reduce long-term risk exposure")
        elif range_analysis['trend_direction'] == 'stable':
            interpretations.append("Uncertainty remains relatively stable throughout forecast period")
            
        # CV interpretation
        if cv_analysis.get('industry_assessment') == 'very_high_uncertainty':
            interpretations.append("Relative uncertainty is very high compared to industry standards")
            risk_factors.append("Exceptionally high production variability")
            recommendations.append("Implement robust risk management strategies and scenario planning")
        elif cv_analysis.get('industry_assessment') == 'low_uncertainty':
            interpretations.append("Relative uncertainty is low, indicating confident production forecasts")
            
        # Ratio interpretation
        if ratio_analysis['ratio_assessment'] == 'very_conservative':
            interpretations.append("P10/P90 ratio suggests very conservative uncertainty estimates")
            recommendations.append("Review if uncertainty estimates adequately capture potential upside")
        elif ratio_analysis['ratio_assessment'] == 'very_optimistic':
            interpretations.append("P10/P90 ratio suggests optimistic uncertainty estimates")
            risk_factors.append("Potentially underestimating downside risks")
            recommendations.append("Consider more conservative uncertainty estimates")
        elif ratio_analysis['ratio_assessment'] == 'industry_typical':
            interpretations.append("P10/P90 ratio aligns with industry standards")
            
        # Distribution interpretation
        if distribution_analysis['uncertainty_concentration'] == 'early_heavy':
            interpretations.append("Uncertainty is concentrated in early forecast years")
            recommendations.append("Focus risk mitigation efforts on near-term production optimization")
        elif distribution_analysis['uncertainty_concentration'] == 'late_heavy':
            interpretations.append("Uncertainty is concentrated in later forecast years")
            recommendations.append("Develop flexible long-term strategies to adapt to evolving conditions")
            
        # Peak uncertainty interpretation (use typical/average for business interpretation)
        peak_year = range_analysis.get('peak_uncertainty_year', 0)
        typical_uncertainty = range_analysis.get('typical_uncertainty_bbl', 0)
        if peak_year < 5:
            interpretations.append(f"Peak uncertainty occurs in year {peak_year:.1f}, suggesting early-stage production challenges. Typical uncertainty: {typical_uncertainty:,.0f} bbl")
        elif peak_year > 20:
            interpretations.append(f"Peak uncertainty occurs in year {peak_year:.1f}, indicating long-term forecast challenges. Typical uncertainty: {typical_uncertainty:,.0f} bbl")
            
        # Overall assessment
        if len(risk_factors) == 0:
            overall_assessment = "low_risk"
        elif len(risk_factors) <= 2:
            overall_assessment = "moderate_risk"
        else:
            overall_assessment = "high_risk"
            
        return {
            'key_interpretations': interpretations,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'overall_assessment': overall_assessment,
            'forecast_confidence': cv_analysis.get('industry_assessment', 'unknown'),
            'uncertainty_timing': distribution_analysis['uncertainty_concentration']
        }
    
    def _get_industry_uncertainty_benchmarks(self) -> Dict[str, Any]:
        """Provide industry benchmark ranges for uncertainty metrics."""
        return {
            'typical_cv_range': {'low': 0.2, 'moderate': 0.5, 'high': 0.8, 'very_high': 1.2},
            'typical_p10_p90_ratio': {'conservative': 1.5, 'industry_standard': 2.2, 'optimistic': 3.0},
            'explanation': 'Based on oil & gas industry standards for production forecasting uncertainty',
            'cv_interpretation': {
                'below_0.3': 'Low uncertainty - high confidence forecasts',
                '0.3_to_0.6': 'Moderate uncertainty - typical for mature assets',
                '0.6_to_1.0': 'High uncertainty - common for early-stage developments',
                'above_1.0': 'Very high uncertainty - requires additional risk assessment'
            },
            'ratio_interpretation': {
                'below_1.5': 'Very conservative estimates',
                '1.5_to_2.5': 'Industry typical range',
                '2.5_to_3.5': 'Optimistic but reasonable',
                'above_3.5': 'Potentially over-optimistic'
            }
        }
    
    def _assess_forecast_quality(self, cv_analysis: Dict, ratio_analysis: Dict) -> Dict[str, Any]:
        """Assess overall forecast quality based on uncertainty characteristics."""
        
        quality_indicators = []
        quality_score = 100  # Start with perfect score
        
        # CV assessment
        cv_assessment = cv_analysis.get('industry_assessment', 'unknown')
        if cv_assessment == 'very_high_uncertainty':
            quality_score -= 30
            quality_indicators.append("High relative uncertainty reduces forecast confidence")
        elif cv_assessment == 'low_uncertainty':
            quality_score += 10
            quality_indicators.append("Low relative uncertainty indicates high forecast confidence")
        elif cv_assessment == 'moderate_uncertainty':
            quality_indicators.append("Moderate uncertainty levels are typical for oil & gas forecasts")
            
        # Ratio assessment - USE ASSET-LEVEL CUMULATIVE RATIO for quality assessment
        # The time-series median ratio is for operational analysis, not overall quality
        asset_cumulative_ratio = self.aggregation_metrics.get('uncertainty_p10_p90_ratio', 0)
        
        # Apply industry standards to asset-level cumulative ratio
        if asset_cumulative_ratio > 0:
            if asset_cumulative_ratio < 1.2:
                cumulative_ratio_assessment = "very_conservative"
            elif asset_cumulative_ratio < 1.8:
                cumulative_ratio_assessment = "conservative"
            elif asset_cumulative_ratio < 2.5:
                cumulative_ratio_assessment = "industry_typical"
            elif asset_cumulative_ratio < 3.5:
                cumulative_ratio_assessment = "optimistic"
            else:
                cumulative_ratio_assessment = "very_optimistic"
        else:
            cumulative_ratio_assessment = "unknown"
            
        if cumulative_ratio_assessment in ['very_conservative', 'very_optimistic']:
            quality_score -= 15
            quality_indicators.append(f"Asset-level P10/P90 ratio ({asset_cumulative_ratio:.1f}x) outside typical industry range")
        elif cumulative_ratio_assessment == 'industry_typical':
            quality_score += 10
            quality_indicators.append(f"Asset-level P10/P90 ratio ({asset_cumulative_ratio:.1f}x) aligns with industry standards")
            
        # Stability assessment
        cv_stability = cv_analysis.get('cv_stability', 'unknown')
        ratio_stability = ratio_analysis.get('ratio_stability', 'unknown')
        
        if cv_stability == 'high' and ratio_stability == 'stable':
            quality_score += 15
            quality_indicators.append("Stable uncertainty metrics indicate robust forecasting")
        elif cv_stability == 'low' or ratio_stability == 'volatile':
            quality_score -= 20
            quality_indicators.append("Volatile uncertainty metrics suggest forecasting challenges")
            
        # Final quality grade
        if quality_score >= 90:
            quality_grade = "Excellent"
        elif quality_score >= 75:
            quality_grade = "Good"
        elif quality_score >= 60:
            quality_grade = "Fair"
        elif quality_score >= 45:
            quality_grade = "Poor"
        else:
            quality_grade = "Very Poor"
            
        return {
            'quality_score': max(0, min(100, quality_score)),
            'quality_grade': quality_grade,
            'quality_indicators': quality_indicators,
            'assessment_criteria': [
                'Relative uncertainty level vs industry standards',
                'Asset-level P10/P90 ratio appropriateness (not time-series median)',
                'Uncertainty metric stability over time'
            ]
        }


def load_well_forecasts_from_directory(forecasts_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load well forecasts from a directory containing forecast CSV files.
    
    Args:
        forecasts_dir: Directory containing forecast CSV files
        
    Returns:
        Dictionary mapping well names to forecast DataFrames
    """
    forecasts_path = Path(forecasts_dir)
    
    if not forecasts_path.exists():
        raise AssetAggregationError(f"Forecasts directory not found: {forecasts_dir}")
    
    well_forecasts = {}
    
    # Look for forecast files
    for forecast_file in forecasts_path.glob("forecast_*.csv"):
        try:
            # Extract well name from filename
            well_name = forecast_file.stem.replace("forecast_", "")
            
            # Load forecast
            forecast_df = pd.read_csv(forecast_file)
            
            # Basic validation
            if not forecast_df.empty:
                well_forecasts[well_name] = forecast_df
                logger.debug(f"Loaded forecast for well {well_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load forecast from {forecast_file}: {str(e)}")
    
    logger.info(f"Loaded forecasts for {len(well_forecasts)} wells from {forecasts_dir}")
    
    return well_forecasts


def aggregate_asset_production(well_forecasts: Dict[str, pd.DataFrame],
                             forecast_months: int = 360,
                             start_date: str = "2025-01-01") -> pd.DataFrame:
    """
    Convenience function to aggregate well forecasts into asset-level production.
    
    Args:
        well_forecasts: Dictionary mapping well names to forecast DataFrames
        forecast_months: Number of months to forecast
        start_date: Start date for forecast period
        
    Returns:
        Asset-level production forecast DataFrame
    """
    aggregator = AssetAggregator(forecast_months=forecast_months)
    
    return aggregator.aggregate_well_forecasts(
        well_forecasts=well_forecasts,
        start_date=start_date
    ) 