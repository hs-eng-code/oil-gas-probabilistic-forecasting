"""
Probabilistic Production Forecasting Pipeline

This pipeline leverages the new ArpsDCA capabilities including:
- Multi-method robust fitting engine
- Enhanced quality metrics and validation
- Structured error handling with ArpsDeclineError
- FitResult dataclass for type-safe operations
- Intelligent model selection based on well characteristics
"""

import logging
import time
import os
import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import traceback

import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

# Add src to path
sys.path.append('src')

# Local imports
from arps_dca import (AdvancedArpsDCA, ArpsDeclineError, DeclineModel, FitResult, ValidationResult, TransitionMethod)
from bayesian_forecaster import ModernizedBayesianForecaster, AssetScaleBayesianForecaster
from aggregator import AssetAggregator
from revenue_calculator import RevenueCalculator
import importlib
from data_loader import WellProductionDataLoader
from reporting import ComprehensiveReporter
from uncertainty_config import UncertaintyConfig

# Reload the entire module
import visualizations  # This should already be imported
importlib.reload(visualizations)
from visualizations import SingleScenarioResultsVisualizer, AcquisitionAnalysisVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecasting_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ProbablisticProductionForecastingPipeline:
    """
    Advanced Oil Production Forecasting Pipeline using ArpsDCA capabilities.

    This pipeline leverages:
    - Multi-method robust fitting engine
    - Enhanced quality metrics and validation
    - Intelligent model selection
    - Structured error handling
    - Type-safe operations with FitResult dataclass
    """

    def __init__(self,
                 well_data_path: str,
                 price_data_path: str,
                 output_dir: str = "output",
                 forecast_years: int = 10,
                 max_workers: int = 4,
                 use_asset_scale_bayesian_processing: bool = None,
                 time_limit_minutes: int = 10,
                 random_seed: int = None,
                 uncertainty_level: str = 'standard'):
        """
        Initialize the forecasting pipeline.

        Args:
            well_data_path: Path to well production data CSV
            price_data_path: Path to price data CSV
            output_dir: Directory for output files
            forecast_years: Number of years to forecast
            max_workers: Maximum number of worker threads
            use_asset_scale_bayesian_processing: Whether to use asset-scale processing (auto-determined if None)
            time_limit_minutes: Time limit for processing in minutes
            random_seed: Random seed for reproducible results (None for random behavior)
            uncertainty_level: Uncertainty level ('standard', 'conservative', 'aggressive', 'high_uncertainty')
        """

        self.well_data_path = well_data_path
        self.price_data_path = price_data_path
        
        # Create dynamic output directory based on Bayesian approach type and uncertainty level
        approach_type = "asset_scale" if use_asset_scale_bayesian_processing else "individual"
        
        # If output_dir doesn't already contain the approach and uncertainty info, create it
        if not (f"_{approach_type}_" in output_dir and f"_{uncertainty_level}" in output_dir):
            dynamic_output_dir = f"{output_dir}/output_{approach_type}Bayesian_{uncertainty_level}Uncertainty"
        else:
            dynamic_output_dir = output_dir
        
        # Convert output_dir to Path object to be consistent with other modules
        self.output_dir = Path(dynamic_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create visualizations subdirectory
        (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        self.forecast_years = forecast_years
        self.forecast_months = forecast_years * 12  # Convert years to months for internal use
        self.max_workers = max_workers
        self.time_limit_minutes = time_limit_minutes
        self.random_seed = random_seed
        self.uncertainty_level = uncertainty_level
        
        # Validate uncertainty level
        available_levels = UncertaintyConfig.list_available_levels()
        if uncertainty_level not in available_levels:
            logger.warning(f"Unknown uncertainty level '{uncertainty_level}'. Available levels: {list(available_levels.keys())}")
            logger.warning(f"Using 'standard' uncertainty level instead.")
            self.uncertainty_level = 'standard'
        else:
            logger.info(f"Using uncertainty level: {uncertainty_level} - {available_levels[uncertainty_level]}")
        
        # Asset-scale processing decision
        self.use_asset_scale_bayesian_processing = use_asset_scale_bayesian_processing  # Will be determined after data loading if None

        # Initialize components
        self.data_loader = None
        self.data_validator = None  # Add missing data_validator
        self.arps_dca = None
        self.bayesian_forecaster = None
        self.asset_forecaster = None
        self.price_forecaster = None
        self.revenue_calculator = None
        self.visualizer = None
        self.reporter = None

        # Pipeline state
        self.well_data = None
        self.price_data = None
        self.processing_stats = {}
        self.processing_times = {}
        self.memory_usage = {}

        # Pipeline initialization
        if self.random_seed is not None:
            logger.info(f"Pipeline initialized with random seed {self.random_seed} for reproducible results")
            logger.info(f"Pipeline initialized with uncertainty level '{self.uncertainty_level}'")

        logger.info(f"Forecasting Pipeline initialized with {forecast_years} year forecast horizon")
        logger.info(f"Output directory: {self.output_dir}")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete forecasting pipeline.

        Returns:
            Dictionary containing all pipeline results and statistics
        """
        pipeline_start_time = time.time()

        try:
            logger.info("="*80)
            logger.info("STARTING FORECASTING PIPELINE")
            logger.info("="*80)

            # Phase 1: Data Loading and Validation
            logger.info("Phase 1: Data Loading and Validation")
            self._load_and_validate_data()

            # Phase 2: Advanced Arps DCA Processing
            logger.info("Phase 2: Advanced Arps DCA Processing")
            self._process_arps_dca()

            # Phase 3: Bayesian Forecasting
            logger.info("Phase 3: Bayesian Forecasting")
            self._generate_probabilistic_forecasts()

            # Phase 4: Asset Aggregation
            logger.info("Phase 4: Asset Aggregation")
            self._aggregate_to_asset_level()

            # Phase 5: Revenue Calculation
            logger.info("Phase 5: Revenue Calculation")
            self._calculate_revenue_forecasts()

            # Phase 6: Comprehensive Validation
            logger.info("Phase 6: Comprehensive Validation")
            self._perform_comprehensive_validation()

            # Phase 7: Advanced Analytics
            logger.info("Phase 7: Advanced Analytics")
            self._generate_advanced_analytics()

            # Phase 8: Final Reporting
            logger.info("Phase 8: Final Reporting")
            self._generate_final_report()

            self.processing_times['total_pipeline'] = time.time() - pipeline_start_time
            self.memory_usage['peak'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total processing time: {self.processing_times['total_pipeline']:.2f} seconds")
            logger.info(f"Peak memory usage: {self.memory_usage['peak']:.1f} MB")
            logger.info("="*80)

            return self._get_pipeline_results()

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _load_and_validate_data(self) -> None:
        """Load and validate input data using enhanced validation."""
        start_time = time.time()

        try:
            # Initialize data validator
            if self.data_validator is None:
                self.data_validator = WellProductionDataLoader()
            
            # Load and validate data using WellProductionDataLoader
            logger.info("Loading and validating well production data...")
            self.well_data = self.data_validator.load_well_data(self.well_data_path)

            logger.info("Loading and validating price data...")
            self.price_data = self.data_validator.load_price_data(self.price_data_path)

            # Cross-validate datasets
            logger.info("Performing cross-validation between datasets...")
            self.validation_report = self.data_validator.cross_validate_datasets()

            # Get validation summary
            validation_summary = self.data_validator.get_validation_summary()
            logger.info(f"Data validation completed:")
            logger.info(f"  Total wells: {validation_summary.get('total_wells', 0)}")
            logger.info(f"  Valid wells: {validation_summary.get('valid_wells', 0)}")
            logger.info(f"  Data quality issues: {len(validation_summary.get('data_quality_issues', []))}")

            self.processing_times['data_validation'] = time.time() - start_time
            self.memory_usage['data_loading'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            self._determine_bayesian_processing_approach()
            logger.info("Data loading and validation completed successfully")

        except Exception as e:
            logger.error(f"Data loading and validation failed: {str(e)}")
            raise

    def _determine_bayesian_processing_approach(self) -> None:
        """
        Determine whether to use asset-scale processing based on data characteristics.
        """
        if self.use_asset_scale_bayesian_processing is not None:
            # User explicitly set the approach
            approach = "asset-scale" if self.use_asset_scale_bayesian_processing else "traditional"
            logger.info(f"Using {approach} processing (user-specified)")
            return

        # Auto-determine based on data characteristics
        well_count = len(self.well_data['WellName'].unique())

        # Decision criteria
        if well_count >= 1000:
            # Large asset: definitely use asset-scale
            self.use_asset_scale_bayesian_processing = True
            logger.info(f"Auto-selected asset-scale processing for {well_count} wells (large asset)")
        elif well_count >= 500 and self.time_limit_minutes <= 15:
            # Medium asset with time constraints: use asset-scale
            self.use_asset_scale_bayesian_processing = True
            logger.info(f"Auto-selected asset-scale processing for {well_count} wells (time constraint: {self.time_limit_minutes}min)")
        else:
            # Small asset: traditional approach is sufficient
            self.use_asset_scale_bayesian_processing = False
            logger.info(f"Auto-selected traditional processing for {well_count} wells (small asset)")

        # Log the processing approach
        if self.use_asset_scale_bayesian_processing:
            logger.info("Asset-scale features enabled:")
            logger.info("  - Hierarchical Bayesian modeling")
            logger.info("  - Fast Approximate Bayesian Computation (ABC)")
            logger.info("  - Vectorized batch processing")
            logger.info("  - Memory-efficient uncertainty propagation")
            logger.info("  - Adaptive quality-based sampling")
        else:
            logger.info("Traditional processing features:")
            logger.info("  - Individual well processing")
            logger.info("  - Full Bayesian inference per well")
            logger.info("  - Comprehensive quality assessment")

    def _process_arps_dca(self) -> None:
        """Process wells using advanced ArpsDCA with intelligent multi-method fitting."""
        start_time = time.time()

        try:
            # Initialize advanced ArpsDCA with optimized parameters
            self.arps_dca = AdvancedArpsDCA(
                terminal_decline_rate=0.05,
                b_factor_max=2.0,
                min_production_months=6,
                oil_termination_rate=1.0,
                max_forecast_years=self.forecast_years,
                r_squared_threshold=0.7,  # Higher threshold - engine handles fallbacks
                pearson_threshold=0.7
            )

            # Get all wells for processing
            all_wells = self.well_data['WellName'].unique().tolist()
            logger.info(f"Processing {len(all_wells)} wells with enhanced data handling")

            # Process wells with intelligent method selection
            processing_results = self._process_wells_with_enhanced_handling(all_wells)

            # Store comprehensive processing statistics
            self.processing_stats['arps_dca'] = processing_results

            # Log summary results
            self._log_processing_summary(processing_results)

            self.processing_times['arps_dca'] = time.time() - start_time
            self.memory_usage['arps_dca'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        except Exception as e:
            logger.error(f"Advanced Arps DCA processing failed: {str(e)}")
            raise

    def _process_wells_with_enhanced_handling(self, all_wells: List[str]) -> Dict[str, Any]:
        """Process wells with enhanced data handling and quality tier classification."""
        successful_wells = []
        failed_wells = []
        quality_tiers = {
            'high': [], 'medium': [], 'low': [], 'very_low': [],
            'unreliable': [],  # For negative R² wells with high error ratios
            'failed': [], 'unknown': []
        }

        # Track negative R² wells specifically for business reporting
        negative_r_squared_wells = []

        for well_name in tqdm(all_wells, desc="Enhanced DCA fitting"):
            try:
                # Fit using enhanced engine with fallback methods
                fit_result = self.arps_dca.fit_decline_curve(
                    self.well_data,
                    well_name
                )

                if fit_result['success']:
                    successful_wells.append(well_name)

                    # Classify by quality tier - HANDLE ALL TIERS INCLUDING UNRELIABLE
                    quality_tier = fit_result.get('quality_tier', 'unknown')

                    # Add to appropriate tier (create new tier if needed)
                    if quality_tier not in quality_tiers:
                        quality_tiers[quality_tier] = []
                    quality_tiers[quality_tier].append(well_name)

                    # Track negative R² wells for business analysis
                    r_squared = fit_result.get('quality_metrics', {}).get('r_squared', 0)
                    if r_squared < 0:
                        negative_r_squared_wells.append({
                            'well_name': well_name,
                            'r_squared': r_squared,
                            'quality_tier': quality_tier,
                            'uncertainty_multiplier': fit_result.get('uncertainty_multiplier', 4.0),
                            'business_confidence': fit_result.get('business_confidence', 'very_low')
                        })
                        logger.info(f"Negative R² well processed: {well_name} (R²={r_squared:.3f}, tier={quality_tier})")

                    # Log detailed results
                    self._log_well_processing_details(well_name, fit_result)

                    # Generate forecast - Include ALL successful wells
                    forecast_result = self.arps_dca.forecast_production(well_name, forecast_months=self.forecast_months)

                    # BUSINESS ENHANCEMENT: Apply uncertainty multiplier to forecasts
                    # This ensures negative R² wells contribute to portfolio with appropriate uncertainty

                else:
                    failed_wells.append(well_name)
                    quality_tiers['failed'].append(well_name)
                    logger.warning(f"Well {well_name}: {fit_result['error']}")

            except Exception as e:
                failed_wells.append(well_name)
                quality_tiers['failed'].append(well_name)
                logger.error(f"Well {well_name} processing failed: {str(e)}")

        # Calculate enhanced statistics including negative R² analysis
        total_wells = len(all_wells)
        successful_count = len(successful_wells)
        failed_count = len(failed_wells)
        negative_r_squared_count = len(negative_r_squared_wells)

        # Calculate method performance
        method_performance = self._analyze_method_performance()

        # BUSINESS REPORTING: Enhanced statistics with negative R² insights
        logger.info("="*60)
        logger.info("ENHANCED ARPS DCA PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total wells processed: {total_wells}")
        logger.info(f"Successful fits: {successful_count} ({successful_count/total_wells*100:.1f}%)")
        logger.info(f"Failed fits: {failed_count} ({failed_count/total_wells*100:.1f}%)")
        logger.info(f"Negative R² wells: {negative_r_squared_count} ({negative_r_squared_count/total_wells*100:.1f}%)")
        logger.info("")
        logger.info("QUALITY TIER DISTRIBUTION:")
        for tier, wells in quality_tiers.items():
            if wells:
                logger.info(f"  {tier.upper()}: {len(wells)} wells ({len(wells)/total_wells*100:.1f}%)")

        # BUSINESS INSIGHT: Negative R² wells analysis
        if negative_r_squared_wells:
            logger.info("")
            logger.info("NEGATIVE R² WELLS BUSINESS ANALYSIS:")
            logger.info(f"  Total negative R² wells: {len(negative_r_squared_wells)}")

            # Group by quality tier
            tier_distribution = {}
            for well_info in negative_r_squared_wells:
                tier = well_info['quality_tier']
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

            for tier, count in tier_distribution.items():
                uncertainty_mult = self.arps_dca._calculate_uncertainty_multiplier_from_quality(tier)
                logger.info(f"    {tier}: {count} wells (uncertainty multiplier: {uncertainty_mult}x)")

            # Average R² for negative wells
            avg_negative_r2 = np.mean([w['r_squared'] for w in negative_r_squared_wells])
            logger.info(f"  Average negative R²: {avg_negative_r2:.3f}")
            logger.info(f"  Business impact: High uncertainty but portfolio value maintained")

        return {
            'total_wells': total_wells,
            'successful_wells': successful_count,
            'failed_wells': failed_count,
            'success_rate': (successful_count / total_wells) * 100,
            'successful_well_names': successful_wells,
            'failed_well_names': failed_wells,
            'quality_tiers': quality_tiers,
            'quality_tier_distribution': {tier: len(wells) for tier, wells in quality_tiers.items()},  # Add expected format for visualization/reporting
            'negative_r_squared_wells': negative_r_squared_wells,  # Business insight
            'negative_r_squared_count': negative_r_squared_count,
            'method_performance': method_performance
        }

    def _log_well_processing_details(self, well_name: str, fit_result: Dict[str, Any]) -> None:
        """Log detailed processing information for individual wells."""
        quality_metrics = fit_result.get('quality_metrics', {})
        quality_tier = fit_result.get('quality_tier', 'unknown')
        method = fit_result.get('method', 'unknown')
        data_points = fit_result.get('data_points_used', 0)
        uncertainty = fit_result.get('uncertainty_multiplier', 1.0)

        logger.debug(f"Well {well_name}: "
                   f"Quality={quality_tier}, "
                   f"Method={method}, "
                   f"R²={quality_metrics.get('r_squared', 0):.3f}, "
                   f"DataPoints={data_points}, "
                   f"Uncertainty={uncertainty:.1f}x")

    def _log_processing_summary(self, processing_results: Dict[str, Any]) -> None:
        """Log comprehensive processing summary."""
        logger.info("="*60)
        logger.info("ENHANCED ARPS DCA PROCESSING RESULTS:")
        logger.info(f"  Total wells processed: {processing_results['total_wells']}")
        logger.info(f"  Successfully fitted: {processing_results['successful_wells']} ({processing_results['success_rate']:.1f}%)")
        logger.info(f"  Failed wells: {processing_results['failed_wells']} ({(processing_results['failed_wells']/processing_results['total_wells'])*100:.1f}%)")
        logger.info("")

        # Quality tier distribution (already logged in enhanced processing)
        if 'quality_tiers' in processing_results:
            logger.info("QUALITY TIER SUMMARY:")
            for tier, wells in processing_results['quality_tiers'].items():
                if wells:
                    percentage = (len(wells) / processing_results['total_wells']) * 100
                    logger.info(f"  {tier.upper()}: {len(wells)} wells ({percentage:.1f}%)")

        # Method performance summary
        if 'method_performance' in processing_results:
            logger.info("")
            logger.info("METHOD PERFORMANCE SUMMARY:")
            for method, stats in processing_results['method_performance'].items():
                if stats.get('count', 0) > 0:
                    success_rate = (stats.get('count', 0) - stats.get('warning_count', 0)) / stats.get('count', 1) * 100
                    logger.info(f"  {method}: {stats.get('count', 0)} wells (success: {success_rate:.1f}%)")

        logger.info("="*60)

    def _analyze_failures_with_validation(self, failed_wells: List[str]) -> Dict:
        """
        Analyze failures using enhanced validation capabilities.

        Note: With enhanced processing, failures should be minimal.
        """
        failure_analysis = {
            'data_quality': {'count': 0, 'subcategories': {}},
            'physical_constraints': {'count': 0, 'subcategories': {}},
            'convergence': {'count': 0, 'subcategories': {}},
            'validation': {'count': 0, 'subcategories': {}}
        }

        for well_name in failed_wells:
            fit_result = self.arps_dca.fit_results.get(well_name)
            if fit_result:
                error_msg = fit_result.error or "Unknown error"

                # Categorize failures (should be rare with enhanced processing)
                if "No production data" in error_msg:
                    failure_analysis['data_quality']['count'] += 1
                    failure_analysis['data_quality']['subcategories'].setdefault(
                        'no_data', []).append(well_name)
                elif "Physical constraints" in error_msg:
                    failure_analysis['physical_constraints']['count'] += 1
                    failure_analysis['physical_constraints']['subcategories'].setdefault(
                        'constraints', []).append(well_name)
                else:
                    failure_analysis['validation']['count'] += 1
                    failure_analysis['validation']['subcategories'].setdefault(
                        'other', []).append(well_name)

        return failure_analysis

    def _analyze_method_performance(self) -> Dict:
        """
        Analyze performance of different fitting methods.

        Returns:
            Dictionary with method performance statistics
        """
        method_stats = {}

        for well_name, fit_result in self.arps_dca.fit_results.items():
            # Include ALL successful fits for accurate method performance
            if fit_result.success and fit_result.method:
                method = fit_result.method
                if method not in method_stats:
                    method_stats[method] = {
                        'count': 0,
                        'avg_r_squared': 0,
                        'avg_pearson_r': 0,
                        'warning_count': 0
                    }

                method_stats[method]['count'] += 1

                if fit_result.quality_metrics:
                    method_stats[method]['avg_r_squared'] += fit_result.quality_metrics.get('r_squared', 0)
                    method_stats[method]['avg_pearson_r'] += fit_result.quality_metrics.get('pearson_r', 0)

                validation = self.arps_dca.validation_results.get(well_name)
                if validation and validation.warnings:
                    method_stats[method]['warning_count'] += len(validation.warnings)

        # Calculate averages
        for method, stats in method_stats.items():
            if stats['count'] > 0:
                stats['avg_r_squared'] /= stats['count']
                stats['avg_pearson_r'] /= stats['count']

        return method_stats

    def _generate_probabilistic_forecasts(self) -> None:
        """Generate probabilistic forecasts using either fast hierarchical or comprehensive individual Bayesian methods."""
        start_time = time.time()

        try:
            successful_wells = self.processing_stats['arps_dca']['successful_well_names']

            if not successful_wells:
                raise ValueError("No successful wells for probabilistic forecasting")

            logger.info(f"Generating probabilistic forecasts for {len(successful_wells)} wells")

            # CLARIFIED WORKFLOW: Choose processing approach based on asset characteristics
            if self.use_asset_scale_bayesian_processing:
                # FAST HIERARCHICAL BAYESIAN: For large assets with time constraints
                logger.info("SELECTED: FAST HIERARCHICAL BAYESIAN PROCESSING")
                logger.info("   Rationale: Large asset size or time constraints require optimized approach")
                logger.info("   Features: Hierarchical modeling, ABC sampling, vectorized processing")
                logger.info("   Target: <10 minute runtime for portfolio-level uncertainty")
                self._generate_fast_hierarchical_bayesian_forecasts(successful_wells, start_time)
            else:
                # COMPREHENSIVE INDIVIDUAL BAYESIAN: For smaller assets requiring detailed analysis
                logger.info("SELECTED: COMPREHENSIVE INDIVIDUAL BAYESIAN PROCESSING")
                logger.info("   Rationale: Smaller asset size allows thorough well-by-well analysis")
                logger.info("   Features: Full Bayesian inference, detailed quality assessment, comprehensive uncertainty")
                logger.info("   Target: Complete probabilistic characterization per well")
                self._generate_comprehensive_individual_bayesian_forecasts(successful_wells, start_time)

        except Exception as e:
            logger.error(f"Probabilistic forecasting failed: {str(e)}")
            raise

    def _generate_fast_hierarchical_bayesian_forecasts(self, successful_wells: List[str], start_time: float) -> None:
        """
        FAST HIERARCHICAL BAYESIAN PROCESSING

        Optimized for large assets (100+ wells) with time constraints (<10 minutes).
        Uses asset-level hierarchical modeling with fast approximate methods.

        Workflow:
        1. Hierarchical Asset Model: Estimate population-level parameters
        2. Well Clustering: Group wells by production characteristics
        3. Fast ABC Sampling: Approximate Bayesian computation for speed
        4. Asset-Scale Aggregation: Direct portfolio-level uncertainty propagation
        """

        logger.info("FAST HIERARCHICAL BAYESIAN PROCESSING INITIATED")
        logger.info(f"Target: <{self.time_limit_minutes} minutes for {len(successful_wells)} wells")
        logger.info("="*60)

        # Initialize fast hierarchical Bayesian forecaster
        self.asset_forecaster = AssetScaleBayesianForecaster(
            n_samples=1000,  # Optimized for speed vs accuracy balance
            confidence_level=0.95,
            arps_dca_instance=self.arps_dca,
            random_seed=self.random_seed,
            uncertainty_level=self.uncertainty_level
        )

        # Phase 1: Hierarchical Asset Model (HIGHEST PRIORITY - biggest speed gain)
        logger.info("Phase 1: Building hierarchical asset model...")
        hierarchical_start = time.time()

        hierarchical_result = self.asset_forecaster.fit_hierarchical_asset_model(self.well_data)

        if hierarchical_result['success']:
            hierarchical_time = time.time() - hierarchical_start
            logger.info(f"Hierarchical model completed: {hierarchical_time:.2f} seconds")
            logger.info(f"   Population parameters estimated from {hierarchical_result['asset_results']['success_rate']:.1f}% of wells")
            logger.info(f"   Well clusters identified: {len(hierarchical_result['well_clusters'])}")

            # Phase 2: Fast batch processing of remaining wells
            batch_start = time.time()
            logger.info("Phase 2: Fast batch processing...")

            batch_result = self.asset_forecaster.batch_fit_asset_wells(self.well_data)

            if batch_result['success']:
                batch_time = time.time() - batch_start
                logger.info(f"Batch processing completed: {batch_time:.2f} seconds")
                logger.info(f"   Wells processed: {batch_result['processed_wells']}")

            # Phase 3: Asset-scale uncertainty propagation (avoids individual well aggregation)
            propagation_start = time.time()
            logger.info("Phase 3: Portfolio-level uncertainty propagation...")

            asset_forecast = self.asset_forecaster.asset_scale_uncertainty_propagation(
                forecast_months=self.forecast_months
            )

            if asset_forecast['success']:
                propagation_time = time.time() - propagation_start
                logger.info(f"Portfolio uncertainty completed: {propagation_time:.2f} seconds")
                logger.info(f"   Wells included: {asset_forecast['wells_included']}")

            # Performance summary
            total_time = time.time() - start_time
            successful_forecasts = list(self.asset_forecaster.fit_results.keys())
            failed_forecasts = [w for w in successful_wells if w not in successful_forecasts]

            forecast_success_rate = len(successful_forecasts) / len(successful_wells) * 100

            logger.info("="*60)
            logger.info("FAST HIERARCHICAL PROCESSING RESULTS:")
            logger.info(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"   Successful forecasts: {len(successful_forecasts)}/{len(successful_wells)} ({forecast_success_rate:.1f}%)")
            logger.info(f"   Performance: {total_time/len(successful_wells):.3f} seconds per well")

            # Method distribution
            stats = self.asset_forecaster.processing_stats
            logger.info("Processing method distribution:")
            logger.info(f"   Hierarchical Bayesian: {stats['hierarchical_wells']} wells")
            logger.info(f"   Fast ABC: {stats['abc_wells']} wells")
            logger.info(f"   Deterministic + uncertainty: {stats['deterministic_wells']} wells")

            # Store processing statistics
            self.processing_stats['probabilistic_forecasting'] = {
                'total_forecasts': len(successful_wells),
                'successful_forecasts': len(successful_forecasts),
                'failed_forecasts': len(failed_forecasts),
                'forecast_success_rate': forecast_success_rate,
                'successful_well_names': successful_forecasts,
                'failed_well_names': failed_forecasts,
                'method': 'Fast Hierarchical Bayesian (Asset-Scale)',
                'processing_approach': 'hierarchical_asset_modeling',
                'processing_time': total_time,
                'performance_per_well': total_time / len(successful_wells),
                'method_distribution': stats,
                'phase_times': {
                    'hierarchical_modeling': hierarchical_time,
                    'batch_processing': batch_time,
                    'uncertainty_propagation': propagation_time
                }
            }

            # Time target assessment
            time_limit_seconds = self.time_limit_minutes * 60
            if total_time <= time_limit_seconds:
                logger.info(f"TIME TARGET MET: {total_time:.1f}s < {time_limit_seconds}s limit")
            else:
                logger.warning(f"TIME TARGET EXCEEDED: {total_time:.1f}s > {time_limit_seconds}s limit")

        else:
            logger.error("Hierarchical asset model failed, falling back to comprehensive processing")
            self._generate_comprehensive_individual_bayesian_forecasts(successful_wells, start_time)

        self.processing_times['probabilistic_forecasting'] = time.time() - start_time
        self.memory_usage['probabilistic_forecasting'] = psutil.Process().memory_info().rss / 1024 / 1024

    def _generate_comprehensive_individual_bayesian_forecasts(self, successful_wells: List[str], start_time: float) -> None:
        """
        COMPREHENSIVE INDIVIDUAL BAYESIAN PROCESSING

        Thorough well-by-well Bayesian analysis for smaller assets.
        Provides detailed uncertainty characterization and quality assessment.

        Workflow:
        1. Individual Well Analysis: Full Bayesian inference per well
        2. Quality-Aware Processing: Uncertainty adjustment based on fit quality
        3. Comprehensive Uncertainty: Detailed confidence intervals and posteriors
        4. Traditional Aggregation: Sum individual well forecasts with correlations
        """

        logger.info("COMPREHENSIVE INDIVIDUAL BAYESIAN PROCESSING INITIATED")
        logger.info(f"Processing {len(successful_wells)} wells with detailed analysis")
        logger.info("="*60)

        # Initialize comprehensive Bayesian forecaster
        self.bayesian_forecaster = ModernizedBayesianForecaster(
            n_samples=1000,
            confidence_level=0.95,
            use_analytical_posteriors=True,
            cache_results=True,
            arps_dca_instance=self.arps_dca,
            random_seed=self.random_seed,
            uncertainty_level=self.uncertainty_level
        )

        # Process wells with comprehensive analysis
        successful_forecasts = []
        failed_forecasts = []
        negative_r2_wells_processed = 0

        logger.info("Processing wells with comprehensive Bayesian analysis...")
        logger.info("CORRECTED WORKFLOW: Using integrated Bayesian fitting + forecasting (no duplicate Monte Carlo)")

        for well_name in tqdm(successful_wells, desc="Comprehensive Bayesian analysis"):
            try:
                # Get fit quality for uncertainty adjustment - Access FitResult dataclass attributes correctly
                fit_result_obj = self.arps_dca.fit_results[well_name]  # This is a FitResult dataclass
                quality_metrics = fit_result_obj.quality_metrics or {}
                r_squared = quality_metrics.get('r_squared', 0.5)

                # Track negative R² wells
                if r_squared < 0:
                    negative_r2_wells_processed += 1
                    logger.debug(f"Processing negative R² well: {well_name} (R²={r_squared:.3f})")

                # Calculate uncertainty adjustment - Calculate quality_tier directly from existing objects (efficient)
                validation_result = self.arps_dca.validation_results.get(well_name)
                quality_tier = self.arps_dca._determine_quality_tier(fit_result_obj, validation_result)
                uncertainty_multiplier = self.arps_dca._calculate_uncertainty_multiplier_from_quality(
                    quality_tier,
                    fit_result_obj.method,
                    validation_result
                )

                # Fit comprehensive Bayesian decline model with integrated forecasting
                well_data = self.well_data[self.well_data['WellName'] == well_name]
                model_result = self.bayesian_forecaster.fit_bayesian_decline(production_data=well_data, well_name=well_name)

                if model_result['success']:
                    # CORRECTED: Use pre-computed Bayesian forecasts instead of Monte Carlo
                    # The forecasts are already generated during Bayesian fitting process
                    bayesian_forecasts = model_result.get('bayesian_forecasts')
                    
                    if bayesian_forecasts and bayesian_forecasts['success']:
                        successful_forecasts.append(well_name)

                        # Enhanced logging for negative R² wells
                        if r_squared < 0:
                            logger.info(f"Negative R² well successfully processed: {well_name}")
                            logger.info(f"   R²={r_squared:.3f}, uncertainty_multiplier={uncertainty_multiplier:.1f}x")

                        logger.debug(f"Well {well_name}: Comprehensive Bayesian forecast successful, "
                                   f"EUR P50={bayesian_forecasts['cumulative_percentiles']['P50'][-1]:.0f}")
                    else:
                        failed_forecasts.append(well_name)
                        logger.warning(f"Well {well_name}: Bayesian forecast generation failed")
                else:
                    failed_forecasts.append(well_name)
                    logger.warning(f"Well {well_name}: Bayesian model fitting failed: {model_result.get('error', 'Unknown error')}")

            except Exception as e:
                failed_forecasts.append(well_name)
                logger.error(f"Well {well_name}: Comprehensive analysis error: {str(e)}")
                continue

        # Performance summary
        forecast_success_rate = len(successful_forecasts) / len(successful_wells) * 100
        total_time = time.time() - start_time

        logger.info("="*60)
        logger.info("COMPREHENSIVE PROCESSING RESULTS:")
        logger.info(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"   Successful forecasts: {len(successful_forecasts)}/{len(successful_wells)} ({forecast_success_rate:.1f}%)")
        logger.info(f"   Failed forecasts: {len(failed_forecasts)}")
        logger.info(f"   Negative R² wells processed: {negative_r2_wells_processed}")
        logger.info(f"   Performance: {total_time/len(successful_wells):.3f} seconds per well")

        # Store comprehensive statistics
        self.processing_stats['probabilistic_forecasting'] = {
            'total_forecasts': len(successful_wells),
            'successful_forecasts': len(successful_forecasts),
            'failed_forecasts': len(failed_forecasts),
            'forecast_success_rate': forecast_success_rate,
            'successful_well_names': successful_forecasts,
            'failed_well_names': failed_forecasts,
            'method': 'Comprehensive Individual Bayesian (Well-by-Well)',
            'processing_approach': 'individual_well_analysis',
            'processing_time': total_time,
            'performance_per_well': total_time / len(successful_wells),
            'negative_r2_wells_processed': negative_r2_wells_processed,
            'comprehensive_analysis_features': [
                'full_bayesian_inference',
                'quality_aware_uncertainty',
                'detailed_posteriors',
                'individual_well_characterization'
            ]
        }

        self.processing_times['probabilistic_forecasting'] = time.time() - start_time
        self.memory_usage['probabilistic_forecasting'] = psutil.Process().memory_info().rss / 1024 / 1024


    def _aggregate_to_asset_level(self) -> None:
        """Aggregate well forecasts to asset level using proper AssetAggregator with enhanced uncertainty handling."""
        start_time = time.time()

        try:
            successful_forecasts = self.processing_stats['probabilistic_forecasting']['successful_well_names']

            if self.use_asset_scale_bayesian_processing and hasattr(self, 'asset_forecaster') and self.asset_forecaster:
                # OPTIMIZATION: Check if asset forecast is already computed to avoid redundant calculations
                logger.info("Extracting forecasts from asset-scale forecaster")

                # Check if asset forecast is already computed in memory
                if hasattr(self.asset_forecaster, 'asset_forecast') and self.asset_forecaster.asset_forecast is not None:
                    self.asset_forecast = self.asset_forecaster.asset_forecast
                    logger.info("Asset-scale forecast retrieved from cache")
                else:
                    # OPTIMIZATION: Only generate if not already computed
                    logger.info("Generating asset-scale forecast (not cached)")
                    asset_results = self.asset_forecaster.asset_scale_uncertainty_propagation(self.forecast_months)

                    if asset_results['success'] and 'asset_forecast_percentiles' in asset_results:
                        # Convert to expected format - Map percentiles to industry convention
                        forecast_percentiles = asset_results['asset_forecast_percentiles']

                        # Industry convention mapping - P10 gets optimistic values, P90 gets conservative values
                        self.asset_forecast = pd.DataFrame({
                            'Date': pd.date_range(start='2025-01-01', periods=self.forecast_months, freq='M'),
                            'P10_Production_bbl': forecast_percentiles.get('P10', np.zeros(self.forecast_months)),  # P10 = optimistic (high)
                            'P50_Production_bbl': forecast_percentiles.get('P50', np.zeros(self.forecast_months)),  # P50 = median
                            'P90_Production_bbl': forecast_percentiles.get('P90', np.zeros(self.forecast_months))   # P90 = conservative (low)
                        })

                        # Add cumulative columns
                        self.asset_forecast['P10_Cumulative_bbl'] = self.asset_forecast['P10_Production_bbl'].cumsum()
                        self.asset_forecast['P50_Cumulative_bbl'] = self.asset_forecast['P50_Production_bbl'].cumsum()
                        self.asset_forecast['P90_Cumulative_bbl'] = self.asset_forecast['P90_Production_bbl'].cumsum()

                        # OPTIMIZATION: Cache the result to avoid recomputation
                        self.asset_forecaster.asset_forecast = self.asset_forecast

                        logger.info("Asset-scale forecast generated and cached successfully")
                    else:
                        logger.warning("Asset-scale forecast failed, falling back to traditional aggregation")
                        # Fall back to proper aggregator-based aggregation
                        self._aggregate_using_asset_aggregator(successful_forecasts)

            else:
                # Use proper AssetAggregator for traditional forecasting
                logger.info("Using AssetAggregator for traditional forecasting aggregation")
                self._aggregate_using_asset_aggregator(successful_forecasts)

            # Store aggregation results
            self.aggregation_results = {
                'asset_forecast': self.asset_forecast,
                'well_count': len(successful_forecasts),
                'aggregation_method': 'Asset-Scale' if (hasattr(self, 'asset_forecaster') and self.asset_forecaster and
                                                       hasattr(self.asset_forecaster, 'fit_results') and self.asset_forecaster.fit_results) else 'AssetAggregator'
            }

            logger.info("Asset aggregation completed successfully")
            logger.info(f"  Wells aggregated: {len(successful_forecasts)}")
            logger.info(f"  Method: {self.aggregation_results['aggregation_method']}")

        except Exception as e:
            logger.error(f"Asset aggregation failed: {str(e)}")
            raise

        self.processing_times['asset_aggregation'] = time.time() - start_time
        self.memory_usage['asset_aggregation'] = psutil.Process().memory_info().rss / 1024 / 1024

    def _aggregate_using_asset_aggregator(self, successful_forecasts: List[str]) -> None:
        """Use proper AssetAggregator class for well forecast aggregation with enhanced uncertainty handling."""

        logger.info("Initializing AssetAggregator for proper aggregation")

        # Initialize AssetAggregator
        aggregator = AssetAggregator(
            forecast_months=self.forecast_months,
            validation_enabled=True
        )

        # Collect well forecasts with enhanced uncertainty handling for negative R² wells
        well_forecasts = {}
        negative_r2_wells_processed = 0

        for well_name in successful_forecasts:
            try:
                # CORRECTED: Use pre-computed Bayesian forecasts instead of Monte Carlo
                bayesian_forecasts = self.bayesian_forecaster.get_bayesian_forecasts(well_name)
                
                if bayesian_forecasts['success']:
                    # Use pre-computed Bayesian forecast percentiles
                    forecast_percentiles = bayesian_forecasts['forecast_percentiles']

                    forecast_df = pd.DataFrame({
                        'Date': pd.date_range(start='2025-01-01', periods=self.forecast_months, freq='M'),
                        'P10_Production_bbl': forecast_percentiles['P10'],
                        'P50_Production_bbl': forecast_percentiles['P50'],
                        'P90_Production_bbl': forecast_percentiles['P90']
                    })

                    # BUSINESS ENHANCEMENT: Apply uncertainty multipliers for negative R² wells
                    if well_name in self.arps_dca.fit_results:
                        fit_result = self.arps_dca.fit_results[well_name]
                        if fit_result.quality_metrics and fit_result.quality_metrics.get('r_squared', 0) < 0:
                            # Apply enhanced uncertainty for negative R² wells
                            quality_tier = self.processing_stats['arps_dca']['quality_tiers']

                            # Find the well's quality tier
                            well_tier = 'very_low'  # default
                            for tier, wells in quality_tier.items():
                                if well_name in wells:
                                    well_tier = tier
                                    break

                            uncertainty_multiplier = self.arps_dca._calculate_uncertainty_multiplier_from_quality(well_tier)

                            # Widen uncertainty bands for negative R² wells
                            p50_values = forecast_df['P50_Production_bbl'].values
                            uncertainty_range = (forecast_df['P10_Production_bbl'] - forecast_df['P90_Production_bbl']).values

                            # Apply enhanced uncertainty
                            enhanced_uncertainty = uncertainty_range * uncertainty_multiplier

                            forecast_df['P10_Production_bbl'] = p50_values + enhanced_uncertainty / 2
                            forecast_df['P90_Production_bbl'] = p50_values - enhanced_uncertainty / 2

                            # Ensure non-negative production
                            forecast_df['P10_Production_bbl'] = np.maximum(forecast_df['P10_Production_bbl'], 0)
                            forecast_df['P90_Production_bbl'] = np.maximum(forecast_df['P90_Production_bbl'], 0)

                            negative_r2_wells_processed += 1
                            logger.info(f"Applied {uncertainty_multiplier}x uncertainty to negative R² well: {well_name}")

                    # Validate forecast data
                    if ((forecast_df['P10_Production_bbl'] < 0).any() or
                        (forecast_df['P50_Production_bbl'] < 0).any() or
                        (forecast_df['P90_Production_bbl'] < 0).any()):
                        logger.warning(f"Skipping well {well_name}: Contains negative production values")
                        continue

                    well_forecasts[well_name] = forecast_df

                else:
                    logger.warning(f"Bayesian forecast failed for well {well_name}: {bayesian_forecasts.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error processing well {well_name} for aggregation: {str(e)}")
                continue

        # Aggregate forecasts using AssetAggregator
        if not well_forecasts:
            raise Exception("No valid well forecasts available for aggregation")

        logger.info(f"Aggregating {len(well_forecasts)} well forecasts using AssetAggregator")
        logger.info(f"CORRECTED: Using pre-computed Bayesian forecasts (not Monte Carlo recomputation)")
        logger.info(f"Enhanced uncertainty applied to {negative_r2_wells_processed} negative R² wells")

        # Use AssetAggregator for proper aggregation
        self.asset_forecast = aggregator.aggregate_well_forecasts(
            well_forecasts=well_forecasts,
            start_date='2025-01-01'
        )

        # Store aggregator results for later use
        self.aggregator = aggregator

        logger.info(f"AssetAggregator completed successfully")
        logger.info(f"  Wells included: {len(well_forecasts)}")
        logger.info(f"  Negative R² wells with enhanced uncertainty: {negative_r2_wells_processed}")

        # Log aggregation summary
        aggregation_summary = aggregator.get_aggregation_summary()
        if aggregation_summary:
            logger.info("Aggregation Summary:")
            for section, metrics in aggregation_summary.items():
                logger.info(f"  {section}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value}")

    def _calculate_revenue_forecasts(self) -> None:
        """Calculate revenue forecasts using aggregated production."""
        start_time = time.time()

        try:
            logger.info("Calculating revenue forecasts...")

            # Initialize revenue calculator with correct parameters
            self.revenue_calculator = RevenueCalculator(
                price_escalation_rate=0.02,
                use_price_escalation=True,
                validation_enabled=True
            )

            # Calculate revenue forecasts using the correct method
            self.revenue_forecast = self.revenue_calculator.calculate_asset_revenue(
                production_forecast=self.asset_forecast,
                price_data=self.price_data
            )

            # Log revenue statistics
            revenue_metrics = self.revenue_calculator.revenue_metrics
            p50_revenue = revenue_metrics.get('total_p50_revenue_usd', 0)
            p10_revenue = revenue_metrics.get('total_p10_revenue_usd', 0)
            p90_revenue = revenue_metrics.get('total_p90_revenue_usd', 0)

            logger.info(f"Revenue forecasts completed:")
            logger.info(f"  P10 Revenue: ${p10_revenue:,.0f}")
            logger.info(f"  P50 Revenue: ${p50_revenue:,.0f}")
            logger.info(f"  P90 Revenue: ${p90_revenue:,.0f}")

            # Export revenue forecast
            output_file = self.output_dir / "asset_revenue_forecast.csv"
            self.revenue_calculator.export_revenue_forecast(str(output_file))

            self.processing_times['revenue_calculation'] = time.time() - start_time
            self.memory_usage['revenue_calculation'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        except Exception as e:
            logger.error(f"Revenue calculation failed: {str(e)}")
            raise

    def _perform_comprehensive_validation(self) -> None:
        """Perform comprehensive validation of results."""
        start_time = time.time()

        try:
            logger.info("Performing comprehensive validation...")

            # Validate physical constraints
            physical_validation = self._validate_physical_constraints()

            # Validate statistical properties
            statistical_validation = self._validate_statistical_properties()

            # Validate business logic
            business_validation = self._validate_business_logic()

            # Validate data quality
            data_quality_validation = self._validate_data_quality()

            # Compile validation results
            self.validation_results = {
                'physical_constraints': physical_validation,
                'statistical_properties': statistical_validation,
                'business_logic': business_validation,
                'data_quality': data_quality_validation,
                'overall_valid': all([
                    physical_validation['valid'],
                    statistical_validation['valid'],
                    len(business_validation['issues']) == 0,  # check issues length instead of 'valid' key
                    data_quality_validation['valid']
                ])
            }

            logger.info("Comprehensive validation completed:")
            logger.info(f"  Physical constraints: {'✓' if physical_validation['valid'] else '✗'}")
            logger.info(f"  Statistical properties: {'✓' if statistical_validation['valid'] else '✗'}")
            logger.info(f"  Business logic: {'✓' if len(business_validation['issues']) == 0 else '✗'}")
            logger.info(f"  Data quality: {'✓' if data_quality_validation['valid'] else '✗'}")
            logger.info(f"  Overall valid: {'✓' if self.validation_results['overall_valid'] else '✗'}")

            self.processing_times['comprehensive_validation'] = time.time() - start_time

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {str(e)}")
            raise

    def _validate_physical_constraints(self) -> Dict[str, Any]:
        """Validate physical constraints using enhanced validation."""
        issues = []
        warnings = []

        # Check ArpsDCA validation results
        for well_name, validation in self.arps_dca.validation_results.items():
            if not validation.valid:
                issues.extend([f"Well {well_name}: {issue}" for issue in validation.issues])
            if validation.warnings:
                warnings.extend([f"Well {well_name}: {warning}" for warning in validation.warnings])

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_wells_validated': len(self.arps_dca.validation_results)
        }

    def _validate_statistical_properties(self) -> Dict[str, Any]:
        """Validate statistical properties of forecasts."""
        issues = []
        warnings = []

        # Check R-squared distribution
        r_squared_values = []
        for well_name, fit_result in self.arps_dca.fit_results.items():
            if fit_result.success and fit_result.quality_metrics:
                r_squared_values.append(fit_result.quality_metrics.get('r_squared', 0))

        if r_squared_values:
            avg_r_squared = np.mean(r_squared_values)
            if avg_r_squared < 0.5:
                issues.append(f"Low average R-squared: {avg_r_squared:.3f}")
            elif avg_r_squared < 0.7:
                warnings.append(f"Moderate average R-squared: {avg_r_squared:.3f}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'avg_r_squared': np.mean(r_squared_values) if r_squared_values else 0,
            'wells_evaluated': len(r_squared_values)
        }

    def _validate_business_logic(self) -> Dict[str, Any]:
        """Validate business logic and forecast reasonableness."""

        # Initialize validation results
        business_validation = {
            'revenue_positive': None,
            'issues': [],
            'warnings': []
        }

        # Business logic validation
        if hasattr(self, 'revenue_forecast') and self.revenue_forecast is not None and not self.revenue_forecast.empty:
            # Check revenue forecast structure and validate appropriately
            try:
                # Look for P50 revenue in the appropriate column
                if 'P50_Revenue' in self.revenue_forecast.columns:
                    p50_revenue = self.revenue_forecast['P50_Revenue'].iloc[-1]
                elif 'Revenue' in self.revenue_forecast.columns:
                    p50_revenue = self.revenue_forecast['Revenue'].iloc[-1]
                else:
                    # Find the first revenue-like column
                    revenue_cols = [col for col in self.revenue_forecast.columns if 'revenue' in col.lower()]
                    if revenue_cols:
                        p50_revenue = self.revenue_forecast[revenue_cols[0]].iloc[-1]
                    else:
                        p50_revenue = None

                if p50_revenue is not None and p50_revenue <= 0:
                    business_validation['revenue_positive'] = False
                    business_validation['issues'].append("Revenue forecast is not positive")
                else:
                    business_validation['revenue_positive'] = True

            except Exception as e:
                logger.warning(f"Could not validate revenue forecast structure: {e}")
                business_validation['revenue_positive'] = None

        return business_validation

    def _validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality."""
        issues = []
        warnings = []

        # Use existing validation report
        if self.validation_report:
            validation_summary = self.validation_report.get('validation_report', {})
            data_quality_issues = validation_summary.get('data_quality_issues', [])

            if data_quality_issues:
                warnings.extend([f"Data quality: {issue}" for issue in data_quality_issues])

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def _generate_advanced_analytics(self) -> None:
        """Generate advanced analytics and visualizations."""
        start_time = time.time()

        try:
            logger.info("Generating advanced analytics and visualizations...")

            # Create visualizations directory
            visualizations_dir = self.output_dir / "visualizations"
            
            if not visualizations_dir.exists():
                visualizations_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created visualizations directory")

            # Initialize visualizer with the correct directory
            visualizer = SingleScenarioResultsVisualizer(output_dir=str(self.output_dir), analysis_dir=str(visualizations_dir))

            # Generate comprehensive analysis
            pipeline_results = self._get_pipeline_results()

            visualizer.generate_single_scenario_visualizations(pipeline_results=pipeline_results, well_data=self.well_data, processing_stats=self.processing_stats)

            logger.info("Advanced analytics and visualizations completed successfully")

            self.processing_times['advanced_analytics'] = time.time() - start_time

            # Save the scenario results to the output directory
            save_scenario_results(self.output_dir, pipeline_results, self.well_data, self.processing_stats)

        except Exception as e:
            logger.error(f"Advanced analytics generation failed: {str(e)}")
            raise

    def _generate_final_report(self) -> None:
        """Generate comprehensive final report using unified reporting system."""
        start_time = time.time()

        try:
            logger.info("Generating comprehensive final report...")

            # Initialize comprehensive reporter
            reporter = ComprehensiveReporter(str(self.output_dir))

            # Compile pipeline results
            pipeline_results = self._get_pipeline_results()

            # Generate unified report
            reporter.generate_unified_report(
                pipeline_results=pipeline_results,
                processing_stats=self.processing_stats,
                validation_results=getattr(self, 'validation_results', {}),
                processing_times=self.processing_times,
                memory_usage=self.memory_usage
            )

            logger.info("Comprehensive final report generated successfully")

            self.processing_times['final_report'] = time.time() - start_time

        except Exception as e:
            logger.error(f"Final report generation failed: {str(e)}")
            raise

    def _analyze_enhanced_quality_statistics(self) -> Dict:
        """Analyze enhanced quality statistics from ArpsDCA results."""
        if not self.arps_dca:
            return {}

        fit_summary = self.arps_dca.get_fit_summary()

        if fit_summary.empty:
            return {}

        # Method distribution
        method_counts = fit_summary['method'].value_counts().to_dict()

        # Quality distribution
        quality_ranges = {
            'excellent': (fit_summary['r_squared'] > 0.8).sum(),
            'good': ((fit_summary['r_squared'] > 0.6) & (fit_summary['r_squared'] <= 0.8)).sum(),
            'fair': ((fit_summary['r_squared'] > 0.4) & (fit_summary['r_squared'] <= 0.6)).sum(),
            'poor': (fit_summary['r_squared'] <= 0.4).sum()
        }

        # Warning statistics
        avg_warnings = fit_summary['warnings'].mean()
        wells_with_warnings = (fit_summary['warnings'] > 0).sum()

        return {
            'method_distribution': method_counts,
            'quality_distribution': quality_ranges,
            'avg_r_squared': fit_summary['r_squared'].mean(),
            'avg_warnings': avg_warnings,
            'wells_with_warnings': wells_with_warnings,
            'total_wells': len(fit_summary)
        }

    def _get_bayesian_statistics(self) -> Dict:
        """Get Bayesian forecasting statistics."""
        if not self.bayesian_forecaster:
            return {}

        bayesian_summary = self.bayesian_forecaster.get_fit_summary()

        if bayesian_summary.empty:
            return {}

        return {
            'total_wells': len(bayesian_summary),
            'avg_parameter_uncertainty': bayesian_summary['parameter_uncertainty'].mean(),
            'avg_forecast_uncertainty': bayesian_summary['forecast_uncertainty'].mean(),
            'wells_high_uncertainty': (bayesian_summary['parameter_uncertainty'] > 0.5).sum(),
            'method': 'Advanced Bayesian inference with quality-aware uncertainty'
        }

    def _calculate_business_metrics(self) -> Dict:
        """Calculate key business metrics."""
        metrics = {}

        if hasattr(self, 'revenue_forecast') and self.revenue_forecast is not None and not self.revenue_forecast.empty:
            # Use the correct column names from the revenue forecast CSV structure
            if 'P10_Cumulative_Revenue_USD' in self.revenue_forecast.columns:
                p10_revenue = self.revenue_forecast['P10_Cumulative_Revenue_USD'].iloc[-1]
                p50_revenue = self.revenue_forecast['P50_Cumulative_Revenue_USD'].iloc[-1]
                p90_revenue = self.revenue_forecast['P90_Cumulative_Revenue_USD'].iloc[-1]
            else:
                # Fallback to monthly revenue if cumulative not available
                p10_revenue = self.revenue_forecast['P10_Revenue_USD'].iloc[-1]
                p50_revenue = self.revenue_forecast['P50_Revenue_USD'].iloc[-1]
                p90_revenue = self.revenue_forecast['P90_Revenue_USD'].iloc[-1]

            metrics.update({
                'total_revenue_p10': p10_revenue,
                'total_revenue_p50': p50_revenue,
                'total_revenue_p90': p90_revenue,
                'revenue_range': p10_revenue - p90_revenue,
                'revenue_uncertainty': (p10_revenue - p90_revenue) / p50_revenue * 100 if p50_revenue > 0 else 0
            })

        if hasattr(self, 'asset_forecast') and self.asset_forecast is not None and not self.asset_forecast.empty:
            # Check if asset_forecast has the expected structure
            if isinstance(self.asset_forecast, dict) and 'cumulative_percentiles' in self.asset_forecast:
                if 'P50' in self.asset_forecast['cumulative_percentiles']:
                    p50_eur = self.asset_forecast['cumulative_percentiles']['P50'][-1]
                    metrics['p50_eur'] = p50_eur
            elif isinstance(self.asset_forecast, pd.DataFrame):
                # Handle DataFrame case - look for P50 cumulative production
                if 'P50_Production_bbl' in self.asset_forecast.columns:
                    p50_eur = self.asset_forecast['P50_Production_bbl'].iloc[-1]
                    metrics['p50_eur'] = p50_eur

        return metrics

    def _generate_executive_summary(self, report: Dict) -> None:
        """Generate executive summary."""
        summary_file = self.output_dir / "executive_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("FORECASTING PIPELINE - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Pipeline Performance
            f.write("PIPELINE PERFORMANCE:\n")
            f.write(f"• Total Processing Time: {report['pipeline_summary']['total_processing_time']:.2f} seconds\n")
            f.write(f"• Peak Memory Usage: {report['pipeline_summary']['peak_memory_usage']:.1f} MB\n")
            f.write(f"• Wells Processed: {report['pipeline_summary']['wells_processed']}\n")
            f.write(f"• Success Rate: {report['pipeline_summary']['success_rate']:.1f}%\n\n")

            # Advanced DCA Results
            f.write("ADVANCED DCA RESULTS:\n")
            method_perf = report['advanced_dca_results']['method_performance']
            for method, stats in method_perf.items():
                f.write(f"• {method}: {stats['count']} wells, Avg R²: {stats['avg_r_squared']:.3f}\n")
            f.write("\n")

            # Business Metrics
            business_metrics = report['business_metrics']
            if business_metrics:
                f.write("BUSINESS METRICS:\n")
                f.write(f"• P10 Revenue: ${business_metrics.get('p10_revenue', 0):,.0f}\n")
                f.write(f"• P50 Revenue: ${business_metrics.get('p50_revenue', 0):,.0f}\n")
                f.write(f"• P90 Revenue: ${business_metrics.get('p90_revenue', 0):,.0f}\n")
                f.write(f"• Revenue Uncertainty: {business_metrics.get('revenue_uncertainty', 0):.1f}%\n")

        logger.info(f"Executive summary saved to {summary_file}")

    def _get_pipeline_results(self) -> Dict[str, Any]:
        """Get comprehensive pipeline results for reporting."""

        # Compile all pipeline results
        pipeline_results = {
            'arps_dca': self.arps_dca,
            'revenue_forecast': getattr(self, 'revenue_forecast', None),
            'asset_forecast': getattr(self, 'asset_forecast', None),
            'revenue_calculator': getattr(self, 'revenue_calculator', None),
            'bayesian_forecaster': getattr(self, 'bayesian_forecaster', None),
            'asset_forecaster': getattr(self, 'asset_forecaster', None),
            'use_asset_scale_bayesian_processing': self.use_asset_scale_bayesian_processing,
            'forecast_months': self.forecast_months,
            'forecast_years': self.forecast_years,
            'well_count': len(self.well_data['WellName'].unique()) if hasattr(self, 'well_data') else 0,
            'price_data': getattr(self, 'price_data', None),
            'output_dir': self.output_dir
        }

        return pipeline_results


def run_forecasting_pipeline(well_data_path: str, price_data_path: str, output_dir: str = "output", forecast_years: int = 30, use_asset_scale_bayesian_processing: bool = None, time_limit_minutes: int = 10, random_seed: int = None, uncertainty_level: str = 'standard') -> Dict[str, Any]:
    """
    Run the complete forecasting pipeline with asset-scale capabilities.

    Args:
        well_data_path: Path to well production data CSV
        price_data_path: Path to price data CSV
        output_dir: Output directory for results
        forecast_years: Number of years to forecast
        use_asset_scale_bayesian_processing: Whether to use asset-scale processing (auto-determined if None)
        time_limit_minutes: Time limit for processing in minutes
        random_seed: Random seed for reproducible results (None for random behavior)
        uncertainty_level: Uncertainty level ('standard', 'conservative', 'aggressive', 'high_uncertainty')

    Returns:
        Dictionary containing all pipeline results and performance statistics
    """
    pipeline = ProbablisticProductionForecastingPipeline(
        well_data_path=well_data_path,
        price_data_path=price_data_path,
        output_dir=output_dir,
        forecast_years=forecast_years,
        use_asset_scale_bayesian_processing=use_asset_scale_bayesian_processing,
        time_limit_minutes=time_limit_minutes,
        random_seed=random_seed,
        uncertainty_level=uncertainty_level
    )

    return pipeline.run_complete_pipeline()

def save_scenario_results(output_dir, pipeline_results, well_dat, processing_stats):
    # Save the results to the output directory
    with open(output_dir / 'pipeline_results.pkl', 'wb') as f:
        pickle.dump(pipeline_results, f)
    with open(output_dir / 'well_dat.pkl', 'wb') as f:
        pickle.dump(well_dat, f)
    with open(output_dir / 'processing_stats.pkl', 'wb') as f:
        pickle.dump(processing_stats, f)

def load_scenario_results(output_dir):
    # Load the results if they exist
    with open(output_dir / 'pipeline_results.pkl', 'rb') as f:
        pipeline_results = pickle.load(f)
    with open(output_dir / 'well_dat.pkl', 'rb') as f:
        well_dat = pickle.load(f)
    with open(output_dir / 'processing_stats.pkl', 'rb') as f:
        processing_stats = pickle.load(f)
    return pipeline_results, well_dat, processing_stats

def save_all_results(all_results: Dict, output_dir: str = "output") -> str:
    """Save all_results to both pickle and JSON formats for reuse."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle (preserves all objects including numpy arrays, DataFrames, etc.)
    pickle_file = output_path / "all_results.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save as JSON (for human readability, but limited data types)
    json_file = output_path / "all_results_summary.json"
    try:
        # Create a JSON-serializable summary
        json_summary = {}
        for scenario, results in all_results.items():
            json_summary[scenario] = {
                'success': results.get('success', False),
                'total_wells': results.get('total_wells', 0),
                'successful_wells': results.get('successful_wells', 0),
                'processing_time': results.get('processing_time', 0),
                'output_dir': str(results.get('output_dir', '')),
                'timestamp': results.get('timestamp', ''),
                'validation_status': results.get('validation_status', {})
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_summary, f, indent=2)
    
    except Exception as e:
        print(f"Warning: Could not save JSON summary: {e}")
    
    print(f"All results saved to: {pickle_file}")
    print(f"Summary saved to: {json_file}")
    return str(pickle_file)

def load_all_results(pickle_file: str = "output/all_results.pkl") -> Dict:
    """Load all_results from pickle file."""
    pickle_path = Path(pickle_file)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Results file not found: {pickle_file}")
    
    with open(pickle_path, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"All results loaded from: {pickle_file}")
    return all_results

# ========== CONFIGURATION ==========
main_configs = {
    'load_existing_results': False,  # False to run the full pipeline
    'name': 'Individual Well-Level Bayesian',
    'use_asset_scale_bayesian_processing': False,
    'uncertainty_levels': ['conservative', 'standard', 'aggressive'],
    'output_dir': 'output',
    'well_data_path': 'data/QCG_DS_Exercise_well_prod_data.csv',
    'price_data_path': 'data/QCG_DS_Exercise_price_data.csv',
    'forecast_years': 30,
    'random_seed': 42
}

# ========== MAIN RUN ==========
if __name__ == "__main__":
    # Check if user wants to load existing results instead of running pipeline
    if main_configs['load_existing_results']:
        print("Loading existing results from output/all_results.pkl...")
        try:
            all_results = load_all_results(f"{main_configs['output_dir']}/all_results.pkl")
            print(f"Successfully loaded results for scenarios: {list(all_results.keys())}")
        except FileNotFoundError:
            print("No existing results found. Please run the full pipeline first.")
            sys.exit(1)
    else: # Run the full pipeline
        all_results = {}
        
        for uncertainty_level in main_configs['uncertainty_levels']:
            print(f"\n{'='*60}")
            print(f"RUNNING: {main_configs['name']} with {uncertainty_level} uncertainty level")
            print(f"{'='*60}")
            
            # Run Pipeline
            results = run_forecasting_pipeline(
                well_data_path=main_configs['well_data_path'],
                price_data_path=main_configs['price_data_path'],
                output_dir=main_configs['output_dir'],
                forecast_years=main_configs['forecast_years'],
                use_asset_scale_bayesian_processing=main_configs['use_asset_scale_bayesian_processing'],
                random_seed=main_configs['random_seed'],  # Set for reproducible results
                uncertainty_level=uncertainty_level # uncertainty_level parameter
            )

            print(f"Forecasting pipeline with {uncertainty_level} uncertainty level completed successfully!")
            print(f"Output saved to: {results['output_dir']}") 

            all_results[uncertainty_level] = results
        
        # Save all results for future use
        save_all_results(all_results, main_configs['output_dir'])
    
    #------------ Individual Scenario Visualizations ------------
    print(f"\n{'='*60}")
    print("GENERATING INDIVIDUAL SCENARIO VISUALIZATIONS")
    print(f"{'='*60}")

    try:
        for uncertainty_level in main_configs['uncertainty_levels']:
            print(f"\n{'='*60}")
            print(f"LOADING: {main_configs['name']} with {uncertainty_level} uncertainty level")
            print(f"{'='*60}")

            # Load the results if they exist
            pipeline_results, well_dat, processing_stats = load_scenario_results(all_results[uncertainty_level]["output_dir"])

            visualizer = SingleScenarioResultsVisualizer(output_dir=str(all_results[uncertainty_level]["output_dir"]), analysis_dir=str(all_results[uncertainty_level]["output_dir"] / "visualizations"))

            # Generate comprehensive analysis
            visualizer.generate_single_scenario_visualizations(pipeline_results=pipeline_results, well_data=well_dat, processing_stats=processing_stats)

            # Print success message
            print(f"Single scenario visualizations for {uncertainty_level} uncertainty level completed successfully!")
            print(f"Visualizations saved to: {all_results[uncertainty_level]['output_dir']}/visualizations/")

    except Exception as e:
        print(f"Warning: Individual Scenario Visualizations generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    #------------ Acquisition Analysis Visualizations ------------
    print(f"\n{'='*60}")
    print("GENERATING ACQUISITION ANALYSIS VISUALIZATIONS")
    print(f"{'='*60}")

    try:
        # Initialize the acquisition analysis visualizer
        acquisition_visualizer = AcquisitionAnalysisVisualizer(output_dir=main_configs['output_dir'])
        
        # Generate all acquisition visualizations using the multi-scenario results
        acquisition_visualizer.generate_all_acquisition_visualizations(all_results)
        
        print("Acquisition analysis visualizations completed successfully!")
        print(f"Visualizations saved to: {main_configs['output_dir']}/visualizations/")

    except Exception as e:
        print(f"Warning: Acquisition Analysis Visualizations generation failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print("PIPELINE WITH ALL UNCERTAINTY LEVELS COMPLETED")
    print(f"{'='*80}")