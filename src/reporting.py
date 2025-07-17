"""
Comprehensive Reporting Module for Modernized Forecasting Pipeline

This module centralizes all report generation to eliminate redundancy and ensure
consistent, accurate reporting across the entire forecasting pipeline.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ComprehensiveReporter:
    """
    Centralized reporting system for the modernized forecasting pipeline.
    
    This class consolidates all report generation into a single, coherent system
    that produces accurate, consistent reports without redundancy.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the comprehensive reporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.pipeline_results = {}
        self.processing_stats = {}
        self.validation_results = {}
        self.performance_metrics = {}
        
    def generate_unified_report(self, 
                               pipeline_results: Dict[str, Any],
                               processing_stats: Dict[str, Any],
                               validation_results: Dict[str, Any],
                               processing_times: Dict[str, float],
                               memory_usage: Dict[str, float]) -> None:
        """
        Generate the unified comprehensive report.
        
        Args:
            pipeline_results: Complete pipeline results
            processing_stats: Processing statistics
            validation_results: Validation results
            processing_times: Processing time metrics
            memory_usage: Memory usage metrics
        """
        logger.info("Generating unified comprehensive report...")
        
        # Store data for report generation
        self.pipeline_results = pipeline_results
        self.processing_stats = processing_stats
        self.validation_results = validation_results
        self.processing_times = processing_times
        self.memory_usage = memory_usage
        
        # Generate the comprehensive report
        comprehensive_report = self._compile_comprehensive_report()
        
        # Save the unified report
        self._save_comprehensive_report(comprehensive_report)
        
        # Clean up redundant files
        self._cleanup_redundant_files()
        
        logger.info("Unified comprehensive report generated successfully")
    
    def _compile_comprehensive_report(self) -> Dict[str, Any]:
        """Compile all report sections into a comprehensive report."""
        
        # Extract key components
        arps_dca = self.pipeline_results.get('arps_dca')
        revenue_forecast = self.pipeline_results.get('revenue_forecast')
        asset_forecast = self.pipeline_results.get('asset_forecast')
        
        # Compile comprehensive report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "Modernized Forecasting Pipeline v2.0",
                "report_type": "Comprehensive Asset Analysis"
            },
            "executive_summary": self._generate_executive_summary(),
            "pipeline_performance": self._generate_pipeline_performance(),
            "arps_dca_analysis": self._generate_arps_dca_analysis(),
            "forecasting_results": self._generate_forecasting_results(),
            "revenue_analysis": self._generate_revenue_analysis(),
            "quality_assessment": self._generate_quality_assessment(),
            "validation_summary": self._generate_validation_summary(),
            "business_metrics": self._generate_business_metrics(),
            "technical_details": self._generate_technical_details()
        }
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary with key metrics."""
        
        # Get ArpsDCA results
        arps_stats = self.processing_stats.get('arps_dca', {})
        
        # Get revenue metrics
        revenue_metrics = self._extract_revenue_metrics()
        
        # Get forecasting statistics
        forecasting_stats = self._extract_forecasting_statistics()
        
        executive_summary = {
            "asset_overview": {
                "total_wells": arps_stats.get('total_wells', 0),
                "successful_wells": arps_stats.get('successful_wells', 0),
                "success_rate_percent": arps_stats.get('success_rate', 0),
                "processing_time_seconds": self.processing_times.get('total_pipeline', 0),
                "peak_memory_mb": self.memory_usage.get('peak', 0)
            },
            "production_forecast": {
                "forecast_period_years": 30,
                "p10_total_production_bbl": forecasting_stats.get('p10_total_production', 0),
                "p50_total_production_bbl": forecasting_stats.get('p50_total_production', 0),
                "p90_total_production_bbl": forecasting_stats.get('p90_total_production', 0)
            },
            "revenue_forecast": {
                "p10_total_revenue_usd": revenue_metrics.get('total_p10_revenue_usd', 0),
                "p50_total_revenue_usd": revenue_metrics.get('total_p50_revenue_usd', 0),
                "p90_total_revenue_usd": revenue_metrics.get('total_p90_revenue_usd', 0),
                "average_oil_price_per_bbl": revenue_metrics.get('average_price_per_bbl', 0),
                "revenue_uncertainty_percent": self._calculate_revenue_uncertainty(revenue_metrics)
            },
            "quality_summary": self._generate_quality_summary(),
            "method_performance": self._generate_method_performance_summary()
        }
        
        return executive_summary
    
    def _generate_pipeline_performance(self) -> Dict[str, Any]:
        """Generate pipeline performance metrics."""
        
        return {
            "processing_times": self.processing_times,
            "memory_usage": self.memory_usage,
            "system_performance": {
                "total_processing_time": self.processing_times.get('total_pipeline', 0),
                "arps_dca_time": self.processing_times.get('arps_dca', 0),
                "bayesian_forecasting_time": self.processing_times.get('bayesian_forecasting', 0),
                "revenue_calculation_time": self.processing_times.get('revenue_calculation', 0),
                "peak_memory_usage": self.memory_usage.get('peak', 0)
            }
        }
    
    def _generate_arps_dca_analysis(self) -> Dict[str, Any]:
        """Generate ArpsDCA analysis results."""
        
        arps_dca = self.pipeline_results.get('arps_dca')
        arps_stats = self.processing_stats.get('arps_dca', {})
        
        analysis = {
            "processing_summary": {
                "total_wells": arps_stats.get('total_wells', 0),
                "successful_wells": arps_stats.get('successful_wells', 0),
                "failed_wells": arps_stats.get('failed_wells', 0),
                "success_rate": arps_stats.get('success_rate', 0)
            },
            "method_performance": arps_stats.get('method_performance', {}),
            "quality_tier_distribution": arps_stats.get('quality_tier_distribution', {}),
            "parameter_statistics": self._generate_parameter_statistics(),
            "failure_analysis": arps_stats.get('failure_analysis', {})
        }
        
        return analysis
    
    def _generate_forecasting_results(self) -> Dict[str, Any]:
        """Generate forecasting results summary."""
        
        asset_forecast = self.pipeline_results.get('asset_forecast')
        forecasting_stats = self._extract_forecasting_statistics()
        
        results = {
            "forecasting_method": "Asset-Scale Bayesian" if self.pipeline_results.get('use_asset_scale') else "Traditional Bayesian",
            "forecast_period_months": 360,
            "production_forecast_summary": {
                "p10_total_production_bbl": forecasting_stats.get('p10_total_production', 0),
                "p50_total_production_bbl": forecasting_stats.get('p50_total_production', 0),
                "p90_total_production_bbl": forecasting_stats.get('p90_total_production', 0),
                "peak_monthly_production_bbl": forecasting_stats.get('peak_monthly_production', 0)
            },
            "uncertainty_analysis": self._generate_uncertainty_analysis()
        }
        
        return results
    
    def _generate_revenue_analysis(self) -> Dict[str, Any]:
        """Generate revenue analysis results."""
        
        revenue_metrics = self._extract_revenue_metrics()
        
        analysis = {
            "revenue_forecast_summary": {
                "p10_total_revenue_usd": revenue_metrics.get('total_p10_revenue_usd', 0),
                "p50_total_revenue_usd": revenue_metrics.get('total_p50_revenue_usd', 0),
                "p90_total_revenue_usd": revenue_metrics.get('total_p90_revenue_usd', 0),
                "revenue_per_barrel": {
                    "p10": revenue_metrics.get('revenue_per_bbl_p10', 0),
                    "p50": revenue_metrics.get('revenue_per_bbl_p50', 0),
                    "p90": revenue_metrics.get('revenue_per_bbl_p90', 0)
                }
            },
            "price_analysis": {
                "average_price_per_bbl": revenue_metrics.get('average_price_per_bbl', 0),
                "min_price_per_bbl": revenue_metrics.get('min_price_per_bbl', 0),
                "max_price_per_bbl": revenue_metrics.get('max_price_per_bbl', 0)
            },
            "revenue_uncertainty": {
                "uncertainty_percent": self._calculate_revenue_uncertainty(revenue_metrics),
                "p10_p90_ratio": revenue_metrics.get('revenue_uncertainty_p10_p90_ratio', 0)
            }
        }
        
        return analysis
    
    def _generate_quality_assessment(self) -> Dict[str, Any]:
        """Generate quality assessment summary."""
        
        arps_dca = self.pipeline_results.get('arps_dca')
        quality_stats = self.processing_stats.get('arps_dca', {})
        
        assessment = {
            "quality_tier_distribution": quality_stats.get('quality_tier_distribution', {}),
            "r_squared_statistics": self._calculate_r_squared_statistics(),
            "method_success_rates": self._calculate_method_success_rates(),
            "parameter_validation": self._generate_parameter_validation_summary()
        }
        
        return assessment
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        
        return {
            "validation_results": self.validation_results,
            "data_quality_checks": self._generate_data_quality_summary(),
            "business_logic_validation": self._generate_business_validation_summary()
        }
    
    def _generate_business_metrics(self) -> Dict[str, Any]:
        """Generate business-focused metrics."""
        
        revenue_metrics = self._extract_revenue_metrics()
        
        metrics = {
            "key_performance_indicators": {
                "total_estimated_reserves_bbl": {
                    "p10": self._extract_forecasting_statistics().get('p10_total_production', 0),
                    "p50": self._extract_forecasting_statistics().get('p50_total_production', 0),
                    "p90": self._extract_forecasting_statistics().get('p90_total_production', 0)
                },
                "total_revenue_potential_usd": {
                    "p10": revenue_metrics.get('total_p10_revenue_usd', 0),
                    "p50": revenue_metrics.get('total_p50_revenue_usd', 0),
                    "p90": revenue_metrics.get('total_p90_revenue_usd', 0)
                },
                "average_revenue_per_barrel": revenue_metrics.get('average_price_per_bbl', 0)
            },
            "investment_metrics": {
                "asset_value_range_usd": {
                    "low": revenue_metrics.get('total_p90_revenue_usd', 0),
                    "base": revenue_metrics.get('total_p50_revenue_usd', 0),
                    "high": revenue_metrics.get('total_p10_revenue_usd', 0)
                },
                "uncertainty_level": self._categorize_uncertainty_level()
            }
        }
        
        return metrics
    
    def _generate_technical_details(self) -> Dict[str, Any]:
        """Generate technical implementation details."""
        
        return {
            "methodology": {
                "decline_curve_analysis": "Advanced ArpsDCA with multiple fitting methods",
                "probabilistic_forecasting": "Bayesian uncertainty quantification",
                "revenue_calculation": "Strip price integration with escalation"
            },
            "quality_control": {
                "parameter_validation": "Physical constraint validation",
                "uncertainty_quantification": "Quality-tier based uncertainty multipliers",
                "fallback_methods": "Industry analog parameters for challenging wells"
            },
            "performance_optimization": {
                "parallel_processing": "Multi-threaded well processing",
                "memory_optimization": "Efficient data structures",
                "computation_time": f"{self.processing_times.get('total_pipeline', 0):.2f} seconds"
            }
        }
    
    def _extract_revenue_metrics(self) -> Dict[str, Any]:
        """Extract revenue metrics from pipeline results."""
        
        # Try to get revenue metrics from multiple sources
        revenue_metrics = {}
        
        # Check revenue calculator
        revenue_calculator = self.pipeline_results.get('revenue_calculator')
        if revenue_calculator and hasattr(revenue_calculator, 'revenue_metrics'):
            revenue_metrics = revenue_calculator.revenue_metrics
        
        # Check direct revenue forecast
        revenue_forecast = self.pipeline_results.get('revenue_forecast')
        if revenue_forecast is not None and not revenue_forecast.empty:
            # Extract metrics from revenue forecast DataFrame
            revenue_metrics.update({
                'total_p10_revenue_usd': revenue_forecast['P10_Cumulative_Revenue_USD'].iloc[-1],
                'total_p50_revenue_usd': revenue_forecast['P50_Cumulative_Revenue_USD'].iloc[-1],
                'total_p90_revenue_usd': revenue_forecast['P90_Cumulative_Revenue_USD'].iloc[-1],
                'average_price_per_bbl': revenue_forecast['Strip_price_Oil'].mean(),
                'total_p10_production_bbl': revenue_forecast['P10_Cumulative_bbl'].iloc[-1],
                'total_p50_production_bbl': revenue_forecast['P50_Cumulative_bbl'].iloc[-1],
                'total_p90_production_bbl': revenue_forecast['P90_Cumulative_bbl'].iloc[-1]
            })
        
        return revenue_metrics
    
    def _extract_forecasting_statistics(self) -> Dict[str, Any]:
        """Extract forecasting statistics from pipeline results."""
        
        stats = {}
        
        # Get asset forecast
        asset_forecast = self.pipeline_results.get('asset_forecast')
        if asset_forecast is not None and not asset_forecast.empty:
            stats.update({
                'p10_total_production': asset_forecast['P10_Cumulative_bbl'].iloc[-1],
                'p50_total_production': asset_forecast['P50_Cumulative_bbl'].iloc[-1],
                'p90_total_production': asset_forecast['P90_Cumulative_bbl'].iloc[-1],
                'peak_monthly_production': max(
                    asset_forecast['P10_Production_bbl'].max(),
                    asset_forecast['P50_Production_bbl'].max(),
                    asset_forecast['P90_Production_bbl'].max()
                )
            })
        
        return stats
    
    def _calculate_revenue_uncertainty(self, revenue_metrics: Dict[str, Any]) -> float:
        """Calculate revenue uncertainty percentage."""
        
        p10_rev = revenue_metrics.get('total_p10_revenue_usd', 0)
        p50_rev = revenue_metrics.get('total_p50_revenue_usd', 0)
        p90_rev = revenue_metrics.get('total_p90_revenue_usd', 0)
        
        if p50_rev > 0:
            return ((p10_rev - p90_rev) / p50_rev) * 100
        return 0
    
    def _generate_quality_summary(self) -> Dict[str, Any]:
        """Generate quality summary metrics."""
        
        arps_stats = self.processing_stats.get('arps_dca', {})
        quality_tiers = arps_stats.get('quality_tier_distribution', {})
        
        total_wells = sum(quality_tiers.values()) if quality_tiers else 1
        
        return {
            "high_quality_wells": quality_tiers.get('high', 0),
            "medium_quality_wells": quality_tiers.get('medium', 0),
            "low_quality_wells": quality_tiers.get('low', 0),
            "very_low_quality_wells": quality_tiers.get('very_low', 0),
            "failed_wells": quality_tiers.get('failed', 0),
            "high_quality_percentage": (quality_tiers.get('high', 0) / total_wells) * 100,
            "overall_quality_score": self._calculate_overall_quality_score(quality_tiers)
        }
    
    def _generate_method_performance_summary(self) -> Dict[str, Any]:
        """Generate method performance summary."""
        
        method_performance = self.processing_stats.get('arps_dca', {}).get('method_performance', {})
        
        summary = {}
        for method, stats in method_performance.items():
            summary[method] = {
                "wells_processed": stats.get('count', 0),
                "avg_r_squared": stats.get('avg_r_squared', 0),
                "avg_pearson_r": stats.get('avg_pearson_r', 0),
                "success_rate": self._calculate_method_success_rate(stats)
            }
        
        return summary
    
    def _calculate_method_success_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate success rate for a fitting method."""
        
        count = stats.get('count', 0)
        warnings = stats.get('warning_count', 0)
        
        if count > 0:
            return ((count - warnings) / count) * 100
        return 0
    
    def _calculate_overall_quality_score(self, quality_tiers: Dict[str, int]) -> float:
        """Calculate overall quality score."""
        
        weights = {'high': 1.0, 'medium': 0.75, 'low': 0.5, 'very_low': 0.25, 'failed': 0.0}
        
        total_wells = sum(quality_tiers.values())
        if total_wells == 0:
            return 0
        
        weighted_score = sum(quality_tiers.get(tier, 0) * weight for tier, weight in weights.items())
        return (weighted_score / total_wells) * 100
    
    def _categorize_uncertainty_level(self) -> str:
        """Categorize uncertainty level for business use."""
        
        revenue_metrics = self._extract_revenue_metrics()
        uncertainty_percent = self._calculate_revenue_uncertainty(revenue_metrics)
        
        if uncertainty_percent < 10:
            return "Low"
        elif uncertainty_percent < 20:
            return "Medium"
        elif uncertainty_percent < 30:
            return "High"
        else:
            return "Very High"
    
    def _calculate_r_squared_statistics(self) -> Dict[str, Any]:
        """Calculate R-squared statistics."""
        
        arps_dca = self.pipeline_results.get('arps_dca')
        if not arps_dca or not arps_dca.fit_results:
            return {}
        
        r_squared_values = []
        for well_name, fit_result in arps_dca.fit_results.items():
            # Include ALL successful fits for accurate R-squared statistics
            if fit_result.success and fit_result.quality_metrics:
                r_squared_values.append(fit_result.quality_metrics.get('r_squared', 0))
        
        if not r_squared_values:
            return {}
        
        return {
            "mean": np.mean(r_squared_values),
            "median": np.median(r_squared_values),
            "std": np.std(r_squared_values),
            "min": np.min(r_squared_values),
            "max": np.max(r_squared_values),
            "wells_above_08": sum(1 for r2 in r_squared_values if r2 > 0.8),
            "wells_above_06": sum(1 for r2 in r_squared_values if r2 > 0.6),
            "wells_negative": sum(1 for r2 in r_squared_values if r2 < 0)
        }
    
    def _calculate_method_success_rates(self) -> Dict[str, float]:
        """Calculate success rates for each fitting method."""
        
        method_performance = self.processing_stats.get('arps_dca', {}).get('method_performance', {})
        
        success_rates = {}
        for method, stats in method_performance.items():
            count = stats.get('count', 0)
            warnings = stats.get('warning_count', 0)
            
            if count > 0:
                success_rates[method] = ((count - warnings) / count) * 100
            else:
                success_rates[method] = 0
        
        return success_rates
    
    def _generate_parameter_statistics(self) -> Dict[str, Any]:
        """Generate parameter statistics."""
        
        arps_dca = self.pipeline_results.get('arps_dca')
        if not arps_dca or not arps_dca.fit_results:
            return {}
        
        qi_values = []
        di_values = []
        b_values = []
        
        for well_name, fit_result in arps_dca.fit_results.items():
            # Include ALL successful fits for accurate parameter statistics
            if fit_result.success:
                qi_values.append(fit_result.qi)
                di_values.append(fit_result.Di)
                b_values.append(fit_result.b)
        
        if not qi_values:
            return {}
        
        return {
            "initial_production_qi": {
                "mean": np.mean(qi_values),
                "median": np.median(qi_values),
                "std": np.std(qi_values),
                "min": np.min(qi_values),
                "max": np.max(qi_values)
            },
            "decline_rate_di": {
                "mean": np.mean(di_values),
                "median": np.median(di_values),
                "std": np.std(di_values),
                "min": np.min(di_values),
                "max": np.max(di_values)
            },
            "b_factor": {
                "mean": np.mean(b_values),
                "median": np.median(b_values),
                "std": np.std(b_values),
                "min": np.min(b_values),
                "max": np.max(b_values)
            }
        }
    
    def _generate_parameter_validation_summary(self) -> Dict[str, Any]:
        """Generate parameter validation summary."""
        
        arps_dca = self.pipeline_results.get('arps_dca')
        if not arps_dca or not arps_dca.validation_results:
            return {}
        
        total_wells = len(arps_dca.validation_results)
        wells_with_issues = sum(1 for v in arps_dca.validation_results.values() if v.issues)
        wells_with_warnings = sum(1 for v in arps_dca.validation_results.values() if v.warnings)
        
        return {
            "total_wells": total_wells,
            "wells_with_issues": wells_with_issues,
            "wells_with_warnings": wells_with_warnings,
            "validation_pass_rate": ((total_wells - wells_with_issues) / total_wells) * 100 if total_wells > 0 else 0
        }
    
    def _generate_data_quality_summary(self) -> Dict[str, Any]:
        """Generate data quality summary."""
        
        validation_results = self.validation_results
        
        return {
            "data_validation_passed": validation_results.get('data_quality', {}).get('valid', False),
            "data_quality_issues": len(validation_results.get('data_quality', {}).get('issues', [])),
            "data_quality_warnings": len(validation_results.get('data_quality', {}).get('warnings', []))
        }
    
    def _generate_business_validation_summary(self) -> Dict[str, Any]:
        """Generate business validation summary."""
        
        validation_results = self.validation_results
        
        return {
            "business_logic_passed": validation_results.get('business_logic', {}).get('valid', False),
            "business_logic_issues": len(validation_results.get('business_logic', {}).get('issues', [])),
            "business_logic_warnings": len(validation_results.get('business_logic', {}).get('warnings', []))
        }
    
    def _generate_uncertainty_analysis(self) -> Dict[str, Any]:
        """Generate uncertainty analysis."""
        
        forecasting_stats = self._extract_forecasting_statistics()
        
        p10_prod = forecasting_stats.get('p10_total_production', 0)
        p50_prod = forecasting_stats.get('p50_total_production', 0)
        p90_prod = forecasting_stats.get('p90_total_production', 0)
        
        if p50_prod > 0:
            production_uncertainty = ((p10_prod - p90_prod) / p50_prod) * 100
        else:
            production_uncertainty = 0
        
        return {
            "production_uncertainty_percent": production_uncertainty,
            "p10_p90_production_ratio": p10_prod / p90_prod if p90_prod > 0 else 0,
            "uncertainty_level": self._categorize_uncertainty_level()
        }
    
    def _save_comprehensive_report(self, report: Dict[str, Any]) -> None:
        """Save the unified comprehensive report - SINGLE FILE ONLY."""
        
        # Save ONLY the comprehensive report - everything consolidated into one file
        report_file = self.output_dir / "comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Unified comprehensive report saved to {report_file}")
        logger.info("SINGLE COMPREHENSIVE REPORT GENERATED - NO SCATTERED FILES")
        
        # NO OTHER FILES ARE CREATED - EVERYTHING IS IN comprehensive_report.json
        # The cleanup method will handle removing any existing scattered files
    
    def _generate_revenue_metrics_json(self, report: Dict[str, Any]) -> None:
        """Generate revenue metrics JSON file for compatibility."""
        
        revenue_analysis = report['revenue_analysis']
        revenue_summary = revenue_analysis['revenue_forecast_summary']
        
        revenue_metrics = {
            "total_p10_revenue_usd": revenue_summary['p10_total_revenue_usd'],
            "total_p50_revenue_usd": revenue_summary['p50_total_revenue_usd'],
            "total_p90_revenue_usd": revenue_summary['p90_total_revenue_usd'],
            "average_price_per_bbl": revenue_analysis['price_analysis']['average_price_per_bbl'],
            "min_price_per_bbl": revenue_analysis['price_analysis']['min_price_per_bbl'],
            "max_price_per_bbl": revenue_analysis['price_analysis']['max_price_per_bbl'],
            "revenue_uncertainty_percent": revenue_analysis['revenue_uncertainty']['uncertainty_percent'],
            "calculation_date": datetime.now().isoformat()
        }
        
        metrics_file = self.output_dir / "revenue_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(revenue_metrics, f, indent=2, default=str)
        
        logger.info(f"Revenue metrics saved to {metrics_file}")
    
    def _cleanup_redundant_files(self) -> None:
        """Clean up ALL scattered report files - consolidate everything into comprehensive_report.json."""
        
        # ALL scattered report files to remove (keep only comprehensive_report.json)
        redundant_files = [
            "executive_summary.txt",  # This was still being generated
            "revenue_metrics.json",
            "asset_revenue_forecast_P10.csv",
            "asset_revenue_forecast_P50.csv", 
            "asset_revenue_forecast_P90.csv",
            "processing_summary.json",
            "validation_report.json"
        ]
        
        for filename in redundant_files:
            file_path = self.output_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed redundant report file: {filename}")
        
        # Clean up visualizations directory scattered reports
        visualizations_dir = self.output_dir / "visualizations"
        if visualizations_dir.exists():
            visualization_redundant_files = [
                "executive_summary.json",  # This was being generated by visualizations
                "revenue_metrics.json",
                "processing_summary.json",
                "method_performance.json",
                "quality_assessment.json"
            ]
            
            for filename in visualization_redundant_files:
                file_path = visualizations_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed redundant visualization file: {filename}")
        
        # Clean up analysis directory scattered reports (legacy)
        analysis_dir = self.output_dir / "analysis"
        if analysis_dir.exists():
            analysis_redundant_files = [
                "executive_summary.json",
                "revenue_metrics.json",
                "processing_summary.json"
            ]
            
            for filename in analysis_redundant_files:
                file_path = analysis_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed redundant analysis file: {filename}")
        
        # Rename analysis directory to visualizations (if it exists and contains visualizations)
        if analysis_dir.exists() and not visualizations_dir.exists():
            analysis_dir.rename(visualizations_dir)
            logger.info("Renamed analysis directory to visualizations")
        
        logger.info("REPORT CONSOLIDATION COMPLETE - ONLY comprehensive_report.json REMAINS")
        logger.info("All scattered reports have been eliminated - single source of truth established") 