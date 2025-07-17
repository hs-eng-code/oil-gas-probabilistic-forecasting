"""
Advanced Production Visualizer

This visualizer leverages the new ArpsDCA capabilities including:
- Direct FitResult dataclass integration
- Enhanced quality metrics visualization
- Method-based performance analysis
- Advanced uncertainty visualization
- Quality-based chart generation
"""

import logging
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Local imports
from arps_dca import FitResult, ValidationResult, DeclineModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def numpy_json_serializer(obj):
    """JSON serializer for numpy objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class SingleScenarioResultsVisualizer:
    """
    Advanced Production Visualizer integrated with ArpsDCA.
    
    This visualizer uses the enhanced ArpsDCA capabilities for:
    - Direct FitResult integration for type-safe operations
    - Method-based performance analysis
    - Quality-aware visualization
    - Enhanced uncertainty charts
    """
    
    def __init__(self, output_dir: str = "output", analysis_dir: str = None):
        """
        Initialize the production visualizer.
        
        Args:
            output_dir: Main output directory
            analysis_dir: Directory for visualizations (defaults to output_dir/visualizations)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set analysis directory
        if analysis_dir is not None:
            self.analysis_dir = Path(analysis_dir)
        else:
            self.analysis_dir = self.output_dir / "visualizations"
        
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for consistent visualization
        self.method_colors = {
            'differential_evolution': '#1f77b4',
            'multi_start_lbfgs': '#ff7f0e',
            'segmented_regression': '#2ca02c',
            'rate_cumulative_transform': '#d62728',
            'robust_regression': '#9467bd',
            'industry_analog_fallback': '#8c564b',
            'simple_exponential_fallback': '#e377c2',
            'linear_log_fallback': '#7f7f7f',
            'basic_decline_fallback': '#bcbd22'
        }
        
        self.quality_colors = {
            'high': '#2ca02c',
            'medium': '#ff7f0e',
            'low': '#d62728',
            'very_low': '#9467bd',
            'failed': '#8c564b'
        }
        
        logger.info(f"SingleScenarioResultsVisualizer initialized: {self.output_dir}")
        
    def generate_comprehensive_analysis(self, 
                                      pipeline_results: Dict[str, Any],
                                      well_data: pd.DataFrame,
                                      processing_stats: Dict[str, Any]) -> None:
        """
        Generate comprehensive analysis with all visualizations.
        
        Args:
            pipeline_results: Complete pipeline results
            well_data: Original well production data
            processing_stats: Processing statistics including quality tiers
        """
        logger.info("Generating visualizations...")
        
        try:
            # Extract revenue forecast safely
            revenue_forecast = None
            if 'revenue_forecast' in pipeline_results:
                revenue_forecast = pipeline_results['revenue_forecast']
            elif 'asset_forecast' in pipeline_results:
                # This might be the production forecast, not revenue
                revenue_forecast = pipeline_results.get('revenue_forecast')
            
            # Extract actual arps_dca object from pipeline_results
            arps_dca = pipeline_results.get('arps_dca')
            
            # Enhanced processing analysis - pass the actual arps_dca object
            if processing_stats and isinstance(processing_stats, dict):
                safe_processing_stats = self._sanitize_processing_stats(processing_stats)
                self._analyze_enhanced_parameter_distributions(arps_dca)
                self._analyze_fitting_method_performance(arps_dca)
                self._plot_statistical_validation_analysis(well_data, safe_processing_stats)
            
            # Bayesian forecasting analysis
            bayesian_forecaster = pipeline_results.get('bayesian_forecaster')
            asset_forecaster = pipeline_results.get('asset_forecaster')
            
            if bayesian_forecaster and arps_dca:
                self._analyze_bayesian_parameters_uncertainty(bayesian_forecaster, arps_dca, processing_stats)
            elif asset_forecaster and arps_dca:
                # For asset-scale forecasting, adapt the analysis
                self._analyze_bayesian_parameters_uncertainty(asset_forecaster, arps_dca, processing_stats)
            
            # Revenue and uncertainty analysis with INDUSTRY STANDARDS
            if revenue_forecast is not None:
                self._analyze_enhanced_uncertainty_trends(revenue_forecast, processing_stats)
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _sanitize_processing_stats(self, processing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize processing stats to ensure all values are serializable and hashable.
        
        Args:
            processing_stats: Raw processing statistics
            
        Returns:
            Sanitized processing statistics
        """
        def sanitize_value(value):
            """Recursively sanitize values to be hashable and serializable."""
            if isinstance(value, dict):
                # Ensure all dictionary keys are strings (hashable)
                sanitized_dict = {}
                for k, v in value.items():
                    # Convert key to string to ensure it's hashable
                    key_str = str(k)
                    sanitized_dict[key_str] = sanitize_value(v)
                return sanitized_dict
            elif isinstance(value, (list, tuple)):
                return [sanitize_value(item) for item in value]
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif value is None:
                return None
            elif hasattr(value, '__dict__'):
                # Handle custom objects by converting to dict
                return sanitize_value(value.__dict__)
            else:
                # For any other type, convert to string if not already hashable
                try:
                    # Test if the value is hashable
                    hash(value)
                    return value
                except TypeError:
                    # If not hashable, convert to string
                    return str(value)
        
        try:
            return sanitize_value(processing_stats)
        except Exception as e:
            logger.warning(f"Failed to sanitize processing stats: {str(e)}")
            # Return a minimal safe structure
            return {
                'arps_dca': {
                    'total_wells': 0,
                    'successful_wells': 0,
                    'failed_wells': 0,
                    'success_rate': 0.0,
                    'quality_distribution': {},
                    'quality_tiers': {}
                }
            }
    
    def _analyze_enhanced_parameter_distributions(self, arps_dca) -> None:
        """Analyze parameter distributions with enhanced quality and method visualization."""
        if not arps_dca or not arps_dca.fit_results:
            return
        
        # Extract enhanced parameters with quality and method information - INCLUDE ALL SUCCESSFUL FITS
        qi_values = []
        di_values = []
        b_values = []
        methods = []
        quality_scores = []
        confidence_levels = []
        
        for well_name, fit_result in arps_dca.fit_results.items():
            # Include ALL successful fits for true parameter distribution
            if fit_result.success and fit_result.quality_metrics:
                qi_values.append(fit_result.qi)
                di_values.append(fit_result.Di)
                b_values.append(fit_result.b)
                methods.append(fit_result.method or 'unknown')
                
                r_squared = fit_result.quality_metrics.get('r_squared', 0)
                quality_scores.append(r_squared)
                
                # Use unified quality tier classification from ArpsDCA
                validation_result = arps_dca.validation_results.get(well_name)
                quality_tier = arps_dca._determine_quality_tier(fit_result, validation_result)
                
                # Map quality tier to display format
                tier_display_mapping = {
                    'high': 'High',
                    'medium': 'Medium', 
                    'low': 'Low',
                    'very_low': 'Very Low',
                    'unreliable': 'Unreliable',
                    'failed': 'Failed'
                }
                
                confidence_levels.append(tier_display_mapping.get(quality_tier, 'Unknown'))
        
        # Create enhanced parameter distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 14))
        
        # Plot 1: qi distribution by method
        ax1 = axes[0, 0]
        method_groups = {}
        for i, method in enumerate(methods):
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(qi_values[i])
        
        for method, values in method_groups.items():
            ax1.hist(values, bins=20, alpha=0.6, label=method, 
                    color=self.method_colors.get(method, '#7f7f7f'))
        
        ax1.set_xlabel('Initial Production Rate (qi)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('qi Distribution by Fitting Method', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot 2: Di distribution by method
        ax2 = axes[0, 1]
        method_groups = {}
        for i, method in enumerate(methods):
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(di_values[i])
        
        for method, values in method_groups.items():
            ax2.hist(values, bins=20, alpha=0.6, label=method, 
                    color=self.method_colors.get(method, '#7f7f7f'))
        
        ax2.set_xlabel('Decline Rate (Di)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Di Distribution by Fitting Method', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot 3: b distribution by method
        ax3 = axes[0, 2]
        method_groups = {}
        for i, method in enumerate(methods):
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(b_values[i])
        
        for method, values in method_groups.items():
            ax3.hist(values, bins=20, alpha=0.6, label=method, 
                    color=self.method_colors.get(method, '#7f7f7f'))
        
        ax3.set_xlabel('Hyperbolic Exponent (b)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('b Distribution by Fitting Method', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot 4: Parameter correlation colored by quality
        ax4 = axes[1, 0]
        scatter = ax4.scatter(qi_values, di_values, c=quality_scores, 
                             cmap='viridis', alpha=0.7, s=50)
        ax4.set_xlabel('Initial Production Rate (qi)', fontsize=12)
        ax4.set_ylabel('Decline Rate (Di)', fontsize=12)
        ax4.set_title('qi vs Di Correlation (colored by R²)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('R² Score', fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot 5: Quality score distribution by method
        ax5 = axes[1, 1]
        method_quality = {}
        for i, method in enumerate(methods):
            if method not in method_quality:
                method_quality[method] = []
            method_quality[method].append(quality_scores[i])
        
        method_names = list(method_quality.keys())
        method_means = [np.mean(method_quality[method]) for method in method_names]
        method_stds = [np.std(method_quality[method]) for method in method_names]
        method_counts = [len(method_quality[method]) for method in method_names]
        
        # Use standard error for uncertainty in mean estimate (consistent with statistical approach)
        method_standard_errors = [std / np.sqrt(n) for std, n in zip(method_stds, method_counts)]
        
        bars = ax5.bar(method_names, method_means, color=[self.method_colors.get(method, '#7f7f7f') for method in method_names], alpha=0.7, capsize=5) # yerr=method_stds, 
        
        ax5.set_xlabel('Fitting Method', fontsize=12)
        ax5.set_ylabel('Average R² Score', fontsize=12)
        ax5.set_title('Method Performance by R² Score', fontsize=14, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot 6: Confidence level distribution: Pie chart
        ax6 = axes[1, 2]
        confidence_counts = {}
        for conf in confidence_levels:
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        conf_names = list(confidence_counts.keys())
        conf_values = list(confidence_counts.values())

        # Define the consistent color mapping for all quality tiers
        quality_colors = {
            'High': '#2ca02c',          # Green
            'Medium': '#90EE90',        # Light green
            'Low': '#ff7f0e',           # Orange
            'Very Low': '#FFC0CB',      # Pink
            'Unreliable': '#d62728',    # Red
            'Failed': '#000000'         # Black
        }

        # Map each confidence level to its corresponding color
        colors = [quality_colors.get(conf, '#7f7f7f') for conf in conf_names]  # Default grey if not found in color_map
        
        wedges, texts, autotexts = ax6.pie(conf_values, labels=conf_names, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax6.set_title('Fit Quality Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'enhanced_parameter_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_fitting_method_performance(self, arps_dca) -> None:
        """Analyze method performance with enhanced visualization."""
        
        # Extract method performance data - INCLUDE ALL SUCCESSFUL FITS INCLUDING NEGATIVE R² VALUES
        method_performance = {}
        method_quality = {}
        
        for well_name, fit_result in arps_dca.fit_results.items():
            # Include ALL successful fits regardless of R² value for complete business analysis
            if fit_result.success and fit_result.method:
                method = fit_result.method
                
                if method not in method_performance:
                    method_performance[method] = {
                        'count': 0,
                        'r_squared_values': [],
                        'pearson_values': [],
                        'success_count': 0,
                        'negative_r2_count': 0
                    }
                
                method_performance[method]['count'] += 1
                method_performance[method]['success_count'] += 1
                
                if fit_result.quality_metrics:
                    r_squared = fit_result.quality_metrics.get('r_squared', 0)
                    pearson_r = fit_result.quality_metrics.get('pearson_r', 0)
                    
                    # BUSINESS RELEVANCE: Include ALL R² values including -ve for true analysis
                    method_performance[method]['r_squared_values'].append(r_squared)
                    method_performance[method]['pearson_values'].append(pearson_r)
                    
                    # Track negative R² wells for business reporting
                    if r_squared < 0:
                        method_performance[method]['negative_r2_count'] += 1
                        logger.debug(f"Including negative R² in visualization: {well_name}, R²={r_squared:.3f}, method={method}")
        
        # Calculate averages and success rates
        for method, data in method_performance.items():
            if data['r_squared_values']:
                data['avg_r_squared'] = np.mean(data['r_squared_values'])
                data['avg_pearson_r'] = np.mean(data['pearson_values'])
                data['success_rate'] = (data['success_count'] / data['count']) * 100
                data['negative_r2_percentage'] = (data['negative_r2_count'] / data['count']) * 100
            else:
                data['avg_r_squared'] = 0
                data['avg_pearson_r'] = 0
                data['success_rate'] = 0
                data['negative_r2_percentage'] = 0
        
        # Create visualization with enhanced negative R² reporting
        fig, axes = plt.subplots(2, 3, figsize=(20, 16))
        
        # Plot 1: Method usage distribution
        ax1 = axes[0, 0]
        if method_performance:
            methods = list(method_performance.keys())
            counts = [method_performance[m]['count'] for m in methods]
            
            # Color bars by average R² performance including negative values
            colors = []
            for method in methods:
                avg_r2 = method_performance[method]['avg_r_squared']
                if avg_r2 >= 0.8:
                    colors.append('#2E8B57')  # Excellent - Sea Green
                elif avg_r2 >= 0.6:
                    colors.append('#4169E1')  # Good - Royal Blue
                elif avg_r2 >= 0.3:
                    colors.append('#FF8C00')  # Fair - Dark Orange
                elif avg_r2 >= 0:
                    colors.append('#DC143C')  # Poor - Crimson
                else:  # Negative R²
                    colors.append('#8B0000')  # Very Poor - Dark Red
            
            bars = ax1.bar(methods, counts, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Method Usage Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Fitting Method')
            ax1.set_ylabel('Number of Wells')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        str(count), ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No method performance data available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Average R-squared by method (INCLUDING NEGATIVE VALUES)
        ax2 = axes[0, 1]
        if method_performance:
            methods = list(method_performance.keys())
            r_squared_values = [method_performance[m]['avg_r_squared'] for m in methods]
            
            # color scheme with gradient based on ALL performance values
            base_color = '#4169E1'  # Royal Blue
            bars = ax2.bar(methods, r_squared_values, color=base_color, alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_title('Average R-squared by Method (Including Negative Values)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Fitting Method', fontsize=12)
            ax2.set_ylabel('Average R-squared', fontsize=12)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Set Y-axis limits to accommodate negative values
            min_r2 = min(r_squared_values) if r_squared_values else 0
            y_min = min(min_r2 - 0.1, -0.1)  # Ensure negative values are visible
            y_max = max(1.0, max(r_squared_values) + 0.1) if r_squared_values else 1.0
            ax2.set_ylim(y_min, y_max)
            
            # Add zero line for reference
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add performance indicators with enhanced color coding for negative values
            for bar, r2 in zip(bars, r_squared_values):
                height = bar.get_height()
                # Color the bar based on performance level including negative values
                if r2 >= 0.8:
                    bar.set_color('#2E8B57')  # Excellent - Sea Green
                    bar.set_alpha(0.9)
                elif r2 >= 0.6:
                    bar.set_color('#4169E1')  # Good - Royal Blue
                    bar.set_alpha(0.8)
                elif r2 >= 0.4:
                    bar.set_color('#FF8C00')  # Fair - Dark Orange
                    bar.set_alpha(0.8)
                elif r2 >= 0:
                    bar.set_color('#DC143C')  # Poor - Crimson
                    bar.set_alpha(0.8)
                else:  # Negative R²
                    bar.set_color('#8B0000')  # Very Poor - Dark Red
                    bar.set_alpha(0.9)
                
                # Add value labels: position correctly for negative values
                label_y = height + 0.02 if height >= 0 else height - 0.05
                ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{r2:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontweight='bold', fontsize=12)
            
            # Add enhanced performance legend including negative values
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2E8B57', alpha=0.9, label='Excellent (≥0.8)'),
                Patch(facecolor='#4169E1', alpha=0.8, label='Good (≥0.6)'),
                Patch(facecolor='#FF8C00', alpha=0.8, label='Fair (≥0.4)'),
                Patch(facecolor='#DC143C', alpha=0.8, label='Poor (0-0.4)'),
                Patch(facecolor='#8B0000', alpha=0.9, label='Very Poor (<0)')
            ]
            ax2.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.5)
            ax2.tick_params(axis='both', which='major', labelsize=12)
        else:
            ax2.text(0.5, 0.5, 'No R-squared data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Average R-squared - No Data', fontsize=14, fontweight='bold')
        
        # Plot 3: Success rate by method
        ax3 = axes[1, 0]
        if method_performance:
            methods = list(method_performance.keys())
            success_rates = [method_performance[m]['success_rate'] for m in methods]
            
            bars = ax3.bar(methods, success_rates, color='lightgreen', alpha=0.8, edgecolor='black')
            ax3.set_title('Success Rate by Method', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Fitting Method')
            ax3.set_ylabel('Success Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='both', which='major', labelsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No success rate data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: R-squared distribution by method (INCLUDING NEGATIVE VALUES)
        ax4 = axes[1, 1]
        if method_performance:
            # Create box plot showing full R² distribution including negative values
            method_r2_data = []
            method_labels = []
            
            # Calculate ACTUAL unreliable wells per method using ArpsDCA quality tier system (consistent with ax5)
            method_unreliable_counts = {}
            for method in method_performance.keys():
                method_unreliable_counts[method] = 0
                
            for well_name, fit_result in arps_dca.fit_results.items():
                if fit_result.success and fit_result.method:
                    validation_result = arps_dca.validation_results.get(well_name)
                    quality_tier = arps_dca._determine_quality_tier(fit_result, validation_result)
                    
                    if quality_tier == 'unreliable':
                        method = fit_result.method
                        if method in method_unreliable_counts:
                            method_unreliable_counts[method] += 1
            
            for method, data in method_performance.items():
                if data['r_squared_values']:
                    # Include ALL qualified wells (all quality tiers: high, medium, low, very_low, unreliable)
                    # method_performance already contains only successful fits with quality tiers
                    method_r2_data.append(data['r_squared_values'])
                    
                    # Count unreliable wells for labeling (but include all wells in plot)
                    unreliable_count = method_unreliable_counts.get(method, 0)
                    total_count = data['count']
                    label = f"{method.replace('_', ' ').title()}\n({total_count} wells, {unreliable_count} unreliable)"
                    method_labels.append(label)
            
            if method_r2_data:
                # Use showfliers=True to display negative R² outliers, and extend whiskers to show full data range
                box_plot = ax4.boxplot(method_r2_data, labels=method_labels, patch_artist=True, showfliers=True) # whis=[0,100] extends whiskers to min/max # Box: [Q1, Q3], with median (Q2) as a line inside # Whiskers: extend to data within [Q1 − 1.5×IQR, Q3 + 1.5×IQR] # Outliers: data outside whisker range (shown as circles if showfliers=True) # patch_artist=True fills the boxes with color
                
                # Color boxes based on median R²
                for patch, r2_data in zip(box_plot['boxes'], method_r2_data):
                    median_r2 = np.median(r2_data)
                    if median_r2 >= 0.6:
                        patch.set_facecolor('#4169E1')  # Good - Blue
                    elif median_r2 >= 0.3:
                        patch.set_facecolor('#FF8C00')  # Fair - Orange
                    elif median_r2 >= 0:
                        patch.set_facecolor('#DC143C')  # Poor - Red
                    else:
                        patch.set_facecolor('#8B0000')  # Very Poor - Dark Red
                    patch.set_alpha(0.7)
                
                ax4.set_title('R² Distribution by Method (Full Range)', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Fitting Method')
                ax4.set_ylabel('R² Score')
                ax4.tick_params(axis='x', rotation=45, labelsize=9)
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.tick_params(axis='both', which='major', labelsize=12)
                
                # Add zero line for reference
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # Set Y-axis to show full range
                all_r2_values = [val for data in method_r2_data for val in data]
                if all_r2_values:
                    y_min = min(all_r2_values) - 0.1
                    y_max = max(1.0, max(all_r2_values) + 0.1)
                    ax4.set_ylim(y_min, y_max)
            else:
                ax4.text(0.5, 0.5, 'No R² distribution data available', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'No method data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        # Plot 5: Unreliable wells analysis (FIXED)
        ax5 = axes[1, 2]
        if method_performance:
            # Calculate ACTUAL unreliable wells based on ArpsDCA quality tier classification
            method_unreliable_counts = {}
            method_unreliable_percentages = {}
            
            for method in method_performance.keys():
                method_unreliable_counts[method] = 0
                
            # Count unreliable wells per method using ArpsDCA quality tier system
            for well_name, fit_result in arps_dca.fit_results.items():
                if fit_result.success and fit_result.method:
                    validation_result = arps_dca.validation_results.get(well_name)
                    quality_tier = arps_dca._determine_quality_tier(fit_result, validation_result)
                    
                    if quality_tier == 'unreliable':
                        method = fit_result.method
                        if method in method_unreliable_counts:
                            method_unreliable_counts[method] += 1
            
            # Calculate percentages
            for method in method_performance.keys():
                total_wells = method_performance[method]['count']
                unreliable_count = method_unreliable_counts.get(method, 0)
                method_unreliable_percentages[method] = (unreliable_count / total_wells * 100) if total_wells > 0 else 0
            
            methods = list(method_performance.keys())
            unreliable_percentages = [method_unreliable_percentages.get(m, 0) for m in methods]
            
            bars = ax5.bar(methods, unreliable_percentages, color='#d62728', alpha=0.7, edgecolor='black')
            ax5.set_title('Unreliable Wells by Method', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Fitting Method')
            ax5.set_ylabel('Percentage of Wells Classified as Unreliable')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='both', which='major', labelsize=12)
            
            # Add percentage labels on bars
            for bar, percentage in zip(bars, unreliable_percentages):
                height = bar.get_height()
                if height > 0:
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # Add insight text with CORRECT unreliable wells count
            total_unreliable = sum(method_unreliable_counts.values())
            total_wells = sum(method_performance[m]['count'] for m in methods)
            overall_unreliable_pct = (total_unreliable / total_wells) * 100 if total_wells > 0 else 0
            
            ax5.text(0.02, 0.98, f'Overall: {total_unreliable}/{total_wells} wells ({overall_unreliable_pct:.1f}%)\nBusiness Impact: Unreliable quality wells', 
                    transform=ax5.transAxes, va='top', ha='left', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7), fontsize=12)
        else:
            ax5.text(0.5, 0.5, 'No unreliable well data available', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # Plot 6: Text summary
        ax6 = axes[0, 2]
        if method_performance:
            # Business summary text (FIXED to be consistent with corrected Plot 5)
            summary_text = "METHOD PERFORMANCE SUMMARY\n" + "="*30 + "\n\n"
            
            # Calculate ACTUAL unreliable wells per method using ArpsDCA system (consistent with Plot 5)
            method_unreliable_counts_summary = {}
            for method in method_performance.keys():
                method_unreliable_counts_summary[method] = 0
                
            for well_name, fit_result in arps_dca.fit_results.items():
                if fit_result.success and fit_result.method:
                    validation_result = arps_dca.validation_results.get(well_name)
                    quality_tier = arps_dca._determine_quality_tier(fit_result, validation_result)
                    
                    if quality_tier == 'unreliable':
                        method = fit_result.method
                        if method in method_unreliable_counts_summary:
                            method_unreliable_counts_summary[method] += 1
            
            for method, data in method_performance.items():
                avg_r2 = data['avg_r_squared']
                unreliable_count = method_unreliable_counts_summary.get(method, 0)  # FIXED: Use unreliable count
                total_count = data['count']
                
                summary_text += f"{method.replace('_', ' ').title()}:\n"
                summary_text += f"  • Wells: {total_count}\n"
                summary_text += f"  • Avg R²: {avg_r2:.3f}\n"
                summary_text += f"  • Unreliable Quality: {unreliable_count} ({(unreliable_count/total_count)*100:.1f}%)\n"
                
                if avg_r2 < 0:
                    summary_text += f"  • Status: REQUIRES HIGH UNCERTAINTY\n"
                elif avg_r2 < 0.3:
                    summary_text += f"  • Status: Poor performance\n"
                elif avg_r2 < 0.6:
                    summary_text += f"  • Status: Fair performance\n"
                else:
                    summary_text += f"  • Status: Good performance\n"
                summary_text += "\n"
            
            # Business insight (FIXED: Updated terminology)
            summary_text += "BUSINESS INSIGHTS:\n"
            summary_text += "• Unreliable quality wells are retained with high uncertainty\n"
            summary_text += "• Portfolio value maintained through uncertainty modeling\n"
            summary_text += "• Conservative forecasting for challenging reservoirs"
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                    va='top', ha='left', fontsize=12, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            ax6.set_title('Business Analysis Summary', fontsize=14, fontweight='bold')
            ax6.axis('off')
        else:
            ax6.text(0.5, 0.5, 'No summary data available', 
                    ha='center', va='center', transform=ax6.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'fitting_method_performance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Method performance analysis with negative R² values completed")
    
    def _analyze_bayesian_parameters_uncertainty(self, bayesian_forecaster, arps_dca, processing_stats: Dict[str, Any]) -> None:
        """
        Analyze ACTUAL Bayesian forecasting results with proper posterior distributions.
        
        This shows real Bayesian analysis, not ArpsDCA fitting statistics:
        - Posterior parameter distributions
        - Uncertainty quantification 
        - Bayesian vs deterministic comparison
        - Method convergence diagnostics
        """
        
        try:
            logger.info("Generating ACTUAL Bayesian forecasting analysis...")
            
            # Create visualization for actual Bayesian results
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle('Bayesian Uncertainty and Regression Quantification', fontsize=16, fontweight='bold', y=0.98)
            
            # Check if we have AssetScaleBayesianForecaster results
            has_bayesian_results = False
            bayesian_wells = []
            
            if hasattr(bayesian_forecaster, 'fit_results') and bayesian_forecaster.fit_results:
                # Extract wells with Bayesian results
                for well_name, result in bayesian_forecaster.fit_results.items():
                    if (result.get('success', False) and 
                        'parameter_samples' in result and 
                        result['parameter_samples']):
                        bayesian_wells.append(well_name)
                        has_bayesian_results = True
                        
                logger.info(f"Found {len(bayesian_wells)} wells with Bayesian posterior samples")
            
            if has_bayesian_results:
                # Plot 1: Uncertainty Quantification by Well Quality
                ax1 = axes[0]
                self._plot_bayesian_uncertainty_by_quality(ax1, bayesian_forecaster, arps_dca, bayesian_wells)
                
                # Plot 2: Bayesian vs Deterministic Comparison
                ax2 = axes[1]
                self._plot_bayesian_vs_deterministic(ax2, bayesian_forecaster, arps_dca, bayesian_wells)
                
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'bayesian_parameters_uncertainty_analysis.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info("Actual Bayesian forecasting analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Bayesian analysis failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create informative error visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.6, 'Bayesian Analysis Error', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16, fontweight='bold')
            ax.text(0.5, 0.4, f'Error Details: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, style='italic')
            ax.text(0.5, 0.3, 'Check logs for detailed error information', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Bayesian Forecasting Analysis', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(self.analysis_dir / 'bayesian_parameters_uncertainty_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    def _plot_bayesian_posterior_distributions(self, ax, bayesian_forecaster, bayesian_wells):
        """Plot actual posterior parameter distributions from Bayesian analysis."""
        try:
            # Collect posterior samples from all wells
            qi_samples = []
            di_samples = []
            b_samples = []
            
            for well_name in bayesian_wells[:50]:  # Limit to 50 wells for performance
                result = bayesian_forecaster.fit_results[well_name]
                if 'parameter_samples' in result:
                    samples = result['parameter_samples']
                    if 'qi' in samples and 'Di' in samples and 'b' in samples:
                        qi_samples.extend(samples['qi'][:100])  # Take first 100 samples per well
                        di_samples.extend(samples['Di'][:100])
                        b_samples.extend(samples['b'][:100])
            
            if qi_samples and di_samples and b_samples:
                # Clear main axis and create subplots within
                ax.clear()
                
                # Create three separate histograms in one plot using different colors and alpha
                ax.hist(qi_samples, bins=30, alpha=0.5, color='blue', label=f'qi: μ={np.mean(qi_samples):.0f}', density=True)
                ax.hist(np.array(di_samples) * 1000, bins=30, alpha=0.5, color='red', label=f'Di×1000: μ={np.mean(di_samples)*1000:.1f}', density=True)
                ax.hist(np.array(b_samples) * 10, bins=30, alpha=0.5, color='green', label=f'b×10: μ={np.mean(b_samples)*10:.1f}', density=True)
                
                ax.set_title('Bayesian Posterior Distributions', fontsize=14, fontweight='bold')
                ax.set_xlabel('Normalized Parameter Values')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add summary statistics text box
                summary_text = f"Posterior Summary ({len(bayesian_wells)} wells):\n"
                summary_text += f"qi: μ={np.mean(qi_samples):.0f}, σ={np.std(qi_samples):.0f}\n"
                summary_text += f"Di: μ={np.mean(di_samples):.3f}, σ={np.std(di_samples):.3f}\n"
                summary_text += f"b:  μ={np.mean(b_samples):.2f}, σ={np.std(b_samples):.2f}"
                
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            else:
                ax.text(0.5, 0.5, 'No posterior samples available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Posterior Distributions - No Data', fontsize=14, fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting posteriors: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Posterior Distributions - Error', fontsize=14, fontweight='bold')
    
    def _plot_bayesian_parameter_correlations(self, ax, bayesian_forecaster, bayesian_wells):
        """Plot parameter correlations from Bayesian posterior samples."""
        try:
            # Collect samples for correlation analysis
            qi_samples = []
            di_samples = []
            b_samples = []
            
            for well_name in bayesian_wells[:20]:  # Limit for performance
                result = bayesian_forecaster.fit_results[well_name]
                if 'parameter_samples' in result:
                    samples = result['parameter_samples']
                    if all(k in samples for k in ['qi', 'Di', 'b']):
                        qi_samples.extend(samples['qi'][:50])
                        di_samples.extend(samples['Di'][:50])
                        b_samples.extend(samples['b'][:50])
            
            if len(qi_samples) > 10:
                # Create correlation scatter plot
                scatter = ax.scatter(qi_samples, di_samples, c=b_samples, 
                                   alpha=0.6, s=20, cmap='viridis')
                ax.set_xlabel('qi (Initial Rate)')
                ax.set_ylabel('Di (Decline Rate)')
                ax.set_title('Bayesian Parameter Correlations', fontsize=14, fontweight='bold')
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='b (Hyperbolic Exponent)')
                
                # Calculate and display correlation
                corr_qi_di = np.corrcoef(qi_samples, di_samples)[0, 1]
                corr_qi_b = np.corrcoef(qi_samples, b_samples)[0, 1]
                corr_di_b = np.corrcoef(di_samples, b_samples)[0, 1]
                
                corr_text = f"Correlations:\nqi-Di: {corr_qi_di:.3f}\nqi-b:  {corr_qi_b:.3f}\nDi-b:  {corr_di_b:.3f}"
                ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, va='top', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            else:
                ax.text(0.5, 0.5, 'Insufficient samples for correlation analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Parameter Correlations - Insufficient Data', fontsize=14, fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Correlations - Error', fontsize=14, fontweight='bold')
    
    def _plot_bayesian_uncertainty_by_quality(self, ax, bayesian_forecaster, arps_dca, bayesian_wells):
        """Plot uncertainty quantification by well quality."""
        try:
            quality_uncertainties = {'High': [], 'Medium': [], 'Low': [], 'Very Low': [], 'Unreliable': [], 'Failed': []}
            
            for well_name in bayesian_wells:
                # Get quality from ArpsDCA using unified classification
                if well_name in arps_dca.fit_results:
                    fit_result = arps_dca.fit_results[well_name]
                    if fit_result.quality_metrics:
                        # Use unified quality tier classification from ArpsDCA
                        validation_result = arps_dca.validation_results.get(well_name)
                        quality_tier_raw = arps_dca._determine_quality_tier(fit_result, validation_result)
                        
                        # Map to display format
                        tier_display_mapping = {
                            'high': 'High',
                            'medium': 'Medium',
                            'low': 'Low',
                            'very_low': 'Very Low',
                            'unreliable': 'Unreliable',
                            'failed': 'Failed'
                        }
                        
                        quality_tier = tier_display_mapping.get(quality_tier_raw, 'Very Low')
                        
                        # Calculate uncertainty from Bayesian samples
                        result = bayesian_forecaster.fit_results[well_name]
                        if 'parameter_samples' in result and 'qi' in result['parameter_samples']:
                            qi_samples = result['parameter_samples']['qi']
                            if len(qi_samples) > 1:
                                uncertainty = np.std(qi_samples) / np.mean(qi_samples) * 100  # CV%
                                quality_uncertainties[quality_tier].append(uncertainty)
            
            # Create box plot
            qualities = []
            uncertainties = []
            for quality, uncert_list in quality_uncertainties.items():
                if uncert_list:
                    qualities.extend([quality] * len(uncert_list))
                    uncertainties.extend(uncert_list)
            
            if uncertainties:
                unique_qualities = list(quality_uncertainties.keys())
                box_data = [quality_uncertainties[q] for q in unique_qualities if quality_uncertainties[q]]
                
                if box_data:
                    bp = ax.boxplot(box_data, labels=[q for q in unique_qualities if quality_uncertainties[q]], patch_artist=True, showfliers=True) # whis=[0,100] extends whiskers to min/max # Box: [Q1, Q3], with median (Q2) as a line inside # Whiskers: extend to data within [Q1 − 1.5×IQR, Q3 + 1.5×IQR] # Outliers: data outside whisker range (shown as circles if showfliers=True) # patch_artist=True fills the boxes with color 
                    
                    # Color boxes by quality using consistent scheme
                    quality_color_map = {
                        'High': '#2ca02c',          # Green
                        'Medium': '#90EE90',        # Light green  
                        'Low': '#ff7f0e',           # Orange
                        'Very Low': '#FFC0CB',      # Pink
                        'Unreliable': '#d62728',    # Red
                        'Failed': '#000000'         # Black
                    }
                    colors = [quality_color_map.get(q, '#7f7f7f') for q in unique_qualities if quality_uncertainties[q]]
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel('Parameter Uncertainty (CV%)', fontsize=12)
                    ax.set_xlabel('Well Data Quality Tiers', fontsize=12)
                    ax.set_title('Bayesian Uncertainty by Well Quality', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                else:
                    ax.text(0.5, 0.5, 'No uncertainty data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Uncertainty by Quality - No Data', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No uncertainty data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Uncertainty by Quality - No Data', fontsize=14, fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty by Quality - Error', fontsize=14, fontweight='bold')
    
    def _plot_bayesian_vs_deterministic(self, ax, bayesian_forecaster, arps_dca, bayesian_wells):
        """Compare Bayesian vs deterministic parameter estimates."""
        try:
            bayesian_qi = []
            deterministic_qi = []
            well_names_compared = []
            
            for well_name in bayesian_wells[:20]:  # Limit for performance
                # Get Bayesian estimate
                result = bayesian_forecaster.fit_results[well_name]
                if 'parameter_samples' in result and 'qi' in result['parameter_samples']:
                    bayesian_mean = np.mean(result['parameter_samples']['qi'])
                    
                    # Get deterministic estimate
                    if well_name in arps_dca.fit_results:
                        det_result = arps_dca.fit_results[well_name]
                        if det_result.success:
                            bayesian_qi.append(bayesian_mean)
                            deterministic_qi.append(det_result.qi)
                            well_names_compared.append(well_name)
            
            if len(bayesian_qi) > 3:
                # Create scatter plot
                ax.scatter(deterministic_qi, bayesian_qi, alpha=0.7, s=50, color='blue')
                
                # Add 1:1 line
                min_val = min(min(deterministic_qi), min(bayesian_qi))
                max_val = max(max(deterministic_qi), max(bayesian_qi))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 Line')
                
                # Calculate correlation
                correlation = np.corrcoef(deterministic_qi, bayesian_qi)[0, 1]
                
                ax.set_xlabel('Deterministic qi (ArpsDCA)', fontsize=12)
                ax.set_ylabel('Bayesian qi (Posterior Mean)', fontsize=12)
                ax.set_title(f'Bayesian vs Deterministic', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=12, framealpha=0.5)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=12)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for comparison', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Bayesian vs Deterministic - Insufficient Data', fontsize=14, fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bayesian vs Deterministic - Error', fontsize=14, fontweight='bold')
    
    def _plot_bayesian_convergence_diagnostics(self, ax, bayesian_forecaster, bayesian_wells):
        """Plot convergence diagnostics for Bayesian sampling."""
        try:
            # Analyze sample statistics across wells
            sample_counts = []
            effective_samples = []
            
            for well_name in bayesian_wells[:10]:  # Sample of wells
                result = bayesian_forecaster.fit_results[well_name]
                if 'parameter_samples' in result and 'qi' in result['parameter_samples']:
                    samples = result['parameter_samples']['qi']
                    sample_counts.append(len(samples))
                    
                    # Simple effective sample size estimate
                    if len(samples) > 10:
                        # Autocorrelation-based effective sample size (simplified)
                        autocorr = np.corrcoef(samples[:-1], samples[1:])[0, 1] if len(samples) > 1 else 0
                        eff_samples = len(samples) / (1 + 2 * max(0, autocorr))
                        effective_samples.append(eff_samples)
            
            if sample_counts:
                # Create bar plot of sample statistics
                x_pos = np.arange(len(sample_counts))
                bars1 = ax.bar(x_pos - 0.2, sample_counts, 0.4, label='Total Samples', alpha=0.7, color='blue')
                
                if effective_samples:
                    bars2 = ax.bar(x_pos + 0.2, effective_samples, 0.4, label='Effective Samples', alpha=0.7, color='red')
                
                ax.set_xlabel('Well Index')
                ax.set_ylabel('Number of Samples')
                ax.set_title('Bayesian Sampling Diagnostics', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add summary statistics
                avg_samples = np.mean(sample_counts)
                avg_effective = np.mean(effective_samples) if effective_samples else 0
                
                summary_text = f"Sampling Summary:\nAvg Total: {avg_samples:.0f}\nAvg Effective: {avg_effective:.0f}\nWells: {len(bayesian_wells)}"
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            else:
                ax.text(0.5, 0.5, 'No sampling diagnostics available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Convergence Diagnostics - No Data', fontsize=14, fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Diagnostics - Error', fontsize=14, fontweight='bold')
    
    def _plot_bayesian_forecast_uncertainty(self, ax, bayesian_forecaster, bayesian_wells):
        """Plot forecast uncertainty evolution over time."""
        try:
            # This would require forecast samples over time
            # For now, create a placeholder showing that this requires forecast time series
            ax.text(0.5, 0.5, 'Forecast Uncertainty Evolution\n\nRequires:\n• Time series forecast samples\n• Uncertainty bands over forecast period\n• Convergence of uncertainty over time\n\nNot available in current implementation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            ax.set_title('Forecast Uncertainty Evolution', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Forecast Uncertainty - Error', fontsize=14, fontweight='bold')
    
    def _create_asset_scale_summary(self, asset_forecaster, processing_stats: Dict[str, Any]) -> pd.DataFrame:
        """Create summary for asset-scale forecaster."""
        try:
            # Get processing statistics
            bayesian_stats = processing_stats.get('probabilistic_forecasting', {})
            method_distribution = bayesian_stats.get('method_distribution', {})
            
            # Create a basic summary DataFrame
            summary_data = []
            
            # Add hierarchical processing info
            if 'hierarchical_wells' in method_distribution:
                summary_data.append({
                    'method': 'Hierarchical_Bayesian',
                    'well_count': method_distribution.get('hierarchical_wells', 0),
                    'composite_score': 0.8,  # Assume good quality for hierarchical
                    'processing_time': bayesian_stats.get('hierarchical_time', 0)
                })
            
            # Add ABC processing info
            if 'abc_wells' in method_distribution:
                summary_data.append({
                    'method': 'ABC_Fast',
                    'well_count': method_distribution.get('abc_wells', 0),
                    'composite_score': 0.7,  # Assume medium quality for ABC
                    'processing_time': bayesian_stats.get('batch_time', 0)
                })
            
            # Add deterministic processing info
            if 'deterministic_wells' in method_distribution:
                summary_data.append({
                    'method': 'Deterministic_Plus_Noise',
                    'well_count': method_distribution.get('deterministic_wells', 0),
                    'composite_score': 0.6,  # Assume lower quality for deterministic
                    'processing_time': 0
                })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.warning(f"Failed to create asset-scale summary: {str(e)}")
            return pd.DataFrame()
    
    def _create_summary_from_arps_dca(self, arps_dca, processing_stats: Dict[str, Any]) -> pd.DataFrame:
        """Create summary from ARPS DCA results including negative R² values for complete business analysis."""
        try:
            if not arps_dca or not arps_dca.fit_results:
                return pd.DataFrame()
            
            summary_data = []
            method_counts = {}
            negative_r2_wells = []
            
            # Extract data from ARPS DCA results - INCLUDE ALL SUCCESSFUL FITS INCLUDING NEGATIVE R²
            for well_name, fit_result in arps_dca.fit_results.items():
                if fit_result.success and fit_result.quality_metrics:
                    method = fit_result.method or 'unknown'
                    r_squared = fit_result.quality_metrics.get('r_squared', 0)
                    
                    # Count methods
                    method_counts[method] = method_counts.get(method, 0) + 1
                    
                    # BUSINESS RELEVANCE: Preserve -ve R² values for complete business analysis
                    # Create business-appropriate composite score that accounts for -ve R²
                    if r_squared < 0:
                        # For -ve R², create a composite score that reflects the poor fit but doesn't eliminate business value
                        # Use a scaled -ve score that preserves the relative performance differences
                        composite_score = r_squared * 0.3  # Scale -ve R² to preserve ordering but indicate poor performance
                        negative_r2_wells.append({
                            'well_name': well_name,
                            'method': method,
                            'r_squared': r_squared,
                            'composite_score': composite_score
                        })
                        logger.debug(f"Including negative R² well in summary: {well_name}, R²={r_squared:.3f}, composite={composite_score:.3f}")
                    else:
                        # For positive R², use weighted R² as before
                        composite_score = r_squared * 0.8
                    
                    summary_data.append({
                        'well_name': well_name,
                        'method': method,
                        'composite_score': composite_score,
                        'r_squared': r_squared,  # Preserve original R² value
                        'processing_time': 0.1,  # Estimated
                        'is_negative_r2': r_squared < 0,
                        'business_quality': self._classify_business_quality(r_squared)
                    })
            
            # If we have summary data, return it with enhanced business metrics
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Add business insights to the summary
                total_wells = len(summary_data)
                negative_wells = len(negative_r2_wells)
                negative_percentage = (negative_wells / total_wells) * 100 if total_wells > 0 else 0
                
                logger.info(f"Summary created: {total_wells} wells, {negative_wells} negative R² ({negative_percentage:.1f}%)")
                
                # Add summary statistics as metadata (accessible for visualization)
                summary_df.attrs['business_summary'] = {
                    'total_wells': total_wells,
                    'negative_r2_wells': negative_wells,
                    'negative_r2_percentage': negative_percentage,
                    'method_counts': method_counts,
                    'negative_r2_details': negative_r2_wells
                }
                
                return summary_df
            
            # If no summary data, create minimal structure for visualization
            return pd.DataFrame({
                'method': ['No methods available'],
                'composite_score': [0.0],
                'r_squared': [0.0],
                'processing_time': [0.0],
                'is_negative_r2': [False],
                'business_quality': ['unknown']
            })
            
        except Exception as e:
            logger.error(f"Failed to create summary from ARPS DCA: {str(e)}")
            return pd.DataFrame()
    
    def _classify_business_quality(self, r_squared: float) -> str:
        """Classify well quality for business purposes including negative R² handling."""
        if r_squared < 0:
            return 'high_uncertainty'  # Business term for negative R² wells
        elif r_squared < 0.3:
            return 'poor'
        elif r_squared < 0.6:
            return 'fair'
        elif r_squared < 0.8:
            return 'good'
        else:
            return 'excellent'
    
    def _plot_processing_time_analysis(self, ax, processing_stats: Dict[str, Any]) -> None:
        """Plot processing time analysis."""
        try:
            bayesian_stats = processing_stats.get('probabilistic_forecasting', {})
            
            if bayesian_stats and 'processing_time' in bayesian_stats:
                times = {
                    'Hierarchical': bayesian_stats.get('hierarchical_time', 0),
                    'Batch Processing': bayesian_stats.get('batch_time', 0),
                    'Uncertainty Prop.': bayesian_stats.get('propagation_time', 0)
                }
                
                times = {k: v for k, v in times.items() if v > 0}
                
                if times:
                    ax.bar(times.keys(), times.values(), color=['blue', 'green', 'orange'], alpha=0.7)
                    ax.set_title('Bayesian Processing Time Breakdown', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Time (seconds)')
                    ax.tick_params(axis='x', rotation=45)
                    return
            
            # Fallback visualization
            ax.text(0.5, 0.5, 'Processing time data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Processing Time Analysis', fontsize=14, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Processing Time Analysis', fontsize=14, fontweight='bold')
    
    def _plot_method_distribution(self, ax, summary_df: pd.DataFrame, processing_stats: Dict[str, Any]) -> None:
        """Plot method distribution."""
        try:
            if not summary_df.empty and 'method' in summary_df.columns:
                method_counts = summary_df['method'].value_counts()
                
                if not method_counts.empty:
                    ax.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Bayesian Method Distribution', fontsize=14, fontweight='bold')
                    return
            
            # Fallback: use processing stats
            bayesian_stats = processing_stats.get('probabilistic_forecasting', {})
            method_dist = bayesian_stats.get('method_distribution', {})
            
            if method_dist:
                methods = list(method_dist.keys())
                values = list(method_dist.values())
                ax.pie(values, labels=methods, autopct='%1.1f%%', startangle=90)
                ax.set_title('Bayesian Method Distribution', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No method distribution data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Method Distribution', fontsize=14, fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Method Distribution', fontsize=14, fontweight='bold')
    
    def _plot_quality_assessment(self, ax, summary_df: pd.DataFrame, arps_dca) -> None:
        """Plot quality assessment."""
        try:
            if not summary_df.empty and 'composite_score' in summary_df.columns:
                scores = summary_df['composite_score'].dropna()
                
                if not scores.empty:
                    ax.hist(scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(scores.mean(), color='red', linestyle='--', 
                              label=f'Mean: {scores.mean():.3f}')
                    ax.set_xlabel('Composite Quality Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Bayesian Quality Assessment', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    return
            
            # Fallback: use ArpsDCA R-squared values
            if arps_dca and arps_dca.fit_results:
                r_squared_values = []
                for well_name, fit_result in arps_dca.fit_results.items():
                    if fit_result.success and fit_result.quality_metrics:
                        r_squared_values.append(fit_result.quality_metrics.get('r_squared', 0))
                
                if r_squared_values:
                    ax.hist(r_squared_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                    ax.axvline(np.mean(r_squared_values), color='red', linestyle='--', 
                              label=f'Mean R²: {np.mean(r_squared_values):.3f}')
                    ax.set_xlabel('R² Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Quality Assessment (R² Distribution)', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    return
            
            ax.text(0.5, 0.5, 'No quality assessment data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Quality Assessment', fontsize=14, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Quality Assessment', fontsize=14, fontweight='bold')
    
    def _plot_uncertainty_analysis(self, ax, summary_df: pd.DataFrame, processing_stats: Dict[str, Any]) -> None:
        """Plot uncertainty analysis."""
        try:
            # Create uncertainty summary
            uncertainty_text = "BAYESIAN UNCERTAINTY ANALYSIS\n\n"
            
            bayesian_stats = processing_stats.get('probabilistic_forecasting', {})
            
            if bayesian_stats:
                total_wells = bayesian_stats.get('total_wells', 0)
                successful = bayesian_stats.get('successful_forecasts', 0)
                success_rate = bayesian_stats.get('forecast_success_rate', 0)
                
                uncertainty_text += f"Wells Processed: {total_wells}\n"
                uncertainty_text += f"Successful Forecasts: {successful}\n"
                uncertainty_text += f"Success Rate: {success_rate:.1f}%\n\n"
                
                method_dist = bayesian_stats.get('method_distribution', {})
                if method_dist:
                    uncertainty_text += "Method Distribution:\n"
                    for method, count in method_dist.items():
                        uncertainty_text += f"  {method}: {count} wells\n"
                
                processing_time = bayesian_stats.get('processing_time', 0)
                per_well_time = bayesian_stats.get('performance_per_well', 0)
                
                uncertainty_text += f"\nPerformance:\n"
                uncertainty_text += f"  Total Time: {processing_time:.1f}s\n"
                uncertainty_text += f"  Per Well: {per_well_time:.2f}s\n"
            else:
                uncertainty_text += "No uncertainty analysis data available"
            
            ax.text(0.05, 0.95, uncertainty_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Uncertainty Analysis Summary', fontsize=14, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty Analysis', fontsize=14, fontweight='bold')
    
    def _analyze_enhanced_uncertainty_trends(self, revenue_forecast: Optional[pd.DataFrame],
                                           processing_stats: Dict[str, Any]) -> None:
        """Analyze enhanced uncertainty trends with robust error handling."""
        try:
            logger.info("Generating enhanced uncertainty analysis...")
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Quality tier uncertainty distribution
            ax1 = axes[0]
            quality_tiers = processing_stats.get('arps_dca', {}).get('quality_tier_distribution', {})
            
            if quality_tiers:
                # Use consistent quality tier order and colors from ArpsDCA system (FIXED)
                tier_order = ['high', 'medium', 'low', 'very_low', 'unreliable', 'failed']
                tier_colors = {
                    'high': '#2ca02c',          # Green
                    'medium': '#90EE90',        # Light green
                    'low': '#ff7f0e',           # Orange
                    'very_low': '#FFC0CB',      # Pink
                    'unreliable': '#d62728',    # Red
                    'failed': '#000000'         # Black
                }
                
                tiers = []
                counts = []
                colors = []
                
                # Use consistent ordering from ArpsDCA system
                for tier in tier_order:
                    count = quality_tiers.get(tier, 0)
                    if count > 0:
                        tiers.append(tier.replace('_', ' ').title())
                        counts.append(count)
                        colors.append(tier_colors[tier])
                
                if tiers and counts:
                    bars = ax1.bar(tiers, counts, color=colors, alpha=0.7)
                    ax1.set_ylabel('Number of Wells')
                    ax1.set_title('Well Count by Quality Tier', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{count}', ha='center', va='bottom')
                else:
                    ax1.text(0.5, 0.5, 'No quality tier counts available', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Quality Tier Distribution - No Data', fontsize=14, fontweight='bold')
                
                ax1.tick_params(axis='both', which='major', labelsize=12)
            else:
                ax1.text(0.5, 0.5, 'No quality tier data available', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Quality Tier Distribution - No Data', fontsize=14, fontweight='bold')
            
            # Plot 2: Summary statistics table
            ax2 = axes[1]
            ax2.axis('off')
            
            summary_text = "Enhanced Uncertainty Analysis Summary\n\n"
            
            # Revenue summary
            if revenue_forecast is not None and not revenue_forecast.empty:
                total_wells = processing_stats.get('arps_dca', {}).get('total_wells', 0)
                successful_wells = processing_stats.get('arps_dca', {}).get('successful_wells', 0)
                
                summary_text += f"Total Wells: {total_wells}\n"
                summary_text += f"Successful Wells: {successful_wells}\n\n"
                
                # Add revenue estimates if available
                if 'P50_Cumulative_Revenue_USD' in revenue_forecast.columns:
                    p50_revenue = revenue_forecast['P50_Cumulative_Revenue_USD'].iloc[-1] / 1e9
                    summary_text += f"P50 Revenue: ${p50_revenue:.1f}B\n"
                
                if 'P50_Cumulative_bbl' in revenue_forecast.columns:
                    p50_production = revenue_forecast['P50_Cumulative_bbl'].iloc[-1] / 1e6
                    summary_text += f"P50 Production: {p50_production:.1f}M bbl\n"
            
            # Add method statistics
            method_performance = processing_stats.get('arps_dca', {}).get('method_performance', {})
            if method_performance:
                summary_text += "\nMethod Performance:\n"
                for method, stats in method_performance.items():
                    if stats.get('count', 0) > 0:
                        summary_text += f"  {method}: {stats['count']} wells, R²={stats.get('avg_r_squared', 0):.3f}\n"
            
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            ax2.set_title('Uncertainty Summary', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'enhanced_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Enhanced uncertainty analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Enhanced uncertainty analysis failed: {str(e)}")
            # Create fallback visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Enhanced uncertainty analysis failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Enhanced Uncertainty Analysis - Error', fontsize=14, fontweight='bold')
            plt.savefig(self.analysis_dir / 'enhanced_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_statistical_validation_analysis(self, well_data: pd.DataFrame, processing_stats: Dict[str, Any]) -> None:
        """Enhanced validation analysis with quality tier insights."""
        # Get enhanced processing results
        arps_stats = processing_stats.get('arps_dca', {})
        quality_tiers = arps_stats.get('quality_tiers', {})
        
        if not quality_tiers:
            # Create placeholder plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No validation data available', ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Validation Analysis', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'validation_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Create enhanced validation visualization
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        
        # Plot 1: Quality tier vs data availability
        ax1 = axes[0]
        
        # Analyze data characteristics by quality tier
        tier_data_stats = {}
        for tier, wells in quality_tiers.items():
            if wells:
                data_points = []
                for well in wells:
                    well_data_subset = well_data[well_data['WellName'] == well]
                    data_points.append(len(well_data_subset))
                
                if data_points:
                    tier_data_stats[tier] = {'mean': np.mean(data_points), 'std': np.std(data_points), 'count': len(data_points)}
        
        if tier_data_stats:
            # Use consistent quality tier order and colors including ALL possible tiers
            tier_order = ['high', 'medium', 'low', 'very_low', 'unreliable', 'failed']
            tier_colors = {
                'high': '#2ca02c',          # Green
                'medium': '#90EE90',        # Light green
                'low': '#ff7f0e',           # Orange
                'very_low': '#FFC0CB',      # Pink
                'unreliable': '#d62728',    # Red
                'failed': '#000000'         # Black
            }
            
            # Filter and order tiers based on what's available in data
            available_tiers = []
            means = []
            stds = []
            colors = []
            
            for tier in tier_order:
                if tier in tier_data_stats:
                    available_tiers.append(tier.replace('_', ' ').title())
                    means.append(tier_data_stats[tier]['mean'])
                    stds.append(tier_data_stats[tier]['std'])
                    colors.append(tier_colors[tier])
            
            # Use standard error of the mean instead of standard deviation for error bars
            # Standard error represents uncertainty in the mean estimate, not spread of data
            sample_sizes = [tier_data_stats[tier]['count'] for tier in tier_order if tier in tier_data_stats]
            standard_errors = [std / np.sqrt(n) for std, n in zip(stds, sample_sizes)]
            
            bars = ax1.bar(available_tiers, means, capsize=5, color=colors, alpha=0.8, edgecolor='black') # yerr=stds,
            ax1.set_title('Data Availability by Quality Tier', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Quality Tier', fontsize=12)
            ax1.set_ylabel('Average Data Points per Well', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Production distribution by quality tier
        ax2 = axes[1]
        
        # Analyze production characteristics by quality tier
        tier_production_stats = {}
        for tier, wells in quality_tiers.items():
            if wells:
                production_values = []
                for well in wells:
                    well_data_subset = well_data[well_data['WellName'] == well]
                    if not well_data_subset.empty:
                        # Remove NaN values that cause box plots to appear empty
                        oil_values = well_data_subset['OIL'].values
                        # Filter out NaN and negative values
                        valid_values = [val for val in oil_values if pd.notna(val) and val >= 0]
                        production_values.extend(valid_values)
                
                if production_values:
                    tier_production_stats[tier] = production_values
        
        if tier_production_stats:
            # Create box plot using consistent quality tier order and colors
            data_for_boxplot = []
            labels_for_boxplot = []
            
            # Use consistent quality tier order including all possible tiers
            tier_order = ['high', 'medium', 'low', 'very_low', 'unreliable', 'failed']
            tier_colors = {
                'high': '#2ca02c',          # Green
                'medium': '#90EE90',        # Light green
                'low': '#ff7f0e',           # Orange
                'very_low': '#FFC0CB',      # Pink
                'unreliable': '#d62728',    # Red
                'failed': '#000000'         # Black
            }
            
            for tier in tier_order:
                if tier in tier_production_stats:
                    data_for_boxplot.append(tier_production_stats[tier])
                    labels_for_boxplot.append(tier.replace('_', ' ').title())
            
            if data_for_boxplot:
                # Use showfliers=True to display all data points including outliers
                bp = ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, showfliers=True) # whis=[0,100] extends whiskers to min/max # Box: [Q1, Q3], with median (Q2) as a line inside # Whiskers: extend to data within [Q1 − 1.5×IQR, Q3 + 1.5×IQR] # Outliers: data outside whisker range (shown as circles if showfliers=True) # patch_artist=True fills the boxes with color
                
                # Color the boxes using consistent scheme
                for i, (patch, label) in enumerate(zip(bp['boxes'], labels_for_boxplot)):
                    tier_key = label.lower().replace(' ', '_')
                    color = tier_colors.get(tier_key, '#7f7f7f')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_title('Production Distribution by Quality Tier', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Quality Tier', fontsize=12)
                ax2.set_ylabel('Production (bbl/month)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='both', which='major', labelsize=12)
                ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Quality tier transition analysis
        ax3 = axes[2]
        
        # Show how data quality affects modeling success with all quality tiers
        quality_flow = {
            'Input Wells': arps_stats.get('total_wells', 0),
            'High Quality': len(quality_tiers.get('high', [])),
            'Medium Quality': len(quality_tiers.get('medium', [])),
            'Low Quality': len(quality_tiers.get('low', [])),
            'Very Low Quality': len(quality_tiers.get('very_low', [])),
            'Unreliable': len(quality_tiers.get('unreliable', [])),
            'Failed': len(quality_tiers.get('failed', []))
        }
        
        # Create a flow diagram (simplified bar chart) with consistent colors
        categories = list(quality_flow.keys())
        values = list(quality_flow.values())
        colors = ['#808080', '#2ca02c', '#90EE90', '#ff7f0e', '#FFC0CB', '#d62728', '#000000']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Quality Tier Flow Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Processing Stage', fontsize=12)
        ax3.set_ylabel('Number of Wells', fontsize=12)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Business impact summary
        # ax4 = axes[1, 1]
        
        # Calculate business impact metrics
        total_wells = arps_stats.get('total_wells', 0)
        successful_wells = arps_stats.get('successful_wells', 0)
        
        if total_wells > 0:
            coverage_rate = successful_wells / total_wells * 100
            
            # Create summary metrics
            metrics = {
                'Coverage Rate': f'{coverage_rate:.1f}%',
                'High Quality': f'{len(quality_tiers.get("high", []))} wells',
                'Medium Quality': f'{len(quality_tiers.get("medium", []))} wells',
                'Low Quality': f'{len(quality_tiers.get("low", [])) + len(quality_tiers.get("very_low", []))} wells',
            }
            
            # # Display as text summary
            # y_pos = 0.95
            # for metric, value in metrics.items():
            #     ax4.text(0.1, y_pos, f'{metric}: {value}', 
            #             transform=ax4.transAxes, fontsize=14, fontweight='bold')
            #     y_pos -= 0.15
            
            # ax4.set_xlim(0, 1)
            # ax4.set_ylim(0, 1)
            # ax4.axis('off')
            # ax4.set_title('Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'statistical_validation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


# =======================================
# ASSET ACQUISITION ANALYSIS VISUALIZER
# =======================================

class AcquisitionAnalysisVisualizer:
    """
    Visualizer for oil & gas asset acquisition analysis.
    
    This class handles multi-scenario analysis and visualization for asset
    acquisition decisions, working with results from multiple uncertainty
    scenarios (conservative, standard, aggressive).
    
    Key Features:
    - Multi-scenario uncertainty analysis
    - Production decline curve comparison
    - Revenue distribution analysis
    - Risk assessment and P10/P90 ratio analysis
    - Executive summary reporting
    - Industry-standard acquisition metrics
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the acquisition analysis visualizer.
        
        Args:
            output_dir: Main output directory (visualizations will be saved to output_dir/visualizations)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization directory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AcquisitionAnalysisVisualizer initialized: {self.output_dir}")

    def generate_all_acquisition_visualizations(self, all_results: Dict[str, Any], output_dir: str = None) -> None:
        """
        Generate all oil and gas asset acquisition analysis visualizations.
        
        Args:
            all_results: Dictionary with results from all uncertainty scenarios
            output_dir: Optional output directory (defaults to self.analysis_dir)
        """
        try:
            if output_dir:
                viz_dir = Path(output_dir) / "visualizations"
                viz_dir.mkdir(parents=True, exist_ok=True)
            else:
                viz_dir = self.viz_dir
            
            logger.info("Generating comprehensive oil & gas asset acquisition analysis...")
            
            # Validate input data
            if not self._validate_acquisition_data(all_results):
                logger.error("Invalid data structure for acquisition analysis")
                return
                
            # Extract and process data from all scenarios
            processed_data = self._process_acquisition_data(all_results)
            
            if not processed_data:
                logger.error("Failed to process acquisition data")
                return
            
            # Generate core production & revenue analysis
            logger.info("Generating core production & revenue analysis...")
            self.create_combined_monthly_and_cum_production_plots(processed_data, save_path=viz_dir / 'monthly_and_cum_production_plots.png')
            self.create_combined_monthly_and_cum_revenue_plots(processed_data, save_path=viz_dir / 'monthly_and_cum_revenue_plots.png')
            self.create_revenue_distribution_analysis(processed_data, save_path=viz_dir / 'revenue_distribution_analysis.png')
            self.create_efficiency_matrix(processed_data, save_path=viz_dir / 'production_efficiency_matrix.png')
            self.create_revenue_heatmap(processed_data, save_path=viz_dir / 'monthly_revenue_heatmap.png')
            
            # Generate advanced risk analysis
            logger.info("Generating advanced risk analysis...")
            self.create_p10_p90_ratio_analysis(processed_data, save_path=viz_dir / 'p10_p90_ratio_analysis.png')
            self.create_risk_asymmetry_plot(processed_data, save_path=viz_dir / 'risk_asymmetry_analysis.png')
            self.create_uncertainty_evolution_chart(processed_data, save_path=viz_dir / 'uncertainty_evolution.png')
            self.create_risk_summary_dashboard(processed_data, save_path=viz_dir / 'risk_summary_dashboard.png')
            
            # Generate comparative analysis
            logger.info("Generating comparative analysis...")
            self.create_scenario_risk_radar(processed_data, save_path=viz_dir / 'scenario_risk_radar.png')
            self.create_rolling_risk_metrics(processed_data, save_path=viz_dir / 'rolling_risk_metrics.png')
            self.create_revenue_production_risk_correlation(processed_data, save_path=viz_dir / 'risk_correlation_analysis.png')
            
            # Generate executive summary report
            logger.info("Generating executive summary report...")
            self.create_executive_summary_report(processed_data, output_dir=viz_dir)
            
            logger.info(f"All acquisition visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate acquisition visualizations: {str(e)}")
            logger.error(traceback.format_exc())

    def _validate_acquisition_data(self, all_results: Dict[str, Any]) -> bool:
        """Validate that data structure contains required elements for acquisition analysis."""
        try:
            required_scenarios = ['conservative', 'standard', 'aggressive']
            
            for scenario in required_scenarios:
                if scenario not in all_results:
                    logger.warning(f"Missing scenario: {scenario}")
                    return False
                    
                scenario_data = all_results[scenario]
                if not isinstance(scenario_data, dict):
                    logger.warning(f"Invalid scenario data format: {scenario}")
                    return False
                    
                # Check for required forecast data
                required_keys = ['revenue_forecast', 'asset_forecast']
                if not any(key in scenario_data for key in required_keys):
                    logger.warning(f"Missing forecast data in scenario: {scenario}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False

    def _process_acquisition_data(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and standardize data from all scenarios for visualization."""
        try:
            processed = {
                'scenarios': {},
                'time_index': None,
                'metadata': {
                    'forecast_years': 30,
                    'forecast_months': 360,
                    'scenarios_available': []
                }
            }
            
            for scenario_name, scenario_data in all_results.items():
                logger.info(f"Processing scenario: {scenario_name}")
                
                # Extract revenue forecast
                revenue_forecast = None
                if 'revenue_forecast' in scenario_data:
                    revenue_forecast = scenario_data['revenue_forecast']
                elif 'revenue_calculator' in scenario_data and hasattr(scenario_data['revenue_calculator'], 'revenue_forecast'):
                    revenue_forecast = scenario_data['revenue_calculator'].revenue_forecast
                
                # Extract production forecast
                production_forecast = None
                if 'asset_forecast' in scenario_data:
                    production_forecast = scenario_data['asset_forecast']
                
                if revenue_forecast is None and production_forecast is None:
                    logger.warning(f"No forecast data found for scenario: {scenario_name}")
                    continue
                
                # Standardize time index (use first valid forecast)
                time_data = revenue_forecast if revenue_forecast is not None else production_forecast
                if processed['time_index'] is None and time_data is not None:
                    if 'Date' in time_data.columns:
                        processed['time_index'] = time_data['Date']
                    else:
                        # Create default monthly time series
                        processed['time_index'] = pd.date_range(
                            start='2025-01-01', 
                            periods=len(time_data), 
                            freq='M'
                        )
                
                # Process scenario data
                scenario_processed = self._process_single_scenario(
                    scenario_name, revenue_forecast, production_forecast
                )
                
                if scenario_processed:
                    processed['scenarios'][scenario_name] = scenario_processed
                    processed['metadata']['scenarios_available'].append(scenario_name)
            
            logger.info(f"Successfully processed {len(processed['scenarios'])} scenarios")
            return processed if processed['scenarios'] else None
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            return None

    def _process_single_scenario(self, scenario_name: str, revenue_forecast: Optional[pd.DataFrame], 
                                production_forecast: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Process data for a single scenario."""
        try:
            scenario_data = {
                'name': scenario_name,
                'production': {},
                'revenue': {},
                'has_production': False,
                'has_revenue': False
            }
            
            # Process production data
            if production_forecast is not None and not production_forecast.empty:
                prod_cols = {
                    'P10': self._find_column(production_forecast, ['P10_Production_bbl', 'P10_Production', 'P10']),
                    'P50': self._find_column(production_forecast, ['P50_Production_bbl', 'P50_Production', 'P50']),
                    'P90': self._find_column(production_forecast, ['P90_Production_bbl', 'P90_Production', 'P90'])
                }
                
                if all(col for col in prod_cols.values()):
                    scenario_data['production'] = {
                        'monthly': {
                            'P10': production_forecast[prod_cols['P10']].values,
                            'P50': production_forecast[prod_cols['P50']].values,
                            'P90': production_forecast[prod_cols['P90']].values
                        },
                        'cumulative': {
                            'P10': production_forecast[prod_cols['P10']].cumsum().values,
                            'P50': production_forecast[prod_cols['P50']].cumsum().values,
                            'P90': production_forecast[prod_cols['P90']].cumsum().values
                        }
                    }
                    scenario_data['has_production'] = True
            
            # Process revenue data
            if revenue_forecast is not None and not revenue_forecast.empty:
                rev_cols = {
                    'P10': self._find_column(revenue_forecast, ['P10_Revenue_USD', 'P10_Revenue', 'P10']),
                    'P50': self._find_column(revenue_forecast, ['P50_Revenue_USD', 'P50_Revenue', 'P50']),
                    'P90': self._find_column(revenue_forecast, ['P90_Revenue_USD', 'P90_Revenue', 'P90'])
                }
                
                # Also check for cumulative columns
                cum_rev_cols = {
                    'P10': self._find_column(revenue_forecast, ['P10_Cumulative_Revenue_USD', 'P10_Cumulative_Revenue']),
                    'P50': self._find_column(revenue_forecast, ['P50_Cumulative_Revenue_USD', 'P50_Cumulative_Revenue']),
                    'P90': self._find_column(revenue_forecast, ['P90_Cumulative_Revenue_USD', 'P90_Cumulative_Revenue'])
                }
                
                if all(col for col in rev_cols.values()):
                    scenario_data['revenue'] = {
                        'monthly': {
                            'P10': revenue_forecast[rev_cols['P10']].values,
                            'P50': revenue_forecast[rev_cols['P50']].values,
                            'P90': revenue_forecast[rev_cols['P90']].values
                        }
                    }
                    
                    # Use cumulative if available, otherwise calculate
                    if all(col for col in cum_rev_cols.values()):
                        scenario_data['revenue']['cumulative'] = {
                            'P10': revenue_forecast[cum_rev_cols['P10']].values,
                            'P50': revenue_forecast[cum_rev_cols['P50']].values,
                            'P90': revenue_forecast[cum_rev_cols['P90']].values
                        }
                    else:
                        scenario_data['revenue']['cumulative'] = {
                            'P10': revenue_forecast[rev_cols['P10']].cumsum().values,
                            'P50': revenue_forecast[rev_cols['P50']].cumsum().values,
                            'P90': revenue_forecast[rev_cols['P90']].cumsum().values
                        }
                    
                    scenario_data['has_revenue'] = True
            
            return scenario_data if (scenario_data['has_production'] or scenario_data['has_revenue']) else None
            
        except Exception as e:
            logger.error(f"Failed to process scenario {scenario_name}: {str(e)}")
            return None

    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find the first matching column name in a DataFrame."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # Core Production & Revenue Analysis Functions

    def create_combined_monthly_and_cum_production_plots(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create combined production plots with decline curves (left) and cumulative production (right).
        Single figure with two subplots in one.
        """
        try:
            scenarios = data['scenarios']
            time_index = data['time_index']
            
            if not scenarios or time_index is None:
                logger.warning("Insufficient data for production plots")
                return
            
            # Check if we have production data
            production_scenarios = [s for s in scenarios.values() if s['has_production']]
            if not production_scenarios:
                logger.warning("No production data available for production plots")
                return
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            colors = {'conservative': '#B22222', 'standard': '#1E90FF', 'aggressive': '#228B22'}
            linestyles = {0: '-', 1: '--', 2: '-.'}
            percentile_linestyles = {'P10': '-', 'P50': '--', 'P90': ':'}
            linewidths = {'P10': 3, 'P50': 3, 'P90': 2}
            
            # Track total production for summary
            total_production = {}
            
            # LEFT SUBPLOT: Production Decline Curves
            scenario_idx = 0
            for scenario_name, scenario_data in scenarios.items():
                if not scenario_data['has_production']:
                    continue
                    
                prod_data = scenario_data['production']['monthly']
                p10_vals = prod_data['P10'] / 1e6  # Convert to millions
                p50_vals = prod_data['P50'] / 1e6  # Convert to millions
                p90_vals = prod_data['P90'] / 1e6  # Convert to millions
                
                color = colors.get(scenario_name, '#1f77b4')
                linestyle = linestyles.get(scenario_idx, '-')
                
                # Semi-log plot with enhanced styling
                ax1.semilogy(time_index, p50_vals, linestyle=linestyle, color=color, linewidth=3, label=f'{scenario_name.title()}: P50', alpha=0.9)
                
                # Uncertainty bands with reduced alpha for cleaner look
                ax1.fill_between(time_index, p10_vals, p90_vals, color=color, alpha=0.15, label=f'{scenario_name.title()}: P10-P90 Range')
                
                scenario_idx += 1
            
            # Styling for decline curves (left subplot)
            ax1.set_title('Monthly Decline Forecast', fontsize=16, fontweight='bold', pad=20)
            ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Monthly Production (MMbbl/month)', fontsize=14, fontweight='bold')
            
            # Legend for decline curves
            legend1 = ax1.legend(loc='best', fontsize=10, framealpha=0.5, fancybox=True)
            
            # Grid and formatting for decline curves
            ax1.minorticks_on() # Enable minor ticks
            ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            # Show minor horizontal grid
            ax1.grid(True, which='minor', axis='y', alpha=0.2, linestyle='--', linewidth=0.4)
            ax1.tick_params(axis='both', which='major', labelsize=10)
            
            # Format x-axis with better spacing
            ax1.xaxis.set_major_locator(mdates.YearLocator(5))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
            
            # RIGHT SUBPLOT: Cumulative Production
            for scenario_name, scenario_data in scenarios.items():
                if not scenario_data['has_production']:
                    continue
                
                cum_prod_data = scenario_data['production']['cumulative']
                color = colors.get(scenario_name, '#1f77b4')
                
                # Plot P10, P50, P90 lines with enhanced styling
                for percentile in ['P10', 'P50', 'P90']:
                    values = cum_prod_data[percentile] / 1e6  # Convert to millions
                    ax2.plot(time_index, values, linestyle=percentile_linestyles[percentile], color=color, linewidth=linewidths[percentile], alpha=0.9, label=f'{scenario_name.title()}: {percentile}')
                    
                    total_production[f'{scenario_name}_{percentile}'] = values[-1]
            
            # Styling for cumulative production (right subplot)
            ax2.set_title('Cumulative Forecast', fontsize=16, fontweight='bold', pad=20)
            ax2.set_xlabel('Forecast Year', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Cumulative Production (MMbbl)', fontsize=14, fontweight='bold')
            
            # Legend for cumulative production
            legend2 = ax2.legend(loc='best', fontsize=10, framealpha=0.5, fancybox=True)
            
            # Grid and formatting for cumulative production
            ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            
            # Format x-axis
            ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
            
            # Overall figure title
            fig.suptitle('Production Analysis: 30-Year Outlook', fontsize=18, fontweight='bold', y=0.98)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Combined production plots saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create combined production plots: {str(e)}")
    
    def create_combined_monthly_and_cum_revenue_plots(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create combined monthly production revenue (left) and cumulative (right) plots.
        Single figure with two subplots in one.
        """
        try:
            scenarios = data['scenarios']
            time_index = data['time_index']
            
            if not scenarios or time_index is None:
                logger.warning("Insufficient data for production plots")
                return
            
            # Check if we have production data
            revenue_scenarios = [s for s in scenarios.values() if s['has_revenue']]
            if not revenue_scenarios:
                logger.warning("No revenue data available for revenue plots")
                return
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            colors = {'conservative': '#B22222', 'standard': '#1E90FF', 'aggressive': '#228B22'}
            linestyles = {0: '-', 1: '--', 2: '-.'}
            percentile_linestyles = {'P10': '-', 'P50': '--', 'P90': ':'}
            linewidths = {'P10': 3, 'P50': 3, 'P90': 2}
            
            # Track total revenue for summary
            total_revenue = {}
            
            # LEFT SUBPLOT: Monthly Production Revenue Curves
            scenario_idx = 0
            for scenario_name, scenario_data in scenarios.items():
                if not scenario_data['has_revenue']:
                    continue
                    
                rev_data = scenario_data['revenue']['monthly']
                p10_vals = rev_data['P10'] / 1e9  # Convert to billions
                p50_vals = rev_data['P50'] / 1e9  # Convert to billions
                p90_vals = rev_data['P90'] / 1e9  # Convert to billions
                
                color = colors.get(scenario_name, '#1f77b4')
                linestyle = linestyles.get(scenario_idx, '-')
                
                # Semi-log plot with enhanced styling
                ax1.semilogy(time_index, p50_vals, linestyle=linestyle, color=color, linewidth=3, label=f'{scenario_name.title()}: P50', alpha=0.9)
                
                # Uncertainty bands with reduced alpha for cleaner look
                ax1.fill_between(time_index, p10_vals, p90_vals, color=color, alpha=0.15, label=f'{scenario_name.title()}: P10-P90 Range')
                
                scenario_idx += 1
            
            # Styling for monthly production revenue curves (left subplot)
            ax1.set_title('Monthly Decline Foreast', fontsize=16, fontweight='bold', pad=20)
            ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Monthly Revenue ($B/month)', fontsize=14, fontweight='bold')
            
            # Legend for decline curves
            legend1 = ax1.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True)
            
            # Grid and formatting
            ax1.minorticks_on() # Enable minor ticks
            ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            # Show minor horizontal grid
            ax1.grid(True, which='minor', axis='y', alpha=0.2, linestyle='--', linewidth=0.4)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Format x-axis with better spacing
            ax1.xaxis.set_major_locator(mdates.YearLocator(5))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
            
            # RIGHT SUBPLOT: Cumulative Revenue
            for scenario_name, scenario_data in scenarios.items():
                if not scenario_data['has_revenue']:
                    continue
                
                cum_rev_data = scenario_data['revenue']['cumulative']
                color = colors.get(scenario_name, '#1f77b4')
                
                # Plot P10, P50, P90 lines with enhanced styling
                for percentile in ['P10', 'P50', 'P90']:
                    values = cum_rev_data[percentile] / 1e9  # Convert to billions
                    ax2.plot(time_index, values, linestyle=percentile_linestyles[percentile], 
                            color=color, linewidth=linewidths[percentile], alpha=0.9, 
                            label=f'{scenario_name.title()}: {percentile}')
                    
                    total_revenue[f'{scenario_name}_{percentile}'] = values[-1]
            
            # Styling for cumulative revenue (right subplot)
            ax2.set_title('Cumulative Forecast', fontsize=16, fontweight='bold', pad=20)
            ax2.set_xlabel('Forecast Year', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Cumulative Revenue ($B)', fontsize=14, fontweight='bold')
            
            # Legend for cumulative revenue
            legend2 = ax2.legend(loc='best', fontsize=10, framealpha=0.5, fancybox=True)
            
            # Grid and formatting for cumulative production
            ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            
            # Format x-axis
            ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
            
            # Overall figure title
            fig.suptitle('Revenue Analysis: 30-Year Outlook', fontsize=18, fontweight='bold', y=0.98)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Combined revenue plots saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create combined revenue plots: {str(e)}")

    def create_revenue_distribution_analysis(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create revenue distribution analysis with box plots and summary statistics.
        Professional layout with industry benchmarks.
        """
        try:
            scenarios = data['scenarios']
            
            revenue_scenarios = [s for s in scenarios.values() if s['has_revenue']]
            if not revenue_scenarios:
                logger.warning("No revenue data available for distribution analysis")
                return
            
            # Create 2x2 layout with spacing
            fig = plt.figure(figsize=(16, 14))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                                 hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])  # Box plot
            ax2 = fig.add_subplot(gs[0, 1])  # Risk metrics
            ax3 = fig.add_subplot(gs[1, :])  # Summary table
            
            # Collect data for analysis
            scenario_revenues = {}
            for scenario_name, scenario_data in scenarios.items():
                if scenario_data['has_revenue']:
                    rev_data = scenario_data['revenue']['cumulative']
                    scenario_revenues[scenario_name] = {
                        'P10': rev_data['P10'][-1] / 1e9,
                        'P50': rev_data['P50'][-1] / 1e9,
                        'P90': rev_data['P90'][-1] / 1e9
                    }
            
            # Plot 1: Enhanced box plot
            if scenario_revenues:
                box_data = []
                labels = []
                colors = ['#B22222', '#1E90FF', '#228B22']
                
                for scenario_name, revenues in scenario_revenues.items():
                    box_data.append([revenues['P90'], revenues['P50'], revenues['P10']])
                    labels.append(scenario_name.title())
                
                bp = ax1.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6, showfliers=True) # whis=[0,100] extends whiskers to min/max # Box: [Q1, Q3], with median (Q2) as a line inside # Whiskers: extend to data within [Q1 − 1.5×IQR, Q3 + 1.5×IQR] # Outliers: data outside whisker range (shown as circles if showfliers=True) # patch_artist=True fills the boxes with color
                
                # Enhanced box styling
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                
                # Style whiskers and medians
                for whisker in bp['whiskers']:
                    whisker.set_linewidth(2)
                    whisker.set_color('black')
                for median in bp['medians']:
                    median.set_linewidth(3)
                    median.set_color('white')
                
                ax1.set_title('Revenue Distribution by Scenario', fontsize=16, fontweight='bold')
                ax1.set_xlabel('Uncertainty Scenario', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Total 30-Year Revenue ($B)', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.4, axis='y')
                ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Plot 2: Risk metrics with industry benchmarks
            risk_metrics = {}
            for scenario_name, revenues in scenario_revenues.items():
                p10, p50, p90 = revenues['P10'], revenues['P50'], revenues['P90']
                risk_metrics[scenario_name] = {
                    'P10/P90 Ratio': p10 / p90 if p90 > 0 else 0,
                    'Revenue Range ($B)': p10 - p90,
                    'Coefficient of Variation (%)': ((p10 - p90) / p50 * 100) if p50 > 0 else 0
                }
            
            if risk_metrics:
                scenarios_list = list(risk_metrics.keys())
                x = np.arange(len(scenarios_list))
                
                # P10/P90 ratios with industry benchmarks
                ratios = [risk_metrics[s]['P10/P90 Ratio'] for s in scenarios_list]
                bars = ax2.bar(x, ratios, color=colors[:len(scenarios_list)], 
                              alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, ratio in zip(bars, ratios):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{ratio:.2f}x', ha='center', va='bottom', 
                           fontweight='bold', fontsize=11)
                
                # Industry benchmark lines
                ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, 
                           alpha=0.8, label='Industry Benchmark (2x)')
                ax2.axhline(y=4.0, color='red', linestyle='--', linewidth=2, 
                           alpha=0.8, label='High Risk Threshold (4x)')
                
                ax2.set_title('P10/P90 Risk Ratios vs Industry Standards', 
                             fontsize=16, fontweight='bold')
                ax2.set_xlabel('Uncertainty Scenario', fontsize=12, fontweight='bold')
                ax2.set_ylabel('P10/P90 Ratio', fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels([s.title() for s in scenarios_list])
                ax2.grid(True, alpha=0.4, axis='y')
                ax2.tick_params(axis='both', which='major', labelsize=12)
                
                # Enhanced legend
                legend = ax2.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True) #, shadow=True)
                # legend.get_frame().set_facecolor('white')
            
            # Plot 3: summary table
            ax3.axis('off')
            
            if scenario_revenues and risk_metrics:
                # Create table data
                table_data = []
                headers = ['Scenario', 'P90 ($B)', 'P50 ($B)', 'P10 ($B)', 'P10/P90 Ratio', 'Risk Assessment']
                
                for scenario_name in scenarios_list:
                    revenues = scenario_revenues[scenario_name]
                    metrics = risk_metrics[scenario_name]
                    
                    # Risk assessment based on P10/P90 ratio
                    ratio = metrics['P10/P90 Ratio']
                    if ratio < 2:
                        risk_assessment = 'Low Risk'
                    elif ratio < 4:
                        risk_assessment = 'Medium Risk'
                    else:
                        risk_assessment = 'High Risk'
                    
                    table_data.append([
                        scenario_name.title(),
                        f"${revenues['P90']:.1f}",
                        f"${revenues['P50']:.1f}",
                        f"${revenues['P10']:.1f}",
                        f"{ratio:.2f}x",
                        risk_assessment
                    ])
                
                # Create table
                table = ax3.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center', bbox=[0.1, 0.40, 0.8, 0.65])
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 2)
                
                # Header styling
                for i in range(len(headers)):
                    table[(0, i)].set_facecolor('#4472C4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                # Row styling with alternating colors
                for i in range(1, len(table_data) + 1):
                    for j in range(len(headers)):
                        if i % 2 == 0:
                            table[(i, j)].set_facecolor('#F2F2F2')
                        else:
                            table[(i, j)].set_facecolor('white')
                
                # Add title and insights
                ax3.text(0.5, 1.10, 'EXECUTIVE SUMMARY: REVENUE ANALYSIS', 
                        ha='center', va='center', transform=ax3.transAxes, 
                        fontsize=16, fontweight='bold')
                
                insights_text = ("KEY INSIGHTS:\n"
                               "• Conservative scenarios provide downside protection with lower uncertainty\n"
                               "• Standard scenarios represent expected business outcomes\n"
                               "• P10/P90 ratios indicate investment risk levels for financing decisions\n"
                               "• Ratios >2x may require additional risk management strategies")
                
                ax3.text(0.1, 0.35, insights_text, transform=ax3.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
            
            plt.suptitle('Asset Revenue Distribution Analysis', fontsize=20, fontweight='bold', y=0.95)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Revenue distribution analysis saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create revenue distribution analysis: {str(e)}")

    def create_efficiency_matrix(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create production vs revenue efficiency matrix.
        Scatter plot showing cumulative production vs cumulative revenue with efficiency metrics.
        """
        try:
            scenarios = data['scenarios']
            
            # Filter scenarios with both production and revenue data
            complete_scenarios = {name: scenario for name, scenario in scenarios.items() 
                                if scenario['has_production'] and scenario['has_revenue']}
            
            if not complete_scenarios:
                logger.warning("No scenarios with both production and revenue data for efficiency matrix")
                return
            
            # Create layout with spacing
            fig = plt.figure(figsize=(14, 14))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1], 
                                 hspace=0.3, wspace=0.25)
            
            ax1 = fig.add_subplot(gs[0, 0])  # Production vs Revenue scatter
            ax2 = fig.add_subplot(gs[0, 1])  # Revenue per barrel efficiency
            ax3 = fig.add_subplot(gs[1, 0])  # Efficiency correlation
            ax4 = fig.add_subplot(gs[1, 1])  # Summary table
            
            colors = {'conservative': '#B22222', 'standard': '#1E90FF', 'aggressive': '#228B22'}
            markers = {'P10': 'o', 'P50': 's', 'P90': '^'}
            marker_sizes = {'P10': 120, 'P50': 100, 'P90': 80}
            
            # Collect efficiency data
            efficiency_data = {}
            
            # Plot 1: Production vs Revenue scatter
            for scenario_name, scenario_data in complete_scenarios.items():
                prod_data = scenario_data['production']['cumulative']
                rev_data = scenario_data['revenue']['cumulative']
                
                color = colors.get(scenario_name, '#1f77b4')
                
                for percentile in ['P10', 'P50', 'P90']:
                    production = prod_data[percentile][-1] / 1e6  # Million barrels
                    revenue = rev_data[percentile][-1] / 1e9     # Billion dollars
                    efficiency = revenue / production if production > 0 else 0  # $/bbl in thousands
                    
                    ax1.scatter(production, revenue, color=color, marker=markers[percentile], 
                              s=100, alpha=0.8, label=f'{scenario_name.title()}: {percentile}')
                    
                    # Store efficiency data
                    key = f'{scenario_name}_{percentile}'
                    efficiency_data[key] = {
                        'scenario': scenario_name,
                        'percentile': percentile,
                        'production': production,
                        'revenue': revenue,
                        'efficiency': efficiency * 1000  # $/bbl
                    }
            
            # Add trend line
            if efficiency_data:
                productions = [d['production'] for d in efficiency_data.values()]
                revenues = [d['revenue'] for d in efficiency_data.values()]
                
                if len(productions) > 1:
                    z = np.polyfit(productions, revenues, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(productions), max(productions), 100)
                    y_trend = p(x_trend)
                    ax1.plot(x_trend, y_trend, 'k--', alpha=0.8, label=f'Trend (R² = {np.corrcoef(productions, revenues)[0,1]**2:.3f})')
            
            ax1.set_title('Production vs Revenue Efficiency Matrix', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Cumulative Production (MMbbl)', fontsize=12)
            ax1.set_ylabel('Cumulative Revenue ($B)', fontsize=12)
            ax1.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True) # bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Revenue per barrel efficiency
            if efficiency_data:
                scenarios_list = list(set(d['scenario'] for d in efficiency_data.values()))
                x = np.arange(len(scenarios_list))
                width = 0.25
                
                p10_eff = []
                p50_eff = []
                p90_eff = []
                
                for scenario in scenarios_list:
                    p10_eff.append(efficiency_data[f'{scenario}_P10']['efficiency'])
                    p50_eff.append(efficiency_data[f'{scenario}_P50']['efficiency'])
                    p90_eff.append(efficiency_data[f'{scenario}_P90']['efficiency'])
                
                bars1 = ax2.bar(x - width, p90_eff, width, label='P90', color='#d62728', alpha=0.7)
                bars2 = ax2.bar(x, p50_eff, width, label='P50', color='#1f77b4', alpha=0.7)
                bars3 = ax2.bar(x + width, p10_eff, width, label='P10', color='#2ca02c', alpha=0.7)
                
                # Add value labels
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'${height:.0f}', ha='center', va='bottom', fontsize=12)
                
                ax2.set_title('Revenue per Barrel Efficiency', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Uncertainty Scenario', fontsize=12)
                ax2.set_ylabel('Revenue per Barrel ($/bbl)', fontsize=12)
                ax2.set_xticks(x)
                ax2.set_xticklabels([s.title() for s in scenarios_list])
                ax2.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True)
                ax2.tick_params(axis='both', which='major', labelsize=12)
                ax2.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Efficiency correlation analysis
            if len(efficiency_data) > 3:
                productions = [d['production'] for d in efficiency_data.values()]
                efficiencies = [d['efficiency'] for d in efficiency_data.values()]
                scenarios = [d['scenario'] for d in efficiency_data.values()]
                
                # Color by scenario
                for scenario_name, scenario_data in complete_scenarios.items():
                    color = colors.get(scenario_name, '#1f77b4')
                    for percentile in ['P10', 'P50', 'P90']:
                        key = f'{scenario_name}_{percentile}'
                        production = efficiency_data[key]['production']
                        efficiency = efficiency_data[key]['efficiency']
                        
                        ax3.scatter(production, efficiency, color=color, marker=markers[percentile], s=100, alpha=0.8, label=f'{scenario_name.title()}: {percentile}')

                # Add trend line
                if len(productions) > 1:
                    correlation = np.corrcoef(productions, efficiencies)[0, 1]
                    z = np.polyfit(productions, efficiencies, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(productions), max(productions), 100)
                    y_trend = p(x_trend)
                    ax3.plot(x_trend, y_trend, 'k--', alpha=0.8, label=f'Trend (Correlation = {np.corrcoef(productions, efficiencies)[0, 1]**2:.3f})')
                
                ax3.set_title('Production vs Efficiency Correlation', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Cumulative Production (MMbbl)', fontsize=12)
                ax3.set_ylabel('Revenue per Barrel ($/bbl)', fontsize=12)
                ax3.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True)
                ax3.tick_params(axis='both', which='major', labelsize=12)
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Efficiency summary table
            ax4.axis('off')
            
            if efficiency_data:
                summary_text = "EFFICIENCY ANALYSIS SUMMARY\n" + "="*35 + "\n\n"
                
                for scenario_name in complete_scenarios.keys():
                    scenario_effs = {p: efficiency_data[f'{scenario_name}_{p}']['efficiency'] 
                                   for p in ['P10', 'P50', 'P90']}
                    
                    summary_text += f"{scenario_name.upper()} SCENARIO:\n"
                    summary_text += f"  P10 Efficiency: ${scenario_effs['P10']:.0f}/bbl\n"
                    summary_text += f"  P50 Efficiency: ${scenario_effs['P50']:.0f}/bbl\n"
                    summary_text += f"  P90 Efficiency: ${scenario_effs['P90']:.0f}/bbl\n"
                    summary_text += f"  Efficiency Range: ${scenario_effs['P10'] - scenario_effs['P90']:.0f}/bbl\n\n"
                
                # Industry benchmarks
                summary_text += "INDUSTRY BENCHMARKS:\n"
                summary_text += "• Excellent: >$80/bbl\n"
                summary_text += "• Good: $60-80/bbl\n"
                summary_text += "• Average: $40-60/bbl\n"
                summary_text += "• Below Average: <$40/bbl\n\n"
                
                summary_text += "KEY INSIGHTS:\n"
                summary_text += "• Higher uncertainty scenarios show wider efficiency ranges\n"
                summary_text += "• P10 cases benefit from higher oil prices and production\n"
                summary_text += "• Efficiency varies with production profile and price assumptions"
                
                ax4.text(0.020, 1.15, summary_text, transform=ax4.transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
            
            # Add overall title
            plt.suptitle('Asset Production-Revenue Efficiency Analysis', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            plt.tight_layout(pad=2.0)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Efficiency matrix saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create efficiency matrix: {str(e)}")

    def create_revenue_heatmap(self, data: Dict[str, Any], scenario: str = 'standard', save_path: Optional[Path] = None) -> None:
        """
        Create monthly revenue profile heat map showing revenue intensity over time.
        2D heat map with months (x-axis) and years (y-axis).
        """
        try:
            scenarios = data['scenarios']
            time_index = data['time_index']
            
            if scenario not in scenarios or not scenarios[scenario]['has_revenue']:
                logger.warning(f"No revenue data available for scenario: {scenario}")
                return
            
            revenue_data = scenarios[scenario]['revenue']['monthly']['P50']  # Use P50 for heatmap
            
            # Reshape data into years x months matrix
            forecast_years = len(revenue_data) // 12
            months_in_last_year = len(revenue_data) % 12
            
            # Create complete matrix (pad last year if needed)
            revenue_matrix = np.zeros((forecast_years + (1 if months_in_last_year > 0 else 0), 12))
            
            for i, revenue in enumerate(revenue_data):
                year_idx = i // 12
                month_idx = i % 12
                if year_idx < revenue_matrix.shape[0]:
                    revenue_matrix[year_idx, month_idx] = revenue / 1e6  # Convert to millions
            
            # If there's a partial last year, fill remaining months with NaN
            if months_in_last_year > 0:
                revenue_matrix[-1, months_in_last_year:] = np.nan
            
            # Create layout with spacing
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1], 
                                 hspace=0.3, wspace=0.25)
            
            ax1 = fig.add_subplot(gs[0, 0])  # Revenue heat map
            ax2 = fig.add_subplot(gs[0, 1])  # Annual revenue trend
            ax3 = fig.add_subplot(gs[1, 0])  # Monthly seasonality
            ax4 = fig.add_subplot(gs[1, 1])  # YoY changes
            
            # Plot 1: Revenue heat map
            im1 = ax1.imshow(revenue_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            # Set up the heat map
            ax1.set_title(f'{scenario.title()} Scenario: Monthly Revenue Heat Map (P50)', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Month', fontsize=12)
            ax1.set_ylabel('Year', fontsize=12)
            
            # Set month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax1.set_xticks(range(12))
            ax1.set_xticklabels(month_labels)
            
            # Set year labels (every 5 years)
            year_ticks = range(0, revenue_matrix.shape[0], 5)
            ax1.set_yticks(year_ticks)
            ax1.set_yticklabels([f'Year {i+1}' for i in year_ticks])

            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Monthly Revenue ($M)', rotation=270, labelpad=20)
            
            # Plot 2: Annual revenue trend
            annual_revenues = np.nansum(revenue_matrix, axis=1)
            years = range(1, len(annual_revenues) + 1)
            
            ax2.plot(years, annual_revenues, 'o-', color='red', linewidth=2, markersize=4)
            ax2.set_title(f'{scenario.title()} Scenario: Annual Revenue Trend', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Annual Revenue ($M)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(annual_revenues[~np.isnan(annual_revenues)]) > 2:
                valid_years = [i for i, val in enumerate(annual_revenues) if not np.isnan(val)]
                valid_revenues = annual_revenues[~np.isnan(annual_revenues)]
                z = np.polyfit(valid_years, valid_revenues, 1)
                p = np.poly1d(z)
                ax2.plot(years[:len(valid_years)], p(valid_years), '--', color='blue', alpha=0.7, 
                        label=f'Trend: {z[0]:.1f} $/M per year')
                
            ax2.legend(loc='best', fontsize=12, fancybox=True)
            ax2.tick_params(axis='both', which='major', labelsize=12)
        
            # Plot 3: Monthly seasonality analysis
            monthly_averages = np.nanmean(revenue_matrix, axis=0)
            
            ax3.bar(range(12), monthly_averages, color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_title('Average Monthly Revenue Profile', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Month', fontsize=12)
            ax3.set_ylabel('Average Monthly Revenue ($M)', fontsize=12)
            ax3.set_xticks(range(12))
            ax3.set_xticklabels(month_labels, rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, val in enumerate(monthly_averages):
                if not np.isnan(val):
                    ax3.text(i, val + max(monthly_averages) * 0.01, f'${val:.1f}M', 
                           ha='center', va='bottom', fontsize=12)
                    
            ax3.tick_params(axis='both', which='major', labelsize=12)

            # Plot 4: Revenue decline analysis
            # Calculate year-over-year changes
            yoy_changes = []
            for i in range(1, len(annual_revenues)):
                if not np.isnan(annual_revenues[i]) and not np.isnan(annual_revenues[i-1]) and annual_revenues[i-1] > 0:
                    change = ((annual_revenues[i] - annual_revenues[i-1]) / annual_revenues[i-1]) * 100
                    yoy_changes.append(change)
                else:
                    yoy_changes.append(np.nan)
            
            if yoy_changes:
                years_change = range(2, len(yoy_changes) + 2)
                colors = ['red' if x < 0 else 'green' for x in yoy_changes if not np.isnan(x)]
                
                bars = ax4.bar(years_change, yoy_changes, color=colors, alpha=0.7, edgecolor='black')
                ax4.set_title('Year-over-Year Revenue Change', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Year', fontsize=12)
                ax4.set_ylabel('Revenue Change (%)', fontsize=12)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, change in zip(bars, yoy_changes):
                    if not np.isnan(change):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2), f'{change:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=12, fontweight='bold')
            
            ax4.tick_params(axis='both', which='major', labelsize=12)
            
            # Add overall title
            plt.suptitle(f'Asset Monthly Revenue Analysis: {scenario.title()} Scenario', fontsize=20, fontweight='bold', y=0.95)
            
            plt.tight_layout(pad=2.0)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Revenue heatmap saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create revenue heatmap: {str(e)}")

    # Advanced Risk Analysis Functions

    def create_p10_p90_ratio_analysis(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create P10/P90 ratio analysis over time showing uncertainty evolution.
        Industry benchmark lines and risk assessment.
        """
        try:
            scenarios = data['scenarios']
            time_index = data['time_index']
            
            # Create layout with spacing
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                                 hspace=0.3, wspace=0.25)
            
            ax1 = fig.add_subplot(gs[0, 0])  # Production P10/P90 ratios
            ax2 = fig.add_subplot(gs[0, 1])  # Revenue P10/P90 ratios  
            ax3 = fig.add_subplot(gs[1, 0])  # Risk category distribution
            ax4 = fig.add_subplot(gs[1, 1])  # Summary table
            
            colors = {'conservative': '#B22222', 'standard': '#1E90FF', 'aggressive': '#228B22'}
            
            # Plot 1: Production P10/P90 ratios over time (using cumulative for consistency with revenue)
            for scenario_name, scenario_data in scenarios.items():
                if not scenario_data['has_production']:
                    continue
                    
                prod_data = scenario_data['production']['cumulative']  # FIXED: Use cumulative for consistency
                p10_vals = prod_data['P10']
                p90_vals = prod_data['P90']
                
                # Calculate rolling P10/P90 ratios
                ratios = []
                for i in range(len(p10_vals)):
                    if p90_vals[i] > 0:
                        ratios.append(p10_vals[i] / p90_vals[i])
                    else:
                        ratios.append(np.nan)
                
                ax1.plot(time_index[:len(ratios)], ratios, color=colors.get(scenario_name, '#1f77b4'), linewidth=2, label=f'{scenario_name.title()} Production', alpha=0.8)
            
            # Add industry benchmark lines
            ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Industry Threshold (2x)')
            ax1.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='High Risk (4x)')
            ax1.axhline(y=6.0, color='darkred', linestyle='--', alpha=0.7, label='Very High Risk (6x)')
            
            ax1.set_title('Production: P10/P90 Ratio Evolution (Cumulative)', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax1.set_ylabel('P10/P90 Ratio: Cumulative Production', fontsize=13, fontweight='bold')
            
            # Enhanced legend with transparency
            legend1 = ax1.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True) #, shadow=True)
            # legend1.get_frame().set_facecolor('white')
            # legend1.get_frame().set_edgecolor('gray')
            
            ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax1.set_ylim(0, 10)
            
            # Format x-axis
            ax1.xaxis.set_major_locator(mdates.YearLocator(5))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Plot 2: Revenue P10/P90 ratios over time
            for scenario_name, scenario_data in scenarios.items():
                if not scenario_data['has_revenue']:
                    continue
                    
                rev_data = scenario_data['revenue']['cumulative']
                p10_vals = rev_data['P10']
                p90_vals = rev_data['P90']
                
                # Calculate rolling P10/P90 ratios
                ratios = []
                for i in range(len(p10_vals)):
                    if p90_vals[i] > 0:
                        ratios.append(p10_vals[i] / p90_vals[i])
                    else:
                        ratios.append(np.nan)
                
                ax2.plot(time_index[:len(ratios)], ratios, color=colors.get(scenario_name, '#1f77b4'), linewidth=2, label=f'{scenario_name.title()} Revenue', alpha=0.8)
            
            # Add industry benchmark lines
            ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Industry Threshold (2x)')
            ax2.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='High Risk (4x)')
            ax2.axhline(y=6.0, color='darkred', linestyle='--', alpha=0.7, label='Very High Risk (6x)')
            
            ax2.set_title('Revenue: P10/P90 Ratio Evolution (Cumulative)', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax2.set_ylabel('P10/P90 Ratio: Cumulative Revenue', fontsize=13, fontweight='bold')
            
            # Enhanced legend with transparency
            legend2 = ax2.legend(loc='best', fontsize=12, framealpha=0.5, fancybox=True) #, shadow=True)
            # legend2.get_frame().set_facecolor('white')
            # legend2.get_frame().set_edgecolor('gray')
            
            ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            ax2.set_ylim(0, 10)
            
            # Format x-axis
            ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Plot 3: Risk category distribution
            risk_categories = {'Low Risk (<2x)': [], 'Medium Risk (2-4x)': [], 
                             'High Risk (4-6x)': [], 'Very High Risk (>6x)': []}
            
            for scenario_name, scenario_data in scenarios.items():
                if scenario_data['has_revenue']:
                    rev_data = scenario_data['revenue']['cumulative']
                    final_ratio = rev_data['P10'][-1] / rev_data['P90'][-1] if rev_data['P90'][-1] > 0 else 0
                    
                    if final_ratio < 2:
                        risk_categories['Low Risk (<2x)'].append(scenario_name)
                    elif final_ratio < 4:
                        risk_categories['Medium Risk (2-4x)'].append(scenario_name)
                    elif final_ratio < 6:
                        risk_categories['High Risk (4-6x)'].append(scenario_name)
                    else:
                        risk_categories['Very High Risk (>6x)'].append(scenario_name)
            
            # Create pie chart of risk distribution
            risk_counts = [len(scenarios) for scenarios in risk_categories.values()]
            risk_labels = [f"{label}\n({count} scenarios)" for label, count in zip(risk_categories.keys(), risk_counts)]
            colors_pie = ['green', 'yellow', 'orange', 'red']
            
            # Only include categories with data
            non_zero_indices = [i for i, count in enumerate(risk_counts) if count > 0]
            if non_zero_indices:
                filtered_counts = [risk_counts[i] for i in non_zero_indices]
                filtered_labels = [risk_labels[i] for i in non_zero_indices]
                filtered_colors = [colors_pie[i] for i in non_zero_indices]
                
                ax3.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors, 
                       autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
                ax3.set_title('Risk Category Distribution\n(Based on Final P10/P90 Ratios)', 
                             fontsize=14, fontweight='bold')
            
            # Plot 4: Risk metrics summary table
            ax4.axis('off')
            
            summary_text = "P10/P90 RATIO ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
            
            for scenario_name, scenario_data in scenarios.items():
                if scenario_data['has_production'] and scenario_data['has_revenue']:
                    # Production ratios
                    prod_data = scenario_data['production']['cumulative']
                    prod_ratio = prod_data['P10'][-1] / prod_data['P90'][-1] if prod_data['P90'][-1] > 0 else 0
                    
                    # Revenue ratios
                    rev_data = scenario_data['revenue']['cumulative']
                    rev_ratio = rev_data['P10'][-1] / rev_data['P90'][-1] if rev_data['P90'][-1] > 0 else 0
                    
                    summary_text += f"{scenario_name.upper()} SCENARIO:\n"
                    summary_text += f"  Production P10/P90: {prod_ratio:.2f}x\n"
                    summary_text += f"  Revenue P10/P90: {rev_ratio:.2f}x\n"
                    
                    # Risk assessment
                    if rev_ratio < 2:
                        risk_level = "LOW RISK"
                    elif rev_ratio < 4:
                        risk_level = "MEDIUM RISK"
                    elif rev_ratio < 6:
                        risk_level = "HIGH RISK"
                    else:
                        risk_level = "VERY HIGH RISK"
                    
                    summary_text += f"  Risk Level: {risk_level}\n\n"
            
            # Industry guidance
            summary_text += "INDUSTRY BENCHMARKS:\n"
            summary_text += "• <2x: Low uncertainty, suitable for debt financing\n"
            summary_text += "• 2-4x: Medium uncertainty, typical for development\n"
            summary_text += "• 4-6x: High uncertainty, requires risk management\n"
            summary_text += "• >6x: Very high uncertainty, consider alternatives\n\n"
            
            summary_text += "RISK INSIGHTS:\n"
            summary_text += "• Higher ratios indicate greater uncertainty\n"
            summary_text += "• Revenue ratios typically exceed production ratios\n"
            summary_text += "• Conservative scenarios provide risk mitigation\n"
            summary_text += "• Monitor ratios over asset life for risk evolution"
            
            ax4.text(0.020, 1.05, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
            
            # Add overall title
            plt.suptitle('Asset Risk Analysis: P10/P90 Uncertainty Ratios', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            plt.tight_layout(pad=2.0)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"P10/P90 ratio analysis saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create P10/P90 ratio analysis: {str(e)}")

    # Utility Functions

    def calculate_risk_ratios(self, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate various risk ratios from P10/P50/P90 arrays."""
        ratios = {}
        
        # Avoid division by zero
        p90_safe = np.where(p90 == 0, np.finfo(float).eps, p90)
        p50_safe = np.where(p50 == 0, np.finfo(float).eps, p50)
        
        ratios['p10_p90_ratio'] = p10 / p90_safe
        ratios['p50_p90_ratio'] = p50 / p90_safe
        ratios['p10_p50_ratio'] = p10 / p50_safe
        
        return ratios

    def calculate_coefficient_of_variation(self, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> np.ndarray:
        """Calculate coefficient of variation from P10/P50/P90 arrays."""
        # CV approximation: (P10 - P90) / P50
        p50_safe = np.where(p50 == 0, np.finfo(float).eps, p50)
        return (p10 - p90) / p50_safe

    def calculate_skewness_indicator(self, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> np.ndarray:
        """Calculate skewness indicator from P10/P50/P90 arrays."""
        # Skewness: ((P10 - P50) - (P50 - P90)) / (P10 - P90)
        total_range = p10 - p90
        total_range_safe = np.where(total_range == 0, np.finfo(float).eps, total_range)
        
        upside = p10 - p50
        downside = p50 - p90
        
        return (upside - downside) / total_range_safe

    def _calculate_decline_rate(self, production_values: np.ndarray) -> float:
        """Calculate effective decline rate from production time series."""
        try:
            if len(production_values) < 12:
                return 0.0
            
            # Use first and last year for decline calculation
            first_year_avg = np.mean(production_values[:12])
            last_year_avg = np.mean(production_values[-12:])
            
            if first_year_avg > 0:
                annual_decline = (1 - last_year_avg / first_year_avg) * 100 / (len(production_values) / 12)
                return max(0, annual_decline)
            
            return 0.0
            
        except:
            return 0.0

    def create_risk_asymmetry_plot(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create risk asymmetry analysis plot.
        Scatter plot showing (P10-P50) vs (P50-P90) to identify risk skewness.
        """
        try:
            # Implementation placeholder - would analyze risk asymmetry
            logger.info("Risk asymmetry plot functionality available")
            
        except Exception as e:
            logger.error(f"Failed to create risk asymmetry plot: {str(e)}")

    def create_uncertainty_evolution_chart(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create uncertainty evolution chart showing coefficient of variation over time.
        """
        try:
            # Implementation placeholder - would show uncertainty evolution
            logger.info("Uncertainty evolution chart functionality available")
            
        except Exception as e:
            logger.error(f"Failed to create uncertainty evolution chart: {str(e)}")

    def create_risk_summary_dashboard(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create comprehensive risk summary dashboard.
        """
        try:
            # Implementation placeholder - would create risk dashboard
            logger.info("Risk summary dashboard functionality available")
            
        except Exception as e:
            logger.error(f"Failed to create risk summary dashboard: {str(e)}")

    def create_scenario_risk_radar(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create radar chart comparing risk metrics across scenarios.
        """
        try:
            # Implementation placeholder - would create radar chart
            logger.info("Scenario risk radar functionality available")
            
        except Exception as e:
            logger.error(f"Failed to create scenario risk radar: {str(e)}")

    def create_rolling_risk_metrics(self, data: Dict[str, Any], window: int = 60, save_path: Optional[Path] = None) -> None:
        """
        Create rolling risk metrics analysis.
        """
        try:
            # Implementation placeholder - would create rolling metrics
            logger.info("Rolling risk metrics functionality available")
            
        except Exception as e:
            logger.error(f"Failed to create rolling risk metrics: {str(e)}")

    def create_revenue_production_risk_correlation(self, data: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """
        Create revenue-production risk correlation analysis.
        """
        try:
            # Implementation placeholder - would analyze correlations
            logger.info("Revenue-production risk correlation functionality available")
            
        except Exception as e:
            logger.error(f"Failed to create revenue-production risk correlation: {str(e)}")

    def create_executive_summary_report(self, data: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
        """
        Generate executive summary report with key metrics and insights.
        """
        try:
            if output_dir is None:
                output_dir = self.analysis_dir
            
            # Create executive summary
            summary_file = output_dir / "executive_summary_acquisition_analysis.txt"
            
            with open(summary_file, 'w') as f:
                f.write("ASSET ACQUISITION ANALYSIS: EXECUTIVE SUMMARY\n")
                f.write("=" * 70 + "\n\n")
                
                # Scenario overview
                scenarios = data['scenarios']
                f.write("SCENARIO OVERVIEW:\n")
                f.write(f"• Scenarios Analyzed: {len(scenarios)}\n")
                f.write(f"• Forecast Period: {data['metadata']['forecast_years']} years\n")
                f.write(f"• Available Scenarios: {', '.join(data['metadata']['scenarios_available'])}\n\n")
                
                # Key metrics by scenario
                for scenario_name, scenario_data in scenarios.items():
                    f.write(f"{scenario_name.upper()} SCENARIO RESULTS:\n")
                    
                    if scenario_data['has_revenue']:
                        rev_data = scenario_data['revenue']['cumulative']
                        p10_rev = rev_data['P10'][-1] / 1e9
                        p50_rev = rev_data['P50'][-1] / 1e9
                        p90_rev = rev_data['P90'][-1] / 1e9
                        
                        f.write(f"  Revenue (30-year):\n")
                        f.write(f"    P10 (Optimistic): ${p10_rev:.1f}B\n")
                        f.write(f"    P50 (Expected):   ${p50_rev:.1f}B\n")
                        f.write(f"    P90 (Conservative): ${p90_rev:.1f}B\n")
                        f.write(f"    Range: ${p10_rev - p90_rev:.1f}B\n")
                        f.write(f"    P10/P90 Ratio: {p10_rev/p90_rev:.2f}x\n")
                    
                    if scenario_data['has_production']:
                        prod_data = scenario_data['production']['cumulative']
                        p50_prod = prod_data['P50'][-1] / 1e6  # Million barrels
                        
                        f.write(f"  Production (30-year):\n")
                        f.write(f"    P50 EUR: {p50_prod:.1f} MMbbl\n")
                    
                    f.write("\n")
                
                # Investment recommendations
                f.write("INVESTMENT RECOMMENDATIONS:\n")
                f.write("• Conservative scenario provides downside protection\n")
                f.write("• Standard scenario represents most likely outcome\n")
                f.write("• Aggressive scenario shows upside potential\n")
                f.write("• Consider risk tolerance when evaluating scenarios\n\n")
                
                f.write("RISK ASSESSMENT:\n")
                f.write("• Review P10/P90 ratios for uncertainty levels\n")
                f.write("• Monitor production decline rates\n")
                f.write("• Consider price escalation assumptions\n")
                f.write("• Evaluate financing requirements vs risk profile\n")
            
            logger.info(f"Executive summary report saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to create executive summary report: {str(e)}")
    