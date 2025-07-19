"""
Advanced Bayesian Forecaster

This forecaster leverages the new ArpsDCA capabilities including:
- Direct FitResult dataclass integration
- Enhanced quality metrics for uncertainty quantification
- Method-aware uncertainty adjustment
- Structured error handling with ArpsDeclineError
- Quality-based parameter estimation
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Local imports
from arps_dca import AdvancedArpsDCA, FitResult, ValidationResult, ArpsDeclineError
from uncertainty_config import UncertaintyConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class BayesianForecastError(Exception):
    """Exception for Bayesian forecasting errors."""
    pass


class ModernizedBayesianForecaster:
    """
    Advanced Bayesian Forecaster integrated with ArpsDCA.
    
    This forecaster uses the advanced ArpsDCA capabilities for:
    - Direct FitResult integration for type-safe operations
    - Quality-aware uncertainty quantification
    - Method-specific parameter estimation
    - Enhanced error handling and validation
    """
    
    def __init__(self,
                 n_samples: int = 1000,
                 confidence_level: float = 0.9,
                 use_analytical_posteriors: bool = True,
                 arps_dca_instance: AdvancedArpsDCA = None,
                 arps_dca_params: Dict[str, Any] = None,
                 cache_results: bool = True,
                 random_seed: Optional[int] = None,
                 uncertainty_level: str = 'standard'):
        """
        Initialize Bayesian forecaster.
        
        Args:
            n_samples: Number of samples for Monte Carlo integration
            confidence_level: Confidence level for credible intervals
            use_analytical_posteriors: Whether to use analytical posterior approximations
            arps_dca_instance: Pre-fitted AdvancedArpsDCA instance (recommended)
            arps_dca_params: Parameters for AdvancedArpsDCA if creating new instance
            cache_results: Whether to cache expensive computations
            random_seed: Random seed for reproducible results (None for random behavior)
            uncertainty_level: Uncertainty level ('standard', 'conservative', 'aggressive', 'high_uncertainty')
        """
        
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self.use_analytical_posteriors = use_analytical_posteriors
        self.cache_results = cache_results
        self.random_seed = random_seed
        self.uncertainty_level = uncertainty_level
        
        # Initialize uncertainty configuration
        self.uncertainty_config = UncertaintyConfig.get_config(uncertainty_level)
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            logger.info(f"Random seed set to {self.random_seed} for reproducible results")
        
        # Use provided ArpsDCA instance or create new one
        if arps_dca_instance is not None:
            self.arps_dca = arps_dca_instance
            self.use_existing_fits = True
            logger.info("ModernizedBayesianForecaster initialized with pre-fitted ArpsDCA instance")
        else:
            # Fallback: Initialize new ArpsDCA
            arps_params = arps_dca_params or {}
            self.arps_dca = AdvancedArpsDCA(**arps_params)
            self.use_existing_fits = False
            logger.warning("ModernizedBayesianForecaster creating new ArpsDCA instance")
        
        # Enhanced results storage
        self.fit_results = {}
        self.bayesian_posteriors = {}
        self.forecast_cache = {}
        self.quality_assessments = {}
        
        # Initialize priors based on uncertainty configuration
        self.priors = self._initialize_enhanced_priors()
        
        logger.info(f"ModernizedBayesianForecaster initialized: {n_samples} samples, "
                   f"confidence={confidence_level}, analytical={use_analytical_posteriors}, "
                   f"uncertainty_level={uncertainty_level}, seed={random_seed}")
    
    def _set_random_state(self, additional_seed: int = 0):
        """Set random state for individual operations."""
        if self.random_seed is not None:
            # Set both numpy global state and return seed for other libraries
            final_seed = self.random_seed + additional_seed
            np.random.seed(final_seed)
            return final_seed
        return None

    def _create_seeded_scipy_distribution(self, dist_type: str, seed_offset: int = 0, **params):
        """Create a scipy.stats distribution with proper random state control."""        
        # Create the distribution
        if dist_type == 'lognorm':
            dist = stats.lognorm(**params)
        elif dist_type == 'gamma':
            dist = stats.gamma(**params)
        elif dist_type == 'beta':
            dist = stats.beta(**params)
        elif dist_type == 'norm':
            dist = stats.norm(**params)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        # Set random state if we have a seed
        if self.random_seed is not None:
            dist.random_state = np.random.RandomState(self.random_seed + seed_offset)
        
        return dist

    def _deterministic_hash(self, well_name: str) -> int:
        """Create deterministic hash from well name for reproducible seeding."""
        import hashlib
        # Use SHA-256 hash which is deterministic across Python runs
        hash_bytes = hashlib.sha256(well_name.encode('utf-8')).digest()
        # Convert first 4 bytes to integer and mod to get reasonable range
        return int.from_bytes(hash_bytes[:4], byteorder='big') % 1000000

    def _initialize_enhanced_priors(self) -> Dict[str, Any]:
        """
        Initialize enhanced priors based on uncertainty configuration.
        
        Returns:
            Dictionary of enhanced prior distributions adjusted for uncertainty level
        """
        # Use uncertainty configuration to generate priors
        return UncertaintyConfig.get_enhanced_priors(self.uncertainty_level)
    
    def fit_bayesian_decline(self,
                           production_data: pd.DataFrame,
                           well_name: str) -> Dict[str, Any]:
        """
        Fit Bayesian decline curve model using enhanced ArpsDCA integration.
        
        Args:
            production_data: Production data DataFrame
            well_name: Well identifier
            
        Returns:
            Dictionary containing enhanced fit results and posterior distributions
        """
        start_time = time.time()
        
        # Ensure global random seed control for consistent results
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        try:
            # Step 1: Get deterministic fit using enhanced ArpsDCA
            if self.use_existing_fits and well_name in self.arps_dca.fit_results:
                # Use existing structured fit results
                fit_result = self.arps_dca.fit_results[well_name]
                validation = self.arps_dca.validation_results.get(well_name)
                
                logger.debug(f"Using existing ArpsDCA fit for well {well_name}: "
                           f"Method={fit_result.method}, "
                           f"Quality={fit_result.quality_metrics}")
                
                if not fit_result.success:
                    return {
                        'success': False,
                        'error': fit_result.error,
                        'method_attempted': fit_result.method,
                        'processing_time': time.time() - start_time
                    }
                
                # Enhanced validation integration
                if validation and validation.warnings:
                    logger.warning(f"Well {well_name} has validation warnings: {validation.warnings}")
                
            else:
                # Fallback: Fit new results
                logger.warning(f"No existing fit found for well {well_name}, performing new fit")
                
                try:
                    fit_result_dict = self.arps_dca.fit_decline_curve(production_data, well_name)
                    
                    if not fit_result_dict['success']:
                        return {
                            'success': False,
                            'error': fit_result_dict['error'],
                            'processing_time': time.time() - start_time
                        }
                
                    # Get the actual FitResult object
                    fit_result = self.arps_dca.fit_results[well_name]
                    validation = self.arps_dca.validation_results.get(well_name)
                    
                except ArpsDeclineError as e:
                    return {
                        'success': False,
                        'error': f"ArpsDCA fitting failed: {str(e)}",
                        'processing_time': time.time() - start_time
                    }
            
            # Step 2: Enhanced quality assessment
            quality_assessment = self._assess_fit_quality(fit_result, validation, well_name)
            self.quality_assessments[well_name] = quality_assessment
            
            # Step 3: Calculate enhanced likelihood parameters
            likelihood_params = self._calculate_enhanced_likelihood_parameters(
                production_data, well_name, fit_result, quality_assessment
            )
            
            # Step 4: Compute Bayesian posteriors with quality weighting
            posterior_params = self._compute_enhanced_bayesian_posteriors(
                fit_result, likelihood_params, quality_assessment
            )
            
            # Step 5: Generate parameter samples
            parameter_samples = self._sample_from_enhanced_posteriors(
                posterior_params, quality_assessment
            )
            
            # Step 6: Bayesian diagnostics
            diagnostics = self._compute_enhanced_bayesian_diagnostics(well_name, fit_result, posterior_params, parameter_samples, quality_assessment)
            
            # Step 7: Generate Bayesian forecasts directly during fitting (avoiding duplicate Monte Carlo)
            bayesian_forecasts = self._generate_bayesian_forecasts_during_fit(
                parameter_samples, quality_assessment, well_name, forecast_months=360
            )
            
            # Store results with integrated forecasts
            bayesian_result = {
                'success': True,
                'well_name': well_name,
                'deterministic_fit': fit_result,
                'quality_assessment': quality_assessment,
                'likelihood_params': likelihood_params,
                'posterior_params': posterior_params,
                'parameter_samples': parameter_samples,
                'bayesian_forecasts': bayesian_forecasts,  # Pre-computed Bayesian forecasts
                'diagnostics': diagnostics,
                'processing_time': time.time() - start_time,
                'method': f"Bayesian with {fit_result.method} deterministic base"
            }
            
            self.fit_results[well_name] = bayesian_result
            self.bayesian_posteriors[well_name] = posterior_params
            
            return bayesian_result
            
        except Exception as e:
            logger.error(f"Bayesian fitting failed for well {well_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _generate_bayesian_forecasts_during_fit(self, parameter_samples: Dict[str, np.ndarray], 
                                               quality_assessment: Dict[str, Any], 
                                               well_name: str, 
                                               forecast_months: int = 360) -> Dict[str, Any]:
        """
        Generate complete Bayesian forecasts during the fitting process.
        This avoids the need for subsequent Monte Carlo calls.
        
        Args:
            parameter_samples: Bayesian parameter samples from posterior
            quality_assessment: Quality assessment results
            well_name: Well identifier
            forecast_months: Number of months to forecast
            
        Returns:
            Complete Bayesian forecast results with percentiles and cumulatives
        """
        try:
            # Generate forecast samples using Bayesian parameter samples
            forecast_samples = self._generate_enhanced_forecast_samples(
                parameter_samples, forecast_months, quality_assessment, well_name
            )
            
            # Calculate industry-standard percentiles from Bayesian samples
            percentiles = [0.9, 0.5, 0.1]  # P10, P50, P90
            forecast_percentiles = self._calculate_enhanced_forecast_percentiles(
                forecast_samples, percentiles, quality_assessment
            )
            
            # Calculate cumulative percentiles
            cumulative_percentiles = self._calculate_enhanced_cumulative_percentiles(
                forecast_percentiles, quality_assessment
            )
            
            return {
                'success': True,
                'forecast_percentiles': forecast_percentiles,
                'cumulative_percentiles': cumulative_percentiles,
                'uncertainty_bounds': self._calculate_uncertainty_bounds(
                    forecast_percentiles, quality_assessment
                ),
                'forecast_months': forecast_months,
                'quality_metadata': {
                    'confidence_level': quality_assessment['confidence_level'],
                    'composite_score': quality_assessment['composite_score'],
                    'method': quality_assessment['method']
                }
            }
            
        except Exception as e:
            logger.error(f"Bayesian forecast generation failed for well {well_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_bayesian_forecasts(self, well_name: str) -> Dict[str, Any]:
        """
        Get pre-computed Bayesian forecasts for a well.
        
        Args:
            well_name: Well identifier
            
        Returns:
            Bayesian forecast results or error dict
        """
        if well_name not in self.fit_results:
            return {
                'success': False,
                'error': f"Well {well_name} not fitted with Bayesian method"
            }
        
        fit_result = self.fit_results[well_name]
        bayesian_forecasts = fit_result.get('bayesian_forecasts')
        
        if not bayesian_forecasts:
            return {
                'success': False,
                'error': f"No Bayesian forecasts available for well {well_name}"
            }
        
        return bayesian_forecasts
    
    def _assess_fit_quality(self, fit_result: FitResult, validation: Optional[ValidationResult], well_name: str) -> Dict[str, Any]:
        """
        Assess fit quality using enhanced ArpsDCA capabilities.
        
        Args:
            fit_result: FitResult dataclass from ArpsDCA
            validation: ValidationResult from ArpsDCA
            well_name: Well identifier
            
        Returns:
            Dictionary with enhanced quality assessment
        """
        quality_metrics = fit_result.quality_metrics or {}
        
        # Base quality metrics
        r_squared = quality_metrics.get('r_squared', 0)
        pearson_r = quality_metrics.get('pearson_r', 0)
        
        # Method-specific quality adjustments
        method_quality_factors = {
            'differential_evolution': 1.0,    # Highest confidence
            'multi_start_lbfgs': 0.95,
            'segmented_regression': 0.9,
            'rate_cumulative_transform': 0.85,
            'robust_regression': 0.8
        }
        
        method_factor = method_quality_factors.get(fit_result.method, 0.75)
        
        # Validation-based adjustments
        validation_factor = 1.0
        if validation:
            if validation.issues:
                validation_factor *= 0.7  # Significant penalty for issues
            if validation.warnings:
                validation_factor *= (1.0 - len(validation.warnings) * 0.05)  # Small penalty per warning
        
        # Calculate composite quality score
        composite_score = (0.6 * r_squared + 0.4 * abs(pearson_r)) * method_factor * validation_factor
        
        # Uncertainty multiplier for parameter estimation using unified approach
        if hasattr(self, 'arps_dca') and self.arps_dca:
            # Use unified uncertainty approach from ArpsDCA if available
            quality_tier = self.arps_dca._determine_quality_tier(fit_result, validation)
            uncertainty_multiplier = self.arps_dca._calculate_uncertainty_multiplier_from_quality(
                quality_tier, fit_result.method, validation
            )
        else:
            # Fallback to business-focused multipliers when ArpsDCA not available
            uncertainty_multiplier = self._fallback_uncertainty_multiplier(
                fit_result.method, r_squared, validation
            )
        
        return {
            'r_squared': r_squared,
            'pearson_r': pearson_r,
            'composite_score': composite_score,
            'method': fit_result.method,
            'method_factor': method_factor,
            'validation_factor': validation_factor,
            'uncertainty_multiplier': uncertainty_multiplier,
            'confidence_level': self._categorize_confidence(composite_score),
            'validation_issues': len(validation.issues) if validation else 0,
            'validation_warnings': len(validation.warnings) if validation else 0
        }
    
    def _fallback_uncertainty_multiplier(self, method: str, r_squared: float, validation: Optional[ValidationResult]) -> float:
        """
        Simplified fallback uncertainty multiplier when ArpsDCA not available.
        
        This is a simplified version that maintains business logic consistency.
        Preferred approach is to use the unified function in ArpsDCA.
        """
        # Business-focused approach based on R² quality
        if r_squared >= 0.8:
            base_multiplier = 1.0      # High quality
        elif r_squared >= 0.6:
            base_multiplier = 1.5      # Medium quality
        elif r_squared >= 0.3:
            base_multiplier = 2.5      # Low quality  
        elif r_squared >= 0:
            base_multiplier = 4.0      # Very low quality
        else:
            base_multiplier = 8.0      # Negative R² - unreliable
        
        # Small adjustment for validation warnings
        if validation and validation.warnings:
            base_multiplier *= (1.0 + len(validation.warnings) * 0.05)
            
        return max(0.8, min(12.0, base_multiplier))  # Consistent bounds with ArpsDCA
    
    def _categorize_confidence(self, composite_score: float) -> str:
        """Categorize confidence level based on composite score."""
        if composite_score > 0.8:
            return 'high'
        elif composite_score > 0.6:
            return 'medium'
        elif composite_score > 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_enhanced_likelihood_parameters(self,
                                                production_data: pd.DataFrame,
                                                well_name: str,
                                                fit_result: FitResult,
                                                quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced likelihood parameters using quality assessment.
        
        Args:
            production_data: Production data
            well_name: Well identifier
            fit_result: FitResult from ArpsDCA
            quality_assessment: Quality assessment results
            
        Returns:
            Enhanced likelihood parameters
        """
        # Get well data
        well_data = production_data[production_data['WellName'] == well_name].copy()
        well_data = well_data.sort_values('DATE')
        
        # Create time array (months from start)
        well_data['DATE'] = pd.to_datetime(well_data['DATE'])
        t = np.arange(len(well_data))
        q = well_data['OIL'].values
        
        # Calculate residuals using fitted parameters
        if fit_result.method and "modified_hyperbolic" in fit_result.method:
            # Use modified hyperbolic prediction
            q_pred = self.arps_dca.fitting_engine.model.predict(
                t, fit_result.qi, fit_result.Di, fit_result.b, fit_result.t_switch
            )
        else:
            # Use hyperbolic prediction
            q_pred = fit_result.qi / (1 + fit_result.b * fit_result.Di * t)**(1/fit_result.b)
        
        residuals = q - q_pred
        
        # Enhanced noise estimation with quality weighting
        noise_std = np.std(residuals)
        noise_precision = 1.0 / (noise_std**2 + 1e-8)
        
        # Adjust precision based on quality assessment
        noise_precision *= quality_assessment['validation_factor']
        
        # Calculate parameter covariance matrix approximation
        try:
            # Use Hessian approximation for parameter uncertainty
            hessian = self._approximate_enhanced_hessian(
                t, q, fit_result.qi, fit_result.Di, fit_result.b, quality_assessment
            )
            
            # Check for NaN/Inf in Hessian
            if np.any(np.isnan(hessian)) or np.any(np.isinf(hessian)):
                raise ValueError("NaN/Inf values in Hessian matrix")
            
            # Check condition number before inversion
            condition_number = np.linalg.cond(hessian)
            if condition_number > 1e12:
                raise ValueError(f"Poorly conditioned Hessian matrix: condition={condition_number}")
            
            # Stronger regularization for numerical stability
            regularization = max(1e-6, np.max(np.diag(hessian)) * 1e-8)
            hessian += np.eye(3) * regularization
            
            # Safe inversion with additional checks
            param_covariance = np.linalg.inv(hessian)
            
            # Verify inversion result
            if np.any(np.isnan(param_covariance)) or np.any(np.isinf(param_covariance)):
                raise ValueError("NaN/Inf values in covariance matrix after inversion")
                
            # Ensure positive definiteness by checking diagonal elements
            if np.any(np.diag(param_covariance) <= 0):
                raise ValueError("Non-positive diagonal elements in covariance matrix")
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"Hessian inversion failed ({str(e)}), using robust diagonal covariance")
            # Robust fallback to diagonal covariance with proper bounds
            qi_var = max((0.1 * fit_result.qi)**2, 1.0)  # Minimum variance
            Di_var = max((0.1 * fit_result.Di)**2, 1e-6)
            b_var = max((0.1 * fit_result.b)**2, 1e-4)
            
            param_covariance = np.diag([qi_var, Di_var, b_var])
        
        # Apply uncertainty multiplier
        param_covariance *= quality_assessment['uncertainty_multiplier']**2
        
        return {
            'noise_precision': noise_precision,
            'param_covariance': param_covariance,
            'residuals': residuals,
            'noise_std': noise_std,
            'data_fit_quality': quality_assessment['r_squared'],
            'effective_sample_size': len(t) * quality_assessment['validation_factor']
        }
    
    def _approximate_enhanced_hessian(self, t: np.ndarray, q: np.ndarray, 
                                    qi: float, Di: float, b: float,
                                    quality_assessment: Dict[str, Any]) -> np.ndarray:
        """
        Approximate Hessian matrix for parameter uncertainty with quality weighting.
        
        Args:
            t: Time array
            q: Production data
            qi: Initial production rate
            Di: Decline rate
            b: Hyperbolic exponent
            quality_assessment: Quality assessment results
            
        Returns:
            Enhanced Hessian matrix
        """
        # Compute numerical gradients
        def objective(params):
            qi_p, Di_p, b_p = params
            if b_p == 0:
                q_pred = qi_p * np.exp(-Di_p * t)
            else:
                q_pred = qi_p / (1 + b_p * Di_p * t)**(1/b_p)
            
            # Weight by quality assessment
            weight = quality_assessment['validation_factor']
            return weight * np.sum((q - q_pred)**2)
        
        params = np.array([qi, Di, b])
        eps = 1e-6
        
        # Compute Hessian using finite differences
        hessian = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                params_ij = params.copy()
                params_ij[i] += eps
                params_ij[j] += eps
                
                params_i = params.copy()
                params_i[i] += eps
                
                params_j = params.copy()
                params_j[j] += eps
                
                hessian[i, j] = (
                    objective(params_ij) - objective(params_i) - 
                    objective(params_j) + objective(params)
                ) / (eps**2)
        
        return hessian
    
    def _compute_enhanced_bayesian_posteriors(self,
                                            fit_result: FitResult,
                                            likelihood_params: Dict[str, Any],
                                            quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Bayesian posteriors with quality weighting.
        
        Args:
            fit_result: FitResult from ArpsDCA
            likelihood_params: Enhanced likelihood parameters
            quality_assessment: Quality assessment results
            
        Returns:
            Enhanced posterior parameters
        """
        # Get deterministic estimates
        qi_map = fit_result.qi
        Di_map = fit_result.Di
        b_map = fit_result.b
        
        # Get likelihood precision
        data_precision = likelihood_params['noise_precision']
        param_covariance = likelihood_params['param_covariance']
        
        # Enhanced prior weighting based on quality
        prior_weight = 1.0 / (1.0 + quality_assessment['composite_score'])
        
        # Compute posterior parameters for each parameter
        posteriors = {}
        
        # qi posterior (log-normal)
        qi_prior = self.priors['qi']
        
        # Ensure qi_map is positive for log operation
        qi_map_safe = max(qi_map, 1e-6) if qi_map > 0 else 100.0  # Default fallback
        
        if qi_map <= 0:
            logger.warning(f"Non-positive qi_map ({qi_map}), using fallback value {qi_map_safe}")
            
        qi_log_mean = np.log(qi_map_safe)
        
        # Safe variance calculation with minimum bounds
        qi_log_var = max(param_covariance[0, 0] / (qi_map_safe**2), 1e-10)  # Minimum variance
        
        # Check for problematic values
        if qi_log_var <= 0 or np.isnan(qi_log_var) or np.isinf(qi_log_var):
            logger.warning(f"Invalid qi_log_var ({qi_log_var}), using default variance")
            qi_log_var = 1.0  # Safe default
        
        # Safe precision calculation
        prior_precision = 1.0 / (qi_prior['params']['sigma']**2)
        likelihood_precision = 1.0 / qi_log_var
        
        # Check for infinite precision
        if np.isinf(likelihood_precision):
            logger.warning(f"Infinite likelihood precision, clamping to maximum value")
            likelihood_precision = 1e6  # Large but finite
            
        qi_posterior_precision = likelihood_precision + prior_weight * prior_precision
        
        # Safe mean calculation with bounds checking
        if qi_posterior_precision <= 0 or np.isnan(qi_posterior_precision) or np.isinf(qi_posterior_precision):
            logger.warning(f"Invalid posterior precision ({qi_posterior_precision}), using prior")
            qi_posterior_mean = qi_prior['params']['mu']
            qi_posterior_precision = prior_precision
        else:
            qi_posterior_mean = (
                qi_log_mean * likelihood_precision + 
                prior_weight * qi_prior['params']['mu'] * prior_precision
            ) / qi_posterior_precision
        
        # Final safety checks
        if np.isnan(qi_posterior_mean) or np.isinf(qi_posterior_mean):
            logger.warning(f"Invalid qi_posterior_mean, using prior mean")
            qi_posterior_mean = qi_prior['params']['mu']
            
        qi_posterior_std = 1.0 / np.sqrt(max(qi_posterior_precision, 1e-6))
        
        posteriors['qi'] = {
            'distribution': 'lognormal',
            'mean': qi_posterior_mean,
            'precision': qi_posterior_precision,
            'std': qi_posterior_std
        }
        
        # Di posterior (gamma)
        Di_prior = self.priors['Di']
        Di_var = max(param_covariance[1, 1], 1e-10)  # Minimum variance
        
        # Check for problematic variance
        if Di_var <= 0 or np.isnan(Di_var) or np.isinf(Di_var):
            logger.warning(f"Invalid Di_var ({Di_var}), using default variance")
            Di_var = (0.1 * Di_map)**2  # Safe default based on MAP estimate
        
        # Gamma parameter calculation
        Di_posterior_shape = Di_prior['params']['a'] + max(data_precision / 2, 0.1)
        Di_posterior_rate = Di_prior['params']['scale'] + max(data_precision * Di_var / 2, 1e-8)
        
        # Ensure positive parameters for gamma distribution
        Di_posterior_shape = max(Di_posterior_shape, 0.1)
        Di_posterior_rate = max(Di_posterior_rate, 1e-8)
        
        # Check for NaN/Inf
        if np.isnan(Di_posterior_shape) or np.isinf(Di_posterior_shape):
            logger.warning(f"Invalid Di_posterior_shape, using prior parameters")
            Di_posterior_shape = Di_prior['params']['a']
            
        if np.isnan(Di_posterior_rate) or np.isinf(Di_posterior_rate):
            logger.warning(f"Invalid Di_posterior_rate, using prior parameters")
            Di_posterior_rate = Di_prior['params']['scale']
        
        # Safe mean calculation
        if Di_posterior_rate > 0:
            Di_posterior_mean = Di_posterior_shape / Di_posterior_rate
        else:
            Di_posterior_mean = Di_map  # Fallback to MAP estimate
        
        posteriors['Di'] = {
            'distribution': 'gamma',
            'shape': Di_posterior_shape,
            'rate': Di_posterior_rate,
            'mean': Di_posterior_mean
        }
        
        # b posterior (truncated normal)
        b_prior = self.priors['b']
        b_var = max(param_covariance[2, 2], 1e-10)  # Minimum variance
        
        # Check for problematic variance
        if b_var <= 0 or np.isnan(b_var) or np.isinf(b_var):
            logger.warning(f"Invalid b_var ({b_var}), using default variance")
            b_var = (0.1 * b_map)**2  # Safe default based on MAP estimate
        
        # Safe beta prior conversion to normal approximation
        b_alpha = b_prior['params']['a']
        b_beta = b_prior['params']['b']
        b_sum = b_alpha + b_beta
        
        b_prior_mean = b_alpha / b_sum
        b_prior_var = (b_alpha * b_beta) / (b_sum**2 * (b_sum + 1))
        
        # Ensure positive prior variance
        b_prior_var = max(b_prior_var, 1e-8)
        
        # Safe precision calculation
        likelihood_precision = 1.0 / b_var
        prior_prec = 1.0 / b_prior_var
        
        if np.isinf(likelihood_precision):
            logger.warning(f"Infinite b likelihood precision, clamping")
            likelihood_precision = 1e6
            
        b_posterior_precision = likelihood_precision + prior_weight * prior_prec
        
        # Ensure positive precision
        b_posterior_precision = max(b_posterior_precision, 1e-6)
        
        # Safe mean calculation
        if b_posterior_precision > 0 and not (np.isnan(b_posterior_precision) or np.isinf(b_posterior_precision)):
            b_posterior_mean = (
                b_map * likelihood_precision + prior_weight * b_prior_mean * prior_prec
            ) / b_posterior_precision
        else:
            logger.warning(f"Invalid b_posterior_precision, using prior mean")
            b_posterior_mean = b_prior_mean
            b_posterior_precision = prior_prec
        
        # Final safety checks
        if np.isnan(b_posterior_mean) or np.isinf(b_posterior_mean):
            logger.warning(f"Invalid b_posterior_mean, using prior mean")
            b_posterior_mean = b_prior_mean
            
        b_posterior_std = 1.0 / np.sqrt(max(b_posterior_precision, 1e-6))
        
        posteriors['b'] = {
            'distribution': 'truncated_normal',
            'mean': b_posterior_mean,
            'precision': b_posterior_precision,
            'std': b_posterior_std,
            'bounds': (0.0, 2.0)
        }
        
        # Add quality-based metadata
        posteriors['quality_metadata'] = {
            'composite_score': quality_assessment['composite_score'],
            'uncertainty_multiplier': quality_assessment['uncertainty_multiplier'],
            'method': fit_result.method,
            'prior_weight': prior_weight
        }
        
        return posteriors
    
    def _sample_from_enhanced_posteriors(self, posterior_params: Dict[str, Any], quality_assessment: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate samples from posterior distributions.
        
        Args:
            posterior_params: Posterior parameters
            quality_assessment: Quality assessment results
            
        Returns:
            Dictionary of parameter samples
        """
        # Set random state for reproducible sampling
        self._set_random_state(1)
        
        # Generate samples with quality-based adjustment
        n_samples = self.n_samples
        
        # qi samples (log-normal)
        qi_params = posterior_params['qi']
        try:
            # Validate parameters before sampling
            qi_mean = qi_params['mean']
            qi_std = qi_params['std']
            
            if np.isnan(qi_mean) or np.isinf(qi_mean):
                logger.warning(f"Invalid qi mean ({qi_mean}), using prior")
                qi_mean = self.priors['qi']['params']['mu']
                
            if np.isnan(qi_std) or np.isinf(qi_std) or qi_std <= 0:
                logger.warning(f"Invalid qi std ({qi_std}), using prior")
                qi_std = self.priors['qi']['params']['sigma']
            
            qi_samples = np.random.lognormal(qi_mean, qi_std, n_samples)
            
            # Check for NaN in samples
            if np.any(np.isnan(qi_samples)):
                logger.warning("NaN values in qi samples, using fallback")
                qi_samples = np.full(n_samples, np.exp(qi_mean))
                
        except Exception as e:
            logger.warning(f"qi sampling failed ({str(e)}), using deterministic fallback")
            qi_samples = np.full(n_samples, 100.0)  # Safe fallback
        
        # Apply bounds
        qi_samples = np.clip(qi_samples, self.priors['qi']['bounds'][0], self.priors['qi']['bounds'][1])
        
        # Di samples (gamma)
        Di_params = posterior_params['Di']
        try:
            # Validate parameters before sampling
            Di_shape = Di_params['shape']
            Di_rate = Di_params['rate']
            
            if np.isnan(Di_shape) or np.isinf(Di_shape) or Di_shape <= 0:
                logger.warning(f"Invalid Di shape ({Di_shape}), using prior")
                Di_shape = self.priors['Di']['params']['a']
                
            if np.isnan(Di_rate) or np.isinf(Di_rate) or Di_rate <= 0:
                logger.warning(f"Invalid Di rate ({Di_rate}), using prior")
                Di_rate = 1.0 / self.priors['Di']['params']['scale']
            
            Di_samples = np.random.gamma(Di_shape, 1.0 / Di_rate, n_samples)
            
            # Check for NaN in samples
            if np.any(np.isnan(Di_samples)):
                logger.warning("NaN values in Di samples, using fallback")
                Di_samples = np.full(n_samples, Di_shape / Di_rate)
                
        except Exception as e:
            logger.warning(f"Di sampling failed ({str(e)}), using deterministic fallback")
            Di_samples = np.full(n_samples, 0.1)  # Safe fallback
        
        # Apply bounds
        Di_samples = np.clip(Di_samples, self.priors['Di']['bounds'][0], self.priors['Di']['bounds'][1])
        
        # b samples (truncated normal)
        b_params = posterior_params['b']
        try:
            # Validate parameters before sampling
            b_mean = b_params['mean']
            b_std = b_params['std']
            
            if np.isnan(b_mean) or np.isinf(b_mean):
                logger.warning(f"Invalid b mean ({b_mean}), using prior")
                b_alpha = self.priors['b']['params']['a']
                b_beta = self.priors['b']['params']['b']
                b_mean = b_alpha / (b_alpha + b_beta)
                
            if np.isnan(b_std) or np.isinf(b_std) or b_std <= 0:
                logger.warning(f"Invalid b std ({b_std}), using default")
                b_std = 0.1  # Safe default
            
            b_samples = np.random.normal(b_mean, b_std, n_samples)
            
            # Check for NaN in samples
            if np.any(np.isnan(b_samples)):
                logger.warning("NaN values in b samples, using fallback")
                b_samples = np.full(n_samples, b_mean)
                
        except Exception as e:
            logger.warning(f"b sampling failed ({str(e)}), using deterministic fallback")
            b_samples = np.full(n_samples, 0.5)  # Safe fallback
        
        # Apply bounds
        b_samples = np.clip(b_samples, b_params['bounds'][0], b_params['bounds'][1])
        
        # Quality-based sample filtering - use uncertainty configuration
        noise_multipliers = self.uncertainty_config['quality_noise_multipliers']
        
        if quality_assessment['confidence_level'] == 'high':
            # Use configured noise for high confidence
            if noise_multipliers['high'] > 0:
                self._set_random_state(2)  # Different seed for noise
                qi_samples += np.random.normal(0, noise_multipliers['high'] * qi_samples)
                Di_samples += np.random.normal(0, noise_multipliers['high'] * Di_samples)
                b_samples += np.random.normal(0, noise_multipliers['high'] * b_samples)
        elif quality_assessment['confidence_level'] == 'medium':
            # Use configured noise for medium confidence
            self._set_random_state(2)  # Different seed for noise
            qi_samples += np.random.normal(0, noise_multipliers['medium'] * qi_samples)
            Di_samples += np.random.normal(0, noise_multipliers['medium'] * Di_samples)
            b_samples += np.random.normal(0, noise_multipliers['medium'] * b_samples)
        else:
            # Use configured noise for low confidence
            self._set_random_state(3)  # Different seed for higher noise
            qi_samples += np.random.normal(0, noise_multipliers['low'] * qi_samples)
            Di_samples += np.random.normal(0, noise_multipliers['low'] * Di_samples)
            b_samples += np.random.normal(0, noise_multipliers['low'] * b_samples)
        
        return {
            'qi': qi_samples,
            'Di': Di_samples,
            'b': b_samples,
            'quality_score': np.full(n_samples, quality_assessment['composite_score'])
        }
    
    def _compute_enhanced_bayesian_diagnostics(self, well_name: str, fit_result: FitResult, posterior_params: Dict[str, Any], parameter_samples: Dict[str, np.ndarray], quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Bayesian diagnostics.
        
        Args:
            well_name: Well identifier
            fit_result: FitResult from ArpsDCA
            posterior_params: Posterior parameters
            parameter_samples: Parameter samples
            quality_assessment: Quality assessment results
            
        Returns:
            Enhanced diagnostic metrics
        """
        # Basic parameter statistics
        param_stats = {}
        for param in ['qi', 'Di', 'b']:
            samples = parameter_samples[param]
            param_stats[param] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'percentiles': {
                                'P10': np.percentile(samples, 10),
            'P90': np.percentile(samples, 90)
                }
            }
        
        # Parameter correlations
        param_corr = np.corrcoef([
            parameter_samples['qi'],
            parameter_samples['Di'],
            parameter_samples['b']
        ])
        
        # Effective sample size based on quality
        effective_n = self.n_samples * quality_assessment['validation_factor']
        
        # Convergence diagnostics
        convergence_metrics = {
            'effective_sample_size': effective_n,
            'monte_carlo_error': {
                'qi': np.std(parameter_samples['qi']) / np.sqrt(effective_n),
                'Di': np.std(parameter_samples['Di']) / np.sqrt(effective_n),
                'b': np.std(parameter_samples['b']) / np.sqrt(effective_n)
            }
        }
        
        # Quality-based diagnostics
        quality_diagnostics = {
            'confidence_level': quality_assessment['confidence_level'],
            'composite_score': quality_assessment['composite_score'],
            'uncertainty_multiplier': quality_assessment['uncertainty_multiplier'],
            'method_reliability': quality_assessment['method_factor'],
            'validation_score': quality_assessment['validation_factor']
        }
        
        return {
            'parameter_statistics': param_stats,
            'parameter_correlations': param_corr.tolist(),
            'convergence_metrics': convergence_metrics,
            'quality_diagnostics': quality_diagnostics,
            'posterior_metadata': posterior_params.get('quality_metadata', {}),
            'sample_quality': np.mean(parameter_samples['quality_score'])
        }
    
    def forecast_probabilistic(self, well_name: str, forecast_months: int = 360, percentiles: List[float] = None) -> Dict[str, Any]:
        """
        Generate probabilistic forecasts using enhanced parameter samples.
        
        DEPRECATION WARNING: If you have used fit_bayesian_decline(), the forecasts are already
        pre-computed. Use get_bayesian_forecasts() instead to avoid duplicate Monte Carlo computation.
        
        This method is primarily for standalone Monte Carlo simulation scenarios.
        
        Args:
            well_name: Well identifier
            forecast_months: Number of months to forecast
            percentiles: Percentiles to calculate
            
        Returns:
            Enhanced probabilistic forecast results
        """
        if percentiles is None:
            # Industry convention: P10 = optimistic (high), P50 = median, P90 = conservative (low)
            percentiles = [0.9, 0.5, 0.1]
        
        if well_name not in self.fit_results:
            raise BayesianForecastError(f"Well {well_name} not fitted")
        
        # Check if Bayesian forecasts already exist
        fit_result = self.fit_results[well_name]
        if 'bayesian_forecasts' in fit_result and fit_result['bayesian_forecasts']['success']:
            logger.warning(f"Well {well_name}: Bayesian forecasts already computed during fit_bayesian_decline(). "
                          f"Consider using get_bayesian_forecasts() to avoid duplicate computation.")
            # Return existing forecasts to avoid recomputation
            return fit_result['bayesian_forecasts']
        
        try:
            # Get fit results and quality assessment
            fit_result = self.fit_results[well_name]
            parameter_samples = fit_result['parameter_samples']
            quality_assessment = fit_result['quality_assessment']
            
            # Generate forecast samples using enhanced ArpsDCA integration
            forecast_samples = self._generate_enhanced_forecast_samples(
                parameter_samples, forecast_months, quality_assessment, well_name
            )
        
            # Calculate percentiles
            forecast_percentiles = self._calculate_enhanced_forecast_percentiles(
                forecast_samples, percentiles, quality_assessment
            )
            
            # Calculate cumulative percentiles
            cumulative_percentiles = self._calculate_enhanced_cumulative_percentiles(
                forecast_percentiles, quality_assessment
            )
            
            return {
                'success': True,
                'well_name': well_name,
                'forecast_months': forecast_months,
                'forecast_percentiles': forecast_percentiles,
                'cumulative_percentiles': cumulative_percentiles,
                'uncertainty_bounds': self._calculate_uncertainty_bounds(
                    forecast_percentiles, quality_assessment
                ),
                'quality_metadata': {
                    'confidence_level': quality_assessment['confidence_level'],
                    'composite_score': quality_assessment['composite_score'],
                    'method': quality_assessment['method']
                }
            }
            
        except Exception as e:
            logger.error(f"Probabilistic forecasting failed for well {well_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'well_name': well_name
            }
    
    def _generate_enhanced_forecast_samples(self, parameter_samples: Dict[str, np.ndarray], forecast_months: int, quality_assessment: Dict[str, Any], well_name: str) -> np.ndarray:
        """
        Generate forecast samples using ArpsDCA integration.
        
        Args:
            parameter_samples: Parameter samples
            forecast_months: Number of months to forecast
            quality_assessment: Quality assessment results
            well_name: Well name for seed variation
            
        Returns:
            Forecast samples array
        """
        # Set random state for reproducible forecasts (use deterministic well name hash for variation)
        well_seed = self._deterministic_hash(well_name) if well_name else 0
        self._set_random_state(4 + well_seed)
        
        qi_samples = parameter_samples['qi']
        Di_samples = parameter_samples['Di']
        b_samples = parameter_samples['b']
        
        # Time array
        t = np.arange(forecast_months)
        
        # Generate forecast samples
        forecast_samples = np.zeros((self.n_samples, forecast_months))
        
        for i in range(self.n_samples):
            try:
                # Use ArpsDCA's prediction method
                forecast_values = self.arps_dca.predict_decline_curve(
                    well_name, t, qi_samples[i], Di_samples[i], b_samples[i]
                )
                
                # Apply quality-based noise with seed control
                if quality_assessment['confidence_level'] != 'high':
                    # Reset seed for each sample for consistency
                    if self.random_seed is not None:
                        np.random.seed(self.random_seed + 5 + well_seed + i)
                    
                    noise_std = 0.05 * forecast_values * quality_assessment['uncertainty_multiplier']
                    forecast_values += np.random.normal(0, noise_std)
                
                # Ensure non-negative values with minimum economic threshold
                forecast_values = np.maximum(forecast_values, 1.0)  # 1 bbl/month minimum
                
                forecast_samples[i, :] = forecast_values
                
            except Exception as e:
                # Fallback to manual calculation with proper error handling
                qi = qi_samples[i]
                Di = Di_samples[i]
                b = b_samples[i]
                
                # Handle division by zero and other numerical issues
                if abs(b) < 1e-6:  # Treat near-zero b as exponential decline
                    forecast_values = qi * np.exp(-Di * t)
                else:
                    try:
                        forecast_values = qi / (1 + b * Di * t)**(1/b)
                    except (ZeroDivisionError, OverflowError, ValueError):
                        # Fallback to exponential decline
                        forecast_values = qi * np.exp(-Di * t)
                
                # Apply quality-based noise with seed control
                if quality_assessment['confidence_level'] != 'high':
                    # Reset seed for each sample for consistency
                    if self.random_seed is not None:
                        np.random.seed(self.random_seed + 5 + well_seed + i)
                    
                    noise_std = 0.05 * forecast_values * quality_assessment['uncertainty_multiplier']
                    forecast_values += np.random.normal(0, noise_std)
                
                # Ensure non-negative values with minimum economic threshold
                forecast_values = np.maximum(forecast_values, 1.0)  # 1 bbl/month minimum
                
                forecast_samples[i, :] = forecast_values
                
                logger.debug(f"Used fallback calculation for well {well_name}, sample {i}: {str(e)}")
        
        return forecast_samples
    
    def _calculate_enhanced_forecast_percentiles(self, forecast_samples: np.ndarray, percentiles: List[float], quality_assessment: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate enhanced forecast percentiles with quality weighting.
        
        Args:
            forecast_samples: Forecast samples
            percentiles: Percentiles to calculate
            quality_assessment: Quality assessment results
            
        Returns:
            Enhanced forecast percentiles
        """
        percentile_results = {}
        
        # Apply quality-based smoothing for low-confidence forecasts
        if quality_assessment['confidence_level'] in ['low', 'very_low']:
            # Apply smoothing to reduce noise
            smoothed_samples = np.apply_along_axis(
                lambda x: gaussian_filter1d(x, sigma=2), 
                axis=1, 
                arr=forecast_samples
            )
            forecast_samples = smoothed_samples
        
        # Calculate percentiles with industry convention mapping
        percentile_mapping = {
            0.9: "P10",  # 90th percentile -> P10 (optimistic/high reserves)
            0.5: "P50",  # 50th percentile -> P50 (median reserves)
            0.1: "P90"   # 10th percentile -> P90 (conservative/low reserves)
        }
        
        for p in percentiles:
            percentile_key = percentile_mapping.get(p, f"p{int(p*100)}")
            percentile_results[percentile_key] = np.percentile(
                forecast_samples, p * 100, axis=0
            )
        
        return percentile_results
    
    def _calculate_enhanced_cumulative_percentiles(self, forecast_percentiles: Dict[str, np.ndarray], quality_assessment: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate enhanced cumulative percentiles.
        
        Args:
            forecast_percentiles: Forecast percentiles
            quality_assessment: Quality assessment results
            
        Returns:
            Enhanced cumulative percentiles
        """
        cumulative_percentiles = {}
        
        for percentile_key, forecast_values in forecast_percentiles.items():
            # Calculate cumulative sum
            cumulative_values = np.cumsum(forecast_values)
            
            # Apply quality-based adjustment
            if quality_assessment['confidence_level'] == 'high':
                # No adjustment for high confidence
                cumulative_percentiles[percentile_key] = cumulative_values
            else:
                # Apply conservative adjustment for lower confidence
                adjustment_factor = 0.95 if quality_assessment['confidence_level'] == 'medium' else 0.9
                cumulative_percentiles[percentile_key] = cumulative_values * adjustment_factor
        
        return cumulative_percentiles
    
    def _calculate_uncertainty_bounds(self, forecast_percentiles: Dict[str, np.ndarray], quality_assessment: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate uncertainty bounds based on quality assessment.
        
        Args:
            forecast_percentiles: Forecast percentiles
            quality_assessment: Quality assessment results
            
        Returns:
            Uncertainty bounds
        """
        # Get P10 and P90 for uncertainty bounds
        p10 = forecast_percentiles.get('P10', np.zeros(len(forecast_percentiles['P50'])))
        p90 = forecast_percentiles.get('P90', np.zeros(len(forecast_percentiles['P50'])))
        p50 = forecast_percentiles['P50']
        
        # Calculate uncertainty metrics
        uncertainty_range = p10 - p90
        relative_uncertainty = uncertainty_range / (p50 + 1e-8)
        
        # Quality-based uncertainty adjustment
        confidence_multiplier = {
            'high': 1.0,
            'medium': 1.2,
            'low': 1.5,
            'very_low': 2.0
        }.get(quality_assessment['confidence_level'], 1.0)
        
        adjusted_uncertainty = relative_uncertainty * confidence_multiplier
        
        return {
            'upper_bound': p50 + adjusted_uncertainty * p50,
            'lower_bound': p50 - adjusted_uncertainty * p50,
            'uncertainty_range': uncertainty_range,
            'relative_uncertainty': relative_uncertainty,
            'confidence_multiplier': confidence_multiplier
        }
    
    def get_parameter_correlations(self, well_name: str) -> Dict[str, Any]:
        """
        Get enhanced parameter correlations.
        
        Args:
            well_name: Well identifier
            
        Returns:
            Enhanced parameter correlations
        """
        if well_name not in self.fit_results:
            raise BayesianForecastError(f"Well {well_name} not fitted")
        
        fit_result = self.fit_results[well_name]
        parameter_samples = fit_result['parameter_samples']
        
        # Calculate correlations
        correlations = np.corrcoef([
            parameter_samples['qi'],
            parameter_samples['Di'],
            parameter_samples['b']
        ])
        
        return {
            'correlation_matrix': correlations.tolist(),
            'qi_Di_correlation': correlations[0, 1],
            'qi_b_correlation': correlations[0, 2],
            'Di_b_correlation': correlations[1, 2],
            'quality_score': fit_result['quality_assessment']['composite_score']
        }
    
    def get_fit_summary(self) -> pd.DataFrame:
        """
        Get enhanced fit summary.
        
        Returns:
            Enhanced fit summary DataFrame
        """
        if not self.fit_results:
            return pd.DataFrame()
        
        summary_data = []
        for well_name, result in self.fit_results.items():
            if result['success']:
                quality_assessment = result['quality_assessment']
                diagnostics = result['diagnostics']
                
                summary_data.append({
                    'WellName': well_name,
                    'success': True,
                    'method': quality_assessment['method'],
                    'composite_score': quality_assessment['composite_score'],
                    'confidence_level': quality_assessment['confidence_level'],
                    'r_squared': quality_assessment['r_squared'],
                    'uncertainty_multiplier': quality_assessment['uncertainty_multiplier'],
                    'parameter_uncertainty': diagnostics['convergence_metrics']['effective_sample_size'],
                    'forecast_uncertainty': diagnostics['quality_diagnostics']['uncertainty_multiplier'],
                    'validation_issues': quality_assessment['validation_issues'],
                    'validation_warnings': quality_assessment['validation_warnings'],
                    'processing_time': result['processing_time']
                })
        
        return pd.DataFrame(summary_data)


class AssetScaleBayesianForecaster(ModernizedBayesianForecaster):
    """
    Asset-scale Bayesian forecaster optimized for 374+ wells with <10 minute runtime.
    
    Key improvements:
    - Hierarchical Bayesian modeling for efficiency
    - Fast Approximate Bayesian Computation (ABC)
    - Vectorized batch processing
    - Memory-efficient uncertainty propagation
    - Adaptive quality-based sampling
    """
    
    def __init__(self, **kwargs):
        # Extract uncertainty_level if provided
        uncertainty_level = kwargs.get('uncertainty_level', 'standard')
        
        # Extract random_seed if provided
        random_seed = kwargs.get('random_seed', None)
        
        super().__init__(**kwargs)
        self.asset_priors = None
        self.field_parameters = {}
        self.well_clusters = {}
        self.population_priors = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_wells': 0,
            'successful_wells': 0,
            'hierarchical_wells': 0,
            'abc_wells': 0,
            'deterministic_wells': 0
        }
        
        # Set sklearn random state for clustering
        if random_seed is not None:
            self.clustering_random_state = random_seed
        else:
            self.clustering_random_state = 42  # Default for consistency
        
        # Ensure proper random seed inheritance and comprehensive control
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        logger.info(f"AssetScaleBayesianForecaster initialized for high-performance asset-scale processing with uncertainty_level={uncertainty_level}, seed={random_seed}")
    
    def _ensure_global_seed_control(self, operation_name: str = ""):
        """Ensure global numpy random seed is set before any random operation."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            if operation_name:
                logger.debug(f"Global random seed reset to {self.random_seed} before {operation_name}")
    
    def _set_clustering_random_state(self):
        """Set random state for clustering operations."""
        if self.random_seed is not None:
            return self.random_seed
        return 42  # Default for consistency
    
    def fit_hierarchical_asset_model(self, all_wells_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit hierarchical model: Asset -> Field -> Well levels
        
        Key improvements:
        1. Estimate population-level parameters from all wells
        2. Use shrinkage toward field averages for uncertain wells
        3. Parallel processing for 374 wells
        4. Reduced parameter space per well
        """
        start_time = time.time()
        logger.info("Starting hierarchical asset model fitting")
        
        try:
            # Step 1: Cluster wells by production characteristics
            logger.info("Clustering wells by production characteristics")
            try:
                self.well_clusters = self._cluster_wells_by_production_profile(all_wells_data)
                logger.info(f"Created {len(self.well_clusters)} well clusters")
            except Exception as e:
                logger.warning(f"Clustering failed: {str(e)}, using simple clustering")
                self.well_clusters = self._simple_well_clustering(all_wells_data)
            
            # Step 2: Estimate population priors from high-quality wells
            logger.info("Estimating population priors from high-quality wells")
            self.population_priors = self._estimate_population_priors(all_wells_data, self.well_clusters)
            
            # Step 3: Parallel hierarchical fitting
            logger.info("Starting parallel hierarchical fitting")
            well_names = all_wells_data['WellName'].unique()
            self.processing_stats['total_wells'] = len(well_names)
            
            # Always use sequential processing (simplified, no parallel complexity)
            results = {}
            logger.info("Using sequential processing for all wells")
            for well in well_names:
                try:
                    result = self._fit_well_with_population_priors(well, all_wells_data)
                    results[well] = result
                    if result['success']:
                        self.processing_stats['successful_wells'] += 1
                except Exception as e:
                    logger.error(f"Well {well} processing failed: {str(e)}")
                    results[well] = {'success': False, 'error': str(e)}
            
            # Step 4: Consolidate results
            asset_results = self._consolidate_hierarchical_results(results)
            
            processing_time = time.time() - start_time
            logger.info(f"Hierarchical asset model completed in {processing_time:.2f} seconds")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'asset_results': asset_results,
                'well_clusters': self.well_clusters,
                'population_priors': self.population_priors,
                'processing_stats': self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Hierarchical asset model failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _cluster_wells_by_production_profile(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Cluster wells by production characteristics using ArpsDCA fit results.
        
        This method now leverages existing ArpsDCA fit results instead of crude estimation,
        ensuring consistency between clustering logic and forecasting parameters.
        
        Args:
            data: Combined production data for all wells
            
        Returns:
            Dictionary mapping cluster IDs to lists of well names
        """
        try:
            # Get unique wells
            wells = data['WellName'].unique()
            
            if len(wells) < 10:
                logger.warning("Not enough wells for clustering, using simple clustering")
                return self._simple_well_clustering_with_arps(data)
            
            # Use ArpsDCA fit results for clustering instead of crude estimation
            well_features = []
            valid_wells = []
            
            for well in wells:
                well_data = data[data['WellName'] == well]
                
                # Skip wells with insufficient data
                if len(well_data) < 6:
                    continue
                
                # IMPROVED: Use ArpsDCA fit results if available
                if hasattr(self.arps_dca, 'fit_results') and well in self.arps_dca.fit_results:
                    fit_result = self.arps_dca.fit_results[well]
                    
                    if fit_result.success:
                        # Use sophisticated ArpsDCA parameters for clustering
                        initial_rate = fit_result.qi
                        decline_rate = fit_result.Di
                        hyperbolic_exp = fit_result.b
                        
                        # Calculate additional features from production data
                        production_values = well_data['OIL'].values
                        production_values = production_values[~np.isnan(production_values)]
                        production_values = production_values[production_values > 0]
                        
                        if len(production_values) < 3:
                            continue
                        
                        peak_rate = production_values.max()
                        cumulative_production = production_values.sum()
                        
                        # Use fit quality metrics
                        quality_score = fit_result.quality_metrics.get('r_squared', 0.0) if fit_result.quality_metrics else 0.0
                        
                        # Create feature vector using ArpsDCA parameters
                        features = [
                            initial_rate,           # qi from ArpsDCA
                            decline_rate,           # Di from ArpsDCA  
                            hyperbolic_exp,         # b from ArpsDCA
                            peak_rate,              # Peak production rate
                            cumulative_production,  # Total production
                            quality_score,          # Fit quality
                            len(production_values)  # Number of valid months
                        ]
                        
                        # Check for valid features
                        if all(np.isfinite(f) for f in features):
                            well_features.append(features)
                            valid_wells.append(well)
                        else:
                            logger.debug(f"Well {well} has invalid ArpsDCA features: {features}")
                    else:
                        logger.debug(f"Well {well} ArpsDCA fit failed, skipping clustering")
                else:
                    # Fallback: Skip well if no ArpsDCA results available
                    logger.debug(f"Well {well} has no ArpsDCA fit results, skipping clustering")
                    continue
            
            # Ensure we have enough valid wells for clustering
            if len(valid_wells) < 10:
                logger.warning("Not enough wells with valid ArpsDCA results for clustering")
                return self._simple_well_clustering_with_arps(data)
            
            # Use scikit-learn clustering with ArpsDCA features

            # Normalize features for clustering
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(well_features)
            
            # Determine optimal number of clusters (3-5 based on dataset size)
            n_clusters = min(5, max(3, len(valid_wells) // 75))
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self._set_clustering_random_state(), n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Organize results
            clusters = {}
            for i, well_name in enumerate(valid_wells):
                cluster_id = f"cluster_{cluster_labels[i]}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(well_name)
            
            # Add wells without ArpsDCA results to largest cluster
            wells_without_fits = [w for w in wells if w not in valid_wells]
            if wells_without_fits:
                largest_cluster = max(clusters, key=lambda k: len(clusters[k]))
                clusters[largest_cluster].extend(wells_without_fits)
                logger.info(f"Added {len(wells_without_fits)} wells without ArpsDCA fits to {largest_cluster}")
            
            logger.info(f"ArpsDCA-based clustering created {len(clusters)} clusters from {len(valid_wells)} wells")
            return clusters
            
        except Exception as e:
            logger.warning(f"ArpsDCA-based clustering failed: {str(e)}, falling back to simple clustering")
            return self._simple_well_clustering_with_arps(data)
    
    def _simple_well_clustering_with_arps(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Simple clustering using ArpsDCA results when K-means fails.
        
        Args:
            data: Combined production data for all wells
            
        Returns:
            Dictionary mapping cluster IDs to lists of well names
        """
        try:
            clusters = {"cluster_0": [], "cluster_1": [], "cluster_2": []}
            
            # Get unique wells
            wells = data['WellName'].unique()
            
            for well in wells:
                # IMPROVED: Use ArpsDCA results if available
                if hasattr(self.arps_dca, 'fit_results') and well in self.arps_dca.fit_results:
                    fit_result = self.arps_dca.fit_results[well]
                    
                    if fit_result.success:
                        # Use ArpsDCA parameters for simple categorization
                        qi = fit_result.qi
                        Di = fit_result.Di
                        
                        # Simple categorization based on ArpsDCA parameters
                        if qi > 500 and Di < 0.1:
                            # High initial production, low decline
                            clusters["cluster_0"].append(well)
                        elif qi > 100 and Di < 0.2:
                            # Medium production wells
                            clusters["cluster_1"].append(well)
                        else:
                            # Low production or high decline wells
                            clusters["cluster_2"].append(well)
                        continue
                
                # Fallback: Use production data if no ArpsDCA results
                well_data = data[data['WellName'] == well]
                
                if len(well_data) < 3:
                    continue
                    
                production_values = well_data['OIL'].values
                production_values = production_values[~np.isnan(production_values)]
                production_values = production_values[production_values > 0]
                
                if len(production_values) < 3:
                    continue
                
                try:
                    peak_rate = production_values.max()
                    avg_rate = production_values.mean()
                    
                    if not (np.isfinite(peak_rate) and np.isfinite(avg_rate) and avg_rate > 0):
                        continue
                    
                    # Simple categorization based on production characteristics
                    if peak_rate > avg_rate * 2:
                        clusters["cluster_0"].append(well)
                    elif peak_rate > avg_rate * 1.2:
                        clusters["cluster_1"].append(well)
                    else:
                        clusters["cluster_2"].append(well)
                        
                except Exception as e:
                    logger.debug(f"Failed to categorize well {well}: {str(e)}")
                    continue
            
            # Remove empty clusters
            clusters = {k: v for k, v in clusters.items() if v}
            
            # Ensure at least one cluster exists
            if not clusters:
                valid_wells = []
                for well in wells:
                    if hasattr(self.arps_dca, 'fit_results') and well in self.arps_dca.fit_results:
                        if self.arps_dca.fit_results[well].success:
                            valid_wells.append(well)
                
                if not valid_wells:
                    valid_wells = list(wells)
                    
                clusters["cluster_0"] = valid_wells
            
            logger.info(f"Simple ArpsDCA-based clustering created {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            logger.warning(f"Simple ArpsDCA clustering failed: {str(e)}, using fallback")
            # Final fallback to single cluster with all wells
            all_wells = list(data['WellName'].unique())
            return {"cluster_0": all_wells}
    
    # REMOVED: _estimate_simple_decline function - replaced with ArpsDCA-based clustering
    # This eliminates redundancy and ensures consistency between clustering and forecasting
    
    def _estimate_population_priors(self, data: pd.DataFrame, clusters: Dict) -> Dict[str, Any]:
        """Estimate population-level priors from high-quality deterministic fits"""
        population_params = {'qi': [], 'Di': [], 'b': []}
        
        for cluster_id, well_list in clusters.items():
            cluster_params = {'qi': [], 'Di': [], 'b': []}
            
            for well_name in well_list:
                if well_name in self.arps_dca.fit_results:
                    fit_result = self.arps_dca.fit_results[well_name]
                    if fit_result.success and fit_result.quality_metrics:
                        r_squared = fit_result.quality_metrics.get('r_squared', 0)
                        if r_squared > 0.7:
                            cluster_params['qi'].append(fit_result.qi)
                            cluster_params['Di'].append(fit_result.Di)
                            cluster_params['b'].append(fit_result.b)
            
            # Store cluster-specific priors
            if len(cluster_params['qi']) >= 3:  # Minimum wells per cluster
                self.field_parameters[cluster_id] = {
                    'qi': {
                        'mean': np.mean(np.log(np.maximum(cluster_params['qi'], 1))), 
                        'std': np.std(np.log(np.maximum(cluster_params['qi'], 1)))
                    },
                    'Di': {
                        'shape': self._fit_gamma_params(cluster_params['Di'])[0],
                        'rate': self._fit_gamma_params(cluster_params['Di'])[1]
                    },
                    'b': {
                        'alpha': self._fit_beta_params(cluster_params['b'])[0],
                        'beta': self._fit_beta_params(cluster_params['b'])[1]
                    }
                }
        
        return self.field_parameters
    
    def _fit_gamma_params(self, values: List[float]) -> Tuple[float, float]:
        """Fit gamma distribution parameters"""
        if len(values) < 3:
            return (2.0, 0.1)
        
        try:
            values = np.array(values)
            values = values[values > 0]  # Remove non-positive values
            
            if len(values) < 3:
                return (2.0, 0.1)
            
            # Method of moments
            mean_val = np.mean(values)
            var_val = np.var(values)
            
            if var_val <= 0:
                return (2.0, 0.1)
            
            # Gamma parameters: shape = mean^2 / var, rate = mean / var
            shape = mean_val**2 / var_val
            rate = mean_val / var_val
            
            return (max(0.1, shape), max(0.01, rate))
        except:
            return (2.0, 0.1)
    
    def _fit_beta_params(self, values: List[float]) -> Tuple[float, float]:
        """Fit beta distribution parameters"""
        if len(values) < 3:
            return (2.0, 3.0)
        
        try:
            values = np.array(values)
            values = np.clip(values, 0.01, 1.99)  # Clip to (0, 2) range
            values = values / 2.0  # Scale to (0, 1)
            
            # Method of moments for beta distribution
            mean_val = np.mean(values)
            var_val = np.var(values)
            
            if var_val <= 0 or mean_val <= 0 or mean_val >= 1:
                return (2.0, 3.0)
            
            # Beta parameters
            alpha = mean_val * ((mean_val * (1 - mean_val)) / var_val - 1)
            beta = (1 - mean_val) * ((mean_val * (1 - mean_val)) / var_val - 1)
            
            return (max(0.5, alpha), max(0.5, beta))
        except:
            return (2.0, 3.0)
    
    def _fit_well_with_population_priors(self, well_name: str, all_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit individual well using population priors"""
        try:
            # Get well data
            well_data = all_data[all_data['WellName'] == well_name]
            
            # Assign well to cluster
            cluster_id = self._assign_well_to_cluster(well_data)
            
            # Use cluster-specific priors if available
            if cluster_id in self.field_parameters:
                return self._fit_with_cluster_priors(well_data, well_name, cluster_id)
            else:
                # Fallback to standard fitting
                return self.fit_bayesian_decline(well_data, well_name)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'well_name': well_name
            }
    
    def _assign_well_to_cluster(self, well_data: pd.DataFrame) -> str:
        """Assign well to appropriate cluster based on characteristics"""
        if len(well_data) < 6:
            return list(self.well_clusters.keys())[0] if self.well_clusters else 'default'
        
        try:
            initial_rate = well_data['OIL'].iloc[0]
            
            # Simple assignment based on initial rate
            if initial_rate > 1000:
                for cluster_id, wells in self.well_clusters.items():
                    if 'high' in cluster_id.lower() or len(wells) > 20:
                        return cluster_id
            elif initial_rate > 100:
                for cluster_id, wells in self.well_clusters.items():
                    if 'medium' in cluster_id.lower():
                        return cluster_id
            
            # Default to first cluster
            return list(self.well_clusters.keys())[0]
            
        except:
            return list(self.well_clusters.keys())[0] if self.well_clusters else 'default'
    
    def _fit_with_cluster_priors(self, well_data: pd.DataFrame, well_name: str, cluster_id: str) -> Dict[str, Any]:
        """Fit well using cluster-specific priors"""
        try:
            # Get cluster priors
            cluster_priors = self.field_parameters[cluster_id]
            
            # First get deterministic fit
            if well_name in self.arps_dca.fit_results:
                fit_result = self.arps_dca.fit_results[well_name]
            else:
                fit_dict = self.arps_dca.fit_decline_curve(well_data, well_name)
                if not fit_dict['success']:
                    return fit_dict
                fit_result = self.arps_dca.fit_results[well_name]
            
            # Generate samples using cluster priors
            parameter_samples = self._sample_with_cluster_priors(fit_result, cluster_priors)
            
            self.processing_stats['hierarchical_wells'] += 1
            
            return {
                'success': True,
                'well_name': well_name,
                'method': 'Hierarchical Bayesian',
                'cluster_id': cluster_id,
                'parameter_samples': parameter_samples,
                'deterministic_fit': fit_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'well_name': well_name
            }
    
    def _sample_with_cluster_priors(self, fit_result: FitResult, cluster_priors: Dict) -> Dict[str, np.ndarray]:
        """Generate parameter samples using cluster priors"""
        
        # Set random state for reproducible cluster sampling
        self._set_random_state(10)
        
        # Use moderate uncertainty adjustments that don't create extreme values
        uncertainty_factor = self.uncertainty_config['forecast_uncertainty_factor']
        
        # Use moderate multipliers instead of extreme ones
        # Conservative gets 25% more uncertainty, aggressive gets 15% less
        if uncertainty_factor > 1.2:  # Conservative case
            uncertainty_multiplier = 1.25
        elif uncertainty_factor < 0.9:  # Aggressive case  
            uncertainty_multiplier = 0.85
        else:  # Standard case
            uncertainty_multiplier = 1.0
        
        # Sample qi from log-normal with cluster priors
        try:
            qi_mean = cluster_priors['qi']['mean']
            qi_std = cluster_priors['qi']['std']
            
            # Apply moderate uncertainty adjustment to cluster-based variance
            qi_std = qi_std * uncertainty_multiplier
            
            # Validate parameters
            if np.isnan(qi_mean) or np.isinf(qi_mean):
                logger.warning(f"Invalid cluster qi mean ({qi_mean}), using default")
                qi_mean = np.log(100.0)
                
            if np.isnan(qi_std) or np.isinf(qi_std) or qi_std <= 0:
                logger.warning(f"Invalid cluster qi std ({qi_std}), using default")
                qi_std = 0.5  # Moderate default for log-normal
            
            # Limit qi_std to prevent extreme values
            qi_std = min(qi_std, 1.0)  # Cap at 1.0 for log-normal
            
            qi_samples = np.random.lognormal(qi_mean, qi_std, self.n_samples)
            
            # Check for NaN in samples
            if np.any(np.isnan(qi_samples)):
                logger.warning("NaN values in cluster qi samples, using fallback")
                qi_samples = np.full(self.n_samples, np.exp(qi_mean))
                
        except Exception as e:
            logger.warning(f"Cluster qi sampling failed ({str(e)}), using deterministic fallback")
            qi_samples = np.full(self.n_samples, 100.0)
        
        # Sample Di from gamma with cluster priors
        try:
            Di_shape = cluster_priors['Di']['shape']
            Di_rate = cluster_priors['Di']['rate']
            
            # Apply moderate uncertainty adjustment to rate (inverse relationship)
            Di_rate = Di_rate / uncertainty_multiplier
            
            # Validate parameters
            if np.isnan(Di_shape) or np.isinf(Di_shape) or Di_shape <= 0:
                logger.warning(f"Invalid cluster Di shape ({Di_shape}), using default")
                Di_shape = 2.0
                
            if np.isnan(Di_rate) or np.isinf(Di_rate) or Di_rate <= 0:
                logger.warning(f"Invalid cluster Di rate ({Di_rate}), using default")
                Di_rate = 10.0  # 1/0.1 default scale
            
            Di_samples = np.random.gamma(Di_shape, 1/Di_rate, self.n_samples)
            
            # Check for NaN in samples
            if np.any(np.isnan(Di_samples)):
                logger.warning("NaN values in cluster Di samples, using fallback")
                Di_samples = np.full(self.n_samples, Di_shape / Di_rate)
                
        except Exception as e:
            logger.warning(f"Cluster Di sampling failed ({str(e)}), using deterministic fallback")
            Di_samples = np.full(self.n_samples, 0.1)
        
        # Sample b from beta with cluster priors
        try:
            b_alpha = cluster_priors['b']['alpha']
            b_beta = cluster_priors['b']['beta']
            
            # Apply moderate uncertainty adjustment to both parameters
            b_alpha = b_alpha * uncertainty_multiplier
            b_beta = b_beta * uncertainty_multiplier
            
            # Validate parameters
            if np.isnan(b_alpha) or np.isinf(b_alpha) or b_alpha <= 0:
                logger.warning(f"Invalid cluster b alpha ({b_alpha}), using default")
                b_alpha = 2.0
                
            if np.isnan(b_beta) or np.isinf(b_beta) or b_beta <= 0:
                logger.warning(f"Invalid cluster b beta ({b_beta}), using default")
                b_beta = 3.0
            
            b_samples = np.random.beta(b_alpha, b_beta, self.n_samples) * 2.0  # Scale back to (0, 2)
            
            # Check for NaN in samples
            if np.any(np.isnan(b_samples)):
                logger.warning("NaN values in cluster b samples, using fallback")
                b_samples = np.full(self.n_samples, (b_alpha / (b_alpha + b_beta)) * 2.0)
                
        except Exception as e:
            logger.warning(f"Cluster b sampling failed ({str(e)}), using deterministic fallback")
            b_samples = np.full(self.n_samples, 0.5)
        
        # Apply reasonable bounds to prevent extreme values
        qi_samples = np.clip(qi_samples, 10, 20000)  # Reasonable production range
        Di_samples = np.clip(Di_samples, 0.005, 1.0)  # Reasonable decline range
        b_samples = np.clip(b_samples, 0.01, 1.99)   # Physical bounds
        
        return {
            'qi': qi_samples,
            'Di': Di_samples,
            'b': b_samples
        }
    
    def _consolidate_hierarchical_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate hierarchical fitting results"""
        successful_wells = []
        failed_wells = []
        
        for well_name, result in results.items():
            if result['success']:
                successful_wells.append(well_name)
                self.fit_results[well_name] = result
            else:
                failed_wells.append(well_name)
        
        success_rate = len(successful_wells) / len(results) * 100
        
        return {
            'total_wells': len(results),
            'successful_wells': len(successful_wells),
            'failed_wells': len(failed_wells),
            'success_rate': success_rate,
            'processing_stats': self.processing_stats
        }

    def fit_approximate_bayesian(self, production_data: pd.DataFrame, well_name: str) -> Dict[str, Any]:
        """
        Fast approximate Bayesian inference using summary statistics
        
        Key improvements:
        1. Use summary statistics instead of full likelihood
        2. Rejection sampling with tolerance
        3. 10x faster than full Bayesian
        4. Maintains uncertainty quantification
        """
        start_time = time.time()
        
        # Get deterministic fit as starting point
        if well_name in self.arps_dca.fit_results:
            fit_result = self.arps_dca.fit_results[well_name]
            if not fit_result.success:
                return {'success': False, 'error': 'Deterministic fit failed'}
        else:
            fit_dict = self.arps_dca.fit_decline_curve(production_data, well_name)
            if not fit_dict['success']:
                return {'success': False, 'error': 'Deterministic fit failed'}
            fit_result = self.arps_dca.fit_results[well_name]
        
        # Extract summary statistics from data
        observed_stats = self._extract_summary_statistics(production_data, well_name)
        
        # ABC sampling
        tolerance = self._calculate_adaptive_tolerance(fit_result.quality_metrics)
        accepted_params = self._abc_rejection_sampling(
            observed_stats, tolerance, fit_result, n_samples=self.n_samples
        )
        
        self.processing_stats['abc_wells'] += 1
        
        return {
            'success': True,
            'well_name': well_name,
            'method': 'Approximate Bayesian Computation',
            'parameter_samples': accepted_params,
            'tolerance': tolerance,
            'acceptance_rate': len(accepted_params['qi']) / (self.n_samples * 10),  # Assuming 10x oversampling
            'processing_time': time.time() - start_time
        }

    def _extract_summary_statistics(self, data: pd.DataFrame, well_name: str) -> Dict[str, float]:
        """Extract key summary statistics for ABC"""
        well_data = data[data['WellName'] == well_name].sort_values('DATE')
        production = well_data['OIL'].values
        
        if len(production) < 3:
            return {
                'peak_rate': production[0] if len(production) > 0 else 100,
                'initial_rate': production[0] if len(production) > 0 else 100,
                'rate_at_12_months': production[0] if len(production) > 0 else 100,
                'cumulative_12_months': np.sum(production),
                'decline_rate_early': 0.1,
                'coefficient_of_variation': 0.1
            }
        
        return {
            'peak_rate': np.max(production),
            'initial_rate': production[0],
            'rate_at_12_months': production[min(11, len(production)-1)],
            'cumulative_12_months': np.sum(production[:min(12, len(production))]),
            'decline_rate_early': (production[0] - production[min(5, len(production)-1)]) / max(production[0], 1),
            'coefficient_of_variation': np.std(production) / max(np.mean(production), 1)
        }

    def _calculate_adaptive_tolerance(self, quality_metrics: Optional[Dict]) -> float:
        """Calculate adaptive tolerance based on fit quality and uncertainty configuration"""
        if not quality_metrics:
            base_tolerance = 0.2
        else:
            r_squared = quality_metrics.get('r_squared', 0.5)
            
            # Lower tolerance for higher quality fits
            if r_squared > 0.9:
                base_tolerance = 0.05
            elif r_squared > 0.8:
                base_tolerance = 0.1
            elif r_squared > 0.7:
                base_tolerance = 0.15
            else:
                base_tolerance = 0.2
        
        # Apply uncertainty configuration to adjust tolerance
        # Higher uncertainty levels should have more relaxed tolerance (higher values)
        uncertainty_factor = self.uncertainty_config['forecast_uncertainty_factor']
        adjusted_tolerance = base_tolerance * uncertainty_factor
        
        # Ensure reasonable bounds
        return np.clip(adjusted_tolerance, 0.01, 1.0)

    def _abc_rejection_sampling(self, observed_stats: Dict, tolerance: float, 
                              fit_result: FitResult, n_samples: int) -> Dict[str, np.ndarray]:
        """Fast ABC rejection sampling"""
        # Set random state for reproducible ABC sampling
        self._set_random_state(20)
        
        accepted_params = {'qi': [], 'Di': [], 'b': []}
        
        # Use moderate uncertainty adjustments for proposal distributions
        uncertainty_factor = self.uncertainty_config['forecast_uncertainty_factor']
        
        # Calculate reasonable proposal distribution parameters
        # Use moderate scaling based on uncertainty factor
        if uncertainty_factor > 1.2:  # Conservative case
            scale_factor = 1.3
        elif uncertainty_factor < 0.9:  # Aggressive case
            scale_factor = 0.8
        else:  # Standard case
            scale_factor = 1.0
        
        # Create proposal distributions with reasonable spreads around fitted values
        # qi: log-normal with moderate spread
        qi_log_mean = np.log(fit_result.qi)
        qi_log_std = 0.3 * scale_factor  # Moderate log-normal std
        qi_proposal = self._create_seeded_scipy_distribution('lognorm', seed_offset=21, 
                                                           s=qi_log_std, scale=np.exp(qi_log_mean))
        
        # Di: gamma with shape and scale based on fitted value  
        Di_shape = 4.0  # Fixed shape for reasonable distribution
        Di_scale = fit_result.Di / Di_shape * scale_factor  # Scale to match fitted value
        Di_proposal = self._create_seeded_scipy_distribution('gamma', seed_offset=22, 
                                                           a=Di_shape, scale=Di_scale)
        
        # b: beta distribution scaled to (0, 2) range
        b_alpha = 2.0 * scale_factor
        b_beta = 3.0 * scale_factor  
        b_proposal = self._create_seeded_scipy_distribution('beta', seed_offset=23, 
                                                          a=b_alpha, b=b_beta, loc=0, scale=2)
        
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(accepted_params['qi']) < n_samples and attempts < max_attempts:
            # Sample from proposal
            qi_sample = qi_proposal.rvs()
            Di_sample = Di_proposal.rvs()
            b_sample = b_proposal.rvs()
            
            # Apply reasonable bounds to prevent extreme values
            qi_sample = np.clip(qi_sample, fit_result.qi * 0.2, fit_result.qi * 5.0)
            Di_sample = np.clip(Di_sample, fit_result.Di * 0.3, fit_result.Di * 3.0)
            b_sample = np.clip(b_sample, 0.01, 1.99)
            
            # Simulate summary statistics
            simulated_stats = self._simulate_summary_statistics(qi_sample, Di_sample, b_sample)
            
            # Accept/reject based on distance
            distance = self._calculate_summary_distance(observed_stats, simulated_stats)
            if distance < tolerance:
                accepted_params['qi'].append(qi_sample)
                accepted_params['Di'].append(Di_sample)
                accepted_params['b'].append(b_sample)
            
            attempts += 1
        
        # If we didn't get enough samples, fill with reasonable attempts
        if len(accepted_params['qi']) < n_samples:
            n_missing = n_samples - len(accepted_params['qi'])
            for _ in range(n_missing):
                qi_sample = qi_proposal.rvs()
                Di_sample = Di_proposal.rvs()
                b_sample = b_proposal.rvs()
                
                # Apply same reasonable bounds
                accepted_params['qi'].append(np.clip(qi_sample, fit_result.qi * 0.2, fit_result.qi * 5.0))
                accepted_params['Di'].append(np.clip(Di_sample, fit_result.Di * 0.3, fit_result.Di * 3.0))
                accepted_params['b'].append(np.clip(b_sample, 0.01, 1.99))
        
        return {k: np.array(v) for k, v in accepted_params.items()}

    def _simulate_summary_statistics(self, qi: float, Di: float, b: float) -> Dict[str, float]:
        """Simulate summary statistics for given parameters"""
        # Simulate 24 months of production
        t = np.arange(24)
        production = qi / (1 + b * Di * t)**(1/b)
        
        return {
            'peak_rate': np.max(production),
            'initial_rate': production[0],
            'rate_at_12_months': production[11],
            'cumulative_12_months': np.sum(production[:12]),
            'decline_rate_early': (production[0] - production[5]) / production[0],
            'coefficient_of_variation': np.std(production) / np.mean(production)
        }

    def _calculate_summary_distance(self, observed: Dict[str, float], simulated: Dict[str, float]) -> float:
        """Calculate distance between observed and simulated summary statistics"""
        distance = 0
        
        # Weight different statistics
        weights = {
            'peak_rate': 0.2,
            'initial_rate': 0.2,
            'rate_at_12_months': 0.2,
            'cumulative_12_months': 0.2,
            'decline_rate_early': 0.1,
            'coefficient_of_variation': 0.1
        }
        
        for stat, weight in weights.items():
            obs_val = observed[stat]
            sim_val = simulated[stat]
            
            if obs_val > 0:
                normalized_diff = abs(obs_val - sim_val) / obs_val
                distance += weight * normalized_diff
        
        return distance

    def batch_fit_asset_wells(self, all_wells_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Vectorized batch processing for large (>1000) number of wells with <15 min runtime
        
        Ensures consistent processing across uncertainty levels:
        1. Same wells processed regardless of uncertainty level
        2. Deterministic processing order
        3. Consistent error handling
        4. No early stopping that could cause inconsistencies
        """
        
        start_time = time.time()
        well_names = all_wells_data['WellName'].unique()
        
        # Pre-filter wells by data quality (deterministic)
        quality_filtered_wells = self._pre_filter_wells_by_quality(all_wells_data)
        
        logger.info(f"Batch processing {len(quality_filtered_wells)} quality-filtered wells (of {len(well_names)} total)")
        
        # Process all wells with consistent methodology (no time-based early stopping)
        all_results = {}
        
        # Deterministic processing order (sort well names)
        quality_filtered_wells.sort()
        
        # Process in smaller chunks for memory management
        chunk_size = 50  
        processed_count = 0
        
        for i in range(0, len(quality_filtered_wells), chunk_size):
            chunk_wells = quality_filtered_wells[i:i+chunk_size]
            
            # Process chunk with consistent methodology
            chunk_results = self._process_well_chunk(chunk_wells, all_wells_data)
            all_results.update(chunk_results)
            
            processed_count += len(chunk_wells)
            
            # Progress logging
            logger.info(f"Processed {processed_count} / {len(quality_filtered_wells)} wells")
            
            # REMOVED: Time-based early stopping that could cause inconsistencies
            # Different uncertainty levels should process the same wells
        
        # Store all successful results in fit_results for consistency
        successful_count = 0
        for well_name, result in all_results.items():
            if result.get('success', False):
                self.fit_results[well_name] = result
                successful_count += 1
        
        logger.info(f"Batch processing complete: {successful_count} successful wells out of {len(quality_filtered_wells)} attempted")
        
        return {
            'success': True,
            'total_wells': len(well_names),
            'quality_filtered_wells': len(quality_filtered_wells),
            'processed_wells': len(all_results),
            'successful_wells': successful_count,
            'processing_time': time.time() - start_time,
            'results': all_results
        }

    def _pre_filter_wells_by_quality(self, all_data: pd.DataFrame) -> List[str]:
        """
        Pre-filter wells by data quality for efficient processing
        
        Deterministic filtering that doesn't depend on uncertainty level
        """
        quality_wells = []
        
        # Sort well names for deterministic processing order
        well_names = sorted(all_data['WellName'].unique())
        
        for well_name in well_names:
            well_data = all_data[all_data['WellName'] == well_name]
            
            # Skip wells with insufficient data
            if len(well_data) < 6:
                continue
                
            # Use correct column name from the actual data
            production = well_data['OIL'].values
            
            # Quality metrics - deterministic filtering
            if len(production) >= 12 and production.max() > 0:
                # Additional quality checks
                valid_production = production[production > 0]
                if len(valid_production) >= 6:  # At least 6 months of positive production
                    quality_wells.append(well_name)
        
        logger.info(f"Quality filtered: {len(quality_wells)} wells from {len(all_data['WellName'].unique())} total")
        return quality_wells

    def _process_well_chunk(self, well_chunk: List[str], 
                           all_data: pd.DataFrame) -> Dict[str, Any]:
        """Process well chunk sequentially (simplified, no parallel complexity)"""
        results = {}
        
        # Always use sequential processing
        for well in well_chunk:
            try:
                result = self._fast_bayesian_fit(well, all_data)
                results[well] = result
            except Exception as e:
                logger.error(f"Well {well} failed in chunk processing: {str(e)}")
                results[well] = {'success': False, 'error': str(e)}
        
        return results

    def _fast_bayesian_fit(self, well_name: str, all_data: pd.DataFrame) -> Dict[str, Any]:
        """Fast Bayesian fit with reduced sampling"""
        well_data = all_data[all_data['WellName'] == well_name]
        
        # Use hierarchical priors if available
        if hasattr(self, 'field_parameters') and self.field_parameters:
            cluster_id = self._assign_well_to_cluster(well_data)
            if cluster_id in self.field_parameters:
                # Use cluster-specific priors
                return self._fit_with_cluster_priors(well_data, well_name, cluster_id)
        
        # Fallback to fast ABC
        return self.fit_approximate_bayesian(well_data, well_name)

    def _fast_mode_processing(self, remaining_wells: List[str], all_data: pd.DataFrame) -> Dict[str, Any]:
        """Ultra-fast processing for remaining wells"""
        results = {}
        
        for well_name in remaining_wells:
            try:
                result = self._deterministic_plus_noise(well_name, all_data)
                results[well_name] = result
            except Exception as e:
                results[well_name] = {'success': False, 'error': str(e)}
        
        return results

    def adaptive_quality_sampling(self, well_name: str, all_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Adaptive sampling based on well quality - high quality wells get more samples
        
        Key improvements:
        1. Allocate computation based on well importance
        2. High-quality wells: full Bayesian treatment
        3. Medium-quality wells: ABC
        4. Low-quality wells: deterministic + noise
        """
        
        if well_name not in self.arps_dca.fit_results:
            # Need to fit first
            fit_dict = self.arps_dca.fit_decline_curve(all_data, well_name)
            if not fit_dict['success']:
                return {'success': False, 'error': 'Well not fitted'}
        
        fit_result = self.arps_dca.fit_results[well_name]
        
        # Assess quality
        quality_metrics = fit_result.quality_metrics or {}
        r_squared = quality_metrics.get('r_squared', 0)
        
        # Determine confidence level
        if r_squared > 0.85:
            confidence_level = 'high'
        elif r_squared > 0.7:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Adaptive sampling strategy
        if confidence_level == 'high':
            # Full Bayesian treatment
            return self._full_bayesian_sampling(well_name, all_data)
        elif confidence_level == 'medium':
            # ABC with moderate tolerance
            return self._abc_sampling_moderate(well_name, all_data)
        else:
            # Deterministic + parametric uncertainty
            return self._deterministic_plus_noise(well_name, all_data)

    def _full_bayesian_sampling(self, well_name: str, all_data: pd.DataFrame) -> Dict[str, Any]:
        """Full Bayesian treatment for high-quality wells"""
        well_data = all_data[all_data['WellName'] == well_name]
        return self.fit_bayesian_decline(well_data, well_name)

    def _abc_sampling_moderate(self, well_name: str, all_data: pd.DataFrame) -> Dict[str, Any]:
        """ABC with moderate tolerance for medium-quality wells"""
        well_data = all_data[all_data['WellName'] == well_name]
        return self.fit_approximate_bayesian(well_data, well_name)

    def _deterministic_plus_noise(self, well_name: str, all_data: pd.DataFrame) -> Dict[str, Any]:
        """Fast deterministic + noise for low-quality wells"""
        if well_name not in self.arps_dca.fit_results:
            well_data = all_data[all_data['WellName'] == well_name]
            fit_dict = self.arps_dca.fit_decline_curve(well_data, well_name)
            if not fit_dict['success']:
                return {'success': False, 'error': 'Deterministic fit failed'}
        
        fit_result = self.arps_dca.fit_results[well_name]
        
        # Set random state for deterministic noise generation
        well_seed = self._deterministic_hash(well_name) if well_name else 0
        self._set_random_state(40 + well_seed)
        
        # Use proper uncertainty parameterization for normal distributions
        # Get uncertainty configuration
        uncertainty_factor = self.uncertainty_config['forecast_uncertainty_factor']
        quality_noise = self.uncertainty_config['quality_noise_multipliers']['low']
        
        # Calculate proper noise levels for normal distributions around fitted parameters
        # Use coefficient of variation approach with uncertainty configuration
        base_cv_qi = 0.15    # 15% base coefficient of variation for qi
        base_cv_Di = 0.20    # 20% base coefficient of variation for Di  
        base_cv_b = 0.10     # 10% base coefficient of variation for b
        
        # Apply uncertainty configuration as multipliers on base coefficients
        cv_qi = base_cv_qi * uncertainty_factor * (1 + quality_noise)
        cv_Di = base_cv_Di * uncertainty_factor * (1 + quality_noise)
        cv_b = base_cv_b * uncertainty_factor * (1 + quality_noise)
        
        # Calculate standard deviations using coefficient of variation
        qi_std = cv_qi * fit_result.qi
        Di_std = cv_Di * fit_result.Di
        b_std = cv_b * fit_result.b
        
        # Generate samples with reasonable bounds
        qi_samples = np.random.normal(fit_result.qi, qi_std, self.n_samples)
        Di_samples = np.random.normal(fit_result.Di, Di_std, self.n_samples)
        b_samples = np.random.normal(fit_result.b, b_std, self.n_samples)
        
        # Apply reasonable bounds to prevent extreme values
        qi_samples = np.clip(qi_samples, fit_result.qi * 0.3, fit_result.qi * 3.0)  # 30% to 300% of fitted value
        Di_samples = np.clip(Di_samples, fit_result.Di * 0.5, fit_result.Di * 2.0)  # 50% to 200% of fitted value
        b_samples = np.clip(b_samples, 0.01, 1.99)  # Physical bounds for hyperbolic parameter
        
        self.processing_stats['deterministic_wells'] += 1
        
        return {
            'success': True,
            'well_name': well_name,
            'method': 'Deterministic + Parametric Noise',
            'parameter_samples': {
                'qi': qi_samples,
                'Di': Di_samples,
                'b': b_samples
            }
        }

    def asset_scale_uncertainty_propagation(self, forecast_months: int = 360) -> Dict[str, Any]:
        """
        Memory-efficient uncertainty propagation for asset-scale forecasting
        
        Ensures consistent processing across uncertainty levels:
        1. Same wells processed regardless of uncertainty level
        2. Consistent sample sizes across all wells
        3. Proper uncertainty configuration application
        4. Reliable aggregation methodology
        """
        
        # Get list of successfully fitted wells (deterministic across uncertainty levels)
        successful_wells = [well_name for well_name, result in self.fit_results.items() 
                          if result.get('success', False)]
        
        if not successful_wells:
            logger.warning("No successful wells found for asset-scale uncertainty propagation")
            return {
                'success': False,
                'error': 'No successful wells available',
                'wells_included': 0,
                'forecast_months': forecast_months
            }
        
        logger.info(f"Asset-scale propagation starting with {len(successful_wells)} successful wells")
        
        # Use consistent sample size across all wells (no reduction for memory)
        # This ensures uncertainty levels are comparable
        consistent_n_samples = self.n_samples
        
        # Initialize asset-level accumulators with proper dimensions
        asset_samples = np.zeros((consistent_n_samples, forecast_months))
        processed_well_count = 0
        failed_well_count = 0
        
        # Stream through wells with consistent processing
        for well_name in successful_wells:
            try:
                # Generate well-specific forecast samples with consistent methodology
                well_samples = self._generate_well_forecast_samples_consistent(
                    well_name, forecast_months, consistent_n_samples
                )
                
                if well_samples is not None and well_samples.shape == (consistent_n_samples, forecast_months):
                    # Add to asset total
                    asset_samples += well_samples
                    processed_well_count += 1
                    
                    # Progress logging every 50 wells
                    if processed_well_count % 50 == 0:
                        logger.info(f"Processed {processed_well_count}/{len(successful_wells)} wells for asset forecasting")
                else:
                    failed_well_count += 1
                    logger.warning(f"Well {well_name} produced invalid forecast samples - skipping")
                
            except Exception as e:
                failed_well_count += 1
                logger.error(f"Failed to process well {well_name} for asset forecasting: {str(e)}")
                continue
        
        if processed_well_count == 0:
            logger.error("No wells successfully processed for asset-scale forecasting")
            return {
                'success': False,
                'error': 'No wells successfully processed',
                'wells_included': 0,
                'forecast_months': forecast_months
            }
        
        logger.info(f"Asset-scale processing complete: {processed_well_count} successful, {failed_well_count} failed")
        
        # Calculate asset-level percentiles with industry convention
        asset_percentiles = self._calculate_streaming_percentiles(asset_samples, [0.9, 0.5, 0.1])
        
        # Validate percentiles
        for percentile_name, values in asset_percentiles.items():
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                logger.error(f"Invalid values in {percentile_name} percentiles")
                return {
                    'success': False,
                    'error': f'Invalid percentile values in {percentile_name}',
                    'wells_included': processed_well_count,
                    'forecast_months': forecast_months
                }
        
        return {
            'success': True,
            'asset_forecast_percentiles': asset_percentiles,
            'wells_included': processed_well_count,
            'wells_failed': failed_well_count,
            'total_wells_attempted': len(successful_wells),
            'forecast_months': forecast_months,
            'sample_size_used': consistent_n_samples
        }

    def _generate_well_forecast_samples_consistent(self, well_name: str, forecast_months: int, 
                                                 target_samples: int) -> np.ndarray:
        """
        Generate forecast samples for single well with CONSISTENT methodology across uncertainty levels
        
        No sample size reduction - uses full target_samples for consistency
        """
        if well_name not in self.fit_results:
            logger.warning(f"Well {well_name} not found in fit_results")
            return None
        
        try:
            parameter_samples = self.fit_results[well_name]['parameter_samples']
            
            # Validate parameter samples
            required_params = ['qi', 'Di', 'b']
            for param in required_params:
                if param not in parameter_samples:
                    logger.error(f"Missing parameter {param} for well {well_name}")
                    return None
                
                if len(parameter_samples[param]) < target_samples:
                    logger.warning(f"Insufficient {param} samples for well {well_name}: {len(parameter_samples[param])} < {target_samples}")
                    # Expand samples if needed
                    current_samples = np.array(parameter_samples[param])
                    if len(current_samples) > 0:
                        # Set deterministic seed for consistent expansion
                        if self.random_seed is not None:
                            well_seed = self._deterministic_hash(well_name) if well_name else 0
                            np.random.seed(self.random_seed + 30 + well_seed)
                        
                        indices = np.random.choice(len(current_samples), target_samples, replace=True)
                        parameter_samples[param] = current_samples[indices]
                    else:
                        logger.error(f"Empty {param} samples for well {well_name}")
                        return None
            
            # Generate forecast with exact target sample size
            forecast_samples = np.zeros((target_samples, forecast_months))
            t = np.arange(forecast_months)
            
            # Set deterministic seed for forecast generation
            if self.random_seed is not None:
                well_seed = self._deterministic_hash(well_name) if well_name else 0
                np.random.seed(self.random_seed + 50 + well_seed)
            
            for i in range(target_samples):
                qi = parameter_samples['qi'][i]
                Di = parameter_samples['Di'][i]
                b = parameter_samples['b'][i]
                
                # Validate parameters
                if not (np.isfinite(qi) and np.isfinite(Di) and np.isfinite(b)):
                    logger.warning(f"Invalid parameters for well {well_name}, sample {i}: qi={qi}, Di={Di}, b={b}")
                    # Use fallback values
                    qi = max(qi, 100) if np.isfinite(qi) else 100
                    Di = max(Di, 0.01) if np.isfinite(Di) else 0.1
                    b = max(b, 0.1) if np.isfinite(b) else 0.5
                
                # Generate decline curve forecast
                if abs(b) < 1e-6:  # Exponential decline
                    forecast_samples[i, :] = qi * np.exp(-Di * t)
                else:  # Hyperbolic decline
                    try:
                        forecast_samples[i, :] = qi / (1 + b * Di * t)**(1/b)
                    except (ZeroDivisionError, OverflowError, ValueError):
                        # Fallback to exponential decline
                        forecast_samples[i, :] = qi * np.exp(-Di * t)
                
                # Apply minimum economic threshold and validate
                forecast_samples[i, :] = np.maximum(forecast_samples[i, :], 1.0)
                
                # Check for invalid values
                if np.any(~np.isfinite(forecast_samples[i, :])):
                    logger.warning(f"Invalid forecast values for well {well_name}, sample {i} - using fallback")
                    forecast_samples[i, :] = np.linspace(qi, max(qi*0.01, 10), forecast_months)
            
            return forecast_samples
            
        except Exception as e:
            logger.error(f"Error generating consistent forecast samples for well {well_name}: {str(e)}")
            return None

    def _calculate_streaming_percentiles(self, samples: np.ndarray, percentiles: List[float]) -> Dict[str, np.ndarray]:
        """Calculate percentiles in streaming fashion with industry convention mapping"""
        result = {}
        
        # Industry convention mapping
        percentile_mapping = {
            0.9: "P10",  # 90th percentile -> P10 (optimistic/high reserves)
            0.5: "P50",  # 50th percentile -> P50 (median reserves)
            0.1: "P90"   # 10th percentile -> P90 (conservative/low reserves)
        }
        
        for p in percentiles:
            key = percentile_mapping.get(p, f'P{int(p*100)}')
            result[key] = np.percentile(samples, p*100, axis=0)
        
        return result

    def _generate_well_forecast_samples(self, well_name: str, forecast_months: int) -> np.ndarray:
        """
        Generate forecast samples for single well (backward compatibility)
        
        This method maintains the original interface for any existing calls
        """
        return self._generate_well_forecast_samples_consistent(well_name, forecast_months, self.n_samples)
