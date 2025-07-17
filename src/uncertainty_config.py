"""
Uncertainty Configuration System for Bayesian Forecasting

This module provides systematic uncertainty control without breaking existing functionality.
Supports multiple uncertainty levels for different business use cases.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class UncertaintyConfig:
    """
    Configuration system for controlling uncertainty levels in Bayesian forecasting.
    
    Provides predefined uncertainty levels and custom configuration support.
    """
    
    # Predefined uncertainty configurations
    UNCERTAINTY_PRESETS = {
        'standard': {
            'description': 'Standard uncertainty for typical planning',
            'prior_multipliers': {
                'qi_sigma': 1.2,      # Standard lognormal spread
                'Di_shape': 3.0,      # Focused gamma distribution  
                'Di_scale': 0.02,
                'b_alpha': 1.5,       # Moderately focused beta
                'b_beta': 4.0,
                'noise_precision_a': 2.0,  # Higher precision (lower noise)
                'noise_precision_scale': 0.1
            },
            'quality_noise_multipliers': {
                'high': 0.0,     # No additional noise for high quality
                'medium': 0.05,  # 5% noise for medium quality
                'low': 0.10      # 10% noise for low quality
            },
            'forecast_uncertainty_factor': 1.0  # Standard forecast uncertainty
        },
        
        'conservative': {
            'description': 'Higher uncertainty for conservative planning and risk assessment',
            'prior_multipliers': {
                'qi_sigma': 1.8,      # +50% uncertainty
                'Di_shape': 2.5,      # Wider spread
                'Di_scale': 0.025,
                'b_alpha': 1.2,       # More uncertain
                'b_beta': 3.0,
                'noise_precision_a': 1.5,  # Lower precision (higher noise)
                'noise_precision_scale': 0.15
            },
            'quality_noise_multipliers': {
                'high': 0.08,    # Even high quality gets uncertainty
                'medium': 0.15,  # 15% noise for medium quality
                'low': 0.25      # 25% noise for low quality
            },
            'forecast_uncertainty_factor': 1.5  # 50% more forecast uncertainty
        },
        
        'aggressive': {
            'description': 'Lower uncertainty for optimistic development scenarios',
            'prior_multipliers': {
                'qi_sigma': 1.0,      # Tighter uncertainty
                'Di_shape': 3.5,      # More focused
                'Di_scale': 0.018,
                'b_alpha': 1.8,       # More focused
                'b_beta': 5.0,
                'noise_precision_a': 2.5,  # Higher precision (lower noise)
                'noise_precision_scale': 0.08
            },
            'quality_noise_multipliers': {
                'high': 0.0,     # No additional noise
                'medium': 0.03,  # 3% noise for medium quality
                'low': 0.07      # 7% noise for low quality
            },
            'forecast_uncertainty_factor': 0.8  # 20% less forecast uncertainty
        },
        
        'high_uncertainty': {
            'description': 'Very high uncertainty for extreme risk assessment',
            'prior_multipliers': {
                'qi_sigma': 2.2,      # +83% uncertainty
                'Di_shape': 2.0,      # Very wide spread
                'Di_scale': 0.03,
                'b_alpha': 1.0,       # Near uniform
                'b_beta': 2.0,
                'noise_precision_a': 1.0,  # Low precision (high noise)
                'noise_precision_scale': 0.2
            },
            'quality_noise_multipliers': {
                'high': 0.12,    # 12% noise even for high quality
                'medium': 0.20,  # 20% noise for medium quality
                'low': 0.35      # 35% noise for low quality
            },
            'forecast_uncertainty_factor': 2.0  # Double forecast uncertainty
        }
    }
    
    @classmethod
    def get_config(cls, uncertainty_level: str = 'standard') -> Dict[str, Any]:
        """
        Get uncertainty configuration for specified level.
        
        Args:
            uncertainty_level: One of 'standard', 'conservative', 'aggressive', 'high_uncertainty'
            
        Returns:
            Dictionary containing uncertainty configuration
        """
        if uncertainty_level not in cls.UNCERTAINTY_PRESETS:
            logger.warning(f"Unknown uncertainty level '{uncertainty_level}', using 'standard'")
            uncertainty_level = 'standard'
        
        config = cls.UNCERTAINTY_PRESETS[uncertainty_level].copy()
        config['level'] = uncertainty_level
        
        logger.info(f"Using uncertainty level: {uncertainty_level} - {config['description']}")
        return config
    
    @classmethod
    def get_enhanced_priors(cls, uncertainty_level: str = 'standard') -> Dict[str, Any]:
        """
        Generate enhanced priors based on uncertainty level.
        
        Args:
            uncertainty_level: Uncertainty level configuration
            
        Returns:
            Dictionary of enhanced prior distributions
        """
        config = cls.get_config(uncertainty_level)
        multipliers = config['prior_multipliers']
        
        return {
            'qi': {
                'distribution': 'lognormal',
                'params': {'mu': 5.0, 'sigma': multipliers['qi_sigma']},
                'bounds': (1, 200000)  # Increased to accommodate realistic well production
            },
            'Di': {
                'distribution': 'gamma',
                'params': {'a': multipliers['Di_shape'], 'scale': multipliers['Di_scale']},
                'bounds': (0.005, 0.8)
            },
            'b': {
                'distribution': 'beta',
                'params': {'a': multipliers['b_alpha'], 'b': multipliers['b_beta']},
                'bounds': (0.0, 2.0)
            },
            'noise_precision': {
                'distribution': 'gamma',
                'params': {'a': multipliers['noise_precision_a'], 'scale': multipliers['noise_precision_scale']},
                'bounds': (0.01, 100)
            }
        }
    
    @classmethod
    def list_available_levels(cls) -> Dict[str, str]:
        """
        List all available uncertainty levels with descriptions.
        
        Returns:
            Dictionary mapping level names to descriptions
        """
        return {level: config['description'] 
                for level, config in cls.UNCERTAINTY_PRESETS.items()}
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate that uncertainty configuration is properly structured.
        
        Args:
            config: Uncertainty configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['prior_multipliers', 'quality_noise_multipliers', 'forecast_uncertainty_factor']
        
        if not all(key in config for key in required_keys):
            logger.error(f"Missing required configuration keys. Required: {required_keys}")
            return False
        
        # Validate prior multipliers
        required_multipliers = ['qi_sigma', 'Di_shape', 'Di_scale', 'b_alpha', 'b_beta', 
                              'noise_precision_a', 'noise_precision_scale']
        if not all(key in config['prior_multipliers'] for key in required_multipliers):
            logger.error(f"Missing prior multipliers. Required: {required_multipliers}")
            return False
        
        # Validate quality noise multipliers
        required_quality_levels = ['high', 'medium', 'low']
        if not all(level in config['quality_noise_multipliers'] for level in required_quality_levels):
            logger.error(f"Missing quality noise multipliers. Required: {required_quality_levels}")
            return False
        
        return True

def create_custom_uncertainty_config(
    qi_sigma_multiplier: float = 1.0,
    quality_noise_increase: float = 1.0,
    forecast_uncertainty_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Create custom uncertainty configuration by scaling standard values.
    
    Args:
        qi_sigma_multiplier: Multiplier for qi prior uncertainty (1.0 = standard)
        quality_noise_increase: Multiplier for quality-based noise (1.0 = standard)
        forecast_uncertainty_multiplier: Multiplier for forecast uncertainty (1.0 = standard)
        
    Returns:
        Custom uncertainty configuration dictionary
    """
    base_config = UncertaintyConfig.get_config('standard')
    
    # Scale prior multipliers
    base_config['prior_multipliers']['qi_sigma'] *= qi_sigma_multiplier
    
    # Scale quality noise multipliers
    for quality_level in base_config['quality_noise_multipliers']:
        base_config['quality_noise_multipliers'][quality_level] *= quality_noise_increase
    
    # Scale forecast uncertainty
    base_config['forecast_uncertainty_factor'] *= forecast_uncertainty_multiplier
    
    base_config['description'] = f"Custom uncertainty (qi×{qi_sigma_multiplier:.1f}, noise×{quality_noise_increase:.1f}, forecast×{forecast_uncertainty_multiplier:.1f})"
    
    return base_config 