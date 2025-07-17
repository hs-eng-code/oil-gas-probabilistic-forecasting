# Probabilistic Oil & Gas Production and Revenue Forecasting System

Probabilistic production and revenue forecasting system for oil & gas assets using Arps decline curve analysis (DCA) and Bayesian methods.

## Overview

This codebase implements an 8-phase system (see `System Architecture: Core Pipeline Phases` section below) that combines advanced Arps DCA with Bayesian uncertainty quantification to generate P10/P50/P90 production and revenue forecasts for oil and gas assets.

## Project Structure

```
probabilistic-forecasting/
├── README.md                                    # This file
├── main.py                                      # Main pipeline orchestrator, plus full pipeline validator (Phase 6)
├── requirements.txt                             # Python dependencies
│
├── src/                                         # Core source modules
│   ├── __init__.py                             # Package initialization
│   ├── data_loader.py                          # Phase 1: Data loading & validation
│   ├── arps_dca.py                             # Phase 2: ARPS DCA analysis
│   ├── bayesian_forecaster.py                  # Phase 3: Bayesian forecasting
│   ├── aggregator.py                           # Phase 4: Asset aggregation
│   ├── revenue_calculator.py                   # Phase 5: Revenue calculation
│   ├── visualizations.py                       # Phase 7: Advanced analytics
│   ├── reporting.py                            # Phase 8: Final reporting
│   ├── uncertainty_config.py                   # Uncertainty level configurations
│
├── data/                                        # Input datasets
│   ├── QCG_DS_Exercise_well_prod_data.csv      # Well production data
│   └── QCG_DS_Exercise_price_data.csv          # Oil price strip data
│
├── docs/                                        # Technical documentation
│   ├── Phase1and2_TechnicalImpementation_Report.md      # Phases 1-2 documentation
│   ├── Phase3_TechnicalImpementation_Report.md          # Phase 3 documentation
│   ├── Phase4through6_TechnicalImpementation_Report.md  # Phases 4-6 documentation
│   └── Phase7through8_TechnicalImpementation_Report.md  # Phases 7-8 documentation
│
├── tests/                                       # Test suite
│   ├── unit/                                   # Unit tests
│   │   ├── test_arps_dca.py                   # ARPS DCA tests
│   │   ├── test_data_loader.py                # Data loader tests
│   │   ├── test_bayesian_forecaster.py        # Bayesian forecaster tests
│   │   ├── test_aggregator.py                 # Aggregator tests 
│   │   ├── test_revenue_calculator.py         # Revenue calculator tests
│   └── integration/
│       ├── test_full_pipeline.py              # Full pipeline integration tests
│
└── output/                                      # Generated results
    ├── all_results.pkl                         # Cross-uncertainty scenarios: Consolidated results across scenarios
    ├── all_results_summary.json               # Cross-uncertainty scenarios: Summary statistics across scenarios
    ├── visualizations/                         # Cross-uncertainty scenarios: Analysis charts across scenarios
    └── output_{approach_type}Bayesian_{uncertainty_level}Uncertainty/   # Scenario results (approach_type=individual; 3 uncertainty_level types)
        ├── comprehensive_report.json          # Unified results report
        ├── asset_revenue_forecast.csv          # P10/P50/P90 revenue forecasts: Horizontal data format for P10, P50, P90 cases
        ├── asset_revenue_forecast.xlsx         # P10/P50/P90 revenue forecasts: One tab for each P10, P50, P90 cases
        ├── pipeline_results.pkl               # Complete pipeline state
        ├── processing_stats.pkl               # Processing statistics
        ├── well_dat.pkl                       # Processed well data
        └── visualizations/                    # Scenario-specific charts
```

## System Architecture: Core Pipeline Phases

```
Phase 1: Data Loading and Validation     → src/data_loader.py
Phase 2: Advanced ARPS DCA Processing    → src/arps_dca.py  
Phase 3: Bayesian Forecasting           → src/bayesian_forecaster.py
Phase 4: Asset Aggregation              → src/aggregator.py
Phase 5: Revenue Calculation             → src/revenue_calculator.py
Phase 6: Comprehensive Validation       → main.py (_perform_comprehensive_validation)
Phase 7: Advanced Analytics             → src/visualizations.py
Phase 8: Final Reporting                → src/reporting.py
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

You can execute the pipeline using one of the following two approachs:
1. **Using IDE**: Directly execute the `main.py` code or use Jupyter (Ctrl+A then Shift+Enter) to run in interactive window. The `main.py` orchestrates the pipeline through `run_forecasting_pipeline` function as follows:
```python
from main import run_forecasting_pipeline

# Run with default settings
results = run_forecasting_pipeline(
    well_data_path='data/QCG_DS_Exercise_well_prod_data.csv',
    price_data_path='data/QCG_DS_Exercise_price_data.csv',
    output_dir='output',
    forecast_years=30,
	use_asset_scale_bayesian_processing=False,
    random_seed=42,
	uncertainty_level='standard'
)
```

2. **Using Terminal**: Open the terminal inside the repository's root and then run:
```bash
python main.py
```

### Data Requirements

- **Monthly Well Production History Data**
- **Monthly Oil Strip Price Data**

## Mathematical Foundation and API Reference

See detailed technical implementation documents for Phase 1 through 8 under /docs.

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

### Test Coverage
- Unit tests for five main modules (`src/data_loader.py`, `src/arps_dca.py`, `src/bayesian_forecaster.py`, `src/aggregator.py`, `src/revenue_calculator.py`)
- Integration test for full pipeline
