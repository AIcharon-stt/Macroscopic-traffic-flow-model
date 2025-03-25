# Enhanced CTM Traffic Flow Model

## Overview

The Enhanced Cell Transmission Model (EnCTM) is an advanced macroscopic traffic flow simulation and prediction model based on Daganzo's cell transmission theory. This model builds upon the classical CTM with multiple enhancements, including multi-class vehicle modeling, adaptive cell systems, intelligent traffic state classification, and machine learning integration. These innovative features enable more accurate traffic flow predictions in complex highway networks.

This model is particularly suitable for short and medium-term (5-30 minutes) prediction of highway traffic flows, effectively handling complex road conditions including ramps and bottlenecks, while accounting for external factors such as weather and time-of-day effects.

## Key Features

- **Multi-class vehicle modeling**: Detailed flow prediction for 6 vehicle classes (B1-B3 passenger vehicles, T1-T3 trucks)
- **Automatic traffic state classification**: Intelligent identification of free-flow, transitional, congested, and hypercongested traffic states
- **Adaptive cell system**: Dynamically adjusts cell size and distribution based on traffic conditions to improve both computational efficiency and accuracy
- **Enhanced bottleneck handling**: Advanced detection and spatial influence modeling of bottleneck regions
- **Time-dependent parameter adjustment**: Automatically tunes model parameters for different periods (morning peak, evening peak, off-peak, etc.)
- **Weather impact modeling**: Accounts for precipitation and visibility effects on traffic flow
- **Historical pattern integration**: Leverages historical data patterns to enhance predictions
- **Machine learning augmentation**: Optional ML integration for improved prediction accuracy
- **Hybrid prediction mode**: Combines physical modeling with machine learning approaches

## Installation Requirements

### Requirements
- Python 3.7 or higher
- NumPy
- Pandas
- SciPy
- scikit-learn (for machine learning features)
- TensorFlow (optional, for advanced ML features)
- matplotlib (optional, for visualization)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/enctm-model.git
cd enctm-model

# Install required packages
pip install -r requirements.txt
```

## Data Format Requirements

### Road Segment Data

The model requires a CSV file with the following columns:

- `id`: Unique road segment identifier
- `up_node`: Upstream gantry/detector ID
- `down_node`: Downstream gantry/detector ID
- `length`: Segment length in kilometers
- `lanes`: Number of lanes (default: 3)
- `speed_limit`: Speed limit in km/h (default: 100)
- `type`: Segment type (1: basic, 2: with on-ramp, 3: with off-ramp, 4: with both)
- `bottleneck_factor` (optional): Bottleneck severity factor (0-1)
- `curvature` (optional): Road curvature factor (0-1)
- `grade` (optional): Road grade in percent (positive for uphill)

Example:
```csv
id,up_node,down_node,length,lanes,speed_limit,type
S001,G101,G102,2.5,3,120,1
S002,G102,G103,1.8,2,100,2
```

### Traffic Flow Data

For each gantry/detector, create a CSV file named `trafficflow_<gantry_id>.csv` with the following columns:

- `Time`: Timestamp in format 'YYYY/MM/DD HH:MM'
- `B1`, `B2`, `B3`, `T1`, `T2`, `T3`: Vehicle counts by type in 5-minute intervals

Example:
```csv
Time,B1,B2,B3,T1,T2,T3
2023/05/01 08:00,45,5,2,8,3,1
2023/05/01 08:05,52,6,1,7,4,2
```

### Weather Data (Optional)

CSV file format:
- `Time`: Timestamp in format 'YYYY/MM/DD HH:MM'
- `precipitation`: Precipitation in mm
- `visibility`: Visibility in meters
- `temperature`: Temperature in Celsius

## Model Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `critical_density` | 27.0 veh/km/lane | Critical density at which traffic flow is maximized |
| `jam_density` | 180.0 veh/km/lane | Jam density at which traffic comes to a complete stop |
| `free_flow_speed` | 110.0 km/h | Free-flow speed at very low densities |
| `wave_speed` | 20.0 km/h | Wave speed at which congestion propagates backward |
| `alpha` | 2.0 | Shape parameter for the fundamental diagram |
| `beta` | 0.25 | Transition smoothness parameter |
| `cell_length` | 0.5 km | Base length of cells for road discretization |

### Vehicle Class Parameters

Each vehicle class has specific parameters:

| Parameter | Description |
|-----------|-------------|
| `length` | Average vehicle length in meters |
| `max_speed` | Maximum speed in km/h |
| `acceleration` | Average acceleration in m/s² |
| `deceleration` | Average deceleration in m/s² |
| `gap_factor` | Inter-vehicle spacing factor |
| `pce` | Passenger Car Equivalent value |
| `flow_smoothing` | Flow smoothing factor for prediction |

## Usage

### Basic Command

```bash
python ctm_model.py --road_data ./path/to/roadETC.csv --flow_dir ./path/to/flow_data
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--road_data` | Path to road segment data CSV file |
| `--flow_dir` | Directory containing traffic flow data files |
| `--output_dir` | Output directory for results (default: './enctm_results') |
| `--weather_data` | Path to weather data file (optional) |
| `--alpha` | Shape parameter for fundamental diagram (default: 2.0) |
| `--beta` | Transition smoothness parameter (default: 0.25) |
| `--adaptive_cells` | Enable adaptive cell system (default: enabled) |
| `--ml_integration` | Enable machine learning integration (default: disabled) |
| `--hybrid_prediction` | Enable hybrid CTM-ML prediction (default: disabled) |
| `--predict_minutes` | List of prediction time horizons in minutes (default: 5 10 15 30) |
| `--calibrate` | Automatically calibrate model parameters |
| `--visualization` | Enable result visualization |
| `--aggregate_by` | Method for aggregating metrics: 'time' or 'vehicle' (default: 'time') |

### Examples

#### Basic Prediction with Default Settings

```bash
python ctm_model.py --road_data ./data/roadETC.csv --flow_dir ./data/flow
```

#### Prediction with Custom Parameters

```bash
python ctm_model.py --road_data ./data/roadETC.csv --flow_dir ./data/flow --alpha 2.2 --beta 0.3
```

#### Full-Featured Prediction with ML and Calibration

```bash
python ctm_model.py --road_data ./data/roadETC.csv --flow_dir ./data/flow --weather_data ./data/weather.csv --ml_integration --visualization --calibrate --predict_minutes 5 10 15 30
```

## Output Files

The model produces the following output files:

1. `enctm.log`: Detailed log of model execution
2. `validation_results.csv`: Complete prediction vs. actual results for all validation samples
3. `overall_metrics.csv`: Aggregated performance metrics (MAE, RMSE, MAPE, R²)
4. `calibration_parameters.json`: Calibrated model parameters (if calibration enabled)
5. `traffic_states.csv`: Classification of traffic states for each prediction
6. `visualization/` directory (if visualization enabled): Graphs and visualizations of results

### Example Metrics Output

```csv
sample_count,MAE,RMSE,MAPE,R2,aggregation_method
3450,2.854,4.127,9.35,0.8742,time_average
```

## Model Operation

The Enhanced CTM model simulates traffic flow through the following key mechanisms:

### 1. Multi-Regime Fundamental Diagram

The model uses a modified Wu's model for the density-flow relationship, which provides a more accurate representation of different traffic states than the traditional triangular diagram:

```
q(ρ) = v_f * ρ * (1 - (ρ/ρ_jam)^α) / (1 + β * (ρ/ρ_crit)^α)
```

Where:
- q(ρ) is the flow at density ρ
- v_f is the free-flow speed
- ρ_jam is the jam density
- ρ_crit is the critical density
- α and β are shape control parameters

### 2. Traffic State Classification

The model automatically classifies traffic states into four categories:
- **Free-flow**: Density < 0.65 × critical_density
- **Transition**: 0.65 × critical_density ≤ Density < 1.15 × critical_density
- **Congested**: 1.15 × critical_density ≤ Density < 0.75 × jam_density
- **Hypercongested**: Density ≥ 0.75 × jam_density

Each state uses optimized parameter sets for improved accuracy.

### 3. Adaptive Cell System

Standard CTM uses fixed-length cells, while Enhanced CTM dynamically adjusts cell sizes:
- Larger cells in free-flow conditions for computational efficiency
- Smaller cells in transition and congested states for accuracy
- Specialized fine cells around bottlenecks

### 4. Sending-Receiving Flow Principle

The core calculation principle of CTM is:

```
flow(i→i+1) = min(sending_flow(i), receiving_flow(i+1))
```

Where:
- sending_flow(i) = min(density of cell i × free-flow speed, capacity)
- receiving_flow(i+1) = min(capacity, wave speed × (jam density - density of cell i+1))
- capacity = critical density × free-flow speed

## Advanced Features

### Time-Period Adaptations

The model adapts parameters based on time of day:
- **Morning peak (7-9am)**: Reduced speeds, faster driver reactions
- **Evening peak (5-7pm)**: Further reduced speeds, stronger anticipation
- **Weekend vs. weekday**: Different parameter adjustments
- **Off-peak/night**: Higher speeds, slower reaction times

### Bottleneck Handling Mechanisms

The model employs advanced bottleneck detection and handling methods:
1. Static bottlenecks: Predefined based on road segment data
2. Dynamic bottlenecks: Automatically identified based on:
   - Significant flow drops between adjacent cells
   - Sharp density increases
   - Transitions from free-flow to congested state

### Weather Impact Modeling

Weather conditions affect the simulation through:
- Precipitation effects: Reductions in free-flow speed and capacity
- Visibility effects: More conservative driving behavior in low visibility
- Driver behavior adjustments: Increased vehicle spacing in adverse conditions

### Machine Learning Integration

When ML features are enabled, the model uses:
- Flow predictors based on historical data
- State transition classifiers
- Parameter tuning models

These models can be automatically trained during validation and improve as more data is collected.

## Theoretical Background

The Enhanced CTM model is based on the following traffic flow theories:

1. **Conservation equation**: Representing the principle that vehicles cannot be created or destroyed
2. **Cell transmission theory**: Extensions of Daganzo's classical CTM
3. **Multi-class interactions**: Consideration of interactions between different vehicle types
4. **Traffic wave propagation theory**: Modeling the formation and propagation of congestion waves

The model synthesizes advancements in traffic flow theory including:
- Multi-regime fundamental diagrams
- Bottleneck dynamics
- Spatial-temporal correlations
- Vehicle heterogeneity

