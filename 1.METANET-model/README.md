# METANET Traffic Flow Model

## Overview

METANET is a macroscopic traffic flow model for highway traffic simulation and prediction. This implementation offers a comprehensive framework for predicting multi-class vehicle flow on highway segments with various geometric characteristics, including on-ramps and off-ramps.

The model features dynamic parameter adjustment based on time of day, prediction horizon length, and advanced handling of different vehicle types through Passenger Car Equivalent (PCE) factors. It also incorporates an automated parameter calibration system to optimize prediction accuracy.

## Key Features

- **Multi-class vehicle prediction**: Differentiates between 6 vehicle classes (B1-B3 for passenger vehicles, T1-T3 for trucks)
- **Time-dependent parameter adjustment**: Adapts to peak/off-peak conditions and weekday/weekend patterns
- **Ramp flow modeling**: Simulates the effects of on-ramps and off-ramps on traffic flow
- **Automatic parameter calibration**: Fine-tunes model parameters for different prediction horizons
- **Comprehensive validation metrics**: MAE, RMSE, MAPE and R² across different dimensions

## Installation

### Requirements

- Python 3.7+
- NumPy
- Pandas
- SciPy
- scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/metanet-model.git
cd metanet-model


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

## Model Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 18.0 seconds | Relaxation time - controls how quickly speeds adjust toward equilibrium |
| `nu` | 35.0 km²/h | Anticipation parameter - models drivers' reaction to downstream conditions |
| `kappa` | 13.0 veh/km/lane | Density smoothing parameter - prevents excessive sensitivity |
| `a` | 2.0 | Speed-density relationship exponent - controls the shape of the curve |
| `critical_density` | 27.0 veh/km/lane | Density at which traffic flow is maximized |
| `jam_density` | 180.0 veh/km/lane | Density at which traffic comes to a complete stop |
| `free_flow_speed` | 110.0 km/h | Speed at very low densities |

### Vehicle PCE Values

PCE (Passenger Car Equivalent) values define the impact of different vehicle types:

| Vehicle Type | Description | PCE Value |
|--------------|-------------|-----------|
| B1 | Small passenger car | 1.0 |
| B2 | Medium passenger car | 1.5 |
| B3 | Large passenger car/bus | 2.0 |
| T1 | Small truck | 1.5 |
| T2 | Medium truck | 2.5 |
| T3 | Large truck | 3.5 |

## Usage

### Basic Command

```bash
python metanet_model.py --road_data ./path/to/roadETC.csv --flow_dir ./path/to/flow_data
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--road_data` | Path to road segment data CSV file |
| `--flow_dir` | Directory containing traffic flow data files |
| `--output_dir` | Directory for output files (default: './metanet_results') |
| `--tau` | Relaxation time in seconds (default: 18.0) |
| `--nu` | Anticipation parameter in km²/h (default: 35.0) |
| `--kappa` | Density smoothing parameter (default: 13.0) |
| `--a` | Speed-density exponent (default: 2.0) |
| `--predict_minutes` | List of prediction horizons in minutes (default: 5 10 15) |
| `--calibrate` | Enable automatic parameter calibration |
| `--aggregate_by` | Method for aggregating metrics: 'time' or 'vehicle' (default: 'time') |

### Examples

#### Basic Prediction with Default Settings

```bash
python metanet_model.py --road_data ./data/roadETC.csv --flow_dir ./data/flow
```

#### Prediction with Custom Parameters

```bash
python metanet_model.py --road_data ./data/roadETC.csv --flow_dir ./data/flow --tau 20.0 --nu 40.0 --a 1.8
```

#### Prediction with Automatic Calibration

```bash
python metanet_model.py --road_data ./data/roadETC.csv --flow_dir ./data/flow --calibrate --predict_minutes 5 10 15
```

## Output Files

The model produces the following output files:

1. `metanet.log`: Detailed log of model execution
2. `validation_results.csv`: Complete prediction vs. actual results for all validation samples
3. `overall_metrics.csv`: Aggregated performance metrics (MAE, RMSE, MAPE, R²)

### Example Metrics Output

```csv
sample_count,MAE,RMSE,MAPE,R2,aggregation_method
3450,2.854,4.127,9.35,0.8742,time_average
```

## Model Operation

The METANET model simulates traffic through the following key mechanisms:

1. **Speed-density relationship**: Uses an exponential function to relate traffic density to equilibrium speed
2. **Dynamic segmentation**: Divides road segments into subsegments for accurate simulation
3. **Time step evolution**: Computes traffic state advancement in small time steps (default: 10s)
4. **Flow calculations**: Uses fundamental traffic flow relationships (density × speed × lanes = flow)
5. **Parameter adaptation**: Adjusts model parameters based on time of day and prediction horizon

The core simulation includes:
- Density update (continuity equation)
- Speed update (momentum equation)
- Handling of boundary conditions
- Merging and diverging flows at ramps

## Theoretical Background

The model is based on the second-order macroscopic traffic flow theory incorporating:

1. **Conservation of vehicles** (continuity equation)
2. **Momentum equation** with:
   - Relaxation term: Adjustment toward equilibrium speed
   - Convection term: Speed changes due to vehicle movement
   - Anticipation term: Reaction to density gradients
   - Ramp effect term: Speed changes due to merging/diverging flows

## Advanced Features

### Time-Period Adaptations

The model adapts parameters based on time of day:
- **Morning peak (7-9am)**: Reduced speeds, faster driver reactions
- **Evening peak (5-7pm)**: Further reduced speeds, stronger anticipation
- **Weekend vs. weekday**: Different parameter adjustments
- **Off-peak/night**: Higher speeds, slower relaxation time

### Horizon-Specific Parameters

Different prediction horizons use tailored parameters:
- **5-minute horizon**: Faster relaxation, stronger anticipation
- **10-minute horizon**: Balanced parameters
- **15-minute horizon**: Slower relaxation, adjusted anticipation

## References

1. Messner, A., & Papageorgiou, M. (1990). METANET: A macroscopic simulation program for motorway networks. Traffic Engineering & Control, 31(8-9), 466-470.
2. Papageorgiou, M., Blosseville, J.-M., & Hadj-Salem, H. (1989). Macroscopic modelling of traffic flow on the Boulevard Périphérique in Paris. Transportation Research Part B: Methodological, 23(1), 29-47.
3. Kotsialos, A., Papageorgiou, M., Diakaki, C., Pavlis, Y., & Middelham, F. (2002). Traffic flow modeling of large-scale motorway networks using the macroscopic modeling tool METANET. IEEE Transactions on Intelligent Transportation Systems, 3(4), 282-292.
