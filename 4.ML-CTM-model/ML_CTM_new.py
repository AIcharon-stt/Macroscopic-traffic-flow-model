import numpy as np
import pandas as pd
import os
import logging
import argparse
import datetime
import json
import pickle
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize


class EnhancedCTMModel:
    def __init__(self,
                 road_data_path: str,
                 traffic_flow_dir: str,
                 output_dir: str = "./enctm_results",
                 weather_data_path: Optional[str] = None,
                 simulation_step: float = 10.0,
                 critical_density: float = 27.0,
                 jam_density: float = 180.0,
                 free_flow_speed: float = 110.0,
                 wave_speed: float = 20.0,
                 alpha: float = 2.0,
                 beta: float = 0.25,
                 cell_length: float = 0.5,
                 default_lanes: int = 3,
                 adaptive_cells: bool = True,
                 ml_integration: bool = False,
                 hybrid_prediction: bool = False,
                 aggregate_by: str = 'time'):

        # Initialize basic parameters
        self.road_data_path = road_data_path
        self.traffic_flow_dir = traffic_flow_dir
        self.weather_data_path = weather_data_path
        self.output_dir = output_dir
        self.aggregate_by = aggregate_by

        os.makedirs(output_dir, exist_ok=True)
        if ml_integration:
            os.makedirs(os.path.join(output_dir, 'ml_models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualization'), exist_ok=True)

        # Core model parameters
        self.simulation_step = simulation_step
        self.critical_density = critical_density
        self.jam_density = jam_density
        self.free_flow_speed = free_flow_speed
        self.wave_speed = wave_speed
        self.alpha = alpha
        self.beta = beta
        self.cell_length = cell_length
        self.default_lanes = default_lanes
        self.adaptive_cells = adaptive_cells
        self.ml_integration = ml_integration
        self.hybrid_prediction = hybrid_prediction and ml_integration  # Hybrid requires ML

        # Setting up logger
        self._setup_logging()

        self.logger.info(f"Initializing Enhanced CTM (EnCTM) traffic flow model")
        self.logger.info(f"Core parameters: critical_density={critical_density}, jam_density={jam_density}, " +
                         f"free_flow_speed={free_flow_speed}, wave_speed={wave_speed}")
        self.logger.info(f"Advanced parameters: alpha={alpha}, beta={beta}, adaptive_cells={adaptive_cells}, " +
                         f"ml_integration={ml_integration}, hybrid_prediction={self.hybrid_prediction}")

        # Load data
        self.road_data = self._load_road_data()
        self.flow_data = self._load_flow_data()
        self.weather_data = self._load_weather_data() if weather_data_path else None

        # Process historical flow patterns to enhance prediction
        self.has_historical_patterns = self._process_historical_patterns()

        self.segments = self._prepare_segments()

        # Vehicle characteristics
        self.vehicle_classes = {
            'B1': {  # Small passenger car
                'length': 4.0,  # meters
                'max_speed': 130.0,  # km/h
                'acceleration': 2.5,  # m/s²
                'deceleration': 3.0,  # m/s²
                'gap_factor': 1.0,
                'pce': 1.0,
                'flow_smoothing': 0.2  # Smoothing factor for prediction
            },
            'B2': {  # Medium passenger car
                'length': 5.0,
                'max_speed': 120.0,
                'acceleration': 2.0,
                'deceleration': 2.8,
                'gap_factor': 1.2,
                'pce': 1.5,
                'flow_smoothing': 0.3
            },
            'B3': {  # Large passenger car/bus
                'length': 12.0,
                'max_speed': 100.0,
                'acceleration': 1.2,
                'deceleration': 2.0,
                'gap_factor': 1.5,
                'pce': 2.0,
                'flow_smoothing': 0.4
            },
            'T1': {  # Small truck
                'length': 8.0,
                'max_speed': 110.0,
                'acceleration': 1.5,
                'deceleration': 2.5,
                'gap_factor': 1.3,
                'pce': 1.5,
                'flow_smoothing': 0.3
            },
            'T2': {  # Medium truck
                'length': 15.0,
                'max_speed': 90.0,
                'acceleration': 1.0,
                'deceleration': 2.0,
                'gap_factor': 1.8,
                'pce': 2.5,
                'flow_smoothing': 0.5
            },
            'T3': {  # Large truck
                'length': 22.0,
                'max_speed': 80.0,
                'acceleration': 0.8,
                'deceleration': 1.8,
                'gap_factor': 2.0,
                'pce': 3.5,
                'flow_smoothing': 0.6
            }
        }

        # Initialize parameters for different prediction horizons
        self.horizon_params = {}
        self._init_horizon_params()

        # Initialize traffic state classification thresholds
        self.traffic_state_thresholds = {
            'free_flow': 0.65 * self.critical_density,  # Adjusted from 0.7
            'transition': 1.15 * self.critical_density,  # Adjusted from 1.1
            'hypercongested': 0.75 * self.jam_density  # Adjusted from 0.8
        }

        # Enhanced bottleneck parameters
        self.bottleneck_params = {
            'capacity_reduction': {
                'light': 0.9,  # 10% reduction
                'moderate': 0.8,  # 20% reduction
                'severe': 0.6  # 40% reduction
            },
            'detection_threshold': 0.75,  # Ratio of upstream to downstream flow
            'recovery_rate': 0.05,  # Recovery rate per cell after bottleneck
            'bottleneck_influence': 3  # Number of cells influenced by bottleneck
        }

        # Improved weather impact parameters
        self.weather_impact = {
            'precipitation': {
                'none': {
                    'free_flow_speed_factor': 1.0,
                    'capacity_factor': 1.0,
                    'wave_speed_factor': 1.0,
                    'driver_behavior_factor': 1.0
                },
                'light': {
                    'free_flow_speed_factor': 0.95,
                    'capacity_factor': 0.97,
                    'wave_speed_factor': 1.05,
                    'driver_behavior_factor': 1.1
                },
                'moderate': {
                    'free_flow_speed_factor': 0.90,
                    'capacity_factor': 0.93,
                    'wave_speed_factor': 1.10,
                    'driver_behavior_factor': 1.2
                },
                'heavy': {
                    'free_flow_speed_factor': 0.80,
                    'capacity_factor': 0.85,
                    'wave_speed_factor': 1.20,
                    'driver_behavior_factor': 1.3
                }
            },
            'visibility': {
                'good': {
                    'free_flow_speed_factor': 1.0,
                    'capacity_factor': 1.0,
                    'wave_speed_factor': 1.0,
                    'driver_behavior_factor': 1.0
                },
                'moderate': {
                    'free_flow_speed_factor': 0.93,
                    'capacity_factor': 0.95,
                    'wave_speed_factor': 1.05,
                    'driver_behavior_factor': 1.1
                },
                'poor': {
                    'free_flow_speed_factor': 0.85,
                    'capacity_factor': 0.90,
                    'wave_speed_factor': 1.15,
                    'driver_behavior_factor': 1.2
                },
                'very_poor': {
                    'free_flow_speed_factor': 0.70,
                    'capacity_factor': 0.80,
                    'wave_speed_factor': 1.25,
                    'driver_behavior_factor': 1.4
                }
            }
        }

        # ML model containers (if enabled)
        if self.ml_integration:
            try:
                import sklearn
                self.sklearn_available = True
                try:
                    import tensorflow as tf
                    self.tf_available = True
                    self.logger.info("TensorFlow available for advanced ML integration")
                except ImportError:
                    self.tf_available = False
                    self.logger.info("TensorFlow not available, using scikit-learn for ML integration")
            except ImportError:
                self.sklearn_available = False
                self.tf_available = False
                self.logger.warning("Neither scikit-learn nor TensorFlow is available, ML integration will be limited")

            self.ml_models = {}
            self._init_ml_models()
        else:
            self.sklearn_available = False
            self.tf_available = False

        # Prediction cache for hybrid approach
        self.prediction_cache = {}

        # Traffic pattern database
        self.traffic_patterns = {
            'weekday_morning_peak': {'start_hour': 7, 'end_hour': 9, 'days': [0, 1, 2, 3, 4]},
            'weekday_evening_peak': {'start_hour': 16, 'end_hour': 19, 'days': [0, 1, 2, 3, 4]},
            'weekend_midday': {'start_hour': 11, 'end_hour': 15, 'days': [5, 6]},
            'night': {'start_hour': 22, 'end_hour': 5, 'days': [0, 1, 2, 3, 4, 5, 6]}
        }

        self.calibrated = False
        self.logger.info("Enhanced CTM model initialization complete")

    def _setup_logging(self):
        self.logger = logging.getLogger('enctm')
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            self.logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        log_file = os.path.join(self.output_dir, 'enctm.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _load_road_data(self):
        self.logger.info(f"Loading road segment data from: {self.road_data_path}")
        try:
            road_data = pd.read_csv(self.road_data_path)
            # Process additional columns if available
            for col in ['bottleneck_factor', 'curvature', 'grade']:
                if col not in road_data.columns:
                    road_data[col] = 0.0  # Default values if columns don't exist

            self.logger.info(f"Successfully loaded {len(road_data)} road segments")
            return road_data
        except Exception as e:
            self.logger.error(f"Failed to load road data: {e}")
            return pd.DataFrame()

    def _load_flow_data(self):
        self.logger.info(f"Loading traffic flow data from directory: {self.traffic_flow_dir}")
        flow_data = {}

        gantry_ids = set()
        for _, row in self.road_data.iterrows():
            gantry_ids.add(row['up_node'])
            gantry_ids.add(row['down_node'])

        for gantry_id in gantry_ids:
            flow_file = os.path.join(self.traffic_flow_dir, f"trafficflow_{gantry_id}.csv")
            if os.path.exists(flow_file):
                try:
                    df = pd.read_csv(flow_file)
                    df['time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M')

                    # Extract time features for ML
                    df['hour'] = df['time'].dt.hour
                    df['day_of_week'] = df['time'].dt.dayofweek
                    df['month'] = df['time'].dt.month
                    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

                    flow_data[gantry_id] = df
                    self.logger.debug(f"Loaded flow data for gantry {gantry_id}: {len(df)} records")
                except Exception as e:
                    self.logger.warning(f"Failed to load flow data for gantry {gantry_id}: {e}")

        self.logger.info(f"Successfully loaded flow data for {len(flow_data)} gantries")
        return flow_data

    def _process_historical_patterns(self):
        """Process historical flow patterns from the data"""
        if not self.flow_data:
            return False

        try:
            self.logger.info("Processing historical traffic patterns...")

            # Initialize pattern storage
            self.historical_patterns = {}

            # Process each gantry's data to extract patterns
            for gantry_id, df in self.flow_data.items():
                if len(df) < 100:  # Skip if too little data
                    continue

                # Calculate hourly patterns by day of week
                patterns = {}
                for day in range(7):
                    day_data = df[df['day_of_week'] == day]

                    if len(day_data) > 0:
                        # Calculate average hourly flows
                        hourly_means = day_data.groupby('hour')[['B1', 'B2', 'B3', 'T1', 'T2', 'T3']].mean()
                        patterns[day] = hourly_means

                self.historical_patterns[gantry_id] = patterns

            pattern_file = os.path.join(self.output_dir, 'historical_patterns.pkl')
            with open(pattern_file, 'wb') as f:
                pickle.dump(self.historical_patterns, f)

            self.logger.info(f"Historical patterns processed for {len(self.historical_patterns)} gantries")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to process historical patterns: {e}")
            return False

    def _load_weather_data(self):
        self.logger.info(f"Loading weather data from: {self.weather_data_path}")
        try:
            weather_data = pd.read_csv(self.weather_data_path)
            weather_data['time'] = pd.to_datetime(weather_data['Time'], format='%Y/%m/%d %H:%M')

            # Categorize precipitation and visibility
            if 'precipitation' in weather_data.columns:
                conditions = [
                    (weather_data['precipitation'] == 0),
                    (weather_data['precipitation'] < 1.0),
                    (weather_data['precipitation'] < 4.0),
                    (weather_data['precipitation'] >= 4.0)
                ]
                values = ['none', 'light', 'moderate', 'heavy']
                weather_data['precipitation_category'] = np.select(conditions, values, default='none')

            if 'visibility' in weather_data.columns:
                conditions = [
                    (weather_data['visibility'] >= 10000),
                    (weather_data['visibility'] >= 5000),
                    (weather_data['visibility'] >= 1000),
                    (weather_data['visibility'] < 1000)
                ]
                values = ['good', 'moderate', 'poor', 'very_poor']
                weather_data['visibility_category'] = np.select(conditions, values, default='good')

            # Add weather severity index (0-10 scale)
            if 'precipitation' in weather_data.columns and 'visibility' in weather_data.columns:
                weather_data['weather_severity'] = (
                                                           np.clip(weather_data['precipitation'] * 2.0, 0, 8) +
                                                           np.clip((10000 - weather_data['visibility']) / 1000, 0, 8)
                                                   ) / 2.0

            self.logger.info(f"Successfully loaded weather data: {len(weather_data)} records")
            return weather_data
        except Exception as e:
            self.logger.error(f"Failed to load weather data: {e}")
            return None

    def _prepare_segments(self):
        self.logger.info("Preparing segment information for simulation")
        segments = {}

        for _, row in self.road_data.iterrows():
            segment_id = row['id']

            segment = {
                'id': segment_id,
                'up_node': row['up_node'],
                'down_node': row['down_node'],
                'length': row['length'],
                'lanes': row.get('lanes', self.default_lanes),
                'speed_limit': row.get('speed_limit', 100),
                'type': int(row.get('type', 1)),
                'bottleneck_factor': float(row.get('bottleneck_factor', 0.0)),
                'curvature': float(row.get('curvature', 0.0)),
                'grade': float(row.get('grade', 0.0))
            }

            segment['v_free'] = min(segment['speed_limit'], self.free_flow_speed)

            # Advanced adjustments to free-flow speed
            # Adjust free-flow speed based on curvature and grade
            if segment['curvature'] > 0:
                curve_factor = 1 - 0.1 * segment['curvature'] - 0.05 * (segment['curvature'] ** 2)
                segment['v_free'] *= max(0.7, curve_factor)

            if segment['grade'] > 0:  # Uphill
                grade_factor = 1 - 0.02 * segment['grade'] - 0.003 * (segment['grade'] ** 2)
                segment['v_free'] *= max(0.8, grade_factor)

            # Base number of cells (will be adjusted if adaptive_cells=True)
            num_cells = max(3, int(segment['length'] / self.cell_length))
            segment['num_cells'] = num_cells
            segment['base_cell_length'] = segment['length'] / num_cells

            # Enhanced bottleneck detection and severity assessment
            if segment['bottleneck_factor'] > 0:
                severity = 'light'
                if segment['bottleneck_factor'] >= 0.3:
                    severity = 'moderate'
                if segment['bottleneck_factor'] >= 0.6:
                    severity = 'severe'

                segment['bottleneck_severity'] = severity
                capacity_factor = self.bottleneck_params['capacity_reduction'][severity]
                segment['capacity_factor'] = capacity_factor

                # Calculate bottleneck position (default to middle, but influenced by grade and curvature)
                bottleneck_position = 0.5  # Default position (middle)
                if segment['grade'] > 0:
                    # Uphill bottlenecks tend to be near the steepest part
                    bottleneck_position = min(0.8, 0.5 + segment['grade'] * 0.05)
                elif segment['curvature'] > 0:
                    # Curve bottlenecks tend to be at the curve
                    bottleneck_position = min(0.7, 0.4 + segment['curvature'] * 0.1)

                segment['bottleneck_position'] = bottleneck_position

                self.logger.debug(f"Segment {segment_id} identified as {severity} bottleneck " +
                                  f"at position {bottleneck_position:.2f} " +
                                  f"(capacity reduction: {(1 - capacity_factor) * 100:.0f}%)")
            else:
                segment['bottleneck_severity'] = None
                segment['capacity_factor'] = 1.0
                segment['bottleneck_position'] = None

            segments[segment_id] = segment
            self.logger.debug(f"Prepared segment {segment_id}: {segment['length']}km, {num_cells} cells")

        self.logger.info(f"Prepared {len(segments)} segments for simulation")
        return segments

    def _init_horizon_params(self):
        self.logger.info("Initializing prediction horizon parameters")

        # 5-minute prediction - fastest response, focus on current conditions
        self.horizon_params[5] = {
            'wave_speed': self.wave_speed * 1.15,  # Increased from 1.1
            'critical_density': self.critical_density * 1.05,
            'alpha': self.alpha * 0.9,  # Changed from 0.95
            'beta': self.beta * 0.85,  # Changed from 0.9
            'historical_weight': 0.3,  # Weight for historical pattern contribution
            'recency_factor': 0.7  # Emphasis on very recent measurements
        }

        # 10-minute prediction - balanced
        self.horizon_params[10] = {
            'wave_speed': self.wave_speed,
            'critical_density': self.critical_density,
            'alpha': self.alpha,
            'beta': self.beta,
            'historical_weight': 0.4,
            'recency_factor': 0.5
        }

        # 15-minute prediction - more conservative
        self.horizon_params[15] = {
            'wave_speed': self.wave_speed * 0.92,  # Changed from 0.95
            'critical_density': self.critical_density * 0.95,
            'alpha': self.alpha * 1.1,  # Changed from 1.05
            'beta': self.beta * 1.15,  # Changed from 1.1
            'historical_weight': 0.5,
            'recency_factor': 0.4
        }

        # 30-minute prediction - even more emphasis on historical patterns
        self.horizon_params[30] = {
            'wave_speed': self.wave_speed * 0.85,  # Changed from 0.9
            'critical_density': self.critical_density * 0.88,  # Changed from 0.9
            'alpha': self.alpha * 1.15,  # Changed from 1.1
            'beta': self.beta * 1.25,  # Changed from 1.2
            'historical_weight': 0.6,
            'recency_factor': 0.3
        }

        self.logger.info(f"Parameters set for 5, 10, 15, and 30 minute prediction horizons")

    def _init_ml_models(self):
        """Initialize machine learning models if ML integration is enabled"""
        if not self.ml_integration:
            return

        self.logger.info("Initializing ML model components")

        # Check if we have scikit-learn available
        if not self.sklearn_available:
            self.logger.warning("ML integration requested but scikit-learn not available")
            self.ml_models = {
                'state_transition': {'type': 'statistical'},
                'ramp_flow': {'type': 'statistical'},
                'bottleneck_detection': {'type': 'statistical'},
                'parameter_adjustment': {'type': 'statistical'}
            }
            return

        # Check if pre-trained models exist
        model_dir = os.path.join(self.output_dir, 'ml_models')
        models_exist = all(os.path.exists(os.path.join(model_dir, f"{model_name}.pkl"))
                           for model_name in ['flow_predictor', 'state_classifier', 'parameter_tuner'])

        if models_exist:
            self.logger.info("Loading pre-trained ML models")
            try:
                # Load each model
                for model_name in ['flow_predictor', 'state_classifier', 'parameter_tuner']:
                    with open(os.path.join(model_dir, f"{model_name}.pkl"), 'rb') as f:
                        self.ml_models[model_name] = pickle.load(f)
                self.logger.info("Pre-trained ML models loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load pre-trained models: {e}")
                self._create_ml_models()
        else:
            self.logger.info("Creating new ML models")
            self._create_ml_models()

        # Initialize statistical components
        self.ml_models['ramp_flow'] = {'type': 'statistical'}
        self.ml_models['bottleneck_detection'] = {'type': 'statistical'}

        self.logger.info("ML model initialization complete")

    def _create_ml_models(self):
        """Create new ML models for traffic prediction"""
        self.logger.info("Creating new ML model components")

        # Base flow predictor - RandomForest regressor
        flow_predictor = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])

        # Traffic state classifier - GradientBoosting classifier
        state_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])

        # Parameter tuning model - RandomForest
        parameter_tuner = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42
            ))
        ])

        # Store the models
        self.ml_models = {
            'flow_predictor': flow_predictor,
            'state_classifier': state_classifier,
            'parameter_tuner': parameter_tuner
        }

        self.logger.info("ML models created successfully (untrained)")

        # Flag that models need training
        self.ml_models_trained = False

    def _get_flow_by_type(self, gantry_id, time_point):
        if gantry_id not in self.flow_data:
            return {}

        df = self.flow_data[gantry_id]

        df['time_diff'] = abs(df['time'] - time_point)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        df.drop('time_diff', axis=1, inplace=True)

        flow_dict = {}
        vehicle_types = ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']
        for vt in vehicle_types:
            if vt in closest_row:
                flow_dict[vt] = closest_row[vt]

        return flow_dict

    def _calculate_pce_flow(self, flow_dict):
        pce_flow = 0
        for veh_type, flow in flow_dict.items():
            pce = self.vehicle_classes[veh_type]['pce']
            pce_flow += flow * pce
        return pce_flow

    def _get_historical_pattern(self, gantry_id, time_point):
        """Get historical flow pattern for the given gantry and time"""
        if not hasattr(self, 'historical_patterns') or gantry_id not in self.historical_patterns:
            return None

        # Convert to Python datetime if needed
        if isinstance(time_point, np.datetime64):
            time_point = pd.Timestamp(time_point).to_pydatetime()

        # Get day of week and hour
        day = time_point.weekday()
        hour = time_point.hour

        patterns = self.historical_patterns[gantry_id]
        if day not in patterns or hour not in patterns[day].index:
            return None

        return patterns[day].loc[hour].to_dict()

    def _get_weather_conditions(self, time_point):
        """Get weather conditions at a specific time point"""
        if self.weather_data is None:
            return {
                'precipitation_category': 'none',
                'visibility_category': 'good',
                'weather_severity': 0.0
            }

        # Convert numpy.datetime64 to Python datetime if needed
        if isinstance(time_point, np.datetime64):
            time_point = pd.Timestamp(time_point).to_pydatetime()

        df = self.weather_data
        df['time_diff'] = abs(df['time'] - time_point)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        df.drop('time_diff', axis=1, inplace=True)

        return {
            'precipitation_category': closest_row.get('precipitation_category', 'none'),
            'visibility_category': closest_row.get('visibility_category', 'good'),
            'weather_severity': closest_row.get('weather_severity', 0.0),
            'precipitation': closest_row.get('precipitation', 0.0),
            'visibility': closest_row.get('visibility', 10000)
        }

    def _apply_weather_impact(self, sim_params, time_point):
        """Apply weather impact to simulation parameters"""
        weather = self._get_weather_conditions(time_point)

        # Get impact factors
        precip_impact = self.weather_impact['precipitation'].get(
            weather['precipitation_category'],
            {'free_flow_speed_factor': 1.0, 'capacity_factor': 1.0, 'wave_speed_factor': 1.0,
             'driver_behavior_factor': 1.0}
        )

        vis_impact = self.weather_impact['visibility'].get(
            weather['visibility_category'],
            {'free_flow_speed_factor': 1.0, 'capacity_factor': 1.0, 'wave_speed_factor': 1.0,
             'driver_behavior_factor': 1.0}
        )

        # Apply the most restrictive factor for each parameter
        free_flow_factor = min(precip_impact['free_flow_speed_factor'], vis_impact['free_flow_speed_factor'])
        capacity_factor = min(precip_impact['capacity_factor'], vis_impact['capacity_factor'])
        wave_speed_factor = max(precip_impact['wave_speed_factor'], vis_impact['wave_speed_factor'])

        # Driver behavior factor affects critical density
        driver_behavior_factor = max(precip_impact['driver_behavior_factor'], vis_impact['driver_behavior_factor'])

        # Apply factors to parameters
        sim_params['free_flow_speed'] *= free_flow_factor
        # Don't apply capacity factor here - will be calculated later
        sim_params['wave_speed'] *= wave_speed_factor

        # Adjust alpha and beta based on weather
        weather_severity = weather.get('weather_severity', 0.0)
        if weather_severity > 3.0:
            # More conservative driving in bad weather
            sim_params['alpha'] *= (1.0 + 0.05 * min(weather_severity, 8.0) / 8.0)
            sim_params['beta'] *= (1.0 + 0.1 * min(weather_severity, 8.0) / 8.0)

        # Apply driver behavior factor to jam density (drivers keep larger gaps in bad weather)
        sim_params['jam_density'] = self.jam_density / driver_behavior_factor

        # Store capacity_factor for later use
        sim_params['weather_capacity_factor'] = capacity_factor

        # Log significant weather impacts
        if free_flow_factor < 0.95 or capacity_factor < 0.95:
            self.logger.info(f"Weather impact applied at {time_point}: " +
                             f"speed factor={free_flow_factor:.2f}, " +
                             f"capacity factor={capacity_factor:.2f}, " +
                             f"wave speed factor={wave_speed_factor:.2f}, " +
                             f"driver behavior factor={driver_behavior_factor:.2f}")

        return sim_params

    def _classify_traffic_state(self, density, critical_density, jam_density):
        """Classify traffic state based on density"""
        if density < self.traffic_state_thresholds['free_flow']:
            return "free_flow"
        elif density < self.traffic_state_thresholds['transition']:
            return "transition"
        elif density < self.traffic_state_thresholds['hypercongested']:
            return "congested"
        else:
            return "hypercongested"

    def _calculate_density_ratio(self, density, traffic_state, critical_density, jam_density):
        """Calculate density ratio within current traffic state"""
        if traffic_state == "free_flow":
            return density / self.traffic_state_thresholds['free_flow']
        elif traffic_state == "transition":
            return (density - self.traffic_state_thresholds['free_flow']) / (
                    self.traffic_state_thresholds['transition'] - self.traffic_state_thresholds['free_flow'])
        elif traffic_state == "congested":
            return (density - self.traffic_state_thresholds['transition']) / (
                    self.traffic_state_thresholds['hypercongested'] - self.traffic_state_thresholds['transition'])
        else:  # hypercongested
            return (density - self.traffic_state_thresholds['hypercongested']) / (
                    jam_density - self.traffic_state_thresholds['hypercongested'])

    def _adjust_parameters_for_traffic_state(self, sim_params, traffic_state, density_ratio):
        """Adjust simulation parameters based on traffic state and density ratio"""
        # Apply state-specific adjustments
        if traffic_state == "free_flow":
            # Free flow adjustments focus on accurate propagation
            sim_params['alpha'] *= 0.9
            sim_params['wave_speed'] *= 0.9

            # Fine-tune based on how close we are to transition
            if density_ratio > 0.5:  # Getting closer to transition
                sim_params['alpha'] *= (1.0 + 0.1 * (density_ratio - 0.5) * 2)

        elif traffic_state == "transition":
            # Transition state needs careful handling
            sim_params['alpha'] *= 1.1
            sim_params['beta'] *= 1.2

            # Add stronger adjustments as we get closer to congestion
            if density_ratio > 0.7:  # Closer to congestion
                sim_params['wave_speed'] *= (1.0 + 0.1 * (density_ratio - 0.7) * 3)

        elif traffic_state == "congested":
            # Congested state adjustments
            sim_params['wave_speed'] *= 1.1
            sim_params['beta'] *= 0.9

            # Fine-tune based on congestion severity
            sim_params['wave_speed'] *= (1.0 + 0.05 * density_ratio)

        elif traffic_state == "hypercongested":
            # Hypercongested state adjustments
            sim_params['wave_speed'] *= 1.2
            sim_params['alpha'] *= 1.2

            # Extreme congestion adjustments
            if density_ratio > 0.9:  # Very high density
                sim_params['wave_speed'] *= 1.1

        return sim_params

    def _create_adaptive_cells(self, segment, density_estimate, traffic_state):
        """Create adaptive cell system based on traffic state"""
        if not self.adaptive_cells:
            # Use uniform cells if adaptive system is disabled
            num_cells = segment['num_cells']
            cell_lengths = np.ones(num_cells) * segment['base_cell_length']
            return num_cells, cell_lengths

        base_num_cells = segment['num_cells']
        total_length = segment['length']

        # Start with base cell configuration
        cell_lengths = np.ones(base_num_cells) * segment['base_cell_length']

        # Adjust based on traffic state
        if traffic_state == "free_flow":
            # Fewer, larger cells in free flow for computation efficiency
            if base_num_cells > 5:
                new_num_cells = max(5, base_num_cells // 2)
                cell_lengths = np.ones(new_num_cells) * (total_length / new_num_cells)
        elif traffic_state in ["transition", "congested"]:
            # More, smaller cells in transition and congested states for accuracy
            if segment['length'] > 1.0:  # Only for longer segments
                new_num_cells = min(base_num_cells * 2, int(segment['length'] / 0.2))
                cell_lengths = np.ones(new_num_cells) * (total_length / new_num_cells)
        elif traffic_state == "hypercongested":
            # Even finer cells for hypercongested state
            if segment['length'] > 1.0:
                new_num_cells = min(base_num_cells * 3, int(segment['length'] / 0.15))
                cell_lengths = np.ones(new_num_cells) * (total_length / new_num_cells)

        # Special handling for bottlenecks
        if segment['bottleneck_factor'] > 0:
            num_cells = len(cell_lengths)

            # Determine bottleneck position
            bottleneck_idx = int(num_cells * segment.get('bottleneck_position', 0.5))
            bottleneck_idx = max(1, min(bottleneck_idx, num_cells - 2))  # Keep within bounds

            # Create finer cells around bottleneck for better accuracy
            influence_range = self.bottleneck_params['bottleneck_influence']
            start_idx = max(0, bottleneck_idx - influence_range)
            end_idx = min(num_cells, bottleneck_idx + influence_range + 1)

            # Reduce cell size around bottleneck (with gradual transition)
            for i in range(start_idx, end_idx):
                # Calculate distance from bottleneck (normalized)
                distance = abs(i - bottleneck_idx) / influence_range
                # Apply size reduction based on distance (closer = smaller)
                size_factor = 0.6 + 0.4 * distance
                cell_lengths[i] *= size_factor

            # Redistribute length to maintain total segment length
            total_length_now = np.sum(cell_lengths)
            cell_lengths = cell_lengths * (total_length / total_length_now)

        return len(cell_lengths), cell_lengths

    def _calculate_capacity(self, params):
        """Calculate capacity (maximum flow) based on current parameters"""
        # Capacity occurs at critical density
        critical_density = params['critical_density']

        # Apply Wu model formula directly without recursion
        ratio = critical_density / params['jam_density']
        term1 = 1 - ratio ** params['alpha']
        term2 = 1 + params['beta'] * (critical_density / params['critical_density']) ** params['alpha']

        # Maximum flow occurs at critical density
        capacity = params['free_flow_speed'] * critical_density * term1 / term2

        # Apply weather capacity factor if present
        if 'weather_capacity_factor' in params:
            capacity *= params['weather_capacity_factor']

        return capacity

    def _modified_wu_fundamental_diagram(self, density, params):
        """Modified Wu's fundamental diagram relating density to flow"""
        if density <= 0:
            return 0

        if density >= params['jam_density']:
            return 0

        # Modified Wu model with smoother transition
        ratio = density / params['jam_density']
        term1 = 1 - ratio ** params['alpha']
        term2 = 1 + params['beta'] * (density / params['critical_density']) ** params['alpha']

        flow = params['free_flow_speed'] * density * term1 / term2

        # Cap flow at capacity if it exists in params
        if 'capacity' in params:
            return min(flow, params['capacity'])
        return flow

    def _compute_sending_flow(self, density, params):
        """Compute sending flow from a cell based on modified Wu model"""
        return self._modified_wu_fundamental_diagram(density, params)

    def _compute_receiving_flow(self, density, params):
        """Compute receiving flow to a cell based on modified Wu model"""
        # Receiving flow is the difference between max flow at jam density and current flow
        if 'capacity' not in params:
            # Calculate capacity if not already in params
            params['capacity'] = self._calculate_capacity(params)

        capacity = params['capacity']
        current_flow = self._modified_wu_fundamental_diagram(density, params)

        # If in free-flow, receiving capacity is just the capacity
        if density < params['critical_density']:
            return capacity

        # Otherwise calculate based on remaining gap to jam density
        jam_gap = params['jam_density'] - density
        if jam_gap <= 0:
            return 0

        # Wave speed determines how quickly congestion propagates
        return min(capacity, params['wave_speed'] * jam_gap)

    def _estimate_ramp_flows(self, segment, upstream_flow, time_point=None):
        """Estimate ramp flows with improved ML enhancement"""
        segment_type = segment['type']
        on_ramp_flow = 0
        off_ramp_flow = 0

        # Base estimates
        if segment_type == 2:  # Upstream has on-ramp
            on_ramp_flow = upstream_flow * 0.3
        elif segment_type == 3:  # Upstream has off-ramp
            off_ramp_flow = upstream_flow * 0.2
        elif segment_type == 4:  # Upstream has both on-ramp and off-ramp
            on_ramp_flow = upstream_flow * 0.25
            off_ramp_flow = upstream_flow * 0.15

        # Apply ML enhancement if enabled and time_point is provided
        if self.ml_integration and time_point is not None and 'ramp_flow' in self.ml_models:
            # Convert numpy.datetime64 to Python datetime if needed
            if isinstance(time_point, np.datetime64):
                time_point = pd.Timestamp(time_point).to_pydatetime()

            # Extract time features for the patterns
            hour = time_point.hour
            weekday = time_point.weekday()
            month = time_point.month
            is_weekend = weekday >= 5

            # Identify traffic pattern
            pattern = None
            for pattern_name, pattern_def in self.traffic_patterns.items():
                if (weekday in pattern_def['days'] and
                        pattern_def['start_hour'] <= hour <= pattern_def['end_hour']):
                    pattern = pattern_name
                    break

            # Apply pattern-specific adjustments
            if pattern == 'weekday_morning_peak':
                # Morning peak - heavy inbound traffic (ramps feeding main roads)
                if segment_type in [2, 4]:  # Has on-ramp
                    on_ramp_flow *= 1.3
                if segment_type in [3, 4]:  # Has off-ramp
                    off_ramp_flow *= 0.8

            elif pattern == 'weekday_evening_peak':
                # Evening peak - heavy outbound traffic (main roads to ramps)
                if segment_type in [2, 4]:  # Has on-ramp
                    on_ramp_flow *= 0.8
                if segment_type in [3, 4]:  # Has off-ramp
                    off_ramp_flow *= 1.3

            elif pattern == 'weekend_midday':
                # Weekend midday - balanced but higher recreational traffic
                if segment_type in [2, 4]:  # Has on-ramp
                    on_ramp_flow *= 1.1
                if segment_type in [3, 4]:  # Has off-ramp
                    off_ramp_flow *= 1.1

            # Weather adjustments for ramp flows
            weather = self._get_weather_conditions(time_point)
            if weather['weather_severity'] > 3.0:
                # Bad weather reduces overall ramp activity
                severity_factor = max(0.7, 1.0 - weather['weather_severity'] * 0.05)
                on_ramp_flow *= severity_factor
                off_ramp_flow *= severity_factor

            # Consider seasonal effects
            if 5 <= month <= 9:  # Summer months
                recreational_factor = 1.1 if is_weekend else 1.05
                off_ramp_flow *= recreational_factor
            elif month in [1, 2, 12]:  # Winter months
                winter_factor = 0.9 if weather['weather_severity'] > 2.0 else 0.95
                on_ramp_flow *= winter_factor
                off_ramp_flow *= winter_factor

        return on_ramp_flow, off_ramp_flow

    def _detect_bottlenecks(self, segment, densities, flows, params):
        """Enhanced bottleneck detection during simulation"""
        num_cells = len(densities)
        bottlenecks = []

        # Static bottleneck from road data
        if segment['bottleneck_factor'] > 0:
            # Convert bottleneck position from relative to cell index
            bottleneck_position = segment.get('bottleneck_position', 0.5)
            bottleneck_idx = min(num_cells - 1, max(0, int(bottleneck_position * num_cells)))
            severity = segment['bottleneck_severity']
            bottlenecks.append((bottleneck_idx, severity))

        # Dynamic bottleneck detection based on flow/density patterns
        for i in range(1, num_cells - 1):
            # Multiple conditions for bottleneck detection:

            # 1. Significant flow drop between cells
            if flows[i] > 0 and flows[i + 1] > 0:
                flow_ratio = flows[i + 1] / flows[i]

                if flow_ratio < self.bottleneck_params['detection_threshold']:
                    # Flow ratio indicates bottleneck
                    severity = 'light'
                    if flow_ratio < 0.6:
                        severity = 'moderate'
                    if flow_ratio < 0.4:
                        severity = 'severe'

                    bottlenecks.append((i, severity))
                    continue

            # 2. Sharp density increase (congestion forming)
            if i < num_cells - 1 and densities[i + 1] > densities[i]:
                density_jump = densities[i + 1] / max(0.1, densities[i])

                if density_jump > 1.5:
                    # Significant density jump
                    severity = 'light'
                    if density_jump > 2.0:
                        severity = 'moderate'
                    if density_jump > 3.0:
                        severity = 'severe'

                    bottlenecks.append((i, severity))
                    continue

            # 3. Transition from free-flow to congested state
            if (densities[i] < params['critical_density'] and
                    i < num_cells - 1 and
                    densities[i + 1] > params['critical_density']):
                severity = 'moderate'
                bottlenecks.append((i, severity))

        # Remove duplicate bottlenecks (keep the most severe)
        if len(bottlenecks) > 1:
            # Group bottlenecks by location (within 2 cells)
            grouped = {}
            for idx, severity in bottlenecks:
                found = False
                for group_idx in grouped.keys():
                    if abs(idx - group_idx) <= 2:
                        # Add to existing group
                        grouped[group_idx].append((idx, severity))
                        found = True
                        break

                if not found:
                    # Create new group
                    grouped[idx] = [(idx, severity)]

            # For each group, keep only the most severe bottleneck
            bottlenecks = []
            severity_rank = {'light': 1, 'moderate': 2, 'severe': 3}

            for group in grouped.values():
                if len(group) == 1:
                    bottlenecks.append(group[0])
                else:
                    # Find most severe
                    most_severe = max(group, key=lambda x: severity_rank[x[1]])
                    bottlenecks.append(most_severe)

        return bottlenecks

    def _apply_bottleneck_effects(self, sending_flows, receiving_flows, bottlenecks, params, densities):
        """Apply enhanced bottleneck effects to sending and receiving flows"""
        num_cells = len(sending_flows)
        influence_range = self.bottleneck_params['bottleneck_influence']
        recovery_rate = self.bottleneck_params['recovery_rate']

        for idx, severity in bottlenecks:
            # Get capacity reduction factor
            capacity_factor = self.bottleneck_params['capacity_reduction'][severity]

            # Apply bottleneck effects with spatial influence
            for i in range(max(0, idx - influence_range), min(num_cells, idx + influence_range + 1)):
                # Calculate distance from bottleneck and influence factor
                distance = abs(i - idx)
                influence = max(0, 1.0 - distance * recovery_rate)

                # Adjusted capacity factor (full effect at bottleneck, gradually recovering)
                adjusted_factor = 1.0 - (1.0 - capacity_factor) * influence

                # Apply to sending flow - more reduction for cells at or after bottleneck
                if i >= idx:
                    sending_flows[i] *= adjusted_factor

                # Apply to receiving flow - more reduction for cells before or at bottleneck
                if i <= idx and i < len(receiving_flows):
                    receiving_flows[i] *= adjusted_factor

            # Additional effect: increase wave speed at bottleneck to model faster
            # backwards propagation of congestion
            if idx < num_cells and densities[idx] > params['critical_density']:
                # Create shockwave at severe bottlenecks in congestion
                sending_flows[idx] *= 0.95  # Further reduce outflow from bottleneck

        return sending_flows, receiving_flows

    def _get_ml_features(self, segment_id, start_time, upstream_flow_dict):
        """Extract features for ML models"""
        # Convert time to Python datetime if needed
        if isinstance(start_time, np.datetime64):
            start_time = pd.Timestamp(start_time).to_pydatetime()

        # Time features
        hour = start_time.hour
        day_of_week = start_time.weekday()
        month = start_time.month
        is_weekend = int(day_of_week >= 5)

        # Calculate time of day feature (0-1)
        time_of_day = (hour + start_time.minute / 60) / 24.0

        # Segment features
        segment = self.segments[segment_id]
        length = segment['length']
        lanes = segment['lanes']
        segment_type = segment['type']
        bottleneck_factor = segment['bottleneck_factor']

        # Flow features
        total_flow = sum(upstream_flow_dict.values())
        total_pce_flow = self._calculate_pce_flow(upstream_flow_dict)

        # Vehicle type distribution (normalized)
        veh_dist = {}
        if total_flow > 0:
            for veh_type in ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']:
                veh_dist[f"{veh_type}_ratio"] = upstream_flow_dict.get(veh_type, 0) / total_flow

        # Weather features
        weather = self._get_weather_conditions(start_time)
        precip = weather.get('precipitation', 0)
        vis = min(1.0, weather.get('visibility', 10000) / 10000)
        weather_severity = weather.get('weather_severity', 0)

        # Assemble feature dictionary
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'time_of_day': time_of_day,
            'length': length,
            'lanes': lanes,
            'segment_type': segment_type,
            'bottleneck_factor': bottleneck_factor,
            'total_flow': total_flow,
            'total_pce_flow': total_pce_flow,
            'precipitation': precip,
            'visibility': vis,
            'weather_severity': weather_severity
        }

        # Add vehicle distribution
        features.update(veh_dist)

        # Historical pattern features
        up_node = segment['up_node']
        historical = self._get_historical_pattern(up_node, start_time)
        if historical:
            for veh_type in ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']:
                features[f"{veh_type}_hist"] = historical.get(veh_type, 0)

        return features

    def _ml_predict_flow(self, segment_id, start_time, predict_minutes, upstream_flow_dict):
        """Make flow prediction using ML model"""
        if not self.ml_integration or 'flow_predictor' not in self.ml_models:
            return None

        try:
            # Extract features
            features = self._get_ml_features(segment_id, start_time, upstream_flow_dict)

            # Add prediction horizon to features
            features['predict_minutes'] = predict_minutes

            # Convert to array format that scikit-learn expects
            feature_names = sorted(features.keys())
            X = np.array([features[name] for name in feature_names]).reshape(1, -1)

            # Get the model
            model = self.ml_models['flow_predictor']

            # If model is untrained, return None
            if not hasattr(model, 'predict') or isinstance(model, dict):
                return None

            # Make prediction (this will return total flow)
            predicted_total = model.predict(X)[0]

            # Distribute to vehicle types based on upstream distribution
            total_upstream = sum(upstream_flow_dict.values())
            predicted_flow = {}

            if total_upstream > 0:
                for veh_type, flow in upstream_flow_dict.items():
                    ratio = flow / total_upstream
                    predicted_flow[veh_type] = ratio * predicted_total

            return predicted_flow

        except Exception as e:
            self.logger.warning(f"ML flow prediction failed: {e}")
            return None

    def predict_flow(self, segment_id, start_time, predict_minutes=5):
        """Predict traffic flow using Enhanced CTM model with ML integration"""
        if segment_id not in self.segments:
            self.logger.error(f"Segment ID not found: {segment_id}")
            return {}

        segment = self.segments[segment_id]
        up_node = segment['up_node']
        down_node = segment['down_node']

        self.logger.info(f"Predicting flow for segment {segment_id}: {up_node} -> {down_node}, t+{predict_minutes}min")

        # Get upstream flow by vehicle type
        upstream_flow_dict = self._get_flow_by_type(up_node, start_time)
        if not upstream_flow_dict:
            self.logger.warning(f"No upstream flow data found for gantry {up_node} at {start_time}")
            return {}

        # Check prediction cache for hybrid approach
        cache_key = f"{segment_id}_{start_time}_{predict_minutes}"
        if self.hybrid_prediction and cache_key in self.prediction_cache:
            self.logger.info(f"Using cached prediction for {cache_key}")
            return self.prediction_cache[cache_key]

        # Try ML prediction first if hybrid mode is enabled
        ml_prediction = None
        if self.hybrid_prediction:
            ml_prediction = self._ml_predict_flow(segment_id, start_time, predict_minutes, upstream_flow_dict)
            if ml_prediction:
                self.logger.info(f"Using ML prediction for {segment_id}: t+{predict_minutes}min")
                # Save to cache
                self.prediction_cache[cache_key] = ml_prediction
                return ml_prediction

        # Calculate equivalent PCE flow
        upstream_pce_flow = self._calculate_pce_flow(upstream_flow_dict)
        upstream_flow = upstream_pce_flow * 12  # Convert 5-min flow to hourly flow

        # Get time period factors based on time of day
        time_factors = self._get_time_period_factors(start_time)
        self.logger.debug(f"Time period factors: {time_factors}")

        # Get parameters for this prediction horizon
        base_params = self.horizon_params.get(predict_minutes, {
            'wave_speed': self.wave_speed,
            'critical_density': self.critical_density,
            'alpha': self.alpha,
            'beta': self.beta,
            'historical_weight': 0.4,
            'recency_factor': 0.5
        })

        # Apply time factors to parameters
        sim_params = {
            'wave_speed': base_params.get('wave_speed', self.wave_speed) * time_factors['wave_speed_factor'],
            'critical_density': base_params.get('critical_density', self.critical_density) * time_factors[
                'critical_density_factor'],
            'jam_density': self.jam_density,
            'free_flow_speed': segment['v_free'] * time_factors['free_flow_speed_factor'],
            'alpha': base_params.get('alpha', self.alpha),
            'beta': base_params.get('beta', self.beta),
            'historical_weight': base_params.get('historical_weight', 0.4),
            'recency_factor': base_params.get('recency_factor', 0.5)
        }

        # Apply weather impact
        sim_params = self._apply_weather_impact(sim_params, start_time)

        # Calculate capacity after weather impact
        sim_params['capacity'] = self._calculate_capacity(sim_params)

        # Apply capacity factor from segment bottleneck if present
        sim_params['capacity'] *= segment['capacity_factor']

        self.logger.debug(f"Simulation parameters: wave_speed={sim_params['wave_speed']:.2f}, " +
                          f"critical_density={sim_params['critical_density']:.2f}, " +
                          f"capacity={sim_params['capacity']:.2f}")

        # Estimate initial density from upstream flow
        upstream_speed = sim_params['free_flow_speed'] * 0.8  # Estimate speed
        if upstream_speed > 0:
            initial_density = upstream_flow / (upstream_speed * segment['lanes'])
        else:
            initial_density = sim_params['critical_density'] * 0.5

        # Ensure density is within bounds
        initial_density = min(max(1.0, initial_density), sim_params['jam_density'] * 0.9)

        # Classify initial traffic state
        traffic_state = self._classify_traffic_state(
            initial_density, sim_params['critical_density'], sim_params['jam_density'])

        # Calculate density ratio for parameter adjustment
        density_ratio = self._calculate_density_ratio(
            initial_density, traffic_state, sim_params['critical_density'], sim_params['jam_density'])

        # Adjust parameters based on traffic state and density ratio
        sim_params = self._adjust_parameters_for_traffic_state(sim_params, traffic_state, density_ratio)

        # Create adaptive cell system
        num_cells, cell_lengths = self._create_adaptive_cells(segment, initial_density, traffic_state)

        # Initialize density array for adaptive cells
        density = np.ones(num_cells) * initial_density

        # Set downstream boundary density based on segment type and traffic state
        if segment['type'] >= 3:  # With off-ramp
            downstream_density = sim_params['critical_density'] * 0.6
        else:
            downstream_density = sim_params['critical_density'] * 0.7

        # Adjust downstream density based on traffic state
        if traffic_state == "congested":
            downstream_density *= 1.2
        elif traffic_state == "hypercongested":
            downstream_density *= 1.4

        # Estimate ramp flows with time information for ML enhancement
        on_ramp_flow, off_ramp_flow = self._estimate_ramp_flows(segment, upstream_flow, start_time)

        # Calculate simulation time steps
        sim_seconds = predict_minutes * 60
        time_steps = int(sim_seconds / self.simulation_step)

        self.logger.debug(f"Starting EnCTM simulation with {time_steps} time steps, " +
                          f"{num_cells} cells, traffic state: {traffic_state}")

        # Run simulation
        for t in range(time_steps):
            density_prev = density.copy()

            # Calculate flow at cell boundaries using modified Wu model
            sending_flows = np.zeros(num_cells)
            receiving_flows = np.zeros(num_cells)

            # Calculate sending and receiving flows for each cell
            for i in range(num_cells):
                sending_flows[i] = self._compute_sending_flow(density_prev[i], sim_params) * segment['lanes']
                receiving_flows[i] = self._compute_receiving_flow(density_prev[i], sim_params) * segment['lanes']

            # Detect bottlenecks with enhanced detection
            bottlenecks = self._detect_bottlenecks(segment, density_prev, sending_flows, sim_params)

            # Apply bottleneck effects with spatial influence
            sending_flows, receiving_flows = self._apply_bottleneck_effects(
                sending_flows, receiving_flows, bottlenecks, sim_params, density_prev)

            # Calculate flows between cells
            flow_boundary = np.zeros(num_cells + 1)

            # Upstream boundary flow (inflow to the first cell)
            # Limited by the receiving flow of the first cell
            flow_boundary[0] = min(upstream_flow, receiving_flows[0])

            # Interior cell boundaries (flows between cells)
            for i in range(1, num_cells):
                flow_boundary[i] = min(sending_flows[i - 1], receiving_flows[i])

            # Downstream boundary (outflow from the last cell)
            flow_boundary[num_cells] = sending_flows[-1]

            # Update density in each cell
            for i in range(num_cells):
                # Basic flow update
                inflow = flow_boundary[i]
                outflow = flow_boundary[i + 1]

                # Handle ramp flows
                r_flow = 0  # On-ramp flow
                s_flow = 0  # Off-ramp flow

                # Position ramps appropriately
                mid_cell = num_cells // 2
                if segment['type'] == 2 and i == mid_cell // 2:  # On-ramp in first half
                    r_flow = on_ramp_flow
                elif segment['type'] == 3 and i == mid_cell + mid_cell // 2:  # Off-ramp in second half
                    s_flow = off_ramp_flow
                elif segment['type'] == 4:
                    if i == mid_cell // 2:  # On-ramp
                        r_flow = on_ramp_flow
                    elif i == mid_cell + mid_cell // 2:  # Off-ramp
                        s_flow = off_ramp_flow

                # CTM update equation
                T = self.simulation_step / 3600  # Time step in hours
                L = cell_lengths[i]  # Cell length in km

                density[i] = density_prev[i] + \
                             T / L * (inflow - outflow + r_flow - s_flow) / segment['lanes']

                # Ensure density stays within bounds
                density[i] = min(max(1.0, density[i]), sim_params['jam_density'])

            # Check if traffic state has changed significantly
            new_traffic_state = self._classify_traffic_state(
                np.mean(density), sim_params['critical_density'], sim_params['jam_density'])

            if new_traffic_state != traffic_state:
                # Traffic state transition detected
                traffic_state = new_traffic_state
                density_ratio = self._calculate_density_ratio(
                    np.mean(density), traffic_state, sim_params['critical_density'], sim_params['jam_density'])

                sim_params = self._adjust_parameters_for_traffic_state(sim_params, traffic_state, density_ratio)
                self.logger.debug(f"Traffic state transition to {traffic_state} at step {t}")

        # Calculate final flow at the end of the segment
        final_density = density[-1]
        final_flow = self._compute_sending_flow(final_density, sim_params)

        # Convert back to 5-minute flow
        flow_5min = final_flow / 12

        self.logger.debug(f"Final simulation state: density={final_density:.2f}, flow={flow_5min:.2f}")

        # Distribute flow among vehicle types
        # Use enhanced distribution considering multiple factors:
        # 1. Original upstream distribution
        # 2. Vehicle class characteristics
        # 3. Traffic state
        predicted_flow = {}

        # Calculate total PCE equivalent
        upstream_total_pce = sum(flow * self.vehicle_classes[vt]['pce'] for vt, flow in upstream_flow_dict.items())

        if upstream_total_pce > 0:
            # Calculate vehicle type ratios
            type_ratios = {}
            for veh_type, flow in upstream_flow_dict.items():
                pce = self.vehicle_classes[veh_type]['pce']

                # Base ratio is PCE-weighted contribution to total flow
                base_ratio = (flow * pce) / upstream_total_pce

                # Adjust ratio based on traffic state
                # Larger vehicles more affected by congestion
                if traffic_state in ["congested", "hypercongested"]:
                    flow_smoothing = self.vehicle_classes[veh_type]['flow_smoothing']
                    # Apply more smoothing in congestion (less in free flow)
                    base_ratio *= (1.0 - flow_smoothing * (1.5 if traffic_state == "hypercongested" else 1.0))

                type_ratios[veh_type] = base_ratio

            # Normalize ratios
            total_ratio = sum(type_ratios.values())
            if total_ratio > 0:
                for veh_type in type_ratios:
                    type_ratios[veh_type] /= total_ratio

            # Apply to final flow and convert back from PCE to actual vehicles
            for veh_type, ratio in type_ratios.items():
                pce = self.vehicle_classes[veh_type]['pce']
                predicted_flow[veh_type] = ratio * flow_5min / pce

        # Apply historical pattern influence if available
        if hasattr(self, 'historical_patterns'):
            historical = self._get_historical_pattern(down_node, start_time + pd.Timedelta(minutes=predict_minutes))

            if historical and predicted_flow:
                # Mix predicted flow with historical pattern
                historical_weight = sim_params['historical_weight']
                for veh_type in predicted_flow:
                    if veh_type in historical:
                        # Weighted average with historical pattern
                        predicted_flow[veh_type] = (1 - historical_weight) * predicted_flow[veh_type] + \
                                                   historical_weight * historical[veh_type]

        # Save traffic state for this prediction
        self._save_traffic_state(segment_id, start_time, predict_minutes, traffic_state,
                                 initial_density, final_density, flow_5min)

        self.logger.info(f"Prediction complete: total flow {sum(predicted_flow.values()):.2f} veh/5min")

        # In hybrid mode, store CTM prediction in cache
        if self.hybrid_prediction:
            self.prediction_cache[cache_key] = predicted_flow

        return predicted_flow

    def _save_traffic_state(self, segment_id, start_time, predict_minutes, traffic_state,
                            initial_density, final_density, final_flow):
        """Save traffic state information for analysis"""
        try:
            # Convert numpy.datetime64 to Python datetime if needed
            if isinstance(start_time, np.datetime64):
                start_time_str = pd.Timestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            else:
                start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

            # Create a record for later analysis
            record = {
                'segment_id': segment_id,
                'start_time': start_time_str,
                'predict_minutes': predict_minutes,
                'traffic_state': traffic_state,
                'initial_density': initial_density,
                'final_density': final_density,
                'final_flow': final_flow,
                'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save to CSV for later analysis
            state_file = os.path.join(self.output_dir, 'traffic_states.csv')
            mode = 'a' if os.path.exists(state_file) else 'w'
            header = mode == 'w'

            df = pd.DataFrame([record])
            df.to_csv(state_file, mode=mode, header=header, index=False)
        except Exception as e:
            self.logger.warning(f"Failed to save traffic state information: {e}")

    def _get_time_period_factors(self, time_point):
        try:
            # Convert numpy.datetime64 to Python datetime if needed
            if isinstance(time_point, np.datetime64):
                time_point = pd.Timestamp(time_point).to_pydatetime()

            hour = time_point.hour
            weekday = time_point.weekday()
            is_weekend = weekday >= 5

            # Morning peak (7-9am)
            if 7 <= hour <= 9:
                if not is_weekend:
                    return {
                        'free_flow_speed_factor': 0.88,  # Changed from 0.9
                        'critical_density_factor': 0.93,  # Changed from 0.95
                        'wave_speed_factor': 1.12  # Changed from 1.1
                    }
                else:
                    return {
                        'free_flow_speed_factor': 0.95,
                        'critical_density_factor': 0.98,
                        'wave_speed_factor': 1.05
                    }

            # Evening peak (5-7pm)
            elif 17 <= hour <= 19:
                if not is_weekend:
                    return {
                        'free_flow_speed_factor': 0.83,  # Changed from 0.85
                        'critical_density_factor': 0.88,  # Changed from 0.9
                        'wave_speed_factor': 1.23  # Changed from 1.2
                    }
                else:
                    return {
                        'free_flow_speed_factor': 0.9,
                        'critical_density_factor': 0.95,
                        'wave_speed_factor': 1.1
                    }

            # Daytime off-peak (9am-5pm)
            elif 9 < hour < 17:
                return {
                    'free_flow_speed_factor': 1.0,
                    'critical_density_factor': 1.0,
                    'wave_speed_factor': 1.0
                }

            # Evening transition (7pm-10pm)
            elif 19 <= hour <= 22:
                return {
                    'free_flow_speed_factor': 1.02,
                    'critical_density_factor': 1.02,
                    'wave_speed_factor': 0.95
                }

            # Nighttime (10pm-7am)
            else:
                return {
                    'free_flow_speed_factor': 1.07,  # Changed from 1.05
                    'critical_density_factor': 1.08,  # Changed from 1.05
                    'wave_speed_factor': 0.88  # Changed from 0.9
                }
        except:
            return {
                'free_flow_speed_factor': 1.0,
                'critical_density_factor': 1.0,
                'wave_speed_factor': 1.0
            }

    def validate(self, predict_minutes_list=[5, 10, 15, 30]):
        self.logger.info(f"Starting Enhanced CTM model validation...")
        self.logger.info(f"Validation for prediction horizons: {predict_minutes_list} minutes")

        results = []

        for segment_id, segment in self.segments.items():
            up_node = segment['up_node']
            down_node = segment['down_node']

            if up_node not in self.flow_data or down_node not in self.flow_data:
                continue

            self.logger.info(f"Validating segment {segment_id}: {up_node} -> {down_node}")

            time_points = self.flow_data[up_node]['time'].unique()
            if len(time_points) > 50:
                time_points = np.random.choice(time_points, 50, replace=False)
                self.logger.info(f"Using random sample of 50 time points for validation")
            else:
                self.logger.info(f"Using all {len(time_points)} available time points for validation")

            for start_time in time_points:
                for predict_minutes in predict_minutes_list:
                    end_time = start_time + pd.Timedelta(minutes=predict_minutes)

                    predicted_flow = self.predict_flow(segment_id, start_time, predict_minutes)
                    if not predicted_flow:
                        continue

                    actual_flow = self._get_flow_by_type(down_node, end_time)
                    if not actual_flow:
                        continue

                    for veh_type in predicted_flow:
                        if veh_type in actual_flow:
                            results.append({
                                'segment_id': segment_id,
                                'start_time': start_time,
                                'end_time': end_time,
                                'predict_minutes': predict_minutes,
                                'vehicle_type': veh_type,
                                'predicted': predicted_flow[veh_type],
                                'actual': actual_flow[veh_type],
                                'segment_type': segment['type']
                            })

        if results:
            results_df = pd.DataFrame(results)

            output_file = os.path.join(self.output_dir, 'validation_results.csv')
            results_df.to_csv(output_file, index=False)
            self.logger.info(f"Validation results saved to: {output_file}")
            self.logger.info(f"Total validation records: {len(results_df)}")

            self._calculate_metrics(results_df)

            # Train ML models on validation data if ML integration is enabled
            if self.ml_integration and self.sklearn_available and len(results_df) > 100:
                self._train_ml_models(results_df)

            return results_df
        else:
            self.logger.warning("No validation results obtained")
            return pd.DataFrame()

    def _train_ml_models(self, validation_df):
        """Train ML models using validation data"""
        if not self.ml_integration or not self.sklearn_available:
            return

        try:
            self.logger.info("Training ML models on validation data...")

            # Prepare training data
            features_list = []
            flow_targets = []

            # Group by prediction
            for _, row in validation_df.iterrows():
                segment_id = row['segment_id']
                start_time = row['start_time']
                predict_minutes = row['predict_minutes']
                veh_type = row['vehicle_type']
                actual_flow = row['actual']

                # Get upstream flow
                segment = self.segments[segment_id]
                up_node = segment['up_node']
                upstream_flow_dict = self._get_flow_by_type(up_node, start_time)

                if not upstream_flow_dict:
                    continue

                # Extract features
                features = self._get_ml_features(segment_id, start_time, upstream_flow_dict)
                features['predict_minutes'] = predict_minutes
                features['vehicle_type'] = veh_type

                features_list.append(features)
                flow_targets.append(actual_flow)

            if len(features_list) < 50:
                self.logger.warning("Insufficient data for ML model training")
                return

            # Convert to DataFrame and normalize
            feature_df = pd.DataFrame(features_list)

            # Process categorical columns
            feature_df = pd.get_dummies(feature_df, columns=['vehicle_type'])

            # Train flow predictor model
            X = feature_df.fillna(0)
            y = np.array(flow_targets)

            # Update flow predictor model
            flow_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

            # Fit model
            flow_predictor.fit(X, y)

            # Save trained model
            self.ml_models['flow_predictor'] = flow_predictor

            # Save models to disk
            model_dir = os.path.join(self.output_dir, 'ml_models')
            with open(os.path.join(model_dir, "flow_predictor.pkl"), 'wb') as f:
                pickle.dump(flow_predictor, f)

            self.logger.info(f"ML models trained successfully on {len(X)} samples")
            self.ml_models_trained = True

        except Exception as e:
            self.logger.warning(f"ML model training failed: {e}")

    def _calculate_metrics(self, results_df):
        self.logger.info(f"Calculating Enhanced CTM model performance metrics:")

        self.logger.info("\nPerformance metrics by prediction horizon:")
        time_metrics = []
        for minutes, group in results_df.groupby('predict_minutes'):
            y_true = group['actual']
            y_pred = group['predicted']

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            self.logger.info(
                f"{minutes}-minute prediction: samples={len(group)}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R^2={r2:.4f}")
            time_metrics.append({
                'predict_minutes': minutes,
                'sample_count': len(group),
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            })

        self.logger.info("\nPerformance metrics by vehicle type:")
        vehicle_metrics = []
        for veh_type, group in results_df.groupby('vehicle_type'):
            y_true = group['actual']
            y_pred = group['predicted']

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            self.logger.info(
                f"Vehicle type {veh_type}: samples={len(group)}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R^2={r2:.4f}")
            vehicle_metrics.append({
                'vehicle_type': veh_type,
                'sample_count': len(group),
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            })

        if self.aggregate_by == 'time':
            avg_mae = np.mean([m['MAE'] for m in time_metrics])
            avg_rmse = np.mean([m['RMSE'] for m in time_metrics])
            avg_mape = np.mean([m['MAPE'] for m in time_metrics if not np.isnan(m['MAPE'])])
            avg_r2 = np.mean([m['R2'] for m in time_metrics])

            total_samples = sum(m['sample_count'] for m in time_metrics)

            self.logger.info("\nOverall performance metrics (averaged by time horizon):")
            self.logger.info(
                f"Samples={total_samples}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, MAPE={avg_mape:.2f}%, R^2={avg_r2:.4f}")

            overall_metrics = {
                'sample_count': total_samples,
                'MAE': avg_mae,
                'RMSE': avg_rmse,
                'MAPE': avg_mape,
                'R2': avg_r2,
                'aggregation_method': 'time_average'
            }
        else:
            avg_mae = np.mean([m['MAE'] for m in vehicle_metrics])
            avg_rmse = np.mean([m['RMSE'] for m in vehicle_metrics])
            avg_mape = np.mean([m['MAPE'] for m in vehicle_metrics if not np.isnan(m['MAPE'])])
            avg_r2 = np.mean([m['R2'] for m in vehicle_metrics])

            total_samples = sum(m['sample_count'] for m in vehicle_metrics)

            self.logger.info("\nOverall performance metrics (averaged by vehicle type):")
            self.logger.info(
                f"Samples={total_samples}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, MAPE={avg_mape:.4f}%, R^2={avg_r2:.4f}")

            overall_metrics = {
                'sample_count': total_samples,
                'MAE': avg_mae,
                'RMSE': avg_rmse,
                'MAPE': avg_mape,
                'R2': avg_r2,
                'aggregation_method': 'vehicle_average'
            }

        pd.DataFrame([overall_metrics]).to_csv(
            os.path.join(self.output_dir, 'overall_metrics.csv'), index=False)
        self.logger.info("Overall metrics saved to 'overall_metrics.csv'")

    def calibrate(self, predict_minutes_list=[5, 10, 15, 30], sample_size=20):
        self.logger.info("Starting Enhanced CTM model parameter calibration...")

        if self.calibrated:
            self.logger.info("Model already calibrated, skipping calibration process")
            return

        valid_segments = []
        for segment_id, segment in self.segments.items():
            up_node = segment['up_node']
            down_node = segment['down_node']
            if up_node in self.flow_data and down_node in self.flow_data:
                valid_segments.append(segment_id)

        if not valid_segments:
            self.logger.warning("No valid segments found for calibration")
            return

        calibration_results = {}

        for predict_minutes in predict_minutes_list:
            self.logger.info(f"Calibrating parameters for {predict_minutes}-minute prediction horizon...")

            calibration_samples = []
            for segment_id in valid_segments:
                segment = self.segments[segment_id]
                up_node = segment['up_node']
                down_node = segment['down_node']

                time_points = self.flow_data[up_node]['time'].unique()
                if len(time_points) > sample_size:
                    selected_times = np.random.choice(time_points, sample_size, replace=False)
                else:
                    selected_times = time_points

                for start_time in selected_times:
                    end_time = start_time + pd.Timedelta(minutes=predict_minutes)

                    upstream_flow = self._get_flow_by_type(up_node, start_time)
                    if not upstream_flow:
                        continue

                    actual_flow = self._get_flow_by_type(down_node, end_time)
                    if not actual_flow:
                        continue

                    total_upstream = sum(upstream_flow.values())
                    total_actual = sum(actual_flow.values())

                    calibration_samples.append({
                        'segment_id': segment_id,
                        'start_time': start_time,
                        'total_upstream': total_upstream,
                        'total_actual': total_actual
                    })

            if not calibration_samples:
                self.logger.warning(f"No calibration samples found for {predict_minutes}-minute prediction")
                continue

            self.logger.info(f"Collected {len(calibration_samples)} calibration samples")

            def objective(params):
                wave_speed, critical_density, alpha, beta = params

                temp_params = {
                    'wave_speed': wave_speed,
                    'critical_density': critical_density,
                    'alpha': alpha,
                    'beta': beta
                }

                errors = []
                for sample in calibration_samples:
                    segment_id = sample['segment_id']
                    start_time = sample['start_time']
                    total_actual = sample['total_actual']

                    old_params = self.horizon_params.get(predict_minutes, {})

                    self.horizon_params[predict_minutes] = temp_params

                    predicted_flow = self.predict_flow(segment_id, start_time, predict_minutes)

                    if old_params:
                        self.horizon_params[predict_minutes] = old_params
                    else:
                        del self.horizon_params[predict_minutes]

                    if predicted_flow:
                        total_predicted = sum(predicted_flow.values())
                        if total_actual > 0:
                            rel_error = abs(total_predicted - total_actual) / total_actual
                            errors.append(rel_error)

                if errors:
                    return sum(errors) / len(errors)
                else:
                    return 1.0

            initial_params = [
                self.wave_speed,  # wave_speed
                self.critical_density,  # critical_density
                self.alpha,  # alpha
                self.beta  # beta
            ]

            bounds = [
                (10.0, 40.0),  # wave_speed: 10-40 km/h
                (15.0, 40.0),  # critical_density: 15-40 veh/km/lane
                (1.0, 4.0),  # alpha: 1.0-4.0
                (0.1, 1.0)  # beta: 0.1-1.0
            ]

            self.logger.info(f"Starting optimization with initial params: " +
                             f"wave_speed={initial_params[0]}, " +
                             f"critical_density={initial_params[1]}, " +
                             f"alpha={initial_params[2]}, " +
                             f"beta={initial_params[3]}")

            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-3, 'maxiter': 25}  # Increased from 20 to 25
            )

            if result.success:
                wave_speed, critical_density, alpha, beta = result.x

                # Additional calibration of historical_weight and recency_factor
                # based on prediction horizon
                historical_weight = min(0.6, 0.3 + 0.1 * (predict_minutes / 10))
                recency_factor = max(0.3, 0.7 - 0.1 * (predict_minutes / 10))

                self.horizon_params[predict_minutes] = {
                    'wave_speed': wave_speed,
                    'critical_density': critical_density,
                    'alpha': alpha,
                    'beta': beta,
                    'historical_weight': historical_weight,
                    'recency_factor': recency_factor
                }

                calibration_results[predict_minutes] = {
                    'wave_speed': wave_speed,
                    'critical_density': critical_density,
                    'alpha': alpha,
                    'beta': beta,
                    'historical_weight': historical_weight,
                    'recency_factor': recency_factor,
                    'error': result.fun
                }

                self.logger.info(
                    f"Calibration successful for {predict_minutes}-minute prediction: " +
                    f"wave_speed={wave_speed:.2f}km/h, critical_density={critical_density:.2f}veh/km/lane, " +
                    f"alpha={alpha:.2f}, beta={beta:.2f}, error={result.fun:.4f}")
            else:
                self.logger.warning(f"Calibration failed for {predict_minutes}-minute prediction")
                self.logger.warning(f"Optimization message: {result.message}")

        self.calibrated = True

        # Save calibration parameters to file
        try:
            with open(os.path.join(self.output_dir, 'calibration_parameters.json'), 'w') as f:
                json.dump(calibration_results, f, indent=4)
            self.logger.info("Calibration parameters saved to 'calibration_parameters.json'")
        except Exception as e:
            self.logger.warning(f"Failed to save calibration parameters: {e}")

        self.logger.info("Enhanced CTM model parameter calibration complete")


def main():
    parser = argparse.ArgumentParser(description="Enhanced CTM Traffic Flow Model")
    parser.add_argument("--road_data", type=str, default="./ETC_data_example/roadETC.csv",
                        help="Path to road segment data file")
    parser.add_argument("--flow_dir", type=str, default="./ETC_data_example/flow",
                        help="Directory containing traffic flow data")
    parser.add_argument("--output_dir", type=str, default="./enctm_results",
                        help="Output directory for results")
    parser.add_argument("--weather_data", type=str, default=None,
                        help="Path to weather data file (optional)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Shape parameter for modified Wu fundamental diagram")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="Transition smoothness parameter")
    parser.add_argument("--adaptive_cells", action="store_true", default=True,
                        help="Enable adaptive cell system")
    parser.add_argument("--ml_integration", action="store_true", default=False,
                        help="Enable machine learning integration")
    parser.add_argument("--hybrid_prediction", action="store_true", default=False,
                        help="Enable hybrid CTM-ML prediction")
    parser.add_argument("--predict_minutes", type=int, nargs="+", default=[5, 10, 15],
                        help="Prediction time windows (minutes)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Automatically calibrate model parameters")
    parser.add_argument("--visualization", action="store_true",
                        help="Enable result visualization")
    parser.add_argument("--aggregate_by", type=str, choices=['time', 'vehicle'], default='time',
                        help="Aggregation method for overall metrics: by time window or by vehicle type")

    args = parser.parse_args()

    model = EnhancedCTMModel(
        road_data_path=args.road_data,
        traffic_flow_dir=args.flow_dir,
        output_dir=args.output_dir,
        weather_data_path=args.weather_data,
        alpha=args.alpha,
        beta=args.beta,
        adaptive_cells=args.adaptive_cells,
        ml_integration=args.ml_integration,
        hybrid_prediction=args.hybrid_prediction,
        aggregate_by=args.aggregate_by
    )

    if args.calibrate:
        model.calibrate(predict_minutes_list=args.predict_minutes)

    validation_results = model.validate(predict_minutes_list=args.predict_minutes)

    print(f"Enhanced CTM model validation complete: {len(validation_results)} records")
    print(f"Results saved to: {args.output_dir}")
    print(f"Metrics aggregated by {args.aggregate_by}")

    if args.visualization and len(validation_results) > 0:
        try:
            import matplotlib.pyplot as plt

            # Create visualization directory
            vis_dir = os.path.join(args.output_dir, 'visualization')
            os.makedirs(vis_dir, exist_ok=True)

            # Visualization 1: Predicted vs Actual by horizon
            df = validation_results
            for minutes in args.predict_minutes:
                subset = df[df['predict_minutes'] == minutes]
                if len(subset) > 0:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(subset['actual'], subset['predicted'], alpha=0.5)
                    max_val = max(subset['actual'].max(), subset['predicted'].max()) * 1.1
                    plt.plot([0, max_val], [0, max_val], 'r--')
                    plt.xlabel('Actual Flow')
                    plt.ylabel('Predicted Flow')
                    plt.title(f'{minutes}-minute Prediction: Actual vs Predicted')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(vis_dir, f'prediction_{minutes}min.png'))
                    plt.close()

            # Visualization 2: Error distribution
            df['abs_error'] = abs(df['predicted'] - df['actual'])
            df['rel_error'] = df['abs_error'] / df['actual'].clip(lower=1)

            plt.figure(figsize=(12, 6))
            plt.hist(df['rel_error'].clip(upper=1), bins=50, alpha=0.7)
            plt.xlabel('Relative Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'error_distribution.png'))
            plt.close()

            # Visualization 3: Error by vehicle type
            plt.figure(figsize=(10, 6))
            vehicle_types = df['vehicle_type'].unique()
            errors_by_type = [df[df['vehicle_type'] == vt]['rel_error'].mean() for vt in vehicle_types]

            plt.bar(vehicle_types, errors_by_type)
            plt.xlabel('Vehicle Type')
            plt.ylabel('Average Relative Error')
            plt.title('Prediction Error by Vehicle Type')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'error_by_vehicle_type.png'))
            plt.close()

            # Visualization 4: Performance by time of day
            if 'start_time' in df.columns:
                df['hour'] = pd.to_datetime(df['start_time']).dt.hour
                hourly_errors = df.groupby('hour')['rel_error'].mean()

                plt.figure(figsize=(12, 6))
                hourly_errors.plot(kind='bar')
                plt.xlabel('Hour of Day')
                plt.ylabel('Average Relative Error')
                plt.title('Prediction Error by Time of Day')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(vis_dir, 'error_by_hour.png'))
                plt.close()

            print(f"Visualizations saved to: {vis_dir}")

        except Exception as e:
            print(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()