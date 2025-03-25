import numpy as np
import pandas as pd
import os
import logging
import argparse
import datetime
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize


class PILWRModel:
    """
    PI-LWR (Predictive Incremental Lighthill-Whitham-Richards) macroscopic traffic flow model
    Based on first-order traffic flow conservation equation, using fundamental diagram relationships
    """

    def __init__(self,
                 road_data_path: str,
                 traffic_flow_dir: str,
                 output_dir: str = "./pi_lwr_results",
                 simulation_step: float = 5.0,  # PI-LWR typically allows smaller time steps
                 critical_density: float = 30.0,  # Critical density (veh/km/lane)
                 jam_density: float = 180.0,  # Jam density (veh/km/lane)
                 free_flow_speed: float = 110.0,  # Free flow speed (km/h)
                 default_lanes: int = 3,  # Default number of lanes
                 enhanced_mode: bool = False,  # Whether to use enhanced version 2
                 aggregate_by: str = 'time',  # Aggregation method: 'time' or 'vehicle'
                 fd_type: str = 'triangular'):  # Fundamental diagram type: 'triangular' or 'greenshield'
        """Initialize PI-LWR model"""
        # Data paths
        self.road_data_path = road_data_path
        self.traffic_flow_dir = traffic_flow_dir
        self.output_dir = output_dir
        self.enhanced_mode = enhanced_mode
        self.aggregate_by = aggregate_by
        self.fd_type = fd_type  # Fundamental diagram type

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Model parameters
        self.simulation_step = simulation_step
        self.critical_density = critical_density
        self.jam_density = jam_density
        self.free_flow_speed = free_flow_speed
        self.default_lanes = default_lanes

        # Calculate backward wave speed (for triangular fundamental diagram)
        if fd_type == 'triangular':
            # Backward wave speed calculation for triangular fundamental diagram
            capacity = free_flow_speed * critical_density
            self.backward_wave_speed = capacity / (jam_density - critical_density)
        else:
            # Greenshield model has no explicit backward wave speed
            self.backward_wave_speed = None

        # Set up logging
        self._setup_logging()

        # Display model version information
        model_type = "Enhanced Version 2" if enhanced_mode else "Enhanced Version 1"
        self.logger.info(f"Initializing {model_type} PI-LWR model")
        self.logger.info(f"Fundamental diagram type: {fd_type}")
        self.logger.info(
            f"Core parameters: Free flow speed={free_flow_speed}km/h, Critical density={critical_density}veh/km/lane, Jam density={jam_density}veh/km/lane")
        if fd_type == 'triangular':
            self.logger.info(f"Backward wave speed: {self.backward_wave_speed:.2f}km/h")
        self.logger.info(f"Aggregation method: {'Time-based' if aggregate_by == 'time' else 'Vehicle-based'}")

        # Load data
        self.road_data = self._load_road_data()
        self.flow_data = self._load_flow_data()
        self.segments = self._prepare_segments()

        # Enhanced version 2 specific parameters
        if enhanced_mode:
            # Vehicle PCE values (Passenger Car Equivalent)
            self.vehicle_pce = {
                'B1': 1.0,  # Small passenger car
                'B2': 1.5,  # Medium passenger car
                'B3': 2.0,  # Large passenger car/bus
                'T1': 1.5,  # Small truck
                'T2': 2.5,  # Medium truck
                'T3': 3.5   # Large truck
            }

            # Parameters for different prediction horizons
            self.horizon_params = {}
            self._init_horizon_params()

            # Calibration flag
            self.calibrated = False

        self.logger.info("PI-LWR model initialization complete")

    def _setup_logging(self):
        """Set up logging"""
        model_type = "enhanced_v2" if self.enhanced_mode else "enhanced_v1"
        self.logger = logging.getLogger(f'pi_lwr_{model_type}')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers = []

        # Add console handler with utf-8 encoding
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # Add file handler with explicit UTF-8 encoding
        log_file = os.path.join(self.output_dir, f'pi_lwr_{model_type}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _load_road_data(self):
        """Load road segment data"""
        self.logger.info(f"Loading road segment data: {self.road_data_path}")
        try:
            road_data = pd.read_csv(self.road_data_path)
            self.logger.info(f"Successfully loaded {len(road_data)} road segments")
            return road_data
        except Exception as e:
            self.logger.error(f"Failed to load road data: {e}")
            return pd.DataFrame()

    def _load_flow_data(self):
        """Load traffic flow data"""
        self.logger.info(f"Loading traffic flow data: {self.traffic_flow_dir}")
        flow_data = {}

        # Get all gantry IDs
        gantry_ids = set()
        for _, row in self.road_data.iterrows():
            gantry_ids.add(row['up_node'])
            gantry_ids.add(row['down_node'])

        # Load flow data for each gantry
        for gantry_id in gantry_ids:
            flow_file = os.path.join(self.traffic_flow_dir, f"trafficflow_{gantry_id}.csv")
            if os.path.exists(flow_file):
                try:
                    df = pd.read_csv(flow_file)
                    # Convert time format
                    df['time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M')
                    flow_data[gantry_id] = df
                except Exception as e:
                    self.logger.warning(f"Failed to load flow data for gantry {gantry_id}: {e}")

        self.logger.info(f"Successfully loaded flow data for {len(flow_data)} gantries")
        return flow_data

    def _prepare_segments(self):
        """Prepare road segment information"""
        segments = {}

        for _, row in self.road_data.iterrows():
            segment_id = row['id']

            # Basic segment information
            segment = {
                'id': segment_id,
                'up_node': row['up_node'],
                'down_node': row['down_node'],
                'length': row['length'],  # km
                'lanes': row.get('lanes', self.default_lanes),
                'speed_limit': row.get('speed_limit', 100),  # km/h
                'type': int(row.get('type', 1))  # Default type 1 (no ramp)
            }

            # Set free flow speed
            segment['v_free'] = min(segment['speed_limit'], self.free_flow_speed)

            # Set up discretization - PI-LWR typically uses finer grid
            segment_length = 0.25  # km
            num_segments = max(4, int(segment['length'] / segment_length))
            segment['num_segments'] = num_segments
            segment['segment_length'] = segment['length'] / num_segments

            # Add to dictionary
            segments[segment_id] = segment

        self.logger.info(f"Prepared information for {len(segments)} road segments")
        return segments

    def _get_total_flow(self, gantry_id, time_point):
        """Get total flow for specified gantry and time"""
        if gantry_id not in self.flow_data:
            return 0

        df = self.flow_data[gantry_id]

        # Find closest time point
        df['time_diff'] = abs(df['time'] - time_point)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        df.drop('time_diff', axis=1, inplace=True)

        # Calculate total flow (all vehicle types)
        vehicle_types = ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']
        total_flow = sum(closest_row.get(vt, 0) for vt in vehicle_types)

        return total_flow

    def _get_flow_by_type(self, gantry_id, time_point):
        """Get flow by vehicle type for specified gantry and time"""
        if gantry_id not in self.flow_data:
            return {}

        df = self.flow_data[gantry_id]

        # Find closest time point
        df['time_diff'] = abs(df['time'] - time_point)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        df.drop('time_diff', axis=1, inplace=True)

        # Extract flow by vehicle type
        flow_dict = {}
        vehicle_types = ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']
        for vt in vehicle_types:
            if vt in closest_row:
                flow_dict[vt] = closest_row[vt]

        return flow_dict

    def _compute_flux(self, density, v_free, rho_crit, rho_jam):
        """
        Compute flow-density relationship (fundamental diagram)

        Parameters:
        density: Density value (veh/km/lane)
        v_free: Free flow speed (km/h)
        rho_crit: Critical density (veh/km/lane)
        rho_jam: Jam density (veh/km/lane)

        Returns:
        Flow (veh/h/lane)
        """
        if density <= 0:
            return 0

        if self.fd_type == 'triangular':
            # Triangular fundamental diagram
            if density <= rho_crit:
                # Free flow region
                return density * v_free
            else:
                # Congested region
                capacity = v_free * rho_crit
                return capacity - self.backward_wave_speed * (density - rho_crit)
        else:
            # Greenshield model (parabolic fundamental diagram)
            return v_free * density * (1 - density / rho_jam)

    def _compute_speed(self, density, v_free, rho_crit, rho_jam):
        """Compute speed-density relationship"""
        if density <= 0:
            return v_free

        if self.fd_type == 'triangular':
            # Speed-density relationship for triangular fundamental diagram
            if density <= rho_crit:
                return v_free
            elif density < rho_jam:
                # Speed reduction in congested region
                capacity = v_free * rho_crit
                return capacity / density
            else:
                return 0
        else:
            # Speed-density relationship for Greenshield model
            return v_free * (1 - density / rho_jam)

    def _init_horizon_params(self):
        """Initialize parameters for different prediction horizons (Enhanced Version 2 only)"""
        # 5-minute prediction parameters
        self.horizon_params[5] = {
            'rho_crit': self.critical_density,
            'v_free': self.free_flow_speed,
            'fd_type': self.fd_type
        }

        # 10-minute prediction parameters
        self.horizon_params[10] = {
            'rho_crit': self.critical_density * 0.95,  # Slightly lower critical density
            'v_free': self.free_flow_speed * 0.98,    # Slightly lower free flow speed
            'fd_type': self.fd_type
        }

        # 15-minute prediction parameters
        self.horizon_params[15] = {
            'rho_crit': self.critical_density * 0.9,  # Lower critical density
            'v_free': self.free_flow_speed * 0.95,   # Lower free flow speed
            'fd_type': self.fd_type
        }

    def _get_time_period_factors(self, time_point):
        """Get adjustment factors based on time period (Enhanced Version 2 only)"""
        try:
            hour = time_point.hour
            weekday = time_point.weekday()
            is_weekend = weekday >= 5

            # Morning peak (7-9 AM)
            if 7 <= hour <= 9:
                if not is_weekend:
                    return {
                        'speed_factor': 0.9,  # Speed reduction
                        'density_factor': 1.1  # Density increase
                    }
                else:
                    return {
                        'speed_factor': 0.95,
                        'density_factor': 1.05
                    }

            # Evening peak (5-7 PM)
            elif 17 <= hour <= 19:
                if not is_weekend:
                    return {
                        'speed_factor': 0.85,  # Significant speed reduction
                        'density_factor': 1.15  # Significant density increase
                    }
                else:
                    return {
                        'speed_factor': 0.9,
                        'density_factor': 1.1
                    }

            # Daytime off-peak (9 AM - 5 PM)
            elif 9 < hour < 17:
                return {
                    'speed_factor': 1.0,
                    'density_factor': 1.0
                }

            # Nighttime (7 PM - 7 AM)
            else:
                return {
                    'speed_factor': 1.05,  # Slightly higher speeds
                    'density_factor': 0.9   # Lower density
                }
        except:
            return {
                'speed_factor': 1.0,
                'density_factor': 1.0
            }

    def _calculate_pce_flow(self, flow_dict):
        """Calculate PCE-adjusted flow (Enhanced Version 2 only)"""
        if not self.enhanced_mode:
            return sum(flow_dict.values())

        pce_flow = 0
        for veh_type, flow in flow_dict.items():
            pce = self.vehicle_pce.get(veh_type, 1.0)
            pce_flow += flow * pce
        return pce_flow

    def _godunov_lwr(self, initial_density, segment, boundary_flow, time_steps,
                     v_free, rho_crit, rho_jam, fd_type=None):
        """
        Solve LWR equation using Godunov scheme

        Parameters:
        initial_density: Initial density array
        segment: Segment information
        boundary_flow: Upstream inflow (veh/h)
        time_steps: Number of simulation steps
        v_free: Free flow speed
        rho_crit: Critical density
        rho_jam: Jam density
        fd_type: Fundamental diagram type

        Returns:
        Density and flow history
        """
        # Record original fundamental diagram type
        original_fd_type = self.fd_type
        if fd_type:
            self.fd_type = fd_type

        # Spatial step
        dx = segment['segment_length']  # km
        # Time step
        dt = self.simulation_step / 3600  # hours

        # CFL condition check
        cfl = dt / dx * v_free
        if cfl > 1.0:
            self.logger.warning(f"CFL condition not satisfied: CFL = {cfl:.2f} > 1.0")
            # Automatically adjust time step to satisfy CFL
            dt = 0.9 * dx / v_free
            time_steps = int(time_steps * self.simulation_step / (dt * 3600))
            self.logger.info(f"Adjusted time step to {dt * 3600:.2f} seconds, {time_steps} steps total")

        num_segments = segment['num_segments']
        lanes = segment['lanes']

        # Set boundary conditions
        upstream_flow_per_lane = boundary_flow / lanes  # veh/h/lane

        # Initialize history arrays
        density_history = np.zeros((time_steps + 1, num_segments))
        flow_history = np.zeros((time_steps + 1, num_segments))

        # Set initial conditions
        density_history[0] = initial_density
        for i in range(num_segments):
            flow_history[0, i] = self._compute_flux(density_history[0, i], v_free, rho_crit, rho_jam) * lanes

        # Solve LWR equation using Godunov scheme
        for t in range(time_steps):
            # Calculate numerical fluxes
            numerical_flux = np.zeros(num_segments + 1)

            # Upstream boundary - direct flow input
            if upstream_flow_per_lane <= v_free * rho_crit:
                # Free flow
                upstream_density = upstream_flow_per_lane / v_free
                numerical_flux[0] = upstream_flow_per_lane
            else:
                # Flow exceeding capacity is limited to capacity
                capacity = v_free * rho_crit
                numerical_flux[0] = capacity

            # Numerical fluxes for interior cells
            for i in range(1, num_segments):
                rho_left = density_history[t, i - 1]
                rho_right = density_history[t, i]

                # Godunov flux: minimum flow between left and right states
                flux_left = self._compute_flux(rho_left, v_free, rho_crit, rho_jam)
                flux_right = self._compute_flux(rho_right, v_free, rho_crit, rho_jam)

                # Determine wave direction and type
                if rho_left <= rho_right:
                    # Check for shock or rarefaction wave
                    if rho_left <= rho_crit and rho_right >= rho_crit:
                        # Crossing critical density, take capacity
                        numerical_flux[i] = v_free * rho_crit
                    else:
                        # Take minimum flow for conservation
                        numerical_flux[i] = min(flux_left, flux_right)
                else:
                    # Take maximum flow
                    numerical_flux[i] = max(flux_left, flux_right)

            # Downstream boundary - free outflow
            numerical_flux[num_segments] = self._compute_flux(density_history[t, -1], v_free, rho_crit, rho_jam)

            # Update densities
            for i in range(num_segments):
                flux_in = numerical_flux[i]
                flux_out = numerical_flux[i + 1]
                density_history[t + 1, i] = density_history[t, i] + dt / dx * (flux_in - flux_out)
                # Ensure density is non-negative and doesn't exceed jam density
                density_history[t + 1, i] = min(max(0, density_history[t + 1, i]), rho_jam)

                # Update flows
                flow_history[t + 1, i] = self._compute_flux(density_history[t + 1, i], v_free, rho_crit,
                                                          rho_jam) * lanes

        # Restore original fundamental diagram type
        self.fd_type = original_fd_type

        return density_history, flow_history

    def predict_flow(self, segment_id, start_time, predict_minutes=5):
        """
        Predict future flow using PI-LWR model

        Parameters:
        segment_id: Segment ID
        start_time: Start time
        predict_minutes: Prediction horizon (minutes)

        Returns:
        Dictionary of predicted flows
        """
        if segment_id not in self.segments:
            self.logger.error(f"Segment ID not found: {segment_id}")
            return {}

        segment = self.segments[segment_id]
        up_node = segment['up_node']
        down_node = segment['down_node']

        # Get upstream flow
        upstream_flow_dict = self._get_flow_by_type(up_node, start_time)
        if not upstream_flow_dict:
            self.logger.warning(f"No flow data found for upstream gantry {up_node} at {start_time}")
            return {}

        # Calculate total flow (considering PCE)
        if self.enhanced_mode:
            upstream_total = self._calculate_pce_flow(upstream_flow_dict)
        else:
            upstream_total = sum(upstream_flow_dict.values())

        upstream_flow = upstream_total * 12  # Convert 5-min flow to hourly rate

        # Get model parameters
        if self.enhanced_mode and predict_minutes in self.horizon_params:
            # Use optimized parameters for different prediction horizons
            params = self.horizon_params[predict_minutes]
            v_free = params['v_free']
            rho_crit = params['rho_crit']
            fd_type = params.get('fd_type', self.fd_type)

            # Consider time period factors (Enhanced Version 2 only)
            time_factors = self._get_time_period_factors(start_time)
            v_free *= time_factors['speed_factor']
            rho_crit *= time_factors['density_factor']
        else:
            # Version 1 uses fixed parameters
            v_free = segment['v_free']
            rho_crit = self.critical_density
            fd_type = self.fd_type

        rho_jam = self.jam_density

        # Estimate initial density based on fundamental diagram
        if upstream_flow / segment['lanes'] <= v_free * rho_crit:
            # Free flow region
            initial_density = upstream_flow / (v_free * segment['lanes'])
        else:
            # Congested region - assume near critical density
            initial_density = rho_crit * 1.05

        # Ensure initial density is in valid range
        initial_density = min(max(1.0, initial_density), rho_jam * 0.9)

        # Set initial state
        num_segments = segment['num_segments']
        initial_densities = np.ones(num_segments) * initial_density

        # Consider ramps (Enhanced Version 2 only)
        if self.enhanced_mode and segment['type'] in [2, 3, 4]:
            # Ramp handling
            mid_segment = num_segments // 2
            if segment['type'] == 2:  # On-ramp
                # On-ramp in first half, slightly increase density
                initial_densities[mid_segment // 2:mid_segment] *= 1.1
            elif segment['type'] == 3:  # Off-ramp
                # Off-ramp in second half, slightly decrease density
                initial_densities[mid_segment:] *= 0.9
            elif segment['type'] == 4:  # On-ramp and off-ramp
                # On-ramp in first half increases density, off-ramp in second half decreases it
                initial_densities[mid_segment // 2:mid_segment] *= 1.1
                initial_densities[mid_segment:] *= 0.9

        # Calculate required simulation steps
        sim_seconds = predict_minutes * 60
        time_steps = int(sim_seconds / self.simulation_step)

        # Solve LWR equation using Godunov method
        density_history, flow_history = self._godunov_lwr(
            initial_densities,
            segment,
            upstream_flow,
            time_steps,
            v_free,
            rho_crit,
            rho_jam,
            fd_type
        )

        # Get terminal flow
        predicted_flow = flow_history[-1, -1]  # veh/h

        # Convert back to 5-minute flow
        flow_5min = predicted_flow / 12

        # Distribute predicted flow by vehicle type based on upstream proportions
        upstream_total_orig = sum(upstream_flow_dict.values())
        result_flow = {}

        if upstream_total_orig > 0:
            for veh_type, flow in upstream_flow_dict.items():
                # Version 1: direct proportional distribution
                if not self.enhanced_mode:
                    ratio = flow / upstream_total_orig
                    result_flow[veh_type] = ratio * flow_5min
                else:
                    # Version 2: consider PCE and adjust ratios for small vs large vehicles
                    # Large vehicles have slightly lower throughput in congestion
                    pce = self.vehicle_pce.get(veh_type, 1.0)
                    ratio = flow / upstream_total_orig

                    # Adjust ratio based on density level
                    avg_density = np.mean(density_history[-1])
                    if avg_density > rho_crit:
                        # In congested state, large vehicles have slightly lower proportion
                        factor = 1.0 - 0.05 * (pce - 1.0) * (avg_density - rho_crit) / rho_crit
                        ratio *= max(0.8, factor)

                    result_flow[veh_type] = ratio * flow_5min

        return result_flow

    def validate(self, predict_minutes_list=[5, 10, 15]):
        """Validate model performance for different prediction horizons"""
        model_type = "Enhanced Version 2" if self.enhanced_mode else "Enhanced Version 1"
        self.logger.info(f"Starting {model_type} PI-LWR model validation...")

        results = []

        # Iterate through all segments
        for segment_id, segment in self.segments.items():
            up_node = segment['up_node']
            down_node = segment['down_node']

            # Ensure both upstream and downstream have flow data
            if up_node not in self.flow_data or down_node not in self.flow_data:
                continue

            self.logger.info(f"Validating segment {segment_id}: {up_node} -> {down_node}")

            # Get time points
            time_points = self.flow_data[up_node]['time'].unique()
            # Take only a subset of time points for validation (to speed up process)
            if len(time_points) > 50:
                time_points = np.random.choice(time_points, 50, replace=False)

            # Validate for each time point and prediction horizon
            for start_time in time_points:
                for predict_minutes in predict_minutes_list:
                    # Calculate prediction end time
                    end_time = start_time + pd.Timedelta(minutes=predict_minutes)

                    # Predict flow
                    predicted_flow = self.predict_flow(segment_id, start_time, predict_minutes)
                    if not predicted_flow:
                        continue

                    # Get actual flow
                    actual_flow = self._get_flow_by_type(down_node, end_time)
                    if not actual_flow:
                        continue

                    # Record predicted and actual values for each vehicle type
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

        # Convert to DataFrame
        if results:
            results_df = pd.DataFrame(results)

            # Save results
            model_name = "enhanced_v2" if self.enhanced_mode else "enhanced_v1"
            output_file = os.path.join(self.output_dir, f'validation_results_{model_name}.csv')
            results_df.to_csv(output_file, index=False)
            self.logger.info(f"Validation results saved to: {output_file}")

            # Calculate performance metrics
            self._calculate_metrics(results_df)

            return results_df
        else:
            self.logger.warning("No validation results obtained")
            return pd.DataFrame()

    def _calculate_metrics(self, results_df):
        """Calculate validation metrics"""
        model_type = "Enhanced Version 2" if self.enhanced_mode else "Enhanced Version 1"
        self.logger.info(f"Calculating {model_type} PI-LWR model performance metrics:")

        # Calculate metrics by prediction horizon
        self.logger.info("\nPerformance metrics by prediction horizon:")
        time_metrics = []
        for minutes, group in results_df.groupby('predict_minutes'):
            y_true = group['actual']
            y_pred = group['predicted']

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            self.logger.info(
                f"{minutes} minute prediction: samples={len(group)}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            time_metrics.append({
                'predict_minutes': minutes,
                'sample_count': len(group),
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })

        # Calculate metrics by segment type
        type_metrics = []
        if 'segment_type' in results_df.columns:
            self.logger.info("\nPerformance metrics by segment type:")
            for type_id, group in results_df.groupby('segment_type'):
                y_true = group['actual']
                y_pred = group['predicted']

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                mask = y_true > 0
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    mape = np.nan

                self.logger.info(
                    f"Segment type {type_id}: samples={len(group)}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
                type_metrics.append({
                    'segment_type': type_id,
                    'sample_count': len(group),
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })

        # Calculate metrics by vehicle type
        self.logger.info("\nPerformance metrics by vehicle type:")
        vehicle_metrics = []
        for veh_type, group in results_df.groupby('vehicle_type'):
            y_true = group['actual']
            y_pred = group['predicted']

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            self.logger.info(
                f"Vehicle type {veh_type}: samples={len(group)}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            vehicle_metrics.append({
                'vehicle_type': veh_type,
                'sample_count': len(group),
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })

        # Calculate overall performance metrics - using specified aggregation method
        if self.aggregate_by == 'time':
            # Average based on time window metrics
            avg_mae = np.mean([m['MAE'] for m in time_metrics])
            avg_rmse = np.mean([m['RMSE'] for m in time_metrics])
            avg_mape = np.mean([m['MAPE'] for m in time_metrics if not np.isnan(m['MAPE'])])

            total_samples = sum(m['sample_count'] for m in time_metrics)

            self.logger.info("\nOverall performance metrics (averaged by time window):")
            self.logger.info(
                f"Samples={total_samples}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, MAPE={avg_mape:.2f}%")

            overall_metrics = {
                'model': model_type,
                'sample_count': total_samples,
                'MAE': avg_mae,
                'RMSE': avg_rmse,
                'MAPE': avg_mape,
                'aggregation_method': 'time_average'
            }
        else:
            # Average based on vehicle type metrics
            avg_mae = np.mean([m['MAE'] for m in vehicle_metrics])
            avg_rmse = np.mean([m['RMSE'] for m in vehicle_metrics])
            avg_mape = np.mean([m['MAPE'] for m in vehicle_metrics if not np.isnan(m['MAPE'])])

            total_samples = sum(m['sample_count'] for m in vehicle_metrics)

            self.logger.info("\nOverall performance metrics (averaged by vehicle type):")
            self.logger.info(
                f"Samples={total_samples}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, MAPE={avg_mape:.2f}%")

            overall_metrics = {
                'model': model_type,
                'sample_count': total_samples,
                'MAE': avg_mae,
                'RMSE': avg_rmse,
                'MAPE': avg_mape,
                'aggregation_method': 'vehicle_average'
            }

    def calibrate(self, predict_minutes_list=[5, 10, 15], sample_size=20):
        """Automatically calibrate model parameters (Enhanced Version 2 only)"""
        if not self.enhanced_mode:
            self.logger.warning("Calibration feature is only available in Enhanced Version 2")
            return

        self.logger.info("Starting PI-LWR model parameter calibration...")

        # Find all available segments
        valid_segments = []
        for segment_id, segment in self.segments.items():
            up_node = segment['up_node']
            down_node = segment['down_node']
            if up_node in self.flow_data and down_node in self.flow_data:
                valid_segments.append(segment_id)

        if not valid_segments:
            self.logger.warning("No valid segments found for calibration")
            return

        # Calibrate parameters for each prediction horizon
        for predict_minutes in predict_minutes_list:
            self.logger.info(f"Calibrating parameters for {predict_minutes} minute prediction...")

            # Select samples
            calibration_samples = []
            for segment_id in valid_segments:
                segment = self.segments[segment_id]
                up_node = segment['up_node']
                down_node = segment['down_node']

                # Get time points
                time_points = self.flow_data[up_node]['time'].unique()
                if len(time_points) > sample_size:
                    selected_times = np.random.choice(time_points, sample_size, replace=False)
                else:
                    selected_times = time_points

                for start_time in selected_times:
                    end_time = start_time + pd.Timedelta(minutes=predict_minutes)

                    # Get upstream flow
                    upstream_flow = self._get_flow_by_type(up_node, start_time)
                    if not upstream_flow:
                        continue

                    # Get actual downstream flow
                    actual_flow = self._get_flow_by_type(down_node, end_time)
                    if not actual_flow:
                        continue

                    # Calculate total flows
                    total_upstream = sum(upstream_flow.values())
                    total_actual = sum(actual_flow.values())

                    calibration_samples.append({
                        'segment_id': segment_id,
                        'start_time': start_time,
                        'total_upstream': total_upstream,
                        'total_actual': total_actual
                    })

            if not calibration_samples:
                self.logger.warning(f"No calibration samples found for {predict_minutes} minute prediction")
                continue

            self.logger.info(f"Collected {len(calibration_samples)} calibration samples")

            # Define objective function
            def objective(params):
                # Unpack parameters
                rho_crit, v_free = params

                # Calculate errors
                errors = []
                for sample in calibration_samples:
                    segment_id = sample['segment_id']
                    start_time = sample['start_time']
                    total_actual = sample['total_actual']

                    # Temporarily set parameters
                    original_params = self.horizon_params.get(predict_minutes, {})
                    self.horizon_params[predict_minutes] = {
                        'rho_crit': rho_crit,
                        'v_free': v_free,
                        'fd_type': self.fd_type
                    }

                    # Predict flow
                    predicted_flow = self.predict_flow(segment_id, start_time, predict_minutes)

                    # Restore original parameters
                    if original_params:
                        self.horizon_params[predict_minutes] = original_params

                    if predicted_flow:
                        total_predicted = sum(predicted_flow.values())
                        if total_actual > 0:
                            rel_error = abs(total_predicted - total_actual) / total_actual
                            errors.append(rel_error)

                # Return mean relative error
                if errors:
                    return sum(errors) / len(errors)
                else:
                    return 1.0

            # Set parameter bounds
            bounds = [
                (20.0, 40.0),  # Critical density (veh/km/lane)
                (80.0, 120.0)  # Free flow speed (km/h)
            ]

            # Initial parameters
            initial_params = [self.critical_density, self.free_flow_speed]

            # Run optimization
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-3, 'maxiter': 15}
            )

            if result.success:
                rho_crit, v_free = result.x

                # Update parameters
                self.horizon_params[predict_minutes] = {
                    'rho_crit': rho_crit,
                    'v_free': v_free,
                    'fd_type': self.fd_type
                }

                self.logger.info(
                    f"{predict_minutes} minute prediction parameters calibrated: Critical density={rho_crit:.2f}veh/km/lane, Free flow speed={v_free:.2f}km/h, Error={result.fun:.4f}")
            else:
                self.logger.warning(f"Parameter calibration failed for {predict_minutes} minute prediction")

        self.calibrated = True
        self.logger.info("PI-LWR model parameter calibration complete")


class EnhancedPILWR(PILWRModel):
    """Enhanced Version 2 of PI-LWR model"""

    def __init__(self, *args, **kwargs):
        """Initialize Enhanced Version 2 PI-LWR model"""
        # Ensure enhanced_mode is True
        kwargs['enhanced_mode'] = True
        super().__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="PI-LWR Traffic Flow Model")
    parser.add_argument("--road_data", type=str, default="./ETC_data_example/roadETC.csv",
                        help="Path to road segment data file")
    parser.add_argument("--flow_dir", type=str, default="./ETC_data_example/flow",
                        help="Directory containing traffic flow data")
    parser.add_argument("--output_dir", type=str, default="./pi_lwr_results",
                        help="Output directory")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use Enhanced Version 2 PI-LWR model")
    parser.add_argument("--fd_type", type=str, choices=['triangular', 'greenshield'], default='greenshield',
                        help="Fundamental diagram type: triangular or Greenshield")
    parser.add_argument("--critical_density", type=float, default=30.0,
                        help="Critical density (veh/km/lane)")
    parser.add_argument("--jam_density", type=float, default=90.0,
                        help="Jam density (veh/km/lane)")
    parser.add_argument("--free_flow_speed", type=float, default=110.0,
                        help="Free flow speed (km/h)")
    parser.add_argument("--predict_minutes", type=int, nargs="+", default=[5, 10, 15],
                        help="Prediction horizons (minutes)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Automatically calibrate model parameters (Enhanced Version 2 only)")
    parser.add_argument("--aggregate_by", type=str, choices=['time', 'vehicle'], default='time',
                        help="Metric aggregation method: average by time window or vehicle type")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    if args.enhanced:
        model = EnhancedPILWR(
            road_data_path=args.road_data,
            traffic_flow_dir=args.flow_dir,
            output_dir=args.output_dir,
            critical_density=args.critical_density,
            jam_density=args.jam_density,
            free_flow_speed=args.free_flow_speed,
            aggregate_by=args.aggregate_by,
            fd_type=args.fd_type
        )

        # Calibrate parameters
        if args.calibrate:
            model.calibrate(predict_minutes_list=args.predict_minutes)
    else:
        model = PILWRModel(
            road_data_path=args.road_data,
            traffic_flow_dir=args.flow_dir,
            output_dir=args.output_dir,
            critical_density=args.critical_density,
            jam_density=args.jam_density,
            free_flow_speed=args.free_flow_speed,
            enhanced_mode=False,
            aggregate_by=args.aggregate_by,
            fd_type=args.fd_type
        )

    # Validate model
    validation_results = model.validate(predict_minutes_list=args.predict_minutes)

    model_type = "Enhanced Version 2" if args.enhanced else "Enhanced Version 1"
    print(f"{model_type} PI-LWR model validation complete with {len(validation_results)} records")
    print(f"Results saved to: {args.output_dir}")
    print(f"Metrics aggregated using {args.aggregate_by} method")

    # If both versions are run for comparison
    if args.enhanced and os.path.exists(os.path.join(args.output_dir, "validation_results_enhanced_v1.csv")):
        v1_results = pd.read_csv(os.path.join(args.output_dir, "validation_results_enhanced_v1.csv"))
        v2_results = pd.read_csv(os.path.join(args.output_dir, "validation_results_enhanced_v2.csv"))

        # Calculate overall metrics for both versions
        v1_mae = mean_absolute_error(v1_results['actual'], v1_results['predicted'])
        v1_rmse = np.sqrt(mean_squared_error(v1_results['actual'], v1_results['predicted']))
        mask = v1_results['actual'] > 0
        v1_mape = np.mean(np.abs((v1_results['actual'][mask] - v1_results['predicted'][mask]) / v1_results['actual'][mask])) * 100

        v2_mae = mean_absolute_error(v2_results['actual'], v2_results['predicted'])
        v2_rmse = np.sqrt(mean_squared_error(v2_results['actual'], v2_results['predicted']))
        mask = v2_results['actual'] > 0
        v2_mape = np.mean(np.abs((v2_results['actual'][mask] - v2_results['predicted'][mask]) / v2_results['actual'][mask])) * 100

        print("\n===== Model Performance Comparison =====")
        print(f"Enhanced Version 1: MAE={v1_mae:.3f}, RMSE={v1_rmse:.3f}, MAPE={v1_mape:.2f}%")
        print(f"Enhanced Version 2: MAE={v2_mae:.3f}, RMSE={v2_rmse:.3f}, MAPE={v2_mape:.2f}%")


if __name__ == "__main__":
    main()