# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2025/3/8 15:31
# ------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import os
import logging
import argparse
import datetime
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize


class METANETModel:
    def __init__(self,
                 road_data_path: str,
                 traffic_flow_dir: str,
                 output_dir: str = "./metanet_results",
                 simulation_step: float = 10.0,
                 tau: float = 18.0,
                 nu: float = 35.0,
                 kappa: float = 13.0,
                 a: float = 2.0,
                 critical_density: float = 27.0,
                 jam_density: float = 180.0,
                 free_flow_speed: float = 110.0,
                 default_lanes: int = 3,
                 aggregate_by: str = 'time'):
        self.road_data_path = road_data_path
        self.traffic_flow_dir = traffic_flow_dir
        self.output_dir = output_dir
        self.aggregate_by = aggregate_by

        os.makedirs(output_dir, exist_ok=True)

        self.simulation_step = simulation_step
        self.tau = tau / 3600
        self.nu = nu
        self.kappa = kappa
        self.a = a
        self.critical_density = critical_density
        self.jam_density = jam_density
        self.free_flow_speed = free_flow_speed
        self.default_lanes = default_lanes

        self._setup_logging()

        self.logger.info(f"Initializing METANET traffic flow model")
        self.logger.info(f"Core parameters: tau={tau}s, nu={nu}km^2/h, kappa={kappa}, a={a}")
        self.logger.info(f"Aggregation method: by {'time window' if aggregate_by == 'time' else 'vehicle type'}")

        self.road_data = self._load_road_data()
        self.flow_data = self._load_flow_data()
        self.segments = self._prepare_segments()

        self.vehicle_pce = {
            'B1': 1.0,  # Small passenger car
            'B2': 1.5,  # Medium passenger car
            'B3': 2.0,  # Large passenger car/bus
            'T1': 1.5,  # Small truck
            'T2': 2.5,  # Medium truck
            'T3': 3.5  # Large truck
        }

        self.horizon_params = {}
        self._init_horizon_params()

        self.calibrated = False

        self.logger.info("METANET model initialization complete")

    def _setup_logging(self):
        self.logger = logging.getLogger('metanet')
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            self.logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        log_file = os.path.join(self.output_dir, 'metanet.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _load_road_data(self):
        self.logger.info(f"Loading road segment data from: {self.road_data_path}")
        try:
            road_data = pd.read_csv(self.road_data_path)
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
                    flow_data[gantry_id] = df
                    self.logger.debug(f"Loaded flow data for gantry {gantry_id}: {len(df)} records")
                except Exception as e:
                    self.logger.warning(f"Failed to load flow data for gantry {gantry_id}: {e}")

        self.logger.info(f"Successfully loaded flow data for {len(flow_data)} gantries")
        return flow_data

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
                'type': int(row.get('type', 1))
            }

            segment['v_free'] = min(segment['speed_limit'], self.free_flow_speed)

            segment_length = 0.5
            num_segments = max(3, int(segment['length'] / segment_length))
            segment['num_segments'] = num_segments
            segment['segment_length'] = segment['length'] / num_segments

            segments[segment_id] = segment
            self.logger.debug(f"Prepared segment {segment_id}: {segment['length']}km, {num_segments} subsegments")

        self.logger.info(f"Prepared {len(segments)} segments for simulation")
        return segments

    def _init_horizon_params(self):
        self.logger.info("Initializing prediction horizon parameters")

        self.horizon_params[5] = {
            'tau': self.tau * 0.9,
            'nu': self.nu * 1.1,
            'kappa': self.kappa,
            'a': self.a * 0.9
        }

        self.horizon_params[10] = {
            'tau': self.tau,
            'nu': self.nu,
            'kappa': self.kappa,
            'a': self.a
        }

        self.horizon_params[15] = {
            'tau': self.tau * 1.1,
            'nu': self.nu * 0.95,
            'kappa': self.kappa,
            'a': self.a * 1.1
        }

        self.logger.info(f"Parameters set for 5, 10, and 15 minute prediction horizons")

    def _get_total_flow(self, gantry_id, time_point):
        if gantry_id not in self.flow_data:
            return 0

        df = self.flow_data[gantry_id]

        df['time_diff'] = abs(df['time'] - time_point)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        df.drop('time_diff', axis=1, inplace=True)

        vehicle_types = ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']
        total_flow = sum(closest_row.get(vt, 0) for vt in vehicle_types)

        return total_flow

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

    def _equilibrium_speed(self, density):
        if density <= 0:
            return self.free_flow_speed

        exp_term = -0.5 * ((density / self.critical_density) ** self.a)
        return self.free_flow_speed * np.exp(exp_term)

    def _get_time_period_factors(self, time_point):
        try:
            hour = time_point.hour
            weekday = time_point.weekday()
            is_weekend = weekday >= 5

            # Morning peak (7-9am)
            if 7 <= hour <= 9:
                if not is_weekend:
                    return {
                        'speed_factor': 0.9,
                        'tau_factor': 0.9,
                        'nu_factor': 1.2
                    }
                else:
                    return {
                        'speed_factor': 0.95,
                        'tau_factor': 1.0,
                        'nu_factor': 1.0
                    }

            # Evening peak (5-7pm)
            elif 17 <= hour <= 19:
                if not is_weekend:
                    return {
                        'speed_factor': 0.85,
                        'tau_factor': 0.85,
                        'nu_factor': 1.3
                    }
                else:
                    return {
                        'speed_factor': 0.9,
                        'tau_factor': 0.95,
                        'nu_factor': 1.1
                    }

            # Daytime off-peak (9am-5pm)
            elif 9 < hour < 17:
                return {
                    'speed_factor': 1.0,
                    'tau_factor': 1.0,
                    'nu_factor': 1.0
                }

            # Nighttime (7pm-7am)
            else:
                return {
                    'speed_factor': 1.05,
                    'tau_factor': 1.1,
                    'nu_factor': 0.9
                }
        except:
            return {
                'speed_factor': 1.0,
                'tau_factor': 1.0,
                'nu_factor': 1.0
            }

    def _estimate_ramp_flows(self, segment, upstream_flow):
        segment_type = segment['type']
        on_ramp_flow = 0
        off_ramp_flow = 0

        if segment_type == 2:  # Upstream has on-ramp
            on_ramp_flow = upstream_flow * 0.3
        elif segment_type == 3:  # Upstream has off-ramp
            off_ramp_flow = upstream_flow * 0.2
        elif segment_type == 4:  # Upstream has both on-ramp and off-ramp
            on_ramp_flow = upstream_flow * 0.25
            off_ramp_flow = upstream_flow * 0.15

        return on_ramp_flow, off_ramp_flow

    def _calculate_pce_flow(self, flow_dict):
        pce_flow = 0
        for veh_type, flow in flow_dict.items():
            pce = self.vehicle_pce.get(veh_type, 1.0)
            pce_flow += flow * pce
        return pce_flow

    def predict_flow(self, segment_id, start_time, predict_minutes=5):
        if segment_id not in self.segments:
            self.logger.error(f"Segment ID not found: {segment_id}")
            return {}

        segment = self.segments[segment_id]
        up_node = segment['up_node']
        down_node = segment['down_node']

        self.logger.info(f"Predicting flow for segment {segment_id}: {up_node} -> {down_node}, t+{predict_minutes}min")

        upstream_flow_dict = self._get_flow_by_type(up_node, start_time)
        if not upstream_flow_dict:
            self.logger.warning(f"No upstream flow data found for gantry {up_node} at {start_time}")
            return {}

        upstream_pce_flow = self._calculate_pce_flow(upstream_flow_dict)
        upstream_flow = upstream_pce_flow * 12  # Convert 5-min flow to hourly flow

        time_factors = self._get_time_period_factors(start_time)
        self.logger.debug(f"Time period factors: {time_factors}")

        upstream_speed = segment['v_free'] * 0.8 * time_factors['speed_factor']

        if upstream_speed > 0:
            initial_density = upstream_flow / (upstream_speed * segment['lanes'])
        else:
            initial_density = self.critical_density * 0.5

        initial_density = min(max(1.0, initial_density), self.jam_density * 0.9)

        num_segments = segment['num_segments']
        density = np.ones(num_segments) * initial_density
        speed = np.ones(num_segments) * upstream_speed

        if segment['type'] >= 3:
            downstream_density = self.critical_density * 0.6
        else:
            downstream_density = self.critical_density * 0.7

        on_ramp_flow, off_ramp_flow = self._estimate_ramp_flows(segment, upstream_flow)

        params = self.horizon_params.get(predict_minutes, {
            'tau': self.tau,
            'nu': self.nu,
            'kappa': self.kappa,
            'a': self.a
        })

        tau = params['tau'] * time_factors['tau_factor']
        nu = params['nu'] * time_factors['nu_factor']
        kappa = params['kappa']
        a = params['a']

        self.logger.debug(f"Simulation parameters: tau={tau * 3600}s, nu={nu}, a={a}")

        sim_seconds = predict_minutes * 60
        time_steps = int(sim_seconds / self.simulation_step)

        self.logger.debug(f"Starting simulation with {time_steps} time steps, " +
                          f"initial density={initial_density:.2f}, speed={upstream_speed:.2f}")

        for t in range(time_steps):
            density_prev = density.copy()
            speed_prev = speed.copy()

            flow = density_prev * speed_prev * segment['lanes']

            for i in range(num_segments):
                if i == 0:
                    q_in = upstream_flow
                    q_out = flow[i]
                else:
                    q_in = flow[i - 1]
                    q_out = flow[i]

                r_flow = 0
                s_flow = 0

                mid_segment = num_segments // 2
                if segment['type'] == 2 and i == mid_segment // 2:
                    r_flow = on_ramp_flow
                elif segment['type'] == 3 and i == mid_segment + mid_segment // 2:
                    s_flow = off_ramp_flow
                elif segment['type'] == 4:
                    if i == mid_segment // 2:
                        r_flow = on_ramp_flow
                    elif i == mid_segment + mid_segment // 2:
                        s_flow = off_ramp_flow

                T = self.simulation_step / 3600
                L = segment['segment_length']

                density[i] = density_prev[i] + \
                             T / L * (q_in - q_out) / segment['lanes'] + \
                             T / L * (r_flow - s_flow) / segment['lanes']

                density[i] = min(max(1.0, density[i]), self.jam_density)

                V_eq = self._equilibrium_speed(density_prev[i])

                relaxation = T / tau * (V_eq - speed_prev[i])

                if i == 0:
                    convection = T / L * speed_prev[i] * (upstream_speed - speed_prev[i])
                else:
                    convection = T / L * speed_prev[i] * (speed_prev[i - 1] - speed_prev[i])

                if i == num_segments - 1:
                    anticipation = -nu * T / L * (downstream_density - density_prev[i]) / (density_prev[i] + kappa)
                else:
                    anticipation = -nu * T / L * (density_prev[i + 1] - density_prev[i]) / (density_prev[i] + kappa)

                ramp_effect = 0
                if r_flow > 0:
                    delta = 0.65
                    ramp_effect = -delta * T * r_flow * speed_prev[i] / (
                            L * segment['lanes'] * (density_prev[i] + kappa))

                speed[i] = speed_prev[i] + relaxation + convection + anticipation + ramp_effect

                speed[i] = min(max(5.0, speed[i]), segment['v_free'])

        final_flow = density[-1] * speed[-1] * segment['lanes']
        flow_5min = final_flow / 12

        self.logger.debug(
            f"Final simulation state: density={density[-1]:.2f}, speed={speed[-1]:.2f}, flow={flow_5min:.2f}")

        upstream_total_pce = sum(flow * self.vehicle_pce.get(vt, 1.0) for vt, flow in upstream_flow_dict.items())
        predicted_flow = {}

        if upstream_total_pce > 0:
            for veh_type, flow in upstream_flow_dict.items():
                pce = self.vehicle_pce.get(veh_type, 1.0)
                ratio = (flow * pce) / upstream_total_pce
                predicted_flow[veh_type] = ratio * flow_5min / pce

        self.logger.info(f"Prediction complete: total flow {sum(predicted_flow.values()):.2f} veh/5min")
        return predicted_flow

    def validate(self, predict_minutes_list=[5, 10, 15]):
        self.logger.info(f"Starting METANET model validation...")
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

            return results_df
        else:
            self.logger.warning("No validation results obtained")
            return pd.DataFrame()

    def _calculate_metrics(self, results_df):
        self.logger.info(f"Calculating METANET model performance metrics:")

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

    def calibrate(self, predict_minutes_list=[5, 10, 15], sample_size=20):
        self.logger.info("Starting METANET model parameter calibration...")

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
                tau, nu, a = params

                temp_params = {
                    'tau': tau / 3600,
                    'nu': nu,
                    'kappa': self.kappa,
                    'a': a
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
                18.0,  # tau (seconds)
                35.0,  # nu (km²/h)
                2.0  # a
            ]

            bounds = [
                (10.0, 30.0),  # tau: 10-30 seconds
                (20.0, 60.0),  # nu: 20-60 km²/h
                (1.0, 3.0)  # a: 1.0-3.0
            ]

            self.logger.info(f"Starting optimization with initial params: tau={initial_params[0]}, " +
                             f"nu={initial_params[1]}, a={initial_params[2]}")

            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-3, 'maxiter': 15}
            )

            if result.success:
                tau, nu, a = result.x

                self.horizon_params[predict_minutes] = {
                    'tau': tau / 3600,
                    'nu': nu,
                    'kappa': self.kappa,
                    'a': a
                }

                self.logger.info(
                    f"Calibration successful for {predict_minutes}-minute prediction: " +
                    f"tau={tau:.2f}s, nu={nu:.2f}km^2/h, a={a:.2f}, error={result.fun:.4f}")
            else:
                self.logger.warning(f"Calibration failed for {predict_minutes}-minute prediction")
                self.logger.warning(f"Optimization message: {result.message}")

        self.calibrated = True
        self.logger.info("METANET model parameter calibration complete")


def main():
    parser = argparse.ArgumentParser(description="METANET Traffic Flow Model")
    parser.add_argument("--road_data", type=str, default="./ETC_data_example/roadETC.csv",
                        help="Path to road segment data file")
    parser.add_argument("--flow_dir", type=str, default="./ETC_data_example/flow",
                        help="Directory containing traffic flow data")
    parser.add_argument("--output_dir", type=str, default="./metanet_results",
                        help="Output directory for results")
    parser.add_argument("--tau", type=float, default=18.0,
                        help="Relaxation time (seconds)")
    parser.add_argument("--nu", type=float, default=35.0,
                        help="Anticipation parameter (km²/h)")
    parser.add_argument("--kappa", type=float, default=13.0,
                        help="Density smoothing parameter (veh/km/lane)")
    parser.add_argument("--a", type=float, default=2.0,
                        help="Speed-density relationship exponent")
    parser.add_argument("--predict_minutes", type=int, nargs="+", default=[5, 10, 15],
                        help="Prediction time windows (minutes)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Automatically calibrate model parameters")
    parser.add_argument("--aggregate_by", type=str, choices=['time', 'vehicle'], default='time',
                        help="Aggregation method for overall metrics: by time window or by vehicle type")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = METANETModel(
        road_data_path=args.road_data,
        traffic_flow_dir=args.flow_dir,
        output_dir=args.output_dir,
        tau=args.tau,
        nu=args.nu,
        kappa=args.kappa,
        a=args.a,
        aggregate_by=args.aggregate_by
    )

    if args.calibrate:
        model.calibrate(predict_minutes_list=args.predict_minutes)

    validation_results = model.validate(predict_minutes_list=args.predict_minutes)

    print(f"METANET model validation complete: {len(validation_results)} records")
    print(f"Results saved to: {args.output_dir}")
    print(f"Metrics aggregated by {args.aggregate_by}")


if __name__ == "__main__":
    main()