import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from discretize import TensorMesh
from SimPEG.data import Data
from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem

# --- CPU THREADING SAFETY ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# 1. COVARIANCE MAPPING
# -----------------------------------------------------------------------------
def get_covariance_transform(mesh, correlation_factor, method='cholesky'):
    """
    Builds the structural mapping matrix L using a non-stationary, 
    thickness-dependent correlation length.
    """
    nC = mesh.nC
    thicknesses = mesh.h[0]          
    depths = mesh.cell_centers_x     
    
    # Pairwise physical distance matrix
    dist = np.abs(depths[:, None] - depths[None, :])
    
    # Non-stationary Correlation Lengths
    correlation_factor = max(0.1, float(correlation_factor))
    L_local = correlation_factor * thicknesses 
    L_pair = 0.5 * (L_local[:, None] + L_local[None, :])
    
    # Covariance Matrix (with 1e-6 nugget)
    C = np.exp(-1.0 * (dist / L_pair)) + np.eye(nC) * 1e-6
    
    # Symmetric SVD Logic
    def compute_symmetric_svd(cov_matrix):
        U, S, Vt = np.linalg.svd(cov_matrix)
        V = Vt.T
        S_sqrt = np.diag(np.sqrt(np.clip(S, a_min=1e-10, a_max=None)))
        return V @ S_sqrt @ Vt 

    if method.lower() == 'svd':
        L_mat = compute_symmetric_svd(C)
    else:
        try:
            L_mat = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L_mat = compute_symmetric_svd(C)
            
    return L_mat

# -----------------------------------------------------------------------------
# 2. DETERMINISTIC WORKER (PEST-Style TRF with Dynamic Beta)
# -----------------------------------------------------------------------------
def solve_deterministic_worker(dobs, uncertainties, payload_data, bg_mean, corr_len=2.0, target_chi2=0.5, transform_type='cholesky', use_IRLS=True):
    try:
        # --- 1. Mesh & Depth Weighting Setup ---
        thicknesses = payload_data['thicknesses']
        nC = len(thicknesses) + 1
        mesh = TensorMesh([np.r_[thicknesses, thicknesses[-1]]], "0")
        
        depths = mesh.cell_centers_x
        # Tipping the scales: making deep perturbations 'cheaper'
        depth_weights =np.sqrt(depths / depths[0])# np.ones_like(depths)
        
        
        # --- 2. Survey Setup ---
        rx_loc = payload_data['rx_loc']
        tx_area = float(payload_data['tx_area'])
        tx_centroid = np.mean(np.atleast_2d(payload_data['tx_shape']), axis=0).flatten() 
        
        rx_lm = tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, payload_data['lm_times'], "z")
        rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, payload_data['hm_times'], "z")
        
        if payload_data['lm_wave_time'] is not None:
            w_lm = tdem.sources.PiecewiseLinearWaveform(payload_data['lm_wave_time'], payload_data['lm_wave_form'])
            w_hm = tdem.sources.PiecewiseLinearWaveform(payload_data['hm_wave_time'], payload_data['hm_wave_form'])
            src_lm = tdem.sources.MagDipole([rx_lm], location=tx_centroid, orientation="z", waveform=w_lm)
            src_hm = tdem.sources.MagDipole([rx_hm], location=tx_centroid, orientation="z", waveform=w_hm)
        else:
            src_lm = tdem.sources.MagDipole([rx_lm], location=tx_centroid, orientation="z")
            src_hm = tdem.sources.MagDipole([rx_hm], location=tx_centroid, orientation="z")
            
        survey = tdem.Survey([src_lm, src_hm])
        sim = tdem.Simulation1DLayered(survey=survey, thicknesses=thicknesses, sigmaMap=maps.ExpMap(mesh))
        
        # Use provided uncertainties properly
        Wd = 1.0 / uncertainties
        eval_count = [0]

        # --- 3. Initial Bounds & Model ---
        bg_mean_start, corr_len_start = bg_mean, corr_len
        lb = np.r_[np.full(nC, -3.0), np.log(0.005), 0.05] 
        ub = np.r_[np.full(nC, 3.0), np.log(0.1), 20.0]
        
        # Stochastic kickstart to ensure non-zero starting gradients
        m_current = np.r_[np.random.randn(nC) * 0.05, bg_mean_start, corr_len_start]
        m_current = np.clip(m_current, lb, ub)
        
        best_overall_m = m_current.copy()
        best_overall_chi2 = np.inf
        prev_chi2 = np.inf
        current_chi2 = np.inf  # Initialized here for the IRLS trigger
        
        # --- 4. Initial Beta Estimation ---
        z, curr_mu, curr_ell = m_current[:nC], m_current[nC], m_current[nC+1]
        L_mat = get_covariance_transform(mesh, curr_ell, method=transform_type)
        ln_sigma = np.clip(L_mat @ z + curr_mu, -12.0, 2.0)
        
        J_sim = sim.getJ(ln_sigma) * tx_area 
        J_z = J_sim @ L_mat
        J_mu = np.sum(J_sim, axis=1, keepdims=True)
        
        delta_ell = 0.5
        L_plus = get_covariance_transform(mesh, curr_ell + delta_ell, method=transform_type)
        J_ell = (sim.dpred(np.clip(L_plus @ z + curr_mu, -12.0, 2.0)) * tx_area - sim.dpred(ln_sigma) * tx_area) / delta_ell
        
        J_data = np.hstack([J_z, J_mu, J_ell.reshape(-1, 1)]) * Wd[:, np.newaxis]
        J_reg_z_unscaled = np.hstack([np.diag(1.0 / (2.0 * depth_weights)), np.zeros((nC, 2))])
        J_reg_mu_unscaled = np.zeros((1, nC + 2)); J_reg_mu_unscaled[0, nC] = 1.0 
        J_reg_ell_unscaled = np.zeros((1, nC + 2)); J_reg_ell_unscaled[0, nC+1] = 1.0 
        J_reg_unscaled = np.vstack([J_reg_z_unscaled, J_reg_mu_unscaled, J_reg_ell_unscaled])
        
        max_H_data = np.max(np.sum(J_data**2, axis=0))
        max_H_reg = np.max(np.sum(J_reg_unscaled**2, axis=0))
        beta = np.sqrt(max_H_data / (max_H_reg + 1e-12)) * 0.5 
        
        action_log = []

        # --- 5. Macro Iteration Loop (PEST Style) ---
        for macro_iter in range(20): 
            
            # --- TWO-STAGE IRLS TRIGGER / L2 TOGGLE ---
            if use_IRLS and current_chi2 <= 1.0:
                # Phase 2: L1 Sparsity to sharpen boundaries (Only if use_IRLS is True)
                z_prev = m_current[:nC]
                irls_weights = 1.0 / np.sqrt(np.abs(z_prev) + 1e-2)
                phase = "[L1]"
            else:
                # Phase 1 / Pure L2: Smooth, flat penalty
                irls_weights = 1.0
                phase = "[L2]"
            # ------------------------------------------
            
            def residual_function(m_ext):
                eval_count[0] += 1
                z, curr_mu, curr_ell = m_ext[:nC], m_ext[nC], m_ext[nC+1]
                L_mat = get_covariance_transform(mesh, curr_ell, method=transform_type)
                ln_sigma = np.clip(L_mat @ z + curr_mu, -12.0, 2.0)
                
                d_pred = sim.dpred(ln_sigma) * tx_area
                data_res = Wd * (d_pred - dobs)
                
                # Apply the IRLS weights to the z penalty
                reg_z = (z / (2.0 * depth_weights)) * beta * irls_weights
                
                reg_mu = (curr_mu - bg_mean_start) * beta
                reg_ell = (curr_ell - corr_len_start) * beta
                
                return np.r_[data_res, reg_z, reg_mu, reg_ell]

            def jacobian_function(m_ext):
                z, curr_mu, curr_ell = m_ext[:nC], m_ext[nC], m_ext[nC+1]
                L_mat = get_covariance_transform(mesh, curr_ell, method=transform_type)
                ln_sigma = np.clip(L_mat @ z + curr_mu, -12.0, 2.0)
                
                J_sim = sim.getJ(ln_sigma) * tx_area 
                J_z = J_sim @ L_mat
                J_mu = np.sum(J_sim, axis=1, keepdims=True)
                
                L_plus = get_covariance_transform(mesh, curr_ell + delta_ell, method=transform_type)
                J_ell = (sim.dpred(np.clip(L_plus @ z + curr_mu, -12.0, 2.0)) * tx_area - sim.dpred(ln_sigma) * tx_area) / delta_ell
                
                J_data = np.hstack([J_z, J_mu, J_ell.reshape(-1, 1)]) * Wd[:, np.newaxis] 
                
                # Apply the IRLS weights to the Jacobian diagonal
                penalty_diag = (1.0 / (2.0 * depth_weights)) * irls_weights
                J_reg_z = np.hstack([np.diag(penalty_diag) * beta, np.zeros((nC, 2))])
                
                J_reg_mu = np.zeros((1, nC + 2)); J_reg_mu[0, nC] = beta 
                J_reg_ell = np.zeros((1, nC + 2)); J_reg_ell[0, nC+1] = beta
                
                return np.vstack([J_data, J_reg_z, J_reg_mu, J_reg_ell])

            # Trust Region Reflective Optimization Step
            res = least_squares(
                residual_function, m_current, 
                bounds=(lb, ub), 
                method='trf', jac=jacobian_function, loss='linear',     
                gtol=1e-5, max_nfev=150 
            )
            
            m_current = res.x
            residuals = residual_function(m_current)
            data_misfit_val = np.sum(residuals[:len(dobs)]**2)
            current_chi2 = data_misfit_val / len(dobs)
            
            if current_chi2 < best_overall_chi2:
                best_overall_chi2 = current_chi2
                best_overall_m = m_current.copy()
            
            chi2_change = abs(prev_chi2 - current_chi2)
            target_reached = False
            
            # --- Gapless Beta Logic ---
            if current_chi2 > target_chi2 * 1.5 and chi2_change < 0.1:
                action = "Stalled -> Reset & Kick"
                beta = np.sqrt(max_H_data / (max_H_reg + 1e-12)) * 0.5 
                m_current[:nC] += np.random.randn(nC) * 0.1
                m_current = np.clip(m_current, lb, ub)
            elif current_chi2 < target_chi2 * 0.85:
                action = "Overfit -> Raise Beta"; beta *= 1.5 
            elif current_chi2 > target_chi2 * 1.15:
                action = "Underfit -> Lower Beta"; beta *= 0.5 
            elif current_chi2 > target_chi2:
                action = "Approaching -> Micro Lower"; beta *= 0.8
            else:
                action = "Target Reached!"
                target_reached = True

            action_log.append(f"Iter {macro_iter+1:<2} | {phase} | Beta: {beta:.2e} | Chi2: {current_chi2:.2f} | {action}")
            
            if target_reached: break
            prev_chi2 = current_chi2
            
        # --- 6. Final Model Packaging ---
        final_z, final_mu, final_ell = best_overall_m[:nC], best_overall_m[nC], best_overall_m[nC+1]
        L_final = get_covariance_transform(mesh, final_ell, method=transform_type)
        final_ln_sigma = L_final @ final_z + final_mu
        
        try:
            final_dpred = sim.dpred(final_ln_sigma) * tx_area
        except:
            final_dpred = np.ones_like(dobs) * np.nan
        
        return {
            'sigma': np.exp(final_ln_sigma), 
            'z': final_z, 'mu': final_mu, 'corr_len': final_ell,
            'phi_d': best_overall_chi2 * len(dobs), 'dpred': final_dpred,
            'nfev': eval_count[0], 'success': True,
            'logs': action_log
        }

    except Exception as e:
        return {'success': False, 'error_msg': str(e)}

# 3. CONTROLLER CLASS
# -----------------------------------------------------------------------------
class DeterministicLatent:
    def __init__(self, bg_mean=np.log(0.01), corr_len=3.0):
        self.bg_mean = bg_mean
        self.corr_len = corr_len

    def run_local(self, Sounding, client=None):
        src0 = Sounding.srv.source_list[0] 
        src1 = Sounding.srv.source_list[1] 
        
        payload_data = {
            'thicknesses': Sounding.inv_thickness,
            'rx_loc': src0.receiver_list[0].locations[0],
            'tx_shape': src0.location, 
            'tx_area': Sounding.tx_area,
            'lm_times': src0.receiver_list[0].times,
            'hm_times': src1.receiver_list[0].times,
            'lm_wave_time': src0.waveform.times if hasattr(src0.waveform, 'times') else None,
            'lm_wave_form': src0.waveform.currents if hasattr(src0.waveform, 'currents') else None, 
            'hm_wave_time': src1.waveform.times if hasattr(src1.waveform, 'times') else None,
            'hm_wave_form': src1.waveform.currents if hasattr(src1.waveform, 'currents') else None
        }   
        
        if client is None:
            res = solve_deterministic_worker(
                Sounding.dobs, Sounding.uncertainties, payload_data, 
                self.bg_mean, corr_len=self.corr_len, target_chi2=0.5, transform_type='svd'
            )
            self.post_process(Sounding, res)
        else:
            fut = client.submit(
                solve_deterministic_worker,
                Sounding.dobs, Sounding.uncertainties, payload_data, 
                self.bg_mean, corr_len=self.corr_len, target_chi2=0.5, transform_type='svd'
            )

            from dask.distributed import Client
            if isinstance(client, Client):
                res = client.gather(fut)
            else:
                res = fut.result()
            self.post_process(Sounding, res)

    def post_process(self, Sounding, result):
        if result['success']:
            self.sigma = result['sigma']
            self.z = result['z']
            self.dpred = result['dpred']
            self.mu = result['mu']                
            self.corr_len = result['corr_len']   
            self.nfev = result['nfev']
            self.logs = result.get('logs', []) 
            self.chival = (result['phi_d']) / len(Sounding.dobs)
        else:
            print(f"Worker Error: {result.get('error_msg', 'Unknown Error')}")
            n_layers = len(Sounding.inv_thickness) + 1
            self.sigma = np.ones(n_layers) * np.nan
            self.z = np.ones(n_layers) * np.nan
            self.chival = np.nan
# -----------------------------------------------------------------------------
# 4. SOUNDING CLASS
# -----------------------------------------------------------------------------
class Sounding:
    def __init__(self, Survey, iline, time, inv_thickness, use_relerr=False, unc=None):
        self.iline = iline
        self.time = time
        self.inv_thickness = inv_thickness
        self.Depths = np.r_[self.inv_thickness.cumsum(), self.inv_thickness.cumsum()[-1] + self.inv_thickness[-1]]
        self.use_relerr = use_relerr
        self.unc = unc

        try:
            self.runc_offset = Survey.Data.runc_offset
        except AttributeError:
            self.runc_offset = 0.03

        self.station_data = Survey.Data.station_data.loc[(iline, time)]
        self.UTMX, self.UTMY = self.station_data.UTMX, self.station_data.UTMY
        self.Elevation = self.station_data.ELEVATION
        self.TX_ALTITUDE, self.RX_ALTITUDE = self.station_data.TX_ALTITUDE, self.station_data.RX_ALTITUDE

        rx_loc = np.array(Survey.rx_offset) + [self.UTMX, self.UTMY, self.RX_ALTITUDE]
        tx_loc = np.array(Survey.tx_shape) + [self.UTMX, self.UTMY, self.TX_ALTITUDE]
        self.tx_area = Survey.tx_area

        if Survey.lm_wave_time is not None:
            src_lm = tdem.sources.LineCurrent(
                tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, Survey.lm_times, "z"),
                tx_loc, waveform=tdem.sources.PiecewiseLinearWaveform(Survey.lm_wave_time, Survey.lm_wave_form)
            )
            src_hm = tdem.sources.LineCurrent(
                tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, Survey.hm_times, "z"),
                tx_loc, waveform=tdem.sources.PiecewiseLinearWaveform(Survey.hm_wave_time, Survey.hm_wave_form)
            )
        else:
            src_lm = tdem.sources.LineCurrent(
                tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, Survey.lm_times, "z"), tx_loc
            )
            src_hm = tdem.sources.LineCurrent(
                tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, Survey.hm_times, "z"), tx_loc
            )
        self.srv = tdem.Survey([src_lm, src_hm])

        lm_d = Survey.Data.lm_data.loc[(iline, time)].values
        hm_d = Survey.Data.hm_data.loc[(iline, time)].values
        self.dobs = -self.tx_area * np.r_[lm_d, hm_d]
        self.times = np.r_[Survey.lm_times, Survey.hm_times]

        noise_floor = 1e-15 
        if (self.use_relerr):
            self.uncertainties = np.sqrt((self.dobs * self.runc_offset) ** 2 + noise_floor**2)
        else:
            lm_std = Survey.Data.lm_std.loc[(iline, time)].values
            hm_std = Survey.Data.hm_std.loc[(iline, time)].values
            self.uncertainties = np.sqrt((self.dobs * 0.03) ** 2 + noise_floor**2)
            #self.uncertainties = np.sqrt((np.r_[lm_std, hm_std] * self.tx_area)**2 + noise_floor**2)

        self.data_object = Data(self.srv, dobs=self.dobs, standard_deviation=self.uncertainties)
        self.mesh = TensorMesh([(np.r_[self.inv_thickness, self.inv_thickness[-1]])], "0")

    def get_deterministic(self, client=None, bg_mean=np.log(0.01), corr_len=3.0,):
        self.Det = DeterministicLatent(bg_mean=bg_mean, corr_len=corr_len)
        self.Det.run_local(self, client)
