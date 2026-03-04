import os
import sys
import json
import functools
import logging
import numpy as np
import pandas as pd
import scipy.linalg
import dask
from scipy.special import erf
from scipy.optimize import minimize, minimize_scalar, Bounds
from scipy.signal import find_peaks
from dask.distributed import Client, LocalCluster
from simpeg.data import Data
from discretize import TensorMesh
from simpeg import maps, data, directives, inverse_problem, inversion, optimization, regularization
import simpeg.electromagnetics.time_domain as tdem
from simpeg.data_misfit import L2DataMisfit

# --- IMPORT ENSEMBLE SMOOTHER ---
try:
    from .ES import LMEnsembleSmoother
except ImportError:
    try:
        from ES import LMEnsembleSmoother
    except ImportError:
        print("Warning: ES.py not found. IES functionality will not work.")

# CPU SAFETY
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
def optimize_waveform_bipolar(times, amps, tol=1e-3):
    """
    Optimizes complex bipolar waveforms by preserving:
    1. Start and End
    2. Local Minima and Maxima (The peaks -0.5 and 1.0)
    3. High curvature points (The sharp corners)
    4. Zero crossings
    """
    keep_mask = np.zeros(len(times), dtype=bool)
    
    # 1. Keep Start and End
    keep_mask[0] = True
    keep_mask[-1] = True
    
    # 2. Keep Local Extrema (Peaks and Valleys)
    # This finds points where the slope changes sign
    diff_sign = np.sign(np.diff(amps))
    sign_change = np.diff(diff_sign)
    # Indices where sign change is non-zero are peaks/valleys
    peak_indices = np.where(sign_change != 0)[0] + 1
    keep_mask[peak_indices] = True
    
    # 3. Keep Zero Crossings (Critical for bipolar)
    # Find where amplitude crosses zero
    zero_cross = np.where(np.diff(np.sign(amps)))[0]
    keep_mask[zero_cross] = True
    keep_mask[zero_cross + 1] = True
    
    # 4. Curvature Fill-in (for the ramp-off and corners)
    grads = np.gradient(amps, times)
    curvature = np.gradient(grads)
    max_curve = np.max(np.abs(curvature))
    if max_curve > 0:
        keep_mask[np.abs(curvature) > (max_curve * tol)] = True

    return times[keep_mask], amps[keep_mask]
# Helper context manager
class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr
    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        self.old_stdout = sys.stdout; self.old_stderr = sys.stderr
    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        self._stdout = sys.stdout; self._stderr = sys.stderr
        self.devnull.close()

def fsim(srv, mesh, pars):
    """Forward simulation wrapper."""
    model_mapping = maps.ExpMap(nP=mesh.nC)
    simulation = tdem.Simulation1DLayered(survey=srv, thicknesses=mesh.h[0][:-1], sigmaMap=model_mapping)
    pred = simulation.make_synthetic_data(np.log(pars))
    return pred

def is_scalar(obj):
    if isinstance(obj, np.ndarray): return obj.shape == ()
    else: return np.isscalar(obj)

# -----------------------------------------------------------------------------
# 1. GEOSTATISTICAL HELPERS
# -----------------------------------------------------------------------------

class GeostatisticalMapping(maps.IdentityMap):
    """Projects stochastic latent vector into physical space using L."""
    def __init__(self, mesh, L_matrix):
        self.L = L_matrix
        self.n_layers = mesh.nC
        super().__init__(mesh=mesh, nP=self.n_layers + 1)
    def _transform(self, m):
        return m[0] + (self.L @ m[1:])
    def deriv(self, m, v=None):
        J = np.hstack([np.ones((self.n_layers, 1)), self.L])
        if v is not None: return J @ v
        return J

def get_cholesky_decomposition(mesh, correlation_factor):
    nC = mesh.nC
    idx = np.arange(nC)
    index_dist = np.abs(idx[:, None] - idx[None, :])
    correlation_factor = max(0.1, float(correlation_factor))
    C = np.exp(-1.0 * (index_dist / correlation_factor))
    C += np.eye(nC) * 1e-6
    try:
        L_mat = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        U, S, _ = np.linalg.svd(C)
        L_mat = U @ np.diag(np.sqrt(S))
    return L_mat

def solve_rml_worker_SIMPEG(seed, dobs, uncertainties, payload_data, sample_mean, sample_corr, chifact=1.0):
    import numpy as np
    from simpeg import (
        maps, data, data_misfit, regularization, 
        optimization, inverse_problem, inversion, directives
    )
    import simpeg.electromagnetics.time_domain as tdem
    from discretize import TensorMesh

    try:
        # --- 1. REBUILD MESH ---
        thicknesses_input = payload_data['thicknesses']
        # Mesh has nC cells (e.g. 60)
        mesh = TensorMesh([np.r_[thicknesses_input, thicknesses_input[-1]]], "0")
        nC = mesh.nC  
        sim_thicknesses = mesh.h[0][:-1] 
        
        # --- 2. REBUILD SURVEY ---
        rx_loc = payload_data['rx_loc']
        tx_shape = payload_data['tx_shape']
        rx_lm = tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, payload_data['lm_times'], "z")
        rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, payload_data['hm_times'], "z")
        w_lm = tdem.sources.PiecewiseLinearWaveform(payload_data['lm_wave_time'], payload_data['lm_wave_form'])
        w_hm = tdem.sources.PiecewiseLinearWaveform(payload_data['hm_wave_time'], payload_data['hm_wave_form'])
        src_lm = tdem.sources.LineCurrent([rx_lm], location=tx_shape, waveform=w_lm)
        src_hm = tdem.sources.LineCurrent([rx_hm], location=tx_shape, waveform=w_hm)
        survey = tdem.Survey([src_lm, src_hm])
        
        # --- 3. DATA SETUP ---
        np.random.seed(seed)
        d_pert = dobs + np.random.randn(len(dobs)) * uncertainties
        data_object = data.Data(survey, dobs=d_pert, standard_deviation=uncertainties)

        # --- 4. MAPPINGS (THE FIX) ---
        # Covariance Matrix for Structure
        idx = np.arange(nC); index_dist = np.abs(idx[:, None] - idx[None, :])
        C = np.exp(-1.0 * (index_dist / max(0.1, float(sample_corr)))) + np.eye(nC) * 1e-6
        try:
            L_mat = np.linalg.cholesky(C)
        except:
            U, S, _ = np.linalg.svd(C); L_mat = U @ np.diag(np.sqrt(S))

        wires = maps.Wires(('z', nC), ('mu', 1))
        
        # Map 1: Structure (nC -> nC)
        map_z = maps.LinearMap(L_mat) * wires.z
        
        # Map 2: Mean (1 -> nC)
        # FIX: We create a matrix P of all ones (shape 60x1)
        # This explicitly multiplies the scalar 'mu' to create a vector [mu, mu, ... mu]
        P = np.ones((nC, 1))
        map_mu = maps.LinearMap(P) * wires.mu
        
        # Combined: Now we are adding (60,) + (60,) -> Safe.
        model_mapping = maps.ExpMap(mesh) * (map_z + map_mu)

        # --- 5. INVERSION SETUP ---
        sim = tdem.Simulation1DLayered(survey=survey, thicknesses=sim_thicknesses, sigmaMap=model_mapping)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
        
        reg_mesh = TensorMesh([np.ones(nC + 1)])
        reg = regularization.WeightedLeastSquares(reg_mesh, mapping=maps.IdentityMap(nP=nC + 1),alpha_s=1,alpha_x=0)
        
        m_prior = np.r_[np.zeros(nC), np.array([sample_mean])]
        reg.reference_model = m_prior
        
        # Weights: 1.0 for z, 0.1 for mu
        #reg_weights = np.r_[np.ones(nC), 0.1]
        #reg.set_weights(prior_weights=np.sqrt(reg_weights))

        # GNCG Optimization
        min_log_cond = np.log(1e-4) 
        max_log_cond = np.log(0.1) 
        lower_b = np.r_[np.ones(nC) * -2.0, min_log_cond]
        upper_b = np.r_[np.ones(nC) * 2.0,  max_log_cond]

        opt = optimization.ProjectedGNCG(
            maxIter=50, lower=lower_b, upper=upper_b, 
            maxIterLS=30, maxIterCG=30, tolCG=1e-5
        )

        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        directives_list = [
            directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
            directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
            directives.TargetMisfit(chifact=chifact)
        ]
        inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)

        # --- 6. RUN ---
        import sys, os
        with open(os.devnull, 'w') as f:
            old_stdout = sys.stdout; sys.stdout = f
            try:
                m_rec = inv.run(m_prior)
            finally:
                sys.stdout = old_stdout
        
        sigma_final = model_mapping * m_rec
        dpred = sim.dpred(m_rec)
        phi_d = 0.5 * np.sum(((dpred - d_pert)/uncertainties)**2)
        
        return {
            'sigma': sigma_final, 'z': m_rec[:-1], 'mu': m_rec[-1], 
            'phi_d': phi_d, 'corr': sample_corr, 'success': True
        }

    except Exception as e:
        return {
            'sigma': np.zeros(len(payload_data['thicknesses'])+1) + 1e-8,
            'z': np.zeros(len(payload_data['thicknesses'])+1),
            'mu': sample_mean,
            'phi_d': 1e9, 
            'corr': sample_corr,
            'success': False,
            'error_msg': str(e)
        }
def solve_rml_worker(seed, dobs, uncertainties, payload_data, sample_mean, sample_corr, chifact=1.0):
    """
    Optimized RML Worker: Rebuilds SimPEG objects locally to avoid pickling hangs.
    """
    import numpy as np
    from scipy.optimize import minimize, Bounds
    from simpeg import maps
    import simpeg.electromagnetics.time_domain as tdem
    from discretize import TensorMesh
    
    # --- 1. REBUILD OBJECTS (Milliseconds) ---
    thicknesses = payload_data['thicknesses']
    n_physics = len(thicknesses) + 1
    
    # Rebuild Mesh
    mesh = TensorMesh([np.r_[thicknesses, thicknesses[-1]]], "0")
    
    # Rebuild Receivers
    rx_loc = payload_data['rx_loc']
    rx_lm = tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, payload_data['lm_times'], "z")
    rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, payload_data['hm_times'], "z")
    
    # Rebuild Waveforms
    if payload_data['lm_wave_time'] is not None:
        w_lm = tdem.sources.PiecewiseLinearWaveform(payload_data['lm_wave_time'], payload_data['lm_wave_form'])
        w_hm = tdem.sources.PiecewiseLinearWaveform(payload_data['hm_wave_time'], payload_data['hm_wave_form'])
        
        # Rebuild Sources
        tx_shape = payload_data['tx_shape']
        
        src_lm = tdem.sources.LineCurrent([rx_lm], location=tx_shape, waveform=w_lm)
        src_hm = tdem.sources.LineCurrent([rx_hm], location=tx_shape, waveform=w_hm)
    else:
        tx_shape = payload_data['tx_shape']
        src_lm = tdem.sources.LineCurrent([rx_lm], location=tx_shape)
        src_hm = tdem.sources.LineCurrent([rx_hm], location=tx_shape)
    
    survey = tdem.Survey([src_lm, src_hm])
    
    # --- 2. SETUP OPTIMIZATION ---
    np.random.seed(seed)
    d_pert = dobs + np.random.randn(len(dobs)) * uncertainties
    Wd = 1.0 / uncertainties
    
    m_ref = np.r_[np.zeros(n_physics), np.array([sample_mean])] 

    min_log_cond = np.log(1e-5)  # approx -11.5
    max_log_cond = np.log(1)  # approx +2.3
    
    # Apply to the vectors
    lower_b = np.r_[np.ones(n_physics) * -10.0, min_log_cond]
    upper_b = np.r_[np.ones(n_physics) * 10.0,  max_log_cond]
    solver_bounds = Bounds(lower_b, upper_b)

    # --- 3. OPTIMIZATION LOOP ---
    def get_cholesky_decomposition(mesh, correlation_factor):
        nC = mesh.nC
        idx = np.arange(nC)
        index_dist = np.abs(idx[:, None] - idx[None, :])
        correlation_factor = max(0.1, float(correlation_factor))
        C = np.exp(-1.0 * (index_dist / correlation_factor)) + np.eye(nC) * 1e-6
        return np.linalg.cholesky(C)

    def optimize_structure(current_L_val, m_start):
        L_mat = get_cholesky_decomposition(mesh, current_L_val)
        wires = maps.Wires(('z', n_physics), ('mu', 1))
        
        # Hardcoded std_scaling=1.0 for now, adjust if needed
        model_map = maps.ExpMap(mesh) * (maps.LinearMap(L_mat) * wires.z + maps.SurjectFull(mesh) * wires.mu)
        
        sim = tdem.Simulation1DLayered(survey=survey, thicknesses=thicknesses, sigmaMap=model_map)
        '''
        def objective(m):
            try:
                # Fields & Forward
                # Note: 'sim.fields' optimizes repeated calls inside Jtvec
                f = sim.fields(m) 
                d_pred = sim.dpred(m, f=f)
                
                res = Wd * (d_pred - d_pert)
                phi_d = 0.5 * np.sum(res**2)
                
                # Gradient: J.T * (W.T * W * r)
                grad_d = sim.Jtvec(m, Wd * res, f=f)
                
                # RML Regularization
                phi_s = 0.5 * np.sum((m - m_ref)**2)
                grad_s = (m - m_ref)
                
                return float(phi_d + 1.0 * phi_s), (grad_d + 1.0 * grad_s).astype(float)
            except:
                return 1e20, np.zeros_like(m)

        res = minimize(objective, m_start, method='L-BFGS-B', jac=True, bounds=solver_bounds,
                       options={'maxiter': 70, 'ftol': 1e-5, 'maxcor': 10})
        return res.x, model_map, res.fun
        '''
        # 1. Define Early Stopping Exception
        class TargetMisfitReached(Exception): pass
        target_phi_d = 0.5 * len(d_pert) * chifact 
        
        best_m = m_start.copy()
        best_phi = 1e20

        def objective(m):
            nonlocal best_m, best_phi
            try:
                f = sim.fields(m) 
                d_pred = sim.dpred(m, f=f)
                
                res = Wd * (d_pred - d_pert)
                phi_d = 0.5 * np.sum(res**2)
                
                # LOWER THIS WEIGHT to speed up convergence (e.g., 0.0 or 0.01)
                beta_weight = 0.0 
                phi_s = 0.5 * beta_weight * np.sum((m - m_ref)**2)
                current_obj = float(phi_d + phi_s)

                # EARLY STOPPING CHECK
                if phi_d <= target_phi_d:
                    best_m = m.copy()
                    best_phi = current_obj
                    raise TargetMisfitReached()
                
                grad_d = sim.Jtvec(m, Wd * res, f=f)
                grad_s = beta_weight * (m - m_ref)
                
                return current_obj, (grad_d + grad_s).astype(float)
            
            except TargetMisfitReached:
                raise
            except Exception as e:
                logging.getLogger(__name__).warning(f"Objective evaluation failed: {e}")
                return 1e20, np.zeros_like(m)

        try:
            res = minimize(objective, m_start, method='L-BFGS-B', jac=True, bounds=solver_bounds,
                           options={'maxiter': 70, 'ftol': 1e-5, 'maxcor': 10})
            final_m, final_phi = res.x, res.fun
        except TargetMisfitReached:
            final_m, final_phi = best_m, best_phi

        return final_m, model_map, final_phi
    # --- 4. EXECUTION ---
    m_init = np.r_[np.random.randn(n_physics)*0.01, np.array([sample_mean])]
    
    m_opt, final_map, final_phi = optimize_structure(sample_corr, m_init)
    
    return {'sigma': final_map * m_opt, 'z': m_opt[:n_physics], 'mu': m_opt[-1], 'phi_d': final_phi, 'corr': sample_corr}


class HRML:
    def __init__(self, nreals=100):
        self.nreals = nreals

    def run_local(self, Sounding, client=None):
        # 1. EXTRACT LIGHTWEIGHT PAYLOAD
        # We grab raw arrays. Pickling arrays is instant.
        
        # Access the first source to grab geometry/waveforms
        src0 = Sounding.srv.source_list[0] # LM
        src1 = Sounding.srv.source_list[1] # HM
        
        # Determine shapes
        # Note: SimPEG 1D usually stores Rx/Tx locs as arrays of shape (N, 3)
        rx_loc = src0.receiver_list[0].locations
        tx_shape = src0.location
        # Use the bipolar-safe optimizer
        #lm_t, lm_a = optimize_waveform_bipolar(src0.waveform.times, src0.waveform.currents)
        #hm_t, hm_a = optimize_waveform_bipolar(src1.waveform.times, src1.waveform.currents)
        payload_data = {
            'thicknesses': Sounding.inv_thickness,
            'rx_loc': rx_loc,
            'tx_shape': tx_shape,
            'lm_times': src0.receiver_list[0].times,
            'hm_times': src1.receiver_list[0].times,
            
            # SAFE EXTRACTION: Check if it's a PiecewiseLinearWaveform with times/currents
            'lm_wave_time': src0.waveform.times if hasattr(src0.waveform, 'times') else None,
            'lm_wave_form': src0.waveform.currents if hasattr(src0.waveform, 'currents') else None, 
            'hm_wave_time': src1.waveform.times if hasattr(src1.waveform, 'times') else None,
            'hm_wave_form': src1.waveform.currents if hasattr(src1.waveform, 'currents') else None
        }   
        
        # 2. CLIENT CHECK
        close_client = False
        if client is None:
            cluster = LocalCluster(n_workers=int(os.cpu_count())-1)
            client = Client(cluster)
            close_client = True
            print(f"Dask Dashboard: {client.dashboard_link}")

        try:
            # 3. SCATTER DATA (Crucial Step!)
            # Even with delayed objects, we scatter the heavy data first so the 
            # scheduler knows it's the same data for every task.
            print("Scattering lightweight arrays...")
            dobs_fut = client.scatter(Sounding.dobs, broadcast=True)
            unc_fut = client.scatter(Sounding.uncertainties, broadcast=True)
            pay_fut = client.scatter(payload_data, broadcast=True)

            # 4. BUILD DELAYED GRAPH
            # We generate the priors locally
            means = np.clip(np.random.normal(np.log(0.01), 1.5, self.nreals), np.log(1e-4), np.log(1))
            corrs = np.clip(np.random.lognormal(np.log(2.0), 0.2, self.nreals), 0.5, 15.0)

            print(f"Building {self.nreals} delayed tasks...")
            lazy_results = []
            for r in range(self.nreals):
                # Wrap the worker in dask.delayed
                # Passing futures (dobs_fut, etc) into delayed works perfectly in Dask
                task = dask.delayed(solve_rml_worker)(
                    r, dobs_fut, unc_fut, pay_fut, 
                    means[r], corrs[r]
                )
                lazy_results.append(task)

            # 5. EXECUTE GRAPH
            print("Computing...")
            results = dask.compute(*lazy_results)
            
            # Post-process expects a list, dask.compute returns a tuple
            self.post_process(Sounding, list(results))

        finally:
            if close_client:
                client.close()
                if 'cluster' in locals(): cluster.close()

    def post_process(self, Sounding, results):
        self.calreals = [r['sigma'] for r in results]
        self.z_stack = np.array([r['z'] for r in results])
        
        # Reduced Chi2 normalization
        n_data = len(Sounding.dobs)
        self.chivals = [(2 * r['phi_d']) / n_data for r in results]
        
        self.calib_factors = [r['corr'] for r in results]
        
        log_reals = np.log10(np.array(self.calreals) + 1e-12)
        self.p50 = 10**np.nanpercentile(log_reals, 50, axis=0)
        self.p5 = 10**np.nanpercentile(log_reals, 5, axis=0)
        self.p95 = 10**np.nanpercentile(log_reals, 95, axis=0)

        self.sfi = np.median(self.z_stack, axis=0)
        self.igp = np.median(np.abs(self.z_stack), axis=0)
        self.pprob = np.sign(self.sfi) * erf(np.abs(self.sfi) / np.sqrt(2))

        self.fits = self.chivals
        self.DOI_mean = 0.0; self.cdf = np.zeros(len(Sounding.Depths))
        self.tprob = np.zeros_like(self.pprob); self.ri_prob = np.zeros_like(self.pprob)

class HRML2:
    def __init__(self, nreals=100):
        self.nreals = nreals

    def run_local(self, Sounding, client=None):
        self.RML = HRML(nreals=self.nreals)
        self.RML.run_local(Sounding, client=client)
        # Copy results back to this instance for compatibility
        self.p50 = self.RML.p50
        self.p5 = self.RML.p5
        self.p95 = self.RML.p95
        self.calreals = self.RML.calreals
        self.chivals = self.RML.chivals
        self.calib_factors = self.RML.calib_factors

# -----------------------------------------------------------------------------
# 3. IES HELPERS AND POST-PROCESSOR
# -----------------------------------------------------------------------------

def run_ies_forward(p_input, physics_payload):
    if isinstance(p_input, dict):
        corr_len = p_input.get('corr_len', 2.5) 
        log_mean = p_input.get('log_mean', -4.6)
        n_layers = physics_payload['n_layers']
        z_vec = np.zeros(n_layers)
        for i in range(n_layers): z_vec[i] = p_input.get(f"z_{i:02d}", 0.0)
    else:
        p_vec = np.array(p_input)
        corr_len, log_mean, z_vec = p_vec[0], p_vec[1], p_vec[2:]

    survey = physics_payload['survey']
    thicknesses = physics_payload['thicknesses']
    n_layers = physics_payload['n_layers']
    mesh = TensorMesh([(np.r_[thicknesses, thicknesses[-1]])], "0")

    L = get_cholesky_decomposition(mesh, corr_len)
    log_cond = log_mean + (L @ z_vec)
    simulation = tdem.Simulation1DLayered(survey=survey, thicknesses=thicknesses, sigmaMap=maps.ExpMap(nP=n_layers))
    
    try:
        dpred = simulation.make_synthetic_data(log_cond, add_noise=False).dobs
    except Exception:
        dpred = np.ones(survey.nD) * np.nan
    return dict(zip([f"d_{i:02d}" for i in range(len(dpred))], dpred))

def post_process_batch_worker(p_vecs, param_names, physics_payload, dobs, unc):
    survey = physics_payload['survey']
    thicknesses = physics_payload['thicknesses']
    n_layers = physics_payload['n_layers']
    mesh = TensorMesh([(np.r_[thicknesses, thicknesses[-1]])], "0")
    simulation = tdem.Simulation1DLayered(survey=survey, thicknesses=thicknesses, sigmaMap=maps.ExpMap(nP=n_layers))
    
    results = []
    for p_vec in p_vecs:
        corr_len, log_mean, z_vec = p_vec[0], p_vec[1], p_vec[2:]
        L = get_cholesky_decomposition(mesh, corr_len)
        log_cond = log_mean + (L @ z_vec)
        sigma = np.exp(log_cond)
        try:
            dpred_obj = simulation.make_synthetic_data(log_cond, add_noise=False)
            dclean = dpred_obj.dobs
            residuals = (dobs - dclean) / unc
            chi2 = np.sum(residuals**2) / len(dobs)
            rele = np.mean(np.abs((dobs - dclean) / dobs))
        except Exception:
            dclean = np.zeros_like(dobs); chi2 = 99999.0; rele = 1.0
        results.append((sigma, dclean, chi2, rele, corr_len))
    return results

def get_cutoff(isounding, S, V, kmin=0.0001, kmax=10):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    Yemp = np.zeros(np.shape(V)[0])
    kt = []
    for s in range(0, np.shape(V)[1]):
        Y = Yemp.copy(); Y[s] = 1 
        Perrc = []
        for w in range(0, len(S)):
            S2E = (isounding.uncertainties**2)[w] 
            YtV_2 = []
            for i2 in range(w + 1, np.shape(V)[1]):
                Vi = V[:, i2]; YtV_2.append((Y.T @ Vi) ** 2)
            P1i = np.sum(YtV_2) * S2k
            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]; S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
                SiyTvi.append(S2inv2YTVi)
            P2i = np.sum(SiyTvi) * S2E
            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))

def cdf_for_value(data, x):
    data_array = np.array(data)
    count = np.sum(data_array <= x)
    total_count = len(data_array)
    if total_count == 0: return 0.0
    return count / total_count

def get_DOI(isounding, Cali, depths=False):
    try:
        if hasattr(Cali, 'values'): model_vals = Cali.values
        else: model_vals = Cali 
        temp_map = maps.ExpMap(nP=isounding.mesh.nC)
        temp_sim = tdem.Simulation1DLayered(survey=isounding.srv, thicknesses=isounding.inv_thickness, sigmaMap=temp_map)
        JW = temp_sim.getJ(m=np.log(model_vals))
        if not np.all(np.isfinite(JW)): raise np.linalg.LinAlgError("Jacobian issues")
        U, S, VT = scipy.linalg.svd(JW, lapack_driver='gesvd')
        V = VT.T
        k = get_cutoff(isounding, S, V, kmin=0.0001, kmax=3)
        V1 = V[:, :k]; U1 = U[:, :k]; S1 = np.diag(S[:k])
        S1inv = np.linalg.inv(S1); R = V1 @ V1.T
        DOIi = []
        for i in range(0, 500):
            E = np.random.normal(0, isounding.uncertainties, size=len(isounding.dobs))
            noise_response = V1 @ S1inv @ U1.T @ E
            rat = []
            for r in range(len(R)):
                kval = np.log(model_vals)[r]
                maxTrue = np.abs(R[r, r] * kval)
                maxNoise = np.abs(noise_response[r])
                if maxNoise < 1e-32: nR = 100.0
                else: nR = maxTrue / maxNoise
                rat.append(nR)
            try:
                valid_indices = np.where(np.array(rat) > 1)[0]
                if len(valid_indices) > 0: idx = valid_indices[-1]; DOIi.append(isounding.Depths[idx])
                else: DOIi.append(isounding.Depths[0])
            except: DOIi.append(isounding.Depths[0])
        final_dois = DOIi
    except (np.linalg.LinAlgError, ValueError):
        final_dois = [0.0] * 500
    if depths == False: cdf_results = [cdf_for_value(final_dois, x) for x in isounding.Depths]
    else: cdf_results = [cdf_for_value(final_dois, x) for x in np.arange(0, np.ceil(isounding.Depths.max()))]
    return cdf_results

# -----------------------------------------------------------------------------
# 4. IES CLASS
# -----------------------------------------------------------------------------
class IES:
    def __init__(self, nreals=50):
        self.nreals = nreals
        self.max_iter = 5
        self.initial_lambda = 1.0
        self.use_regularization = False

    def run_local(self, Sounding, client=None):
        obs_df = pd.DataFrame({"value": Sounding.dobs, "std": Sounding.uncertainties}, 
                              index=[f"d_{i:02d}" for i in range(len(Sounding.dobs))])
        
        param_list = [{"name": "corr_len", "prior_mean": 2, "prior_std": 2.0, "pmin": 1.0, "pmax": 30.0},
                      {"name": "log_mean", "prior_mean": np.log(0.01), "prior_std": 1.0, "pmin": np.log(1e-4), "pmax": np.log(0.1)}]
        for i in range(Sounding.mesh.nC):
            param_list.append({"name": f"z_{i:02d}", "prior_mean": 0.0, "prior_std": 1.0, "pmin": -5, "pmax": 5})
        param_df = pd.DataFrame(param_list).set_index("name")
        
        # --- FIX: Store param_names in the IES class instance ---
        self.param_names = param_df.index.tolist()
        # --------------------------------------------------------

        self.physics_payload = {'survey': Sounding.srv, 'thicknesses': Sounding.inv_thickness, 'n_layers': Sounding.mesh.nC}
        model_func = functools.partial(run_ies_forward, physics_payload=self.physics_payload)
        
        self.smoother = LMEnsembleSmoother(model_func=model_func, param_df=param_df, obs_df=obs_df, 
                                           num_ensemble=self.nreals, client=client, transform_parameters=True)
        self.smoother.initialize_priors(phys_bounds=param_df[['pmin', 'pmax']])
        self.smoother.solve(max_iterations=self.max_iter, initial_lambda=self.initial_lambda, enforce_bounds=param_df[['pmin', 'pmax']])
        
        self.post_process(Sounding)

    def post_process(self, Sounding, client=None):
        print("Starting Batched Parallel Post-Processing...")
        P_real = self.smoother.inverse_transform(self.smoother.P)
        if hasattr(P_real, "values"): P_real = P_real.values
        elif hasattr(P_real, "to_numpy"): P_real = P_real.to_numpy()
        
        # Z-Metrics
        z_ensemble = P_real[:, 2:] 
        self.sfi = np.median(z_ensemble, axis=0)
        self.igp = np.median(np.abs(z_ensemble), axis=0)
        self.pprob = np.sign(self.sfi) * erf(np.abs(self.sfi) / np.sqrt(2))
        
        # --- FIX: Use self.param_names instead of self.smoother.param_df ---
        param_names = self.param_names
        # -------------------------------------------------------------------
        
        BATCH_SIZE = 10
        lazy_results = []
        for i in range(0, self.nreals, BATCH_SIZE):
            batch_vecs = P_real[i : i + BATCH_SIZE]
            task = dask.delayed(post_process_batch_worker)(
                batch_vecs, param_names, self.physics_payload, Sounding.dobs, Sounding.uncertainties
            )
            lazy_results.append(task)
            
        nested_results = dask.compute(*lazy_results)
        flat_results = [item for sublist in nested_results for item in sublist]
        
        self.calreals = []; self.preds = []; self.chivals = []; self.fits = []; self.calib_factors = []; self.DOIs = []
        class PredWrapper:
            def __init__(self, d): self.dclean = d

        for res in flat_results:
            sigma, dclean, chi2, rele, c_len = res
            self.calreals.append(sigma)
            self.preds.append(PredWrapper(dclean))
            self.chivals.append(chi2)
            self.fits.append(rele)
            self.calib_factors.append(c_len)
            self.DOIs.append([0.0]*len(Sounding.Depths)) 

        log_reals = np.log10(np.array(self.calreals) + 1e-12)
        self.p50 = 10**np.quantile(log_reals, 0.5, axis=0)
        self.p5 = 10**np.quantile(log_reals, 0.05, axis=0)
        self.p95 = 10**np.quantile(log_reals, 0.95, axis=0)
        
        p_peak = np.zeros(len(Sounding.Depths)); p_trough = np.zeros(len(Sounding.Depths))  
        p_rise = np.zeros(len(Sounding.Depths)); p_fall = np.zeros(len(Sounding.Depths))

        for sigma in self.calreals:
            log_real = np.log10(sigma)
            idx_peaks, _ = find_peaks(log_real, prominence=0.001, width=2)
            p_peak[idx_peaks] += 1
            idx_troughs, _ = find_peaks(-log_real, prominence=0.001, width=2)
            p_trough[idx_troughs] += 1
            grad = np.gradient(log_real)
            p_rise += grad > 0.01
            p_fall += grad < -0.01

        self.tprob = p_trough / self.nreals    
        self.ri_prob = p_rise / self.nreals    
        self.fa_prob = p_fall / self.nreals    
        self.cdf = np.zeros(len(Sounding.Depths))
        self.DOI_mean = 0; self.DOI_std = 0
        print("Post-processing complete.")
# -----------------------------------------------------------------------------
# 5. SOUNDING CLASS
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

        # Define Geometry
        rx_loc = np.array(Survey.rx_offset) + [self.UTMX, self.UTMY, self.RX_ALTITUDE]
        tx_loc = np.array(Survey.tx_shape) + [self.UTMX, self.UTMY, self.TX_ALTITUDE]
        self.tx_area = Survey.tx_area

        # FIXED Waveform Logic
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
                tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, Survey.lm_times, "z"),
                tx_loc
            )
            src_hm = tdem.sources.LineCurrent(
                tdem.receivers.PointMagneticFluxTimeDerivative(rx_loc, Survey.hm_times, "z"),
                tx_loc
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
            self.uncertainties = np.sqrt((np.r_[lm_std, hm_std] * self.tx_area)**2 + noise_floor**2)

        self.data_object = Data(self.srv, dobs=self.dobs, standard_deviation=self.uncertainties)
        self.mesh = TensorMesh([(np.r_[self.inv_thickness, self.inv_thickness[-1]])], "0")

    def get_IES_reals(self, nreals, client=None):
        self.RML = IES(nreals=nreals)
        self.RML.run_local(self, client=client)

    def get_HRML_reals(self, nreals, client=None):
        self.RML = HRML(nreals=nreals)
        self.RML.run_local(self, client=client)
        
    def get_RML_reals(self, nreals, Lrange=20, ival=0.05, lower=0.00001, upper=10, tpw=1, memlim="4GB"):
        """Prepares the RML stochastic ensemble (Legacy)."""
        self.RML = RML(Lrange=Lrange, ival=ival, lower=lower, upper=upper, tpw=tpw, memlim=memlim)
        self.RML.setup_hierarchical_priors(self, nreals)
        self.RML.get_perturbed_data(self, nreals)
        self.RML.prep_parruns(self, nreals)

# -----------------------------------------------------------------------------
# 6. RML / CALIBRATION (Full Legacy Restoration)
# -----------------------------------------------------------------------------

class RML:
    def __init__(self, Lrange, ival, lower, upper, tpw, memlim):
        self.Lrange = Lrange; self.ival = ival; self.lower = lower; self.upper = upper; self.tpw = tpw; self.memlim = memlim
    def setup_hierarchical_priors(self, Sounding, nreals):
        self.Depths = Sounding.Depths; self.nreals = nreals
        self.stochastic_params_list = []
        for i in range(self.nreals):
            seed = 1000 + i; rng = np.random.RandomState(seed=seed)
            self.stochastic_params_list.append({'seed': seed, 'corr_factor': 2, 'mean_prior': rng.uniform(np.log(0.005), np.log(0.01)), 'std_scale': 1.0})
    def get_perturbed_data(self, Sounding, nreals):
        pobs = []
        for index in range(len(Sounding.dobs)):
            obsreals = np.random.normal(Sounding.dobs[index], Sounding.uncertainties[index], nreals)
            pobs.append(obsreals)
        self.pobs = np.array(pobs).T
        return self.pobs
    def prep_parruns(self, Sounding, nreals):
        self.lazy_results = []
        for i in range(nreals):
            Cbi = Calibration(); Cbi.lower = self.lower; Cbi.upper = self.upper
            Sounding.data_object = Data(Sounding.srv, dobs=self.pobs[i], standard_deviation=Sounding.uncertainties)
            self.lazy_results.append(dask.delayed(Cbi.calibrate)(Sounding, self.stochastic_params_list[i]))

class Calibration:
    use_weights = True; maxIter = 30; tolCG = 1e-5; beta0_ratio = 1e-2; coolEpsFact = 2; verbose = False
    def __init__(self): pass
    def calibrate(self, Sounding, stochastic_params):
        try:
            seed = stochastic_params['seed']
            rng = np.random.RandomState(seed=seed)
            L_mat = get_cholesky_decomposition(Sounding.mesh, stochastic_params['corr_factor'])
            geo_map = GeostatisticalMapping(Sounding.mesh, L_mat)
            phys_map = maps.ExpMap(nP=Sounding.mesh.nC); model_mapping = phys_map * geo_map 
            z_init = rng.randn(Sounding.mesh.nC) * stochastic_params['std_scale']
            m_latent_init = np.r_[stochastic_params['mean_prior'], z_init]
            self.simulation = tdem.Simulation1DLayered(survey=Sounding.srv, thicknesses=Sounding.inv_thickness, sigmaMap=model_mapping)
            self.dmis = L2DataMisfit(simulation=self.simulation, data=Sounding.data_object)
            if self.use_weights: self.dmis.W = 1.0 / Sounding.uncertainties
            reg_mesh = TensorMesh([np.ones(Sounding.mesh.nC + 1)])
            self.reg = regularization.WeightedLeastSquares(reg_mesh, mapping=maps.IdentityMap(nP=Sounding.mesh.nC + 1))
            self.reg.set_weights(prior_weights=np.sqrt(np.r_[0.1, np.ones(Sounding.mesh.nC)]))
            self.reg.reference_model = m_latent_init
            self.opt = optimization.ProjectedGNCG(maxIter=self.maxIter, tolCG=self.tolCG)
            self.inv_prob = inverse_problem.BaseInvProblem(self.dmis, self.reg, self.opt)
            self.directives_list = [directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio), directives.BetaSchedule(coolingFactor=self.coolEpsFact, coolingRate=1), directives.TargetMisfit(chifact=1.0)]
            self.inv = inversion.BaseInversion(self.inv_prob, self.directives_list)
            self.recovered_latent = self.inv.run(m_latent_init)
            self.values = model_mapping * self.recovered_latent
            pred = fsim(Sounding.srv, Sounding.mesh, self.values)
            residuals = (Sounding.dobs - pred.dclean) / Sounding.uncertainties
            self.CHi2 = np.sum(residuals**2) / len(Sounding.dobs)
            self.rele = np.mean(np.abs((Sounding.dobs - pred.dclean) / Sounding.dobs))
            self.DOI = get_DOI(isounding=Sounding, Cali=self)
            return {"values": self.values, "rele": self.rele, "pred": pred, "DOI": self.DOI, "CHI2": self.CHi2, "corr_factor": stochastic_params['corr_factor'], "success": True}
        except Exception: return {"success": False}

# -----------------------------------------------------------------------------
# 7. OUTPUT UTILS
# -----------------------------------------------------------------------------

def adjust_dtype(var): return int(var) if isinstance(var, np.integer) else var

def proc_output(out, fd_output_sounding):
    time, isounding = out
    os.makedirs(fd_output_sounding, exist_ok=True)
    
    # Robust retrieval for both IES and HRML metrics
    df_rml = pd.DataFrame({
        "depth": isounding.inv_thickness.cumsum(),
        "p5": getattr(isounding.RML, 'p5', isounding.RML.p50)[:-1],
        "p50": isounding.RML.p50[:-1],
        "p95": getattr(isounding.RML, 'p95', isounding.RML.p50)[:-1],
        "pprob": isounding.RML.pprob[:-1],
        "igp": getattr(isounding.RML, 'igp', np.zeros_like(isounding.RML.pprob))[:-1],
        "tprob": getattr(isounding.RML, 'tprob', np.zeros_like(isounding.RML.pprob))[:-1],
        "ri_prob": getattr(isounding.RML, 'ri_prob', np.zeros_like(isounding.RML.pprob))[:-1],
        "fa_prob": getattr(isounding.RML, 'fa_prob', np.zeros_like(isounding.RML.pprob))[:-1],
        "doicdf": isounding.RML.cdf,
    })

    edict = {}
    edict["times"] = isounding.times
    edict["obs"] = isounding.dobs
    if hasattr(isounding.RML, 'preds') and len(isounding.RML.preds) > 0:
        for i in range(0, len(isounding.RML.preds)):
            edict["real_" + str(i)] = isounding.RML.preds[i].dclean
    df_obs = pd.DataFrame(edict)

    df_calreals = pd.DataFrame(isounding.RML.calreals)

    dic_vars = {
        "lineno": adjust_dtype(isounding.iline),
        "time": adjust_dtype(isounding.time),
        "easting": adjust_dtype(isounding.UTMX),
        "northing": adjust_dtype(isounding.UTMY),
        "mean_relerr": adjust_dtype(isounding.RML.fits),
        "calibration_factors": [adjust_dtype(x) for x in isounding.RML.calib_factors] 
    }

    df_rml.to_parquet(os.path.join(fd_output_sounding, "rml.gz.parquet"), index=False)
    df_obs.to_parquet(os.path.join(fd_output_sounding, "obs.gz.parquet"), index=False)
    df_calreals.to_parquet(os.path.join(fd_output_sounding, "preds.gz.parquet"), index=False)
    with open(os.path.join(fd_output_sounding, "variables.json"), "w") as f:
        json.dump(dic_vars, f, indent=4)
