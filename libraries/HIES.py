import json
import os
import sys
import scipy.linalg
import dask
import numpy as np
import pandas as pd
import simpeg.electromagnetics.time_domain as tdem
from dask.distributed import Client, LocalCluster
from discretize import TensorMesh
from scipy.signal import find_peaks
from simpeg import (
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)
from simpeg.data import Data
from simpeg.data_misfit import L2DataMisfit
import functools

# --- IMPORT ENSEMBLE SMOOTHER ---
try:
    from .ES import LMEnsembleSmoother
except ImportError:
    try:
        from ES import LMEnsembleSmoother
    except ImportError:
        print("Warning: ES.py not found. IES functionality will not work.")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Helper context manager to suppress stdout/stderr
class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self.devnull.close()


def fsim(srv, mesh, pars):
    """Forward simulation wrapper."""
    model_mapping = maps.ExpMap(nP=mesh.nC)
    simulation = tdem.Simulation1DLayered(
        survey=srv, thicknesses=mesh.h[0][:-1], sigmaMap=model_mapping
    )
    pred = simulation.make_synthetic_data(np.log(pars))
    return pred


def is_scalar(obj):
    if isinstance(obj, np.ndarray):
        return obj.shape == ()
    else:
        return np.isscalar(obj)


# -----------------------------------------------------------------------------
# GEOSTATISTICAL HELPERS
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
        if v is not None:
            return J @ v
        return J

def get_cholesky_decomposition(mesh, correlation_factor):
    """
    Computes Cholesky factorization.
    Index-based (Layers) = Non-Stationary in physical space.
    """
    nC = mesh.nC
    idx = np.arange(nC)
    index_dist = np.abs(idx[:, None] - idx[None, :])

    # Ensure correlation factor is safe
    correlation_factor = max(0.1, float(correlation_factor))

    # Correlation decays based on INDEX distance (Layers)
    C = np.exp(-1.0 * (index_dist / correlation_factor))
    C += np.eye(nC) * 1e-6

    try:
        L_mat = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        U, S, _ = np.linalg.svd(C)
        L_mat = U @ np.diag(np.sqrt(S))
    return L_mat


# -----------------------------------------------------------------------------
# WORKER FUNCTIONS (Lightweight & Robust)
# -----------------------------------------------------------------------------

def run_ies_forward(p_input, physics_payload):
    """
    Worker for Inversion Loop (Single Realization).
    Robustly handles both Dictionary and Numpy Array inputs.
    """
    # 1. Parse Input (Handle both Dict and Array)
    if isinstance(p_input, dict):
        # Input is Dictionary (Safe, named)
        corr_len = p_input.get('corr_len', 2.5)
        log_mean = p_input.get('log_mean', -4.6)

        # Extract z_vec from keys like 'z_00', 'z_01'
        n_layers = physics_payload['n_layers']
        z_vec = np.array([p_input.get(f"z_{i:02d}", 0.0) for i in range(n_layers)])

    else:
        # Input is Numpy Array (Fast, unnamed)
        # We assume strict order: [0]=corr_len, [1]=log_mean, [2:]=z_vec
        p_vec = np.array(p_input)
        corr_len = p_vec[0]
        log_mean = p_vec[1]
        z_vec = p_vec[2:]

    # 2. Setup Physics
    survey = physics_payload['survey']
    thicknesses = physics_payload['thicknesses']
    n_layers = physics_payload['n_layers']

    # Rebuild Mesh locally
    mesh = TensorMesh([(np.r_[thicknesses, thicknesses[-1]])], "0")

    # 3. Hierarchical Mapping
    L = get_cholesky_decomposition(mesh, corr_len)
    log_cond = log_mean + (L @ z_vec)

    # 4. Run Simulation
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=maps.ExpMap(nP=n_layers)
    )

    try:
        # Pass log_cond directly
        dpred_obj = simulation.make_synthetic_data(log_cond, add_noise=False)
        dpred = dpred_obj.dobs
    except Exception:
        dpred = np.ones(survey.nD) * np.nan

    # Return Dict
    obs_names = [f"d_{i:02d}" for i in range(len(dpred))]
    return dict(zip(obs_names, dpred))


def post_process_batch_worker(p_vecs, param_names, physics_payload, dobs, unc):
    """
    Worker for Post-Processing Loop (Batched).
    Processes multiple realizations in one call to amortize SimPEG setup overhead.
    """
    # 1. Setup Physics (ONCE per batch)
    survey = physics_payload['survey']
    thicknesses = physics_payload['thicknesses']
    n_layers = physics_payload['n_layers']
    mesh = TensorMesh([(np.r_[thicknesses, thicknesses[-1]])], "0")

    # Pre-allocate simulation object
    simulation = tdem.Simulation1DLayered(
        survey=survey, thicknesses=thicknesses, sigmaMap=maps.ExpMap(nP=n_layers)
    )

    results = []

    # 2. Loop through the batch
    for p_vec in p_vecs:
        # We know p_vec is an array from the batched slice
        # Map indices manually for speed: [0]=corr, [1]=mean, [2:]=z
        corr_len = p_vec[0]
        log_mean = p_vec[1]
        z_vec = p_vec[2:]

        # Transform
        L = get_cholesky_decomposition(mesh, corr_len)
        log_cond = log_mean + (L @ z_vec)
        sigma = np.exp(log_cond)

        # Forward Physics
        try:
            dpred_obj = simulation.make_synthetic_data(log_cond, add_noise=False)
            dclean = dpred_obj.dobs

            residuals = (dobs - dclean) / unc
            chi2 = np.sum(residuals**2) / len(dobs)
            rele = np.mean(np.abs((dobs - dclean) / dobs))
        except Exception:
            dclean = np.zeros_like(dobs)
            chi2 = 99999.0
            rele = 1.0

        results.append((sigma, dclean, chi2, rele, corr_len))

    return results


# -----------------------------------------------------------------------------
# DOI HELPERS
# -----------------------------------------------------------------------------
def get_cutoff(isounding, S, V, kmin=0.0001, kmax=10):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    Yemp = np.zeros(np.shape(V)[0])
    kt = []
    for s in range(0, np.shape(V)[1]):
        Y = Yemp.copy()
        Y[s] = 1
        Perrc = []
        for w in range(0, len(S)):
            S2E = (isounding.uncertainties**2)[w]
            YtV_2 = []
            for i2 in range(w + 1, np.shape(V)[1]):
                Vi = V[:, i2]
                YtV_2.append((Y.T @ Vi) ** 2)
            P1i = np.sum(YtV_2) * S2k
            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]
                S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
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
        # Handle both RML (Cali object) and IES (numpy array) inputs
        if hasattr(Cali, 'values'):
            model_vals = Cali.values
        else:
            model_vals = Cali

        temp_map = maps.ExpMap(nP=isounding.mesh.nC)
        temp_sim = tdem.Simulation1DLayered(
            survey=isounding.srv,
            thicknesses=isounding.inv_thickness,
            sigmaMap=temp_map
        )

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
                if len(valid_indices) > 0:
                    idx = valid_indices[-1]
                    DOIi.append(isounding.Depths[idx])
                else:
                    DOIi.append(isounding.Depths[0])
            except:
                DOIi.append(isounding.Depths[0])
        final_dois = DOIi
    except (np.linalg.LinAlgError, ValueError):
        final_dois = [0.0] * 500

    if depths == False:
        cdf_results = [cdf_for_value(final_dois, x) for x in isounding.Depths]
    else:
        cdf_results = [cdf_for_value(final_dois, x) for x in np.arange(0, np.ceil(isounding.Depths.max()))]
    return cdf_results


# -----------------------------------------------------------------------------
# IES CLASS
# -----------------------------------------------------------------------------

class IES:
    def __init__(self, nreals=50):
        self.nreals = nreals
        self.max_iter = 5
        self.initial_lambda = 1.0
        self.use_regularization = False

    def run_local(self, Sounding, cluster=None, client=None):
        self.ncores = int(os.cpu_count()) - 1

        # 1. Setup Dataframes
        obs_names = [f"d_{i:02d}" for i in range(len(Sounding.dobs))]
        self.obs_df = pd.DataFrame({
            "value": Sounding.dobs,
            "std": Sounding.uncertainties
        }, index=obs_names)

        param_list = []
        # A. Correlation Length (Hierarchical Parameter)
        # Using Safe Bounds (1.0 to 10.0)
        param_list.append({
            "name": "corr_len", "prior_mean": 10, "prior_std": 2.0,
            "pmin": 1.0, "pmax": 30.0
        })
        # B. Mean Conductivity
        param_list.append({
            "name": "log_mean", "prior_mean": np.log(0.01), "prior_std": 2.0,
            "pmin": np.log(1e-4), "pmax": np.log(1.0)
        })
        # C. Latent Variables (Structure)
        # Using Safe Bounds (-3.5 to 3.5)
        for i in range(Sounding.mesh.nC):
            param_list.append({
                "name": f"z_{i:02d}", "prior_mean": 0.0, "prior_std": 1.0,
                "pmin": -3.5, "pmax": 3.5
            })

        self.param_df = pd.DataFrame(param_list).set_index("name")
        self.bounds_df = self.param_df[['pmin', 'pmax']]

        # 2. Dask Setup
        self.closeflag = False
        if (cluster is None) and (client is None):
            self.closeflag = True
            cluster = LocalCluster(n_workers=self.ncores)
            client = Client(cluster)

        # 3. Create Payload (Critical for stability)
        self.physics_payload = {
            'survey': Sounding.srv,
            'thicknesses': Sounding.inv_thickness,
            'n_layers': Sounding.mesh.nC
        }

        # 4. Initialize & Run Smoother
        model_func = functools.partial(run_ies_forward, physics_payload=self.physics_payload)

        self.smoother = LMEnsembleSmoother(
            model_func=model_func,
            param_df=self.param_df,
            obs_df=self.obs_df,
            num_ensemble=self.nreals,
            client=client,
            random_seed=42,
            regularize_parameters=self.use_regularization,
            reg_weight=0.0,
            transform_parameters=True
        )

        self.smoother.initialize_priors(phys_bounds=self.bounds_df)

        final_params = self.smoother.solve(
            max_iterations=self.max_iter,
            initial_lambda=self.initial_lambda,
            enforce_bounds=self.bounds_df,
            apply_jitter=False
        )

        # 5. Post-Process (Batched & Lazy)
        self.post_process(Sounding, client)

        # 6. Cleanup
        if self.closeflag:
            client.close()
            cluster.close()

    def post_process(self, Sounding, client=None):
        """
        Batched parallel post-processing to calculate physical properties and statistics.
        """
        print("Starting Batched Parallel Post-Processing...")

        # 1. Get Final Parameters
        P_real = self.smoother.inverse_transform(self.smoother.P)

        # --- CRITICAL FIX: Ensure it is a Numpy Array ---
        if hasattr(P_real, "values"):
            P_real = P_real.values
        elif hasattr(P_real, "to_numpy"):
            P_real = P_real.to_numpy()

        param_names = self.param_df.index.tolist()

        # 2. Define Batch Size
        BATCH_SIZE = 10
        lazy_results = []
        total_reals = self.nreals

        # 3. Create Batches
        for i in range(0, total_reals, BATCH_SIZE):
            batch_vecs = P_real[i : i + BATCH_SIZE]

            task = dask.delayed(post_process_batch_worker)(
                batch_vecs,
                param_names,
                self.physics_payload,
                Sounding.dobs,
                Sounding.uncertainties
            )
            lazy_results.append(task)

        # 4. Compute (Returns list of lists)
        nested_results = dask.compute(*lazy_results)

        # 5. Flatten Results
        flat_results = [item for sublist in nested_results for item in sublist]

        # 6. Unpack
        self.calreals = []
        self.preds = []
        self.chivals = []
        self.fits = []
        self.calib_factors = []
        self.DOIs = []

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

        # 7. Compute Ensemble Statistics
        log_reals = np.log10(np.array(self.calreals) + 1e-12)
        self.p50 = 10**np.quantile(log_reals, 0.5, axis=0)
        self.p5 = 10**np.quantile(log_reals, 0.05, axis=0)
        self.p95 = 10**np.quantile(log_reals, 0.95, axis=0)

        # 8. Feature Probability Calculation
        p_peak = np.zeros(len(Sounding.Depths))
        p_trough = np.zeros(len(Sounding.Depths))
        p_rise = np.zeros(len(Sounding.Depths))
        p_fall = np.zeros(len(Sounding.Depths))

        for sigma in self.calreals:
            log_real = np.log10(sigma)
            idx_peaks, _ = find_peaks(log_real, prominence=0.001, width=2)
            p_peak[idx_peaks] += 1
            idx_troughs, _ = find_peaks(-log_real, prominence=0.001, width=2)
            p_trough[idx_troughs] += 1
            grad = np.gradient(log_real)
            p_rise += grad > 0.01
            p_fall += grad < -0.01

        self.pprob = p_peak / self.nreals
        self.tprob = p_trough / self.nreals
        self.ri_prob = p_rise / self.nreals
        self.fa_prob = p_fall / self.nreals
        self.igp = (p_rise - p_fall) / self.nreals

        self.cdf = np.zeros(len(Sounding.Depths))
        self.DOI_mean = 0; self.DOI_std = 0

        print("Post-processing complete.")


# -----------------------------------------------------------------------------
# SOUNDING CLASS
# -----------------------------------------------------------------------------

class Sounding:
    def __init__(self, Survey, iline, time, inv_thickness, use_relerr=False, unc=None):
        self.iline = iline
        self.time = time
        self.inv_thickness = inv_thickness
        self.Depths = np.r_[
            self.inv_thickness.cumsum(),
            self.inv_thickness.cumsum()[-1] + self.inv_thickness[-1],
        ]
        self.use_relerr = use_relerr
        self.unc = unc

        try:
            self.runc_offset = Survey.Data.runc_offset
        except AttributeError:
            self.runc_offset = 0.03

        # 1. LOAD STATION DATA
        self.station_data = Survey.Data.station_data[
            Survey.Data.station_data.index == (iline, time)
        ]
        self.UTMX = self.station_data.UTMX.values[0]
        self.UTMY = self.station_data.UTMY.values[0]
        self.Elevation = self.station_data.ELEVATION.values[0]
        self.TX_ALTITUDE = self.station_data.TX_ALTITUDE.values[0]
        self.RX_ALTITUDE = self.station_data.RX_ALTITUDE.values[0]

        self.station_lm_data = Survey.Data.lm_data[
            Survey.Data.lm_data.index == (iline, time)
        ].to_numpy(dtype=float)[0]
        self.station_hm_data = Survey.Data.hm_data[
            Survey.Data.hm_data.index == (iline, time)
        ].to_numpy(dtype=float)[0]

        self.station_lm_std = Survey.Data.lm_std[
            Survey.Data.lm_std.index == (iline, time)
        ].to_numpy(dtype=float)[0]
        self.station_hm_std = Survey.Data.hm_std[
            Survey.Data.hm_std.index == (iline, time)
        ].to_numpy(dtype=float)[0]

        # 2. DEFINE SYSTEM GEOMETRY
        tx_shape_array = np.array(Survey.tx_shape)
        offset_array = np.array([self.UTMX, self.UTMY, self.TX_ALTITUDE])

        self.tx_loc = tx_shape_array + offset_array
        self.rx_loc = np.array(Survey.rx_offset) + np.array(
            [self.UTMX, self.UTMY, self.RX_ALTITUDE]
        )
        self.tx_area = Survey.tx_area

        rx_lm = tdem.receivers.PointMagneticFluxTimeDerivative(
            self.rx_loc, Survey.lm_times, orientation="z"
        )
        lm_wave = tdem.sources.PiecewiseLinearWaveform(
            Survey.lm_wave_time, Survey.lm_wave_form
        )
        src_lm = tdem.sources.LineCurrent(rx_lm, self.tx_loc, waveform=lm_wave)

        rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
            self.rx_loc, Survey.hm_times, orientation="z"
        )
        hm_wave = tdem.sources.PiecewiseLinearWaveform(
            Survey.hm_wave_time, Survey.hm_wave_form
        )
        src_hm = tdem.sources.LineCurrent(rx_hm, self.tx_loc, waveform=hm_wave)

        # 3. CONSOLIDATE DATA
        self.srv = tdem.Survey([src_lm, src_hm])
        self.dobs = -self.tx_area * np.r_[self.station_lm_data, self.station_hm_data]
        self.times = np.r_[Survey.lm_times, Survey.hm_times]

        noise_floor = 1e-15

        if (self.use_relerr):
            self.relerr = np.ones_like(self.dobs) * self.runc_offset
            self.uncertainties = np.sqrt((self.dobs * self.runc_offset) ** 2 + noise_floor**2)
        else:
            self.relerr = np.abs((np.r_[self.station_lm_std, self.station_hm_std])/(np.r_[self.station_lm_data, self.station_hm_data]))
            self.uncertainties = np.sqrt(
                (self.dobs * self.relerr) ** 2 + noise_floor**2
            )

        self.data_object = Data(
            self.srv, dobs=self.dobs, standard_deviation=self.uncertainties
        )

        self.mesh = TensorMesh(
            [(np.r_[self.inv_thickness, self.inv_thickness[-1]])], "0"
        )
        self.model_mapping = maps.ExpMap(nP=self.mesh.nC)
        self.simulation = tdem.Simulation1DLayered(
            survey=self.srv, thicknesses=self.inv_thickness, sigmaMap=self.model_mapping
        )

    def get_RML_reals(
        self, nreals, Lrange=20, ival=0.05, lower=0.00001, upper=10, tpw=1, memlim="4GB"
    ):
        """Prepares the RML stochastic ensemble."""
        self.RML = RML(
            Lrange=Lrange, ival=ival, lower=lower, upper=upper, tpw=tpw, memlim=memlim
        )
        self.RML.setup_hierarchical_priors(self, nreals)
        self.RML.get_perturbed_data(self, nreals)
        self.RML.prep_parruns(self, nreals)

    def get_IES_reals(self, nreals, client=None):
        """Prepares and runs the IES stochastic ensemble."""
        self.RML = IES(nreals=nreals)
        self.RML.run_local(self, client=client)


# -----------------------------------------------------------------------------
# RML CLASS (Legacy Support)
# -----------------------------------------------------------------------------

class RML:
    def __init__(self, Lrange, ival, lower, upper, tpw, memlim):
        self.Lrange = Lrange
        self.ival = ival
        self.lower = lower
        self.upper = upper
        self.tpw = tpw
        self.memlim = memlim

    def setup_hierarchical_priors(self, Sounding, nreals):
        self.Depths = Sounding.Depths
        self.nreals = nreals
        self.stochastic_params_list = []
        for i in range(self.nreals):
            seed = 1000 + i
            rng = np.random.RandomState(seed=seed)
            random_corr_factor = 3
            mean_val_log = rng.uniform(np.log(0.005), np.log(0.01))
            params = {
                'seed': seed,
                'corr_factor': random_corr_factor,
                'mean_prior': mean_val_log,
                'std_scale': 1.0
            }
            self.stochastic_params_list.append(params)

    def get_perturbed_data(self, Sounding, nreals):
        pobs = []
        for index in range(len(Sounding.dobs)):
            obs = Sounding.dobs[index]
            std = Sounding.uncertainties[index]
            obsreals = np.random.normal(obs, std, nreals)
            pobs.append(obsreals)
        self.pobs = np.array(pobs).T
        return self.pobs

    def prep_parruns(self, Sounding, nreals):
        self.lazy_results = []
        for i in range(nreals):
            Cbi = Calibration()
            Cbi.lower = self.lower
            Cbi.upper = self.upper
            params = self.stochastic_params_list[i]
            pert_obs = self.pobs[i]
            Sounding.data_object = Data(
                Sounding.srv, dobs=pert_obs, standard_deviation=Sounding.uncertainties
            )
            lazy_result = dask.delayed(Cbi.calibrate)(Sounding, params)
            self.lazy_results.append(lazy_result)

    def run_local(self, cluster=None, client=None):
        self.ncores = int(os.cpu_count()) - 1
        if (cluster is None) and (client is None):
            self.closeflag = True
            cluster = LocalCluster(
                threads_per_worker=self.tpw,
                n_workers=self.ncores,
                memory_limit=self.memlim,
            )
        else:
            self.closeflag = False

        if client is not None:
            results = dask.compute(*self.lazy_results)
        else:
            with Client(cluster) as client:
                results = dask.compute(*self.lazy_results)
            if self.closeflag:
                cluster.close()

        valid_results = [res for res in results if res.get("success", False)]
        self.nreals = len(valid_results)
        self.calreals = [x["values"] for x in valid_results]
        self.fits = [x["rele"] for x in valid_results]
        self.preds = [x["pred"] for x in valid_results]
        self.DOIs = [x["DOI"] for x in valid_results]
        self.chivals = [x["CHI2"] for x in valid_results]
        self.calib_factors = [x["corr_factor"] for x in valid_results]

        log_reals = np.log10(np.array(self.calreals) + 1e-12)
        self.p50 = 10**np.quantile(log_reals, 0.5, axis=0)
        self.p5 = 10**np.quantile(log_reals, 0.05, axis=0)
        self.p95 = 10**np.quantile(log_reals, 0.95, axis=0)

        # Feature Prob calc (Same as IES class)
        # ... (Omitted for brevity, redundant logic)

    def generate_prior_ensemble(self, Sounding):
        if not hasattr(self, "stochastic_params_list"):
            raise RuntimeError("Run setup_hierarchical_priors() first.")

        self.prior_reals = []
        for params in self.stochastic_params_list:
            seed = params['seed']
            corr_factor = params['corr_factor']
            mean_prior = params['mean_prior']
            std_scale = params['std_scale']

            rng = np.random.RandomState(seed=seed)
            L_mat = get_cholesky_decomposition(Sounding.mesh, corr_factor)
            geo_map = GeostatisticalMapping(Sounding.mesh, L_mat)
            phys_map = maps.ExpMap(nP=Sounding.mesh.nC)
            model_mapping = phys_map * geo_map

            z_init = rng.randn(Sounding.mesh.nC) * std_scale
            m_latent_init = np.r_[mean_prior, z_init]
            m_physical = model_mapping * m_latent_init
            self.prior_reals.append(m_physical)

        self.prior_reals = np.array(self.prior_reals)
        return self.prior_reals


# -----------------------------------------------------------------------------
# CALIBRATION CLASS (Legacy Support)
# -----------------------------------------------------------------------------
class Calibration:
    use_weights = True
    maxIter = 30
    tolCG = 1e-5
    beta0_ratio = 1e-2
    coolEpsFact = 2
    verbose = False
    def __init__(self): pass
    def calibrate(self, Sounding, stochastic_params):
        try:
            def calfunc(self, Sounding, stochastic_params):
                seed = stochastic_params['seed']
                corr_factor = stochastic_params['corr_factor']
                mean_prior = stochastic_params['mean_prior']
                std_scale = stochastic_params['std_scale']
                self.corr_factor = corr_factor
                rng = np.random.RandomState(seed=seed)
                L_mat = get_cholesky_decomposition(Sounding.mesh, corr_factor)
                geo_map = GeostatisticalMapping(Sounding.mesh, L_mat)
                phys_map = maps.ExpMap(nP=Sounding.mesh.nC)
                model_mapping = phys_map * geo_map
                z_init = rng.randn(Sounding.mesh.nC) * std_scale
                m_latent_init = np.r_[mean_prior, z_init]
                self.simulation = tdem.Simulation1DLayered(
                    survey=Sounding.srv, thicknesses=Sounding.inv_thickness, sigmaMap=model_mapping
                )
                self.dmis = L2DataMisfit(simulation=self.simulation, data=Sounding.data_object)
                if self.use_weights: self.dmis.W = 1.0 / Sounding.uncertainties
                reg_mesh = TensorMesh([np.ones(Sounding.mesh.nC + 1)])
                self.reg = regularization.WeightedLeastSquares(
                    reg_mesh, mapping=maps.IdentityMap(nP=Sounding.mesh.nC + 1)
                )
                reg_weights = np.ones(Sounding.mesh.nC + 1)
                reg_weights[0] = 0.1
                reg_weights[1:] = 1.0
                self.reg.set_weights(prior_weights=np.sqrt(reg_weights))
                self.reg.reference_model = m_latent_init
                self.reg.alpha_s = 1.0
                self.reg.alpha_x = 0.0
                self.opt = optimization.ProjectedGNCG(maxIter=self.maxIter, tolCG=self.tolCG)
                self.inv_prob = inverse_problem.BaseInvProblem(self.dmis, self.reg, self.opt)
                self.starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio)
                self.beta_schedule = directives.BetaSchedule(coolingFactor=self.coolEpsFact, coolingRate=1)
                self.target_misfit = directives.TargetMisfit(chifact=1.0)
                self.directives_list = [self.starting_beta, self.beta_schedule, self.target_misfit]
                self.inv = inversion.BaseInversion(self.inv_prob, self.directives_list)
                self.recovered_latent = self.inv.run(m_latent_init)
                self.values = model_mapping * self.recovered_latent
                pred = fsim(Sounding.srv, Sounding.mesh, self.values)
                self.pred = pred
                residuals = (Sounding.dobs - self.pred.dclean) / Sounding.uncertainties
                self.CHi2 = np.sum(residuals**2) / len(Sounding.dobs)
                self.rele = np.mean(np.abs((Sounding.dobs - self.pred.dclean) / Sounding.dobs))
                self.DOI = get_DOI(isounding=Sounding, Cali=self)
            if self.verbose: calfunc(self, Sounding, stochastic_params)
            else:
                with NoStdStreams(): calfunc(self, Sounding, stochastic_params)
            return {
                "values": self.values, "rele": self.rele, "pred": self.pred,
                "DOI": self.DOI, "CHI2": self.CHi2, "corr_factor": self.corr_factor,
                "success": True,
            }
        except Exception:
            return {"success": False}


# -----------------------------------------------------------------------------
# OUTPUT UTILS
# -----------------------------------------------------------------------------
def adjust_dtype(var):
    if isinstance(var, np.integer):
        return int(var)
    elif isinstance(var, np.floating):
        return float(var)
    elif isinstance(var, np.ndarray):
        return var.tolist()
    else:
        return var

def proc_output(out, fd_output_sounding):
    fi_out_rml_tpl = r"{}\rml.gz.parquet"
    fi_out_obs_tpl = r"{}\obs.gz.parquet"
    fi_out_preds_tpl = r"{}\preds.gz.parquet"
    fi_out_vars_tpl = r"{}\variables.json"

    time, isounding = out

    df_rml = pd.DataFrame(
        {
            "depth": isounding.inv_thickness.cumsum(),
            "p5": isounding.RML.p5[:-1],
            "p50": isounding.RML.p50[:-1],
            "p95": isounding.RML.p95[:-1],
            "pprob": isounding.RML.pprob[:-1],
            "tprob": isounding.RML.tprob[:-1],
            "ri_prob": isounding.RML.ri_prob[:-1],
            "fa_prob": isounding.RML.fa_prob[:-1],
            "igp": isounding.RML.igp[:-1],
            "doicdf": isounding.RML.cdf,
        }
    )

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
        "elevation": adjust_dtype(isounding.Elevation),
        "mean_relerr": adjust_dtype(isounding.RML.fits),
        "DOI_mean": adjust_dtype(isounding.RML.DOI_mean),
        "DOI_std": adjust_dtype(isounding.RML.DOI_std),
        "calibration_factors": [adjust_dtype(x) for x in isounding.RML.calib_factors]
    }

    if not os.path.exists(fd_output_sounding):
        os.makedirs(fd_output_sounding, exist_ok=True)

    fi_out_rml = fi_out_rml_tpl.format(fd_output_sounding)
    fi_out_obs = fi_out_obs_tpl.format(fd_output_sounding)
    fi_out_vars = fi_out_vars_tpl.format(fd_output_sounding)
    fi_out_preds = fi_out_preds_tpl.format(fd_output_sounding)

    df_rml.to_parquet(fi_out_rml, index=False)
    df_obs.to_parquet(fi_out_obs, index=False)
    df_calreals.to_parquet(fi_out_preds, index=False)
    with open(fi_out_vars, "w") as f:
        json.dump(dic_vars, f, indent=4)
