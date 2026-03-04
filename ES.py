import sys
import numpy as np
import logging
import pandas as pd

# Check for Dask
try:
    from dask.distributed import Client, LocalCluster
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# =============================================================================
# IES CLASS DEFINITION (Optimized with Parameter Blocking)
# =============================================================================
class LMEnsembleSmoother:
    def __init__(self, model_func, param_df, obs_df, num_ensemble=50, random_seed=None, 
                 client=None, regularize_parameters=False, reg_weight=0.1, 
                 transform_parameters=True): 
        """
        Initialize the Ensemble Smoother (Optimized).
        """
        if random_seed: np.random.seed(random_seed)
        self.model_func = model_func
        self.ne = num_ensemble
        self.client = client
        self.do_transform = transform_parameters 
        
        # --- STORE DATAFRAME ---
        self.df_params = param_df.copy()

        self.par_names = param_df.index.tolist()
        self.npar = len(self.par_names)
        
        # --- STATISTICAL DEFINITIONS ---
        self.par_init = param_df["prior_mean"].values 
        p_mean = param_df["prior_mean"].values
        p_std  = param_df["prior_std"].values
        self.p_info = np.column_stack((p_mean, p_std))

        # --- OBSERVATION DEFINITIONS ---
        self.obs_names = obs_df.index.tolist()
        self.nobs_real = len(self.obs_names)
        self.obs_vals = obs_df["value"].values
        self.obs_std = obs_df["std"].values
        
        # --- REGULARIZATION ---
        self.do_reg = regularize_parameters
        self.reg_weight = reg_weight
        
        if self.do_reg:
            print(f"  [IES Init] Regularization ON. Tethering {self.npar} parameters (Weight={self.reg_weight}).")
            reg_vals = self.par_init
            reg_std = np.abs(p_std) * self.reg_weight
            reg_std[reg_std < 1e-6] = 0.1 
            self.obs_vals = np.concatenate([self.obs_vals, reg_vals])
            self.obs_std = np.concatenate([self.obs_std, reg_std])
            
        self.nobs_total = len(self.obs_vals)
        self.weights = 1.0 / (self.obs_std + 1e-16)
        self.obs_cov_diag = np.power(self.obs_std, 2)
        
        self.P = None; self.S = None

    # --- TRANSFORMATION METHODS ---
    def transform(self, real_vals):
        """Converts Real Space -> Solver Space"""
        if not self.do_transform: return real_vals
        means = self.p_info[:, 0]; stds = self.p_info[:, 1]
        if real_vals.ndim > 1: return (real_vals - means) / stds
        return (real_vals - means) / stds

    def inverse_transform(self, solver_vals):
        """Converts Solver Space -> Real Space"""
        if not self.do_transform: return solver_vals
        means = self.p_info[:, 0]; stds = self.p_info[:, 1]
        if solver_vals.ndim > 1: return (solver_vals * stds) + means
        return (solver_vals * stds) + means

    # --- CORE METHODS ---
    def initialize_priors(self, phys_bounds=None, inflation=1.0):
        target_mean = self.p_info[:, 0]
        target_std  = self.p_info[:, 1] * inflation
        priors = np.zeros((self.ne, self.npar))
        
        for i, name in enumerate(self.par_names):
            mu = target_mean[i]; sigma = target_std[i]
            # Log-Normal logic
            if ('_A' in name and mu > 0): 
                try: priors[:, i] = np.exp(np.random.normal(np.log(mu) - 0.5 * np.sqrt(np.log(1 + (sigma/mu)**2))**2, np.sqrt(np.log(1 + (sigma/mu)**2)), self.ne))
                except: priors[:, i] = np.random.normal(mu, sigma, self.ne)
            elif ('_A' in name and mu < 0): 
                try: priors[:, i] = -np.exp(np.random.normal(np.log(abs(mu)) - 0.5 * np.sqrt(np.log(1 + (sigma/abs(mu))**2))**2, np.sqrt(np.log(1 + (sigma/abs(mu))**2)), self.ne))
                except: priors[:, i] = np.random.normal(mu, sigma, self.ne)
            else: 
                priors[:, i] = np.random.normal(mu, sigma, self.ne)
        
        # Mean Center & Clip
        current_mean = np.mean(priors, axis=0)
        priors = priors + (target_mean - current_mean)
        
        if phys_bounds is not None:
            for i, name in enumerate(self.par_names):
                if name in phys_bounds.index:
                    p_min = phys_bounds.loc[name, 'pmin']; p_max = phys_bounds.loc[name, 'pmax']
                    priors[:, i] = np.clip(priors[:, i], p_min, p_max)
            
        self.P = self.transform(priors) 
        return self.P

    # --- WORKER (BATCHED) ---
    @staticmethod
    def _batch_worker(model_func, batch_params, par_names, obs_names, p_info, do_transform):
        """
        Runs a batch of simulations on a single worker to reduce scheduler overhead.
        """
        results = []
        for params_array in batch_params:
            # 1. Transform (Solver -> Real)
            if do_transform:
                means = p_info[:, 0]; stds = p_info[:, 1]
                real_params = (params_array * stds) + means
            else:
                real_params = params_array

            # 2. Run Model
            p_dict = dict(zip(par_names, real_params))
            try:
                res_dict = model_func(p_dict)
                # Robust extraction
                row = [res_dict.get(name, np.nan) for name in obs_names]
                results.append(row)
            except:
                results.append([np.nan] * len(obs_names))
        
        return results

    # --- RUN ENSEMBLE (OPTIMIZED) ---
    def run_ensemble(self, parameters):
        """
        Runs the ensemble using Batched Parallelism.
        OPTIMIZED: Dynamically calculates batch size based on active workers.
        """
        # Ensure parameters is array
        if hasattr(parameters, "values"):
            raw_matrix = parameters.values
        elif hasattr(parameters, "to_numpy"):
            raw_matrix = parameters.to_numpy()
        else:
            raw_matrix = parameters
            
        total_runs = raw_matrix.shape[0]
        
        # 1. PARALLEL EXECUTION
        if self.client and HAS_DASK:
            # --- DYNAMIC BATCH SIZING ---
            try:
                # Get total number of threads/cores across the entire cluster
                n_workers = sum(self.client.nthreads().values())
            except Exception:
                # Fallback if scheduler info is unavailable
                n_workers = 1
            
            # Safety clamp
            if n_workers < 1: n_workers = 1

            # Calculate optimal size: ceil(total_runs / n_workers)
            # This ensures every worker gets exactly 1 large chunk of work (1 round trip)
            # e.g., 50 runs / 12 workers = 5 runs per batch.
            BATCH_SIZE = -(-total_runs // n_workers)
            
            # Optional: Prevent tiny batches if you have more workers than runs
            if BATCH_SIZE < 1: BATCH_SIZE = 1
            # ----------------------------

            # Prepare Batches
            batches = []
            for i in range(0, total_runs, BATCH_SIZE):
                batch_slice = raw_matrix[i : i + BATCH_SIZE]
                # Convert to list of arrays for serialization
                batch_list = [batch_slice[k, :] for k in range(batch_slice.shape[0])]
                batches.append(batch_list)
            
            # Submit Batches
            # We map the _batch_worker function across the list of batches
            futures = self.client.map(
                self._batch_worker, 
                [self.model_func]*len(batches),
                batches,
                [self.par_names]*len(batches),
                [self.obs_names[:self.nobs_real]]*len(batches),
                [self.p_info]*len(batches),
                [self.do_transform]*len(batches)
            )
            
            # Gather Results (Instant Transfer)
            nested_results = self.client.gather(futures)
            
            # Flatten List of Lists
            real_results = [item for sublist in nested_results for item in sublist]
            
        else:
            # Serial Fallback
            real_results = []
            # Use batch worker locally to avoid code duplication
            all_params_list = [raw_matrix[i, :] for i in range(total_runs)]
            real_results = self._batch_worker(
                self.model_func, all_params_list, self.par_names, 
                self.obs_names[:self.nobs_real], self.p_info, self.do_transform
            )

        # 2. ASSEMBLE S MATRIX
        real_S = np.array(real_results)
        
        # 3. HANDLE REGULARIZATION (Append Params if ON)
        if self.do_reg:
            real_params = self.inverse_transform(parameters)
            return np.hstack([real_S, real_params])
            
        return real_S

    def _check_failures(self, S, safe_min=-1e9, safe_max=1e9):
        nan_mask = np.isnan(S).any(axis=1)
        obs_S = S[:, :self.nobs_real]
        large_mask = ((np.nanmin(obs_S, axis=1) < safe_min) | (np.nanmax(obs_S, axis=1) > safe_max))
        bad_mask = nan_mask | large_mask
        fail_count = np.sum(bad_mask)

        if fail_count > 0:
            print(f"  [IES WARNING] Found {fail_count} unstable runs.")
            good_indices = np.where(~bad_mask)[0]
            if len(good_indices) == 0:
                print("  [CRITICAL FAILURE] All runs failed. Cannot proceed.")
                return False
            safe_mean = np.mean(S[good_indices], axis=0)
            S[bad_mask] = safe_mean
        return True

    def get_phi(self, S):
        return np.mean(np.sum(((S - self.obs_vals)**2) * (self.weights**2), axis=1))

    def get_perturbations(self):
        d_pert = np.zeros((self.ne, self.nobs_total))
        for i in range(self.nobs_total):
            d_pert[:, i] = self.obs_vals[i] + np.random.normal(0, self.obs_std[i], self.ne)
        return d_pert

    def get_update(self, P_curr, S_curr, lam, d_pert, param_mask=None):
        """
        Calculates the Kalman Update.
        If param_mask is provided, it zeroes out the update for parameters where mask is False.
        """
        P_prime = P_curr - np.mean(P_curr, axis=0)
        S_prime = S_curr - np.mean(S_curr, axis=0)
        fact = 1.0 / (self.ne - 1)
        
        # Calculate Cross-Covariance
        Cpd = np.dot(P_prime.T, S_prime) * fact
        
        # --- APPLY PARAMETER BLOCKING ---
        if param_mask is not None:
            # Zero out rows for frozen parameters
            # This ensures they do not receive information from the data residuals
            Cpd[~param_mask, :] = 0.0
        # --------------------------------

        Cdd = np.dot(S_prime.T, S_prime) * fact
        
        R = np.diag(self.obs_cov_diag)
        Matrix = Cdd + lam * R
        
        try:
            Inv = np.linalg.pinv(Matrix, rcond=1e-6)
        except np.linalg.LinAlgError:
            Identity = np.eye(Matrix.shape[0]) * 1e-6
            Inv = np.linalg.inv(Matrix + Identity)

        innov = d_pert - S_curr
        return np.dot(np.dot(Cpd, Inv), innov.T).T, np.dot(Cdd, np.dot(Inv, innov.T)).T

    def solve(self, max_iterations=4, initial_lambda=50.0, apply_jitter=False, 
              jitter_std=0.01, subset_frac=None, safe_min=-1e9, safe_max=1e9,
              enforce_bounds=None): 
        
        print(f"\n[IES] Starting. Transform={self.do_transform}, Iters={max_iterations}, Lambda={initial_lambda}")
        if self.P is None: 
            print("Error: Priors not initialized. Call .initialize_priors() first.")
            return None
        
        # 1. Initial Run
        self.S = self.run_ensemble(self.P)
        if not self._check_failures(self.S, safe_min, safe_max): return False
        
        # 2. PHYSICAL PRUNING (Initial Bounds Check)
        if enforce_bounds is not None and not self.do_transform:
             for i, name in enumerate(self.par_names):
                 if name in enforce_bounds.index:
                     self.P[:, i] = np.clip(self.P[:, i], enforce_bounds.loc[name, 'pmin'], enforce_bounds.loc[name, 'pmax'])

        # 3. STATISTICAL PRUNING
        curr_phi_vec = np.sum(((self.S - self.obs_vals)**2) * (self.weights**2), axis=1)
        curr_phi = np.mean(curr_phi_vec)
        print(f"Iter 0 | Phi: {curr_phi:.4e}")
        curr_lam = initial_lambda
        d_pert = self.get_perturbations()
        
        # --- OUTER LOOP ---
        for k in range(max_iterations):
            if apply_jitter and k > 0:
                print(f"  [Jitter] Adding noise (std={jitter_std})")
                if self.do_transform: self.P += np.random.normal(0, jitter_std, self.P.shape)
                else: self.P *= (1.0 + np.random.normal(0, jitter_std, self.P.shape))
            
            # --- PARAMETER BLOCKING STRATEGY ---
            # Iteration 0: Freeze 'z_' parameters to force update on Global params (corr_len)
            current_mask = None
            if k == 0:
                print("  [Strategy] BLOCKING ACTIVE: Freezing Latent Variables (z_) to force Global update.")
                # Create mask: True = Update, False = Freeze
                # We assume latent vars have 'z_' in the name (from your other code)
                current_mask = np.array(['z_' not in name for name in self.par_names])
            # -----------------------------------

            step_success = False
            
            # --- INNER LOOP (Lambda Search) ---
            for attempt in range(5):
                print(f"\n  [Iter {k+1} | Try {attempt+1}] Testing Lambda {curr_lam:.1f}...")
                
                # Pass the mask to get_update
                shift, obs_shift = self.get_update(self.P, self.S, curr_lam, d_pert, param_mask=current_mask)
                
                P_new = self.P + shift
                S_pred = self.S + obs_shift 

                # --- ENFORCE BOUNDS ---
                if enforce_bounds is not None and not self.do_transform:
                    for i, name in enumerate(self.par_names):
                        if name in enforce_bounds.index:
                            p_min = enforce_bounds.loc[name, 'pmin']
                            p_max = enforce_bounds.loc[name, 'pmax']
                            P_new[:, i] = np.clip(P_new[:, i], p_min, p_max)
                
                # --- SUBSET LOGIC ---
                if subset_frac and subset_frac < 1.0:
                    n_sub = max(5, int(self.ne * subset_frac))
                    sub_idx = np.random.choice(self.ne, n_sub, replace=False)
                    rem_idx = np.setdiff1d(np.arange(self.ne), sub_idx)
                    
                    # Run Subset
                    S_new_sub = self.run_ensemble(P_new[sub_idx])
                    
                    if np.isnan(S_new_sub).any():
                        print(f"  > SUBSET CRASHED -> Increasing Lambda")
                        curr_lam *= 2.0; continue

                    # Calculate Subset Rho
                    phi_old_sub = self.get_phi(self.S[sub_idx])
                    phi_new_sub = self.get_phi(S_new_sub)
                    pred_drop_sub = phi_old_sub - self.get_phi(S_pred[sub_idx]) + 1e-16
                    actual_drop_sub = phi_old_sub - phi_new_sub
                    rho_sub = actual_drop_sub / pred_drop_sub
                    
                    if rho_sub < 0:
                        print(f"  > SUBSET REJECTED (Rho: {rho_sub:.2f}) -> Skipping Remainder.")
                        curr_lam *= 2.0; continue
                    
                    print(f"  > SUBSET PASSED (Rho: {rho_sub:.2f}) -> Running Remaining {len(rem_idx)}...")
                    S_new_rem = self.run_ensemble(P_new[rem_idx])
                    S_new_act = np.zeros_like(self.S)
                    S_new_act[sub_idx] = S_new_sub; S_new_act[rem_idx] = S_new_rem
                else:
                    # Run Full Ensemble
                    S_new_act = self.run_ensemble(P_new)

                # --- GLOBAL CHECK ---
                self._check_failures(S_new_act, safe_min, safe_max)
                
                phi_new = self.get_phi(S_new_act)
                pred_drop = curr_phi - self.get_phi(S_pred) + 1e-16
                actual_drop = curr_phi - phi_new
                rho = actual_drop / pred_drop

                if rho > 0 and phi_new < curr_phi:
                    drop_pct = (curr_phi - phi_new) / curr_phi * 100
                    print(f"  > ACCEPTED (Phi: {phi_new:.4e} | Drop: {drop_pct:.1f}% | Rho: {rho:.2f})")
                    self.P = P_new; self.S = S_new_act; curr_phi = phi_new
                    if rho > 0.75: curr_lam = max(1.0, curr_lam / 2.0)
                    elif rho < 0.25: curr_lam *= 1.5 
                    step_success = True; break 
                else:
                    print(f"  > GLOBAL REJECTED (Rho: {rho:.2f}) -> Increasing Lambda")
                    curr_lam *= 2.0 
            
            if not step_success:
                print(f"\n[IES] FAILED to find a valid step.")
                break
            print(f"Iter {k+1} | Final Phi: {curr_phi:.4e}")
            
        return self.inverse_transform(self.P)

    # =========================================================================
    # RECENTER & POLISH
    # =========================================================================
    def recenter(self, final_params, sigma_fraction=0.05, reg_weight=1.0, n_iters=3, enforce_bounds=None):
        """
        Updates priors to center on previous results, rebuilds regularization targets,
        shrinks uncertainty, and re-runs.
        """
        print(f"\n[Recenter] Updating priors to match previous posterior Mean.")
        print(f"[Recenter] Shrinking uncertainty to {sigma_fraction*100}% of value.")

        # 1. Calculate New Means
        df_results = pd.DataFrame(final_params, columns=self.par_names)
        new_means = df_results.mean()

        # 2. Update Internal Parameter DataFrame
        for idx in self.df_params.index:
            mu_new = new_means[idx]
            self.df_params.loc[idx, 'prior_mean'] = mu_new
            
            sigma_new = abs(mu_new) * sigma_fraction
            sigma_new = max(sigma_new, 1e-4) # Safety floor
            if 'constant_d' in idx: sigma_new = max(sigma_new, 0.5)

            self.df_params.loc[idx, 'prior_std'] = sigma_new

        # 3. Sync p_info
        p_mean = self.df_params["prior_mean"].values
        p_std  = self.df_params["prior_std"].values
        self.p_info = np.column_stack((p_mean, p_std))

        # 4. REBUILD REGULARIZATION VECTORS
        print(f"[Recenter] Rebuilding Regularization Targets (Weight={reg_weight})...")
        self.do_reg = True # Force Regularization ON
        self.reg_weight = reg_weight

        # A. Reset to just the Real Observations
        real_obs = self.obs_vals[:self.nobs_real]
        real_std = self.obs_std[:self.nobs_real]

        # B. Create New Dummy Observations (Target = New Mean)
        reg_vals = p_mean 

        # C. Create New Dummy Weights
        reg_std = np.abs(p_std) * self.reg_weight
        reg_std[reg_std < 1e-6] = 0.1 

        # D. Concatenate
        self.obs_vals = np.concatenate([real_obs, reg_vals])
        self.obs_std = np.concatenate([real_std, reg_std])

        # E. Re-calculate Global Matrices
        self.nobs_total = len(self.obs_vals)
        self.weights = 1.0 / (self.obs_std + 1e-16)
        self.obs_cov_diag = np.power(self.obs_std, 2)

        # 5. Re-Initialize Priors
        self.initialize_priors(phys_bounds=enforce_bounds)

        # 6. Solve
        return self.solve(
            max_iterations=n_iters, 
            initial_lambda=5.0, 
            subset_frac=0.5,
            enforce_bounds=enforce_bounds,
            apply_jitter=False 
        )