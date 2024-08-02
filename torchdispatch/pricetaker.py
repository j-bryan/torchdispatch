import numpy as np
from scipy.stats import norm
import cvxpy

from torchdispatch.base import OptimizationResult, ConfidenceInterval


class PriceTakerDispatch:
    def __init__(self):  # provides 5.5 hours of boost at 500 MW
        self.storage_level = None
        # TODO: Move these to the cashflows.py file, where other aspects of the cash flows (e.g. tax
        #       rates, discount rates, depreciation schedules, etc.) are defined.
        # Costs for the BOP size (upper_bound) and storage capacity (Ecap)
        # These are scaled down to daily costs for the optimization problem.
        # BOP cost is tricky since our sensitivity analysis rolled BOP capex into the total nuclear capex.
        # We can go back to the values from Hill to ballpark this, which cite 5-7 USD/kWe for BOP capex.
        self.bop_capex = 6e3 * 0.07765 / 365  # 6 USD/kWe * CRF
        # Two-tank thermal energy storage estimated to have capex between 5-30 USD/kWhe energy or
        # 400-2400 USD/kWe power, whichever is greater. Ecap is in units of GWhth. Assume
        # 1 GWhth = 0.4 GWhe. Also, we must scale the total capex to an annualized value using a
        # capital recovery factor (CRF) with a lifetime of 30 years. The CRF is ~0.07765.
        # This assumes r_t=0.25, r_i=0.025, r_d=0.08.
        self.storage_capex = (5e3 + 30e3) / 2 * 0.07765 / 365  # 5 USD/kWhe * 1e3 MW/kW * 0.4 GWhe/GWht * CRF
        # Nuclear total capex just to get a ballpark for actual revenue. Will use 5750 USD/kWe
        self.nuclear_capex_tot = 840e3 * 5750 * 0.4 * 0.07062856550691406 / 365  # 840 USD/kWe * 0.4 We/Wt * CRF

        self.results = None

    def solve(self, prices: np.ndarray, BOP_ub: float = 500, TES_ub: float = 900, alpha=0.95) -> OptimizationResult:
        """
        Determine the optimal dispatch schedule for the price-taker dispatch optimization. This is
        solved as a linear programming problem.

        @param prices: Price scenarios for the optimization problem. This is a 2D numpy array.
        @param BOP_ub: Upper bound on dispatch capacity
        @param TES_ub: Upper bound on storage capacity
        @param alpha: Confidence level for bootstrapped confidence intervals
        @return result: An OptimizationResult object containing the results of the optimization.
        """
        batch_size, n_time_steps = prices.shape

        # Optimization variables
        x = cvxpy.Variable((batch_size, n_time_steps))  # dispatch variables
        dispatch_ub = cvxpy.Variable(1)  # dispatch upper bound
        storage_ub = cvxpy.Variable(1)  # storage upper bound
        dispatch_lb = 200.
        dispatch_ss = 840 * 0.4

        # Problem parameters
        price = cvxpy.Parameter((batch_size, n_time_steps), value=prices)  # price as a parameter
        # Create constraints
        # sqrt_eta = 0.9 ** 0.5  # square root of round trip efficiency
        # storage_level = cvxpy.cumsum(cvxpy.pos(dispatch_ss - x) * sqrt_eta + cvxpy.neg(dispatch_ss - x) / sqrt_eta, axis=1)
        storage_level = cvxpy.cumsum(dispatch_ss - x, axis=1)
        constraints = [
            x >= dispatch_lb,
            x <= dispatch_ub,
            dispatch_ub >= dispatch_ss,
            dispatch_ub <= BOP_ub,
            storage_ub >= 0,
            storage_ub <= TES_ub,
            # cvxpy.sum(cvxpy.pos(dispatch_ss - x) * sqrt_eta + cvxpy.neg(dispatch_ss - x) / sqrt_eta, axis=1) == 0,
            storage_level[:, -1] == 0,
            cvxpy.max(storage_level, axis=1) - cvxpy.min(storage_level, axis=1) <= storage_ub
        ]

        # Create objective
        revenue = cvxpy.sum(cvxpy.multiply(price, x))
        costs = dispatch_ub * self.bop_capex + storage_ub * self.storage_capex + self.nuclear_capex_tot
        objective = cvxpy.Maximize(revenue - batch_size * costs)
        # Form the problem object
        problem = cvxpy.Problem(objective, constraints)
        res = problem.solve(ignore_dpp=True)

        # Extract storage usage from optimal dispatch
        self.storage_level = storage_level.value
        if self.storage_level.ndim == 1:
            self.storage_level = np.r_[0, self.storage_level]
        else:
            self.storage_level = np.c_[np.zeros(batch_size), self.storage_level]

        # seg_rev = (price.value * x.value).sum(axis=1) - costs.value
        # mean_rev_confint = bootstrap((seg_rev,), np.mean, n_resamples=max(9999, 10*batch_size))
        # Calculate confidence intervals about expected segment revenue. The segments are independent,
        # so we can calculate the confidence of the sample mean analytically.
        mean_rev = np.mean((price.value * x.value).sum(axis=1) - costs.value)
        std_err = np.std((price.value * x.value).sum(axis=1), ddof=1) / np.sqrt(batch_size)
        z_score = norm.ppf(alpha)
        confint = ConfidenceInterval(mean=mean_rev,
                          std=std_err,
                          confint=(mean_rev - z_score * std_err, mean_rev + z_score * std_err),
                          alpha=alpha)

        result = OptimizationResult(
            x=[x.value, dispatch_ub.value, storage_ub.value],
            storage_level=self.storage_level,
            y=res,
            penalty=0,
            success=True,
            nfev=1,
            niter=1,
            message="Success",
            confint=confint
        )

        self.results = result

        return result
