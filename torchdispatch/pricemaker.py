import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from collections import defaultdict
from tqdm import trange

from torchtransformers.preprocessing import *
from torchtransformers.regression import *
from torchtransformers.compose import *

from torchdispatch.base import OptimizationResult, ConfidenceInterval
from torchdispatch.pricetaker import PriceTakerDispatch


class PriceMakerDispatch:
    def __init__(self,
                 price_model: torch.nn.Module,
                 dispatch_lb: float = 200.,
                 dispatch_ub: float = 500.,
                 dispatch_ss: float = 840*0.4,
                 storage_lb: float = 0.,
                 storage_ub: float = 900,  # provides 5.5 hours of boost at 500 MW
                 rte: float = 0.93,
                 dispatch_mult: float = 1.,
                 pw_period: float = 1e2,
                 pw_cap: float = 1e2,
                 device: str = "cuda"):
        """
        Class constructor for the dispatch optimization problem. Handles storing dispatch and storage
        bounds and initializing the Adam optimizer.

        @param price_model: A torch.nn.Module that takes in synthetic historical data and capacities
                            and returns a tensor of shape (batch_size, time_steps, 1) containing the
                            price of electricity at each time step.
        @param dispatch_lb: The lower bound on the dispatch schedule.
        @param dispatch_ub: The upper bound on the dispatch schedule.
        @param dispatch_ss: The steady state dispatch schedule.
        @param storage_lb: The lower bound on the storage capacity level.
        @param storage_ub: The upper bound on the storage capacity level.
        @param rte: The round trip efficiency of the storage system.
        @param dispatch_mult: The multiplier for the dispatch schedule in the price model. Used for
                              adjusting the price based on multiples of the dispatch schedule.
        @param pw_period: The weight for the penalty function enforcing periodic boundary conditions.
        @param pw_cap: The weight for the penalty function enforcing storage capacity constraint.
        @param num_steps: The number of optimization steps to run.
        @param device: The device to run the optimization on.
        """
        # The optimization is much faster on GPU, so use that if available.
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = "cpu"
        self.device = device

        # Price model used to generate price data from price drivers
        self.price_model = price_model.to(self.device)

        # Initialization of dispatch and storage variables. These will be updated during optimization.
        self.storage_level = None

        # Constraint values
        self.dispatch_lb = dispatch_lb
        self.dispatch_ss = dispatch_ss
        self.dispatch_ub = dispatch_ub
        self.storage_lb = storage_lb
        self.storage_ub = storage_ub
        # self.optimize_capacities = optimize_capacities
        # if optimize_capacities:
        #     self.dispatch_ub = torch.tensor(dispatch_ub).float().to(self.device)
        #     self.dispatch_ub.requires_grad = True
        #     self.storage_ub = torch.tensor(storage_ub).float().to(self.device)
        #     self.storage_ub.requires_grad = True
        # else:
        #     self.dispatch_ub = dispatch_ub
        #     self.storage_ub = storage_ub

        self.sqrt_rte = rte ** 0.5
        self.dispatch_mult = dispatch_mult

        # Penalty function weights
        self.pw_period = pw_period  # Weight for penalty function enforcing periodic boundary conditions
        self.pw_cap = pw_cap  # Weight for penalty function enforcing storage capacity constraint

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

        self.pricetaker_dispatch = None
        self.results = None

    def get_adjusted_prices(self, x: torch.Tensor, synth_hist: torch.Tensor) -> torch.Tensor:
        """
        Calculates the price of electricity for histories synth_hist, adjusted by dispatch x.
        """
        price_input = synth_hist.clone()
        price_input[..., 0] = price_input[..., 0] - self.dispatch_mult * x
        price = self.price_model(price_input)
        return price

    def objective(self, x: torch.Tensor, synth_hist: torch.Tensor) -> torch.Tensor:
        """
        Calculates the objective function for the dispatch optimization problem. The objective function
        is the average revenue generated by the dispatch schedule x. Since the values in x are independent
        for each sample in the batch, maximizing the average revenue is equivalent to maximizing the
        revenue for each sample.

        @param x: A tensor of shape (batch_size, time_steps) containing the dispatch schedule.
        @param synth_hist: A tensor of shape (batch_size, time_steps, vars) containing synthetic histories
                           of (demand, wind capacity factor, solar capacity factor).
        @return net_revenue: A tensor of shape (batch_size, 1) containing the revenue generated by the dispatch schedule x.
        """
        price = self.get_adjusted_prices(x, synth_hist)
        revenue = torch.bmm(x.unsqueeze(1), price).squeeze()  # shape (batch_size,)
        costs = self.bop_capex * self.dispatch_ub + self.storage_capex * self.storage_ub + self.nuclear_capex_tot
        net_revenue = revenue - costs
        return net_revenue

    def get_storage_level(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the storage level at each time step given the dispatch schedule x. Storage level is
        the cumulative sum of the difference between the steady state dispatch and the actual dispatch.

        @param x: A tensor of shape (batch_size, time_steps) containing the dispatch schedule.
        @return storage_level: A tensor of shape (batch_size, time_steps) containing the storage level at each time step.
        """
        storage_level = torch.hstack((torch.zeros(x.size(0), 1, device=self.device),
                                      torch.cumsum(self.sqrt_rte * torch.nn.functional.relu(self.dispatch_ss - x) - 1 / self.sqrt_rte * torch.nn.functional.relu(x - self.dispatch_ss), dim=-1)))
        return storage_level

    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        A penalty function to enforce periodic boundary conditions and storage capacity constraints
        for the system dispatch, calculated as a weighted sum of terms for each constraint.

        @param x: The dispatch schedule
        @return penalty: The penalty term to be added to the objective function.
        """
        storage_level = self.get_storage_level(x)
        # Penalize the final storage level to enforce periodic boundary conditions by forcing the
        # storage level at the final time step to be equal to the initial storage level (zero).
        periodic_penalty = torch.abs(storage_level[..., -1])
        # Penalize storage usage that exceeds the storage capacity to enforce the storage capacity
        # constraint. It's okay for the storage to not be fully utilized, as this is effectively
        # penalized by the cost of the storage capacity.
        capacity_penalty = torch.nn.functional.relu(torch.max(storage_level, dim=-1).values - torch.min(storage_level, dim=-1).values - self.storage_ub)
        # Total penalty term is a weighted sum of these two penalties.
        # The per-sample revenue these will be weighted against will be on the order of 1e5.
        # periodic_penalty and capacity_penalty could both be as large as 1e5, so we'll use weights
        # of 10 for both terms and adjust as needed.
        penalty = self.pw_period * periodic_penalty + self.pw_cap * capacity_penalty
        # Hold onto the storage level so it can be accessed later in the optimization process and
        # after optimization is complete for analysis.
        self.storage_level = storage_level
        return penalty

    def solve(self,
              price_model_inputs: torch.Tensor,
              capacity_opt_bounds: dict[str, tuple[float, float]] | None = None,
              optim_args: dict | None = None,
              max_iter: int = 1000,
              max_iter_no_improvement: int = 30,
              improvement_check_start: int = 500,
              init_with_pricetaker: bool = True,
              f_rtol: float = 1e-5,
              f_atol: float = 1e-8,
              x_rtol: float = 1e-5,
              x_atol: float = 1e-8,
              alpha: float = 0.95) -> OptimizationResult:
        """
        Determine the optimal dispatch schedule for the price-maker dispatch optimization. It is assumed
        that the dispatch schedule should be subtracted from the first variable in the price model
        when determining the price of electricity.

        @param price_model_inputs: A tensor of shape (*, time_steps, n_vars) containing input data
                                   for the price model.
        @param capacity_opt_bounds: A dictionary of bounds for optimizing system parameters not in
                                    the dispatch schedule. The keys are the parameter names and the
                                    values are tuples of the form (lower_bound, upper_bound). If None,
                                    no additional parameters beyond the dispatch schedule are optimized.
        @param optim_args: A dictionary of keyword arguments to pass to the optimizer.
        @param max_iter: The maximum number of iterations to run the optimization.
        @param max_iter_no_improvement: The maximum number of iterations without improvement before
                                        stopping the optimization.
        @param improvement_check_start: The iteration at which to start checking for improvement.
        @param init_with_pricetaker: Whether to initialize the optimized variables with the pricetaker solution.
        @param f_rtol: The relative tolerance for the objective function value.
        @param f_atol: The absolute tolerance for the objective function value.
        @param x_rtol: The relative tolerance for the dispatch schedule.
        @param x_atol: The absolute tolerance for the dispatch schedule.
        @param alpha: The confidence level for confidence intervals.
        @return result: An OptimizationResult object containing the results of the optimization.
        """
        #================
        # Initialization
        #================

        # Initialize dispatch variables with steady state values
        # We optimize a transformed variable z to enforce bounds on the dispatch schedule x in a differentiable way.
        batch_size, n_time_steps, n_vars = price_model_inputs.shape
        price_model_inputs = price_model_inputs.to(self.device)
        if init_with_pricetaker:
            pricetaker_dispatch = PriceTakerDispatch()
            pt_result = pricetaker_dispatch.solve(self.price_model(price_model_inputs).squeeze().detach().cpu().numpy(),
                                                  BOP_ub=self.dispatch_ub,
                                                  TES_ub=self.storage_ub)
            self.pricetaker_dispatch = pricetaker_dispatch
            x0 = pt_result.x[0]
            z = 6 * (torch.tensor(x0, device=self.device).float() - self.dispatch_lb) / (self.dispatch_ub - self.dispatch_lb) - 3  # shape (batch_size, time_steps)
            z.requires_grad = True
        else:
            z = torch.ones((batch_size, n_time_steps), device=self.device).float() * (6 * (self.dispatch_ss - self.dispatch_lb) / (self.dispatch_ub - self.dispatch_lb) - 3)  # shape (batch_size, time_steps)
            z.requires_grad = True

        # Previous values for calculating early stopping criteria
        prev_value = torch.tensor(1.0).float()
        best_value = torch.inf
        prev_x = torch.zeros_like(z)
        steps_since_improvement = 0

        #=================
        # Optimizer Setup
        #=================

        if optim_args is None:
            optim_args = {}
        # Also allow for different optimization parameters for different optimized model parameters
        # (specified in the capacity_opt_bounds dictionary). These can be specified in the optim_args
        # dictionary as "<param_name>__<optim_param>". If not specified, the "top-level" values are
        # used. If not specified at all, the default value of the optimize class is used.
        param_specific_optim_params = defaultdict(dict)
        keys_to_remove = []
        for k, v in optim_args.items():
            if "__" not in k:
                continue
            param_name, param_optim_param = k.split("__")
            param_specific_optim_params[param_name][param_optim_param] = v
            keys_to_remove.append(k)
        # Remove the keys that have been processed from the optim_args dictionary.
        for k in keys_to_remove:
            del optim_args[k]

        # Only the dispatch is for sure optimized, so we'll set the optimizer parameters for that here.
        optimizer_param_list = [{"params": z, **param_specific_optim_params.get("z", {})}]
        # Now we can add any additional parameters to the optimization.
        if capacity_opt_bounds is None:
            capacity_opt_bounds = {}
        for k, v in capacity_opt_bounds.items():
            attr = getattr(self, k)
            if isinstance(attr, (float, int)):
                new_attr_val = torch.tensor([np.float32(attr)], device=self.device, requires_grad=True)
                setattr(self, k, new_attr_val)
            optimizer_param_list.append({"params": getattr(self, k), **param_specific_optim_params.get(k, {})})
        # Now we can instantiate the optimizer with the parameter list.
        optimizer = torch.optim.Adam(optimizer_param_list, **optim_args)

        #===================
        # Optimization Loop
        #===================

        for i in trange(max_iter):
            # Calculate gradients for the dispatch schedule. The objective function is the average
            # revenue generated by the dispatch schedule x for each sample in the batch, minus a
            # penalty term that enforces the storage capacity constraint and periodic boundary conditions.
            optimizer.zero_grad()
            x = torch.nn.functional.hardsigmoid(z) * (self.dispatch_ub - self.dispatch_lb) + self.dispatch_lb
            value = torch.mean(-self.objective(x, price_model_inputs).squeeze() + self.penalty(x))
            value.backward(retain_graph=True)
            optimizer.step()
            # Enforce bounds on optimized capacity values, if applicable.
            with torch.no_grad():
                # self.storage_ub.clamp_(min=0.0, max=2000.0)
                # self.dispatch_ub.clamp_(min=self.dispatch_ss, max=800.0)
                for k, v in capacity_opt_bounds.items():
                    getattr(self, k).clamp_(min=v[0], max=v[1])

            if i % 100 == 0:
                print(f"Iteration: {i:3d}     Objective: {value.item():.3f}")

            # Added stopping criteria here and upped the iteration limit significantly.
            if value < best_value:
                steps_since_improvement = 0
                best_value = value
            else:
                steps_since_improvement += 1

            is_objective_converged = torch.isclose(value, prev_value, rtol=f_rtol, atol=f_atol).all()
            is_dispatch_converged = torch.isclose(x, prev_x, rtol=x_rtol, atol=x_atol).all()
            if is_objective_converged and is_dispatch_converged:
                print(f"Converged after {i + 1} iterations")
                niter = i + 1
                message = "Converged"
                success = True
                break
            if steps_since_improvement > max_iter_no_improvement and i > improvement_check_start:
                print(f"No improvement in the last {max_iter_no_improvement} steps. Stopping optimization.")
                niter = i + 1
                message = "No improvement"  # Not necessarily a failure, but a distinct stopping criterion.
                success = True
                break
            prev_value = value.clone()
            prev_x = x.clone()
        else:
            niter = max_iter
            message = "Iteration limit reached."
            success = False
            print("Iteration limit reached.")
        print("Final objective value:", value.item())

        net_revenue = self.objective(x, price_model_inputs).squeeze()
        penalty = self.penalty(x)

        # Bootstrapped confidence intervals about expected segment revenue
        # seg_rev = net_revenue.detach().cpu().numpy()
        # mean_rev_confint = bootstrap((seg_rev,), np.mean, n_resamples=max(9999, 10 * batch_size))
        # Calculate confidence intervals about expected segment revenue. The segments are independent,
        # so we can calculate the confidence of the sample mean analytically.
        rev = net_revenue.detach().cpu().numpy()
        mean_rev = rev.mean()
        std_err = np.std(rev, ddof=1) / np.sqrt(batch_size)
        z_score = norm.ppf(alpha)
        confint = ConfidenceInterval(mean=mean_rev,
                          std=std_err,
                          confint=(mean_rev - z_score * std_err, mean_rev + z_score * std_err),
                          alpha=alpha)

        result_opt_params = [x]
        for i in range(1, len(optimizer_param_list)):
            result_opt_params.extend(optimizer_param_list[i]["params"])
        for i in range(len(result_opt_params)):
            result_opt_params[i] = result_opt_params[i].detach().cpu()
        result = OptimizationResult(
            x=result_opt_params,
            storage_level=self.storage_level.detach().cpu(),
            y=net_revenue,
            penalty=penalty,
            success=success,
            nfev=niter,
            niter=niter,
            message=message,
            confint=confint
        )

        self.results = result

        return result
