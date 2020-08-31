import inspect

import matplotlib.pyplot as plt
import numpy as np
from ortools.linear_solver import pywraplp


class Mazzolani:
    def __init__(
        self,
        vol_day,
        vol_nf,
        is_holiday,
        N=None,
        **kwargs,
    ):
        """Initializes optimization class, based on Mazzolanis paper.

        Parameters
        ----------
        vol_day : list or np.array
            Daily inflow volume
        vol_nf : list or np.array
            Night inflow volume
        is_holiday : list, or np.array
            Binary indicator, equal to 1 when there is a holiday, 0 otherwise
        N : int, optional
            Numbber of time periods considered, by default None. Calculated based on
            other inputs
        """

        # Initialize attributes as defaults
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        # Check for consistency
        self.check()

    def set_params(self, verbose=False, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        return vars(self)

    def __str__(self):
        return (
            f"Object {self.__class__.__name__} with following parameters:\n{vars(self)}"
        )

    def check(self):
        # Number of data points
        self.N = len(self.vol_day)
        assert self.N == len(self.vol_nf), "Length of volumes do not match"
        assert self.N == len(
            self.is_holiday
        ), "Length of volumes and holiday indicator do not match"

    def optimizeK3(self, Ln, debug=False):
        """Find the optimal value of leakage

        Parameters
        ----------
        Ln : float
            Negative value indicating leakages
        debug : bool, optional
            If True, optimziation problem is printed in detail. By default False

        Returns
        -------
        dict
            Solution as a named dictionary
        """

        # Define linear solver
        solver = pywraplp.Solver.CreateSolver("linear_programming", "GLOP")

        # Create variables to be optimized
        K_h = solver.NumVar(0, solver.infinity(), "K_h")
        K_w = solver.NumVar(0, solver.infinity(), "K_w")

        eps_pos, eps_neg = [-999] * self.N, [-999] * self.N
        for i in range(self.N):
            eps_pos[i] = solver.NumVar(0, solver.infinity(), f"eps_pos_{i}")
            eps_neg[i] = solver.NumVar(0, solver.infinity(), f"eps_neg_{i}")

        # Constraints
        constraints = [0] * self.N
        for i in range(self.N):
            c_term = self.vol_nf[i] - Ln
            constraints[i] = solver.Constraint(c_term, c_term)
            constraints[i].SetCoefficient(eps_pos[i], -1)
            constraints[i].SetCoefficient(eps_neg[i], 1)
            if self.is_holiday[i] == 1:
                constraints[i].SetCoefficient(K_h, (self.vol_day[i] - Ln))
            else:
                constraints[i].SetCoefficient(K_w, (self.vol_day[i] - Ln))

        # Objective function: sum(eps)
        objective = solver.Objective()
        for i in range(self.N):
            objective.SetCoefficient(eps_pos[i], 1)
            objective.SetCoefficient(eps_neg[i], 1)
        objective.SetMinimization()

        # Solve the system.
        solved = solver.Solve()
        if solved == solver.OPTIMAL:
            result_status = "optimal"
        elif solved == solver.INFEASIBLE:
            result_status = "No solution found"
        elif solved == solver.POSSIBLE_OVERFLOW:
            result_status = "Integer overflow"
        else:
            result_status = solver.POSSIBLE_OVERFLOW

        # [START print_solution]
        # The value of each variable in the solution.
        eps_pos_sol = [-999] * self.N
        eps_neg_sol = [-999] * self.N
        for i in range(self.N):
            eps_pos_sol[i] = eps_pos[i].solution_value()
            eps_neg_sol[i] = eps_pos[i].solution_value()

        # The objective value of the solution.
        obj_value = np.sum(eps_pos_sol) + np.sum(eps_neg_sol)

        if debug:
            print("MODEL FORMULATION:\n-----")
            print(solver.ExportModelAsLpFormat(False))
            print("----")

        return {
            "K_h": K_h.solution_value(),
            "K_w": K_w.solution_value(),
            "eps_pos_sol": eps_pos_sol,
            "eps_neg_sol": eps_pos_sol,
            "obj_value": obj_value,
            "result_status": result_status,
            "n_var": solver.NumVariables(),
            "n_const": solver.NumConstraints(),
        }

    def find_k(self, ln_min=-50, ln_max=10, n_grid=1000, do_plot=False):

        # Define possible values of Ln
        Ln_list = np.linspace(ln_min, ln_max, n_grid)

        # Store solution as lists
        obj, K_w, K_h = [], [], []
        for Ln in Ln_list:
            sol = self.optimizeK3(Ln=Ln)
            obj.append(sol["obj_value"])
            K_h.append(sol["K_h"])
            K_w.append(sol["K_w"])

        if do_plot:
            self._plot(Ln_list, obj, K_w, K_h)

        return Ln_list, obj, K_w, K_h

    def _plot(self, Ln_list, obj, K_w, K_h):
        plt.plot(Ln_list, obj)
        plt.title("Objective function vs Ln")
        plt.show()
        plt.plot(Ln_list, K_w)
        plt.plot(Ln_list, K_h)
        plt.legend(("K_w", "K_h"), loc="upper right")
        plt.title("K vs Ln")
        plt.show()