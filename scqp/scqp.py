import numpy as np
from scipy.linalg import lu_solve, lu_factor
import scipy.sparse as spa
import qdldl
import time


class SCQP:
    def __init__(self, Q, p, A, lb, ub, control):
        # --- input space:
        self.Q = Q
        self.p = p
        self.A = A
        self.lb = lb
        self.ub = ub
        self.control = control

        # --- solution storage:
        self.sol = {}

    def solve(self):
        # --- for warm-start:
        x = self.sol.get('x')
        z = self.sol.get('z')
        y = self.sol.get('y')
        if self.control.get('warm_start'):
            rho = self.sol.get('rho', self.control.get('rho'))
            self.control['rho'] = rho
        # --- LU caching:
        if self.control.get('cache_factor'):
            ATA = self.sol.get("ATA")
            M_lu = self.sol.get("M_lu")
        else:
            ATA = None
            M_lu = None
        sol = scqp_solve(Q=self.Q, p=self.p, A=self.A, lb=self.lb, ub=self.ub, control=self.control,
                         x=x, z=z, y=y, ATA=ATA, M_lu=M_lu)
        self.sol = sol
        x = sol.get('x')
        return x

    def update(self, Q=None, p=None, A=None, lb=None, ub=None, control=None):
        if Q is not None:
            self.Q = Q
        if p is not None:
            self.p = p
        if A is not None:
            self.A = A
        if lb is not None:
            self.lb = lb
        if ub is not None:
            self.ub = ub
        if control is not None:
            self.control = control
        return None

    def compute_objective(self, x=None, Q=None, p=None):
        if x is None:
            x = self.sol.get('x')
        if Q is None:
            Q = self.Q
        if p is None:
            p = self.p
        x = make_matrix(x)[:, 0]
        p = make_matrix(p)[:, 0]
        obj_value = compute_objective(x, Q, p)
        return obj_value

    def compute_primal_tol(self, x=None, z=None, A=None, eps_abs=None, eps_rel=None):
        if x is None:
            x = self.sol.get('x')
        if z is None:
            z = self.sol.get('z')
        if A is None:
            A = self.A
        if eps_abs is None:
            eps_abs = self.control.get('eps_abs')
        if eps_rel is None:
            eps_rel = self.control.get('eps_rel')
        x = make_matrix(x)[:, 0]
        z = make_matrix(z)[:, 0]
        primal_tol = compute_primal_tol(x=x, z=z, A=A, eps_abs=eps_abs, eps_rel=eps_rel)
        return primal_tol

    def compute_dual_tol(self, x=None, y=None, Q=None, p=None, A=None, eps_abs=None, eps_rel=None):
        if x is None:
            x = self.sol.get('x')
        if y is None:
            y = self.sol.get('y')
        if Q is None:
            Q = self.Q
        if p is None:
            p = self.p
        if A is None:
            A = self.A
        if eps_abs is None:
            eps_abs = self.control.get('eps_abs')
        if eps_rel is None:
            eps_rel = self.control.get('eps_rel')
        x = make_matrix(x)[:, 0]
        y = make_matrix(y)[:, 0]
        p = make_matrix(p)[:, 0]
        dual_tol = compute_dual_tol(x=x, y=y, Q=Q, p=p, A=A, eps_abs=eps_abs, eps_rel=eps_rel)
        return dual_tol

    def compute_primal_error(self, x=None, A=None, lb=None, ub=None):
        if x is None:
            x = self.sol.get('x')
        if A is None:
            A = self.A
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        x = make_matrix(x)[:, 0]
        lb = make_matrix(lb)[:, 0]
        ub = make_matrix(ub)[:, 0]
        primal_error = compute_primal_error(x=x, A=A, lb=lb, ub=ub)
        return primal_error

    def compute_dual_error(self, x=None, y=None, Q=None, p=None, A=None):
        if x is None:
            x = self.sol.get('x')
        if y is None:
            y = self.sol.get('y')
        if Q is None:
            Q = self.Q
        if p is None:
            p = self.p
        if A is None:
            A = self.A
        x = make_matrix(x)[:, 0]
        y = make_matrix(y)[:, 0]
        p = make_matrix(p)[:, 0]
        dual_error = compute_dual_error(x=x, y=y, Q=Q, p=p, A=A)
        return dual_error


def scqp_solve(Q, p, A, lb, ub, control, x=None, z=None, y=None, ATA=None, M_lu=None):
    #######################################################################
    # Solve a QP in form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx
    #             subject to lb <= Ax <= ub
    #
    # Q:  A (n_x,n_x) SPD matrix
    # p:  A (n_x,1) matrix.
    # A:  A (m, n_x) matrix.
    # lb:  A (m,1) vector
    # ub:  A (m,1) vector
    # Returns: x_star:  A (n_x) vector and solution information
    #######################################################################
    # --- unpacking control:
    max_iters = control.get('max_iters', 10_000)
    eps_abs = control.get('eps_abs', 1e-3)
    eps_rel = control.get('eps_rel', 1e-3)
    eps_infeas = control.get('eps_infeas', 1e-4)
    check_solved = control.get('check_solved', 25)
    check_feasible = control.get('check_feasible', check_solved)
    check_feasible = max(round(check_feasible / check_solved), 1) * check_solved
    alpha = control.get('alpha', 1.2)
    alpha_iter = control.get('alpha_iter', 100)
    rho = control.get('rho')
    rho_min = control.get('rho_min', 1e-6)
    rho_max = control.get('rho_max', 1e6)
    adaptive_rho = control.get('adaptive_rho', False)
    adaptive_rho_tol = control.get('adaptive_rho_tol', 5)
    adaptive_rho_iter = control.get('adaptive_rho_iter', 100)
    adaptive_rho_iter = max(round(adaptive_rho_iter / check_solved), 1) * check_solved
    adaptive_rho_max_iter = control.get('adaptive_rho_max_iter', 10_000)
    sigma = control.get('sigma', 1e-6)
    sigma = max(sigma, 0)
    verbose = control.get('verbose', False)
    scale = control.get('scale', True)
    beta = control.get('beta')
    warm_start = control.get('warm_start', False)
    cache_factor = control.get('cache_factor', False)

    # ---- prep:
    n_A = A.shape[0]
    n_x = A.shape[1]
    is_A_sparse = spa.issparse(A)
    is_Q_sparse = spa.issparse(Q)
    if not is_A_sparse == is_Q_sparse and Q is not None:
        raise Exception('if A or Q are sparse then both must be sparse.')
    is_sparse = is_A_sparse
    if p is None:
        p = np.zeros(n_x)
    else:
        p = make_matrix(p)
        p = p[:, 0]
    p_norm = np.linalg.norm(p, ord=np.inf)
    lb = prep_bound(lb, n_x=n_A, default=-float("inf"))
    lb = lb[:, 0]
    ub = prep_bound(ub, n_x=n_A, default=float("inf"))
    ub = ub[:, 0]
    any_lb = lb.max() > -float("inf")
    any_ub = ub.min() < float("inf")
    idx_lb_finite = np.isfinite(lb)
    idx_ub_finite = np.isfinite(ub)

    # --- scaling and pre-conditioning:
    if scale:
        Q, p, A, lb, ub, D, E = scqp_scale(Q=Q, p=p, A=A, lb=lb, ub=ub, beta=beta)
    else:
        D = 1.0
        E = 1.0

    # --- storage AT
    AT = A.T
    if ATA is None:
        ATA = AT.dot(A)


    # --- parameter selection:
    if rho is None:
        if is_sparse:
            A_norm = spa.linalg.norm(ATA)
        else:
            A_norm = np.linalg.norm(ATA)
        if Q is not None:
            if is_sparse:
                Q_norm = spa.linalg.norm(Q)
            else:
                Q_norm = np.linalg.norm(Q)
            rho = Q_norm / A_norm
        else:
            rho = 1.0 / A_norm
        rho = rho * np.sqrt(n_A / n_x)
        rho = clamp(rho, rho_min, rho_max)

    # --- warm-starting:
    has_x = x is not None
    has_z = z is not None
    has_y = y is not None
    if warm_start and has_x and has_z and has_y:
        x = 0.95 * (x / D)
        z = 0.95 * (E * z)
        y = 0.95 * (y / E)
        u = y / rho
    else:
        x = np.zeros(n_x)
        z = np.zeros(n_A)
        u = np.zeros(n_A)

    # --- LU factorization:
    has_lu = ATA is not None and M_lu is not None
    if not cache_factor or not has_lu:
        if Q is not None:
            M = Q + rho * ATA
        else:
            M = rho * ATA
        if sigma > 0:
            if is_sparse:
                M.setdiag(M.diagonal() + sigma)
            else:
                np.fill_diagonal(M, M.diagonal() + sigma)
        if is_sparse:
            M_lu = qdldl.Solver(M.T)
        else:
            M_lu = lu_factor(M)

    # --- preambles:
    is_optimal = False
    is_primal_infeas = False
    is_dual_infeas = False
    primal_error = dual_error = Ax_norm = z_norm = ATy_norm = Qx_norm = iter = 1.0

    # --- main loop
    for i in range(max_iters):
        # --- adaptive rho:
        if adaptive_rho and i % adaptive_rho_iter == 0 and 0 < i < adaptive_rho_max_iter:
            num = primal_error / max(Ax_norm, z_norm)
            denom = dual_error / max(ATy_norm, Qx_norm, p_norm)
            denom = clamp(denom, x_min=1e-12)
            ratio = (num / denom) ** 0.5
            if ratio > adaptive_rho_tol or ratio < (1 / adaptive_rho_tol):
                rho = rho * ratio
                rho = clamp(rho, rho_min, rho_max)
                if Q is not None:
                    M = Q + rho * ATA
                else:
                    M = rho * ATA
                if sigma > 0:
                    if is_sparse:
                        M.setdiag(M.diagonal() + sigma)
                    else:
                        np.fill_diagonal(M, M.diagonal() + sigma)
                if is_sparse:
                    M_lu.update(M)
                else:
                    M_lu = lu_factor(M)

        # --- projection to sub-space:
        x_prev = x
        rhs = -p + rho * AT.dot(z - u) + sigma * x_prev
        if is_sparse:
            x = M_lu.solve(rhs)
        else:
            x = lu_solve(M_lu, rhs)
        if i > alpha_iter:
            x = alpha * x + (1 - alpha) * x_prev
        # --- proximal projection:
        Ax = A.dot(x)
        z = Ax + u
        if any_lb:
            z = np.maximum(z, lb)
        if any_ub:
            z = np.minimum(z, ub)
        # --- update residual:
        r = Ax - z
        u_prev = u
        u = u_prev + r

        # --- check solved:
        if i % check_solved == 0 or i >= (max_iters - 1):
            # --- update dual variable:
            y_prev = rho * u_prev
            y = rho * u
            # --- dual residual elements:
            ATy = AT.dot(y)
            if Q is not None:
                Qx = Q.dot(x)
            else:
                Qx = 0.0
            s = Qx + p + ATy

            # --- primal dual residuals unscaled:
            primal_error = np.linalg.norm(r / E, ord=np.inf)
            dual_error = np.linalg.norm(s / D, ord=np.inf)
            if verbose:
                print('iteration = {:d}'.format(i))
                print('|| primal_error|| = {:f}'.format(primal_error))
                print('|| dual_error|| = {:f}'.format(dual_error))

            # --- primal:
            Ax_norm = np.linalg.norm(Ax / E, ord=np.inf)
            z_norm = np.linalg.norm(z / E, ord=np.inf)

            # --- dual:
            ATy_norm = np.linalg.norm(ATy / D, ord=np.inf)
            if Q is not None:
                Qx_norm = np.linalg.norm(Qx / D, ord=np.inf)
            else:
                Qx_norm = 0.0

            tol_primal = eps_abs + eps_rel * max(Ax_norm, z_norm)
            tol_dual = eps_abs + eps_rel * max(ATy_norm, Qx_norm, p_norm)
            # --- check for optimality
            do_stop = primal_error < tol_primal and dual_error < tol_dual
            if do_stop:
                is_optimal = True
                break

        # --- check for feasibility:
        if i % check_feasible == 0 or i >= (max_iters - 1):
            # --- delta y and delta x:
            dy = y - y_prev
            dy_neg = np.minimum(dy, 0)
            dy_pos = np.maximum(dy, 0)
            dx = x - x_prev
            dy_norm = np.linalg.norm(E * dy, ord=np.inf)
            dx_norm = np.linalg.norm(D * dx, ord=np.inf)

            # --- feasibility thresholds:
            tol_primal_feas = eps_infeas * dy_norm
            tol_dual_feas = eps_infeas * dx_norm

            # --- primal infeasibility:
            primal_infeas_1 = np.linalg.norm(AT.dot(dy) / D, ord=np.inf)
            if primal_infeas_1:
                if any_lb:
                    lb_dy = (lb[idx_lb_finite] * dy_neg[idx_lb_finite]).sum()
                else:
                    lb_dy = 0.0
                if any_ub:
                    ub_dy = (ub[idx_ub_finite] * dy_pos[idx_ub_finite]).sum()
                else:
                    ub_dy = 0.0
                primal_infeas_2 = lb_dy + ub_dy
            else:
                primal_infeas_2 = np.inf

            # --- dual infeasibility:
            if Q is not None:
                dual_infeas_1 = np.linalg.norm(Q.dot(dx) / D, ord=np.inf)
            else:
                dual_infeas_1 = 0.0
            dual_infeas_2 = np.linalg.norm(p * dx, ord=np.inf)
            dual_infeas_3 = np.linalg.norm(A.dot(dx) / E, ord=np.inf)

            # --- check for primal or dual infeasibility conditions:
            is_primal_infeas = primal_infeas_1 < tol_primal_feas and primal_infeas_2 < tol_primal_feas
            is_dual_infeas = dual_infeas_1 < tol_dual_feas and dual_infeas_2 < tol_dual_feas and dual_infeas_3 < tol_dual_feas

            if is_primal_infeas or is_dual_infeas:
                break

    # --- reverse the scaling:
    iter = i
    x = D * x
    z = z / E
    y = E * y
    y_neg = np.minimum(y, 0)  # was -np.minimum(y, 0)
    y_pos = np.maximum(y, 0)
    # --- return solution:
    sol = {"x": x, "z": z, "y": y, "y_pos": y_pos, "y_neg": y_neg,
           "primal_error": primal_error, "dual_error": dual_error, "rho": rho,
           "iter": iter, "is_optimal": is_optimal, "status": int(is_optimal),
           "is_primal_infeas": is_primal_infeas, "is_dual_infeas": is_dual_infeas}
    if cache_factor:
        sol["ATA"] = ATA
        sol["M_lu"] = M_lu

    return sol


def clamp(x, x_min=-float('inf'), x_max=float('inf')):
    return min(max(x, x_min), x_max)


def make_matrix(x):
    if x is not None:
        x = np.asarray(x)
        shape = x.shape
        if len(shape) < 2:
            x = x.reshape(-1, 1)
    return x


def prep_bound(x, n_x, default=None):
    if x is None:
        x = default
    x = make_matrix(x)
    if x.shape[0] < n_x:
        x = x.repeat(n_x)
        x = make_matrix(x)
    return x


def compute_objective(x, Q, p):
    if Q is None:
        quad = 0
    else:
        quad = Q.dot(x).T.dot(x)
    value = 0.5 * quad + x.T.dot(p)
    return value


def compute_primal_tol(x, z, A, eps_abs, eps_rel):
    Ax_norm = np.linalg.norm(A.dot(x), ord=np.inf)
    z_norm = np.linalg.norm(z, ord=np.inf)
    primal_tol = eps_abs + eps_rel * max(Ax_norm, z_norm)
    return primal_tol


def compute_dual_tol(x, y, Q, p, A, eps_abs, eps_rel):
    ATy_norm = np.linalg.norm(A.T.dot(y), ord=np.inf)
    Qx_norm = np.linalg.norm(Q.dot(x), ord=np.inf)
    p_norm = np.linalg.norm(p, ord=np.inf)
    dual_tol = eps_abs + eps_rel * max(ATy_norm, Qx_norm, p_norm)
    return dual_tol


def compute_primal_error(x, A, lb, ub):
    Ax = A.dot(x)
    error_1 = np.maximum(Ax - ub, 0)
    error_2 = np.minimum(Ax - lb, 0)
    primal_error = np.linalg.norm(error_1 + error_2, ord=np.inf)
    return primal_error


def compute_dual_error(x, y, Q, p, A):
    dual_error = np.linalg.norm(Q.dot(x) + p + A.T.dot(y), ord=np.inf)
    return dual_error


def scqp_scale(Q, p, A, lb, ub, beta):
    is_Q_sparse = spa.issparse(Q)
    is_A_sparse = spa.issparse(A)
    # --- compute Q scaling:
    if Q is not None:
        # --- compute Q_norm:
        if is_Q_sparse:
            Q_norm = spa.linalg.norm(Q, ord=np.inf, axis=0)
        else:
            Q_norm = np.linalg.norm(Q, ord=np.inf, axis=0)
        # --- compute D:
        if Q_norm.min() <= 0.0:
            Q_norm[Q_norm == 0] = Q_norm.mean()
        D = np.sqrt(1 / Q_norm)
        if beta is None:
            v = np.quantile(D, [0.10, 0.90])
            beta = 1 - v[0] / v[1]
        D = (1 - beta) * D + beta * D.mean()
        if is_Q_sparse:
            Q = (Q.multiply(D).T.multiply(D)).T
        else:
            Q = D[:, None] * Q * D
    else:
        D = 1.0
    # --- scale p:
    p = D * p
    # --- A scaling:
    if is_A_sparse:
        AD = A.multiply(D)
        AD_norm = spa.linalg.norm(AD, ord=np.inf, axis=1)
    else:
        AD = A * D
        AD_norm = np.linalg.norm(AD, ord=np.inf, axis=1)
    if AD_norm.min() <= 0.0:
        AD_norm[AD_norm == 0] = AD_norm.mean()
    E = 1 / AD_norm
    if is_A_sparse:
        A = AD.T.multiply(E).T
    else:
        A = E[:, None] * AD
    # --- lb, ub scaling
    lb = E * lb
    ub = E * ub

    return Q, p, A, lb, ub, D, E
