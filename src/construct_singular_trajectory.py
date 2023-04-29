import matplotlib.pyplot as plt
import numpy as np
from casadi import (
    MX, DM, vertcat, horzcat, sin, cos,
    simplify, substitute, pi, jacobian, 
    nlpsol, Function, pinv, evalf
)
from scipy.integrate import solve_ivp
from scipy.interpolate import (
    make_interp_spline, splrep
)
from dynamics import parameters, Dynamics
from scipy.optimize import brentq


class ServoConnectionParametrized:
    def __init__(self):
        k = MX.sym('k', 3) # parameters of the servo-connection
        theta = MX.sym('theta')
        Q = vertcat(
            k[0] * (theta + np.pi/2) + k[1] * (theta + np.pi/2)**2 + k[2] * (theta + np.pi/2)**3,
            theta
        )
        self.theta = theta
        self.parameters = k
        self.parameters_min = [-20, -20, -20]
        self.parameters_max = [20, 20, 20]
        self.Q = Function('Q', [theta], [Q])

    def subs(self, parameters):
        arg = MX.sym('dummy')
        Q = substitute(self.Q(arg), self.parameters, parameters)
        return Function('Q', [arg], [Q])
    
    def __call__(self, arg):
        return self.Q(arg)

class ReducedDynamics:
    def __init__(self, dynamics, connection):
        theta = MX.sym('theta')
        Q = connection(theta)
        dQ = jacobian(Q, theta)
        ddQ = jacobian(dQ, theta)

        alpha = dynamics.B_perp(Q) @ dynamics.M(Q) @ dQ
        beta = dynamics.B_perp(Q) @ (dynamics.M(Q) @ ddQ + dynamics.C(Q, dQ) @ dQ)
        gamma = dynamics.B_perp(Q) @ dynamics.G(Q)

        self.theta = theta
        self.alpha = Function('alpha', [theta], [alpha])
        self.beta = Function('beta', [theta], [beta])
        self.gamma = Function('gamma', [theta], [gamma])

def find_singular_connection(theta_s, theta_l, theta_r, dynamics, parametrized_connection):
    '''
        @brief Find values of parameters of `parametrized_connection` 
        which give the reduced dynamics a smooth trajectory
    '''
    rd = ReducedDynamics(dynamics, parametrized_connection)

    theta_s = MX(theta_s)
    d_alpha = rd.alpha.jac()
    d_alpha_s = d_alpha(theta_s, 0)
    alpha_s = rd.alpha(theta_s)
    beta_s = rd.beta(theta_s)

    smoothness = 3.
    npts = 11

    constraints = [
        -smoothness/2 * d_alpha_s - beta_s
    ]
    pts = np.concatenate((
        np.linspace(theta_l, float(theta_s), npts)[:-1],
        np.linspace(float(theta_s), theta_r, npts)[1:]
    ))
    for p in pts:
        p = MX(p)
        constraints += [
            rd.gamma(p) - 1e-5,
            rd.alpha(p) * (p - theta_s) - 1e-5,
        ]
    constraints = vertcat(*constraints)
    nlp = {
        'x': parametrized_connection.parameters,
        'f': 0,
        'g': constraints
    }
    S = nlpsol('S', 'ipopt', nlp)
    x1 = np.array(parametrized_connection.parameters_min, float)
    x2 = np.array(parametrized_connection.parameters_max, float)
    sol = S(
        x0 = x1 + (x2 - x1) * np.random.rand(*x1.shape), 
        lbg = 0,
        lbx = parametrized_connection.parameters_min,
        ubx = parametrized_connection.parameters_max
    )

    assert np.all(sol['g'] > -1e-2), 'Solution was not found'

    k_found = sol['x']
    print(f'parameters found {k_found}')
    return parametrized_connection.subs(k_found)

def rd_traj_reverse(rd_traj):
    t = rd_traj['t']
    theta = rd_traj['theta']
    dtheta = rd_traj['dtheta']
    ddtheta = rd_traj['ddtheta']
    theta_s = rd_traj['theta_s']
    dtheta_s = rd_traj['dtheta_s']
    ddtheta_s = rd_traj['ddtheta_s']

    return {
        't': t[-1] - t[::-1],
        'theta': theta[::-1],
        'dtheta': -dtheta[::-1],
        'ddtheta': ddtheta[::-1],
        'theta_s': theta_s,
        'dtheta_s': dtheta_s,
        'ddtheta_s': ddtheta_s,
        't_s': t[-1] - rd_traj['t_s'][::-1]
    }

def solve_singular_2(rd, theta_s, theta0, dtheta0, step=1e-3):
    theta = MX.sym('theta')
    y = MX.sym('y')
    alpha = rd.alpha(theta)
    beta = rd.beta(theta)
    gamma = rd.gamma(theta)
    dy = (-2 * beta * y - gamma) / alpha
    rhs = Function('rhs', [theta, y], [dy])
    dy_s_expr = (jacobian(beta, theta) * gamma - beta * jacobian(gamma, theta)) / \
        (jacobian(alpha, theta) * beta + 2 * beta**2)

    # value at singular point
    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))
    dtheta_s = np.sqrt(2 * y_s)
    rhs_s = jacobian(-gamma/(2*beta), theta)
    dy_s = float(evalf(substitute(dy_s_expr, theta, theta_s)))
    ddtheta_s = float(evalf(dy_s))

    # integrate left and right half-trajectories
    if theta0 < theta_s - step:
        sol = solve_ivp(rhs, [theta0, theta_s - step], [0], max_step=step)
    elif theta0 > theta_s + step:
        sol = solve_ivp(rhs, [theta0, theta_s + step], [0], max_step=step)
    else:
        assert False

    theta = np.concatenate((sol['t'], [theta_s]))
    y = np.concatenate((sol['y'][0], [y_s]))
    dy = np.reshape(rhs(sol['t'], sol['y'][0]), (-1,))
    dy = np.concatenate((dy, [dy_s]))
    dtheta = np.sqrt(2 * y)
    ddtheta = np.reshape(dy, (-1))

    if theta0 > theta_s:
        dtheta = -dtheta

    # find time
    h = 2 * np.diff(theta) / (dtheta[1:] + dtheta[:-1])
    t = np.concatenate(([0], np.cumsum(h)))
    sp = make_interp_spline(
        t, theta, k=5,
        bc_type=([(1, dtheta[0]), (2, ddtheta[0])], [(1, dtheta[-1]), (2, ddtheta[-1])])
    )
    ts = t[-1]

    # evaluate at uniform time-grid
    timestep = ts / 500
    npts = int((t[-1] - t[0]) / timestep + 1.5)
    tt = np.linspace(t[0], t[-1], npts)

    rd_traj = {
        't': tt,
        'theta': sp(tt),
        'dtheta': sp(tt, 1),
        'ddtheta': sp(tt, 2),
        'theta_s': theta_s,
        'dtheta_s': dtheta_s,
        'ddtheta_s': ddtheta_s,
        't_s': np.array([ts])
    }

    return rd_traj

def solve_singular(rd, theta_s, theta0, step=1e-3):
    '''
        @brief Find trajectory of reduced singular dynamics

        `theta_s` is s.t. alpha(theta_s) = 0
        `theta_0` is the initial point of the trajectory
    '''
    theta = MX.sym('theta')
    y = MX.sym('y')
    alpha = rd.alpha(theta)
    beta = rd.beta(theta)
    gamma = rd.gamma(theta)
    dy = (-2 * beta * y - gamma) / alpha
    rhs = Function('rhs', [theta, y], [dy])
    dy_s_expr = (jacobian(beta, theta) * gamma - beta * jacobian(gamma, theta)) / \
        (jacobian(alpha, theta) * beta + 2 * beta**2)

    # value at singular point
    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))
    dtheta_s = np.sqrt(2 * y_s)
    rhs_s = jacobian(-gamma/(2*beta), theta)
    dy_s = float(evalf(substitute(dy_s_expr, theta, theta_s)))
    ddtheta_s = float(evalf(dy_s))

    # integrate left and right half-trajectories
    if theta0 < theta_s - step:
        sol = solve_ivp(rhs, [theta0, theta_s - step], [0], max_step=step)
    elif theta0 > theta_s + step:
        sol = solve_ivp(rhs, [theta0, theta_s + step], [0], max_step=step)
    else:
        assert False

    theta = np.concatenate((sol['t'], [theta_s]))
    y = np.concatenate((sol['y'][0], [y_s]))
    dy = np.reshape(rhs(sol['t'], sol['y'][0]), (-1,))
    dy = np.concatenate((dy, [dy_s]))
    dtheta = np.sqrt(2 * y)
    ddtheta = np.reshape(dy, (-1))

    # forward and backward motions
    theta = np.concatenate((theta[:0:-1], theta))
    dtheta = np.concatenate((-dtheta[:0:-1], dtheta))
    ddtheta = np.concatenate((ddtheta[:0:-1], ddtheta))
    if theta0 > theta_s:
        dtheta = -dtheta

    # find time
    h = 2 * np.diff(theta) / (dtheta[1:] + dtheta[:-1])
    t = np.concatenate(([0], np.cumsum(h)))
    sp = make_interp_spline(
        t, theta, k=5,
        bc_type=([(1, dtheta[0]), (2, ddtheta[0])], [(1, dtheta[-1]), (2, ddtheta[-1])])
    )
    ts = t[-1]

    # evaluate at uniform time-grid
    timestep = ts / 500
    npts = int((t[-1] - t[0]) / timestep + 1.5)
    tt = np.linspace(t[0], t[-1], npts)

    rd_traj = {
        't': tt,
        'theta': sp(tt),
        'dtheta': sp(tt, 1),
        'ddtheta': sp(tt, 2),
        'theta_s': theta_s,
        'dtheta_s': dtheta_s,
        'ddtheta_s': ddtheta_s,
        't_s': np.array([0, ts])
    }

    return rd_traj

def get_trajectory(dynamics, constraint, reduced_trajectory):
    R'''
        Get phase trajectory and reference control corresponding the 
        trajectory `reduced_trajectory` of the reduced dynamics
    '''
    theta = MX.sym('theta')
    dtheta = MX.sym('dtheta')
    ddtheta = MX.sym('ddtheta')
    Q = constraint(theta)
    dQ = jacobian(Q, theta)
    ddQ = jacobian(dQ, theta)
    dq_expr = dQ * dtheta
    dq_fun = Function('dq', [theta, dtheta], [dq_expr])
    ddq_expr = dQ * ddtheta + ddQ * dtheta**2
    ddq_fun = Function('ddq', [theta, dtheta, ddtheta], [ddq_expr])

    B = dynamics.B(Q)
    M = dynamics.M(Q)
    C = dynamics.C(Q, dq_expr)
    G = dynamics.G(Q)

    u_fun= Function('u', [theta, dtheta, ddtheta], 
                    [pinv(B) @ (M @ ddq_expr + C @ dq_expr + G)])
    theta = DM(reduced_trajectory['theta']).T
    dtheta = DM(reduced_trajectory['dtheta']).T
    ddtheta = DM(reduced_trajectory['ddtheta']).T
    u = np.array(u_fun(theta, dtheta, ddtheta)).T
    q = np.array(constraint(theta)).T
    dq = np.array(dq_fun(theta, dtheta)).T
    ddq = np.array(ddq_fun(theta, dtheta, ddtheta)).T
    x = np.concatenate([q, dq], axis=1)
    traj = {
        't': reduced_trajectory['t'],
        'state': x,
        'q': q,
        'dq': dq,
        'ddq': ddq,
        'u': u,
    }

    if 'theta_s' in reduced_trajectory:
        theta_s = reduced_trajectory['theta_s']
        dtheta_s = reduced_trajectory['dtheta_s']
        ddtheta_s = reduced_trajectory['ddtheta_s']
        qs = constraint(theta_s)
        dqs = dq_fun(theta_s, dtheta_s)
        ddqs = ddq_fun(theta_s, dtheta_s, ddtheta_s)

        traj['q_s'] = np.reshape(qs, (-1,))
        traj['dq_s'] = np.reshape(dqs, (-1,))
        traj['ddq_s'] = np.reshape(ddqs, (-1,))
        traj['u_s'] = np.reshape(u_fun(qs, dqs, ddqs), (-1,))
        traj['t_s'] = reduced_trajectory['t_s']

    return traj

def join_trajectories(reduced_trajectory_1, reduced_trajectory_2):
    t1 = reduced_trajectory_1['t']
    theta1 = reduced_trajectory_1['theta']
    dtheta1 = reduced_trajectory_1['dtheta']
    ddtheta1 = reduced_trajectory_1['ddtheta']
    ts1 = reduced_trajectory_1['t_s']

    t2 = reduced_trajectory_2['t']
    theta2 = reduced_trajectory_2['theta']
    dtheta2 = reduced_trajectory_2['dtheta']
    ddtheta2 = reduced_trajectory_2['ddtheta']
    ts2 = reduced_trajectory_2['t_s']

    assert np.allclose(theta1[-1], theta2[0])
    assert np.allclose(dtheta1[-1], dtheta2[0])
    assert np.allclose(ddtheta1[-1], ddtheta2[0])
    theta = np.concatenate((theta1[:-1], theta2))
    dtheta = np.concatenate((dtheta1[:-1], dtheta2))
    ddtheta = np.concatenate((ddtheta1[:-1], ddtheta2))
    t = np.concatenate((t1[:-1], t1[-1] + t2))
    ts = np.concatenate((ts1, ts2 + t1[-1]))
    ts = np.unique(ts)

    traj = reduced_trajectory_1.copy()
    traj['t'] = t
    traj['theta'] = theta
    traj['dtheta'] = dtheta
    traj['ddtheta'] = ddtheta
    traj['t_s'] = ts
    return traj

def join_several(*args):
    if len(args) == 1:
        return args[0]
    return join_several(join_trajectories(args[0], args[1]), *args[2:])

def find_singularity(rd : ReducedDynamics, sdiap : tuple):
    f = lambda s: rd.alpha(s)
    x0,r = brentq(f, sdiap[0], sdiap[1], full_output=True)
    assert r.converged, 'Can\'t find zero of alpha'
    return x0

class ServoConnection:
    def __init__(self, q_diap):
        s = MX.sym('s')
        p = MX.sym('p', 6)
        x = p[0] + p[1] * s + p[2] * s**2
        theta = p[3] + p[4] * s + p[5] * s**2
        q = vertcat(x, theta)

        self.theta = s
        self.parameters = p
        self.Q = Function('Q', [theta], [Q])

def find_singular_periodic_trajectory(dynamics):
    sc = ServoConnectionParametrized()
    theta_s = -2.1
    theta_l = theta_s - 0.3
    theta_r = theta_s + 0.5
    Q = find_singular_connection(theta_s, theta_l, theta_r, dynamics, sc)
    rd = ReducedDynamics(dynamics, Q)

    theta_l = theta_s - 0.2
    theta_r = theta_s + 0.4
    theta_s = find_singularity(rd, [theta_l, theta_r])
    rt1 = solve_singular(rd, theta_s, theta_l)
    rt2 = solve_singular(rd, theta_s, theta_r)
    rt = join_trajectories(rt1, rt2)
    return get_trajectory(dynamics, Q, rt)

def test1():
    dynamics = Dynamics(parameters)
    sc = ServoConnectionParametrized()
    theta_s = -2.2
    theta_l = -2.5
    theta_r = -1.7
    Q = find_singular_connection(theta_s, theta_l, theta_r, dynamics, sc)
    rd = ReducedDynamics(dynamics, Q)
    theta_s = find_singularity(rd, [theta_l, theta_r])
    rt1 = solve_singular(rd, theta_s, theta_l)
    rt2 = solve_singular(rd, theta_s, theta_r)
    rt = join_trajectories(rt1, rt2)

    plt.subplot(121, title='trajectory')
    plt.plot(rt['theta'], rt['dtheta'])
    plt.axvline(rt['theta_s'])
    plt.xlabel(R'$\theta$')
    plt.ylabel(R'$\dot\theta$')
    plt.grid(True)

    plt.subplot(122, title='coefficientss')
    theta = np.linspace(theta_l, theta_r)
    plt.plot(theta, rd.alpha(theta), label=R'$\alpha$')
    plt.plot(theta, rd.beta(theta), label=R'$\beta$')
    plt.plot(theta, rd.gamma(theta), label=R'$\gamma$')
    plt.xlabel(R'$\theta$')
    plt.grid(True)
    plt.legend()
    plt.axvline(theta_s)

    plt.tight_layout()
    plt.show()

def test2():
    dynamics = Dynamics(parameters)
    tr = find_singular_periodic_trajectory(dynamics)

    _,axes = plt.subplots(2, 2)

    plt.sca(axes[0,0])
    plt.plot(tr['q'][:,1], tr['q'][:,0])
    plt.xlabel(R'$\phi$')
    plt.ylabel(R'$x$')
    plt.axvline(tr['q_s'][1], ls='--', color='r')
    plt.grid(True)

    plt.sca(axes[0,1])
    plt.plot(tr['q'][:,0], tr['dq'][:,0])
    plt.xlabel(R'$x$')
    plt.ylabel(R'$\dot x$')
    plt.axvline(tr['q_s'][0], ls='--', color='r')
    plt.grid(True)

    plt.sca(axes[1,0])
    axes[1,0].sharex(axes[0,0])
    plt.plot(tr['q'][:,1], tr['dq'][:,1])
    plt.xlabel(R'$\phi$')
    plt.ylabel(R'$\dot \phi$')
    plt.axvline(tr['q_s'][1], ls='--', color='r')
    plt.grid(True)

    plt.sca(axes[1,1])
    axes[1,1].sharex(axes[0,0])
    plt.plot(tr['q'][:,1], tr['u'])
    plt.xlabel(R'$\phi$')
    plt.axvline(tr['q_s'][1], ls='--', color='r')
    plt.ylabel(R'$u$')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test1()
    test2()
