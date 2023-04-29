from construct_singular_trajectory import find_singular_periodic_trajectory
from anim import animate
from dynamics import Dynamics, parameters
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

def main():
    dynamics = Dynamics(parameters)
    tr = find_singular_periodic_trajectory(dynamics)
    t1 = (tr['t_s'][1] + tr['t_s'][0]) / 2
    t2 = t1 + 3 * tr['t'][-1]
    t = np.arange(t1, t2, 0.001)
    sp = make_interp_spline(tr['t'], tr['state'], k=3, bc_type='periodic')
    st = sp(t)
    sp = make_interp_spline(tr['t'], tr['u'], k=3, bc_type='periodic')
    u = sp(t)
    a = animate(t, st, u, speed=0.25, fps=30, name='fig/anim.gif')
    plt.show()

if __name__ == '__main__':
    main()
