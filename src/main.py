from construct_singular_trajectory import find_singular_periodic_trajectory
from anim import animate
from dynamics import Dynamics, parameters
import matplotlib.pyplot as plt


def main():
    dynamics = Dynamics(parameters)
    tr = find_singular_periodic_trajectory(dynamics)
    a = animate(tr['t'], tr['state'], tr['u'], speed=0.25, fps=30, name='fig/anim.gif')
    plt.show()

if __name__ == '__main__':
    main()
