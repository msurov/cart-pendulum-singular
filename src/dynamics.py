from dataclasses import dataclass
from casadi import SX, MX, vertcat, horzcat, sin, cos, Function, pinv, cross, inv, substitute

@dataclass
class Parameters:
    R'''
        @brief Parameters of the cart-pendulum
    '''
    pend_mass : float
    cart_mass : float
    gravity_accel : float
    pend_length : float

'''
    Default parameters
'''
parameters = Parameters(
    pend_mass=0.1,
    cart_mass=0.4,
    gravity_accel=9.8,
    pend_length=1
)

class Dynamics:
    def __init__(self, parameters):
        # parameters
        m = parameters.pend_mass
        m_cart = parameters.cart_mass
        g = parameters.gravity_accel
        l = parameters.pend_length

        # phase coordinates
        x = MX.sym('x')
        dx = MX.sym('dx')
        phi = MX.sym('phi')
        dphi = MX.sym('dphi')
        u = MX.sym('u')
        
        self.u = u
        self.q = vertcat(x, phi)
        self.dq = vertcat(dx, dphi)
        self.state = vertcat(self.q, self.dq)

        M = MX.zeros((2,2))
        M[0,0] = (m + m_cart) / (m * l)
        M[0,1] = \
        M[1,0] = -cos(phi)
        M[1,1] = l

        C = MX.zeros((2,2))
        C[0,1] = sin(phi)*dphi

        G = MX.zeros((2,1))
        G[1] = -g*sin(phi)

        B = MX.zeros((2,1))
        B[0] = 1

        B_perp = MX.zeros((1, 2))
        B_perp[0,1] = 1

        self.M = Function('M', [self.q], [M])
        self.B = Function('B', [self.q], [B])
        self.C = Function('C', [self.q, self.dq], [C])
        self.G = Function('G', [self.q], [G])
        self.B_perp = Function('B_perp', [self.q], [B_perp])

        ddq = inv(M) @ (-C @ self.dq - G + B @ u)
        rhs = vertcat(self.dq, ddq)
        self.rhs = rhs
