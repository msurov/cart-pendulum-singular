from scipy.integrate import ode
import numpy as np
from copy import copy


class Simulator:
    def __init__(self, F, feedback, step, stopcnd=None):
        R'''
            `feedback`: is a of type `function(x, u) -> dx/dt`

        '''
        self.F = F
        self.feedback = feedback
        self.step = step
        self.stopcnd = stopcnd
        self.t = None
        self.u = None
        self.x = None

    def run(self, xstart, tstart, tend):
        self.t = float(tstart)
        self.x = np.asanyarray(xstart)
        xdim, = self.x.shape
        self.u = self.feedback(self.x)
        udim, = self.u.shape

        rhs = lambda _,x: np.reshape(self.F(x, self.u), xdim)
        integrator = ode(rhs)
        integrator.set_initial_value(self.x, tstart)

        solt = [self.t]
        solx = [self.x]
        solu = [self.u]
        solfb = []

        while True and self.t < tend:
            if not integrator.successful():
                print('Warn: integrator doesn\'t feel good')
            integrator.integrate(self.t + self.step)
            self.t += self.step
            self.x = integrator.y
            self.u = np.reshape(self.feedback(self.x), udim)
            solt += [self.t]
            solx += [self.x.copy()]
            solu += [self.u.copy()]

            if hasattr(self.feedback, 'state'):
                solfb += [copy(self.feedback.state)]

            if self.stopcnd is not None and self.stopcnd():
                print('stopped by callback status')
                break

        solt = np.asanyarray(solt)
        solx = np.asanyarray(solx)
        solu = np.asanyarray(solu)
        return solt, solx, solu, solfb
