
class ServoConnectionParametrized:
    def __init__(self):
        k = MX.sym('k', 5) # parameters of the servo-connection
        theta = MX.sym('theta')

        Q = vertcat(
            k[2] + k[3] * theta + k[4] * theta**2,
            theta
        )
        self.theta = theta
        self.parameters = k
        self.parameters_min = [-1, -2, 0, -2, -2]
        self.parameters_max = [1, 2, np.pi, 2, 2]
        self.Q = Function('Q', [theta], [Q])
    
    def subs(self, parameters):
        arg = MX.sym('dummy')
        Q = substitute(self.Q(arg), self.parameters, parameters)
        return Function('Q', [arg], [Q])
    
    def __call__(self, arg):
        return self.Q(arg)
