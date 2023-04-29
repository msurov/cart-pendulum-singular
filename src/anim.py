import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np
from scipy.interpolate import make_interp_spline


class CartPend:
  def __init__(self):
    self.wheel1, = plt.plot([-0.1], [-0.05], 'o', markersize=20, color='black')
    self.wheel2, = plt.plot([0.1], [-0.05], 'o', markersize=20, color='black')
    self.cart, = plt.plot([-0.1, 0.1], [0,0], linewidth=30, color='cyan')
    self.line, = plt.plot([0,0], [0,1], linewidth=5, color='red')
    self.ball1, = plt.plot([0], [1], 'o', markersize=15, color='black')
    self.ball2, = plt.plot([0], [0], 'o', markersize=8, color='black')

  def move(self, x, phi):
    st = np.sin(phi)
    ct = np.cos(phi)

    self.wheel1.set_data([x + -0.1], [-0.08])
    self.wheel2.set_data([x + 0.1], [-0.08])
    self.cart.set_data([x + -0.1, x + 0.1], [0,0])
    self.line.set_data([x, x - st], [0,ct])
    self.ball1.set_data([x - st], [ct])
    self.ball2.set_data([x], [0])
  
  def elems(self):
    return self.wheel1, self.wheel2, self.cart, self.line, self.ball1, self.ball2

class PlotAnim:
  def __init__(self, **plot_properties):
    self.line, = plt.plot([], [], **plot_properties)

  def update(self, x, y):
    xarr,yarr = self.line.get_data()
    xarr = np.append(xarr, [x])
    yarr = np.append(yarr, [y])
    self.line.set_data(xarr, yarr)
  
  def set(self, xarr, yarr):
    self.line.set_data(xarr, yarr)

def get_cart_bounds(x, phi):
  x1 = x - np.sin(phi)
  y1 = np.cos(phi)
  xmin = min(np.min(x1), np.min(x))
  xmax = max(np.max(x1), np.max(x))
  ymin = min(np.min(y1), 0)
  ymax = max(np.max(y1), 0)
  dx = xmax - xmin
  dy = ymax - ymin
  xmax += 0.1 * dx
  xmin -= 0.1 * dx
  ymax += 0.1 * dy
  ymin -= 0.1 * dy
  return xmin, xmax, ymin, ymax

def get_var_bounds(state):
  stmin = np.min(state, axis=0)
  stmax = np.max(state, axis=0)
  dst = stmax - stmin
  stmin -= 0.1 * dst
  stmax += 0.1 * dst
  return stmin, stmax

def animate(t, state, control, fps=60, speed=1., name=None):
  x = state[:,0]
  phi = state[:,1]
  state_fun = make_interp_spline(t, state, k=3)
  control_fun = make_interp_spline(t, control, k=3)

  xmin, xmax, ymin, ymax = get_cart_bounds(x, phi)

  fig = plt.figure('Cart-Pendulum Sngular Trajectory', figsize=(12, 8))
  ax1 = plt.subplot(211)
  plt.sca(ax1)
  plt.grid(True)
  cp = CartPend()
  ax1.set_aspect(1)
  ax1.set_xbound((xmin, xmax))
  ax1.set_ybound((ymin, ymax))

  stmin,stmax = get_var_bounds(state)
  ctmin,ctmax = get_var_bounds(control)

  ax2 = plt.subplot(234)
  plt.xlabel(R'$\phi$')
  plt.ylabel(R'$x$')
  plt.grid(True)
  ax2.set_xbound((stmin[1], stmax[1]))
  ax2.set_ybound((stmin[0], stmax[0]))
  anim_q = PlotAnim(alpha=0.5)

  ax3 = plt.subplot(235)
  plt.xlabel(R'$\phi$')
  plt.ylabel(R'$\dot\phi$')
  plt.grid(True)
  ax3.set_xbound((stmin[1], stmax[1]))
  ax3.set_ybound((stmin[3], stmax[3]))
  anim_dq = PlotAnim(alpha=0.5)

  ax4 = plt.subplot(236)
  plt.xlabel(R'$\phi$')
  plt.ylabel(R'$u$')
  plt.grid(True)
  ax4.set_xbound((stmin[1], stmax[1]))
  ax4.set_ybound((ctmin, ctmax))
  anim_u = PlotAnim(alpha=0.5)

  plt.tight_layout()

  def drawframe(iframe):
    ti = (iframe / fps) * speed + t[0]
    x,phi,*_ = state_fun(ti)
    cp.move(x, phi)

    st = state[t < ti]
    ct = control[t < ti]
    anim_q.set(st[:,1], st[:,0])
    anim_dq.set(st[:,1], st[:,3])
    anim_u.set(st[:,1], ct)
    return cp.elems() + (anim_q.line, anim_dq.line, anim_u.line)

  anim_time = (t[-1] - t[0]) / speed
  nframes = int(anim_time * fps)
  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
  if name is not None:
    anim.save(name, writer='pillow', fps=fps)
  return anim
