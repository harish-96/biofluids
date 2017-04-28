import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import jit


p0 = 0
r0 = 1
rho = 1
nu = 0.046
delta = 0.01
freq = 1


# Initialize the problem. Declare variables for Area and flow rate.
# Arguments: Number of grid points, number of time steps, list giving
# the start and end co ordinates of the pipe and a list for start and end
# time of simulation
def init(gridpts, timesteps, spaceDomain, timeDomain, ic):
    x = np.linspace(spaceDomain[0], spaceDomain[1], gridpts)
    t = np.linspace(timeDomain[0], timeDomain[1], timesteps)
    A = np.zeros((gridpts, timesteps))
    q = np.zeros((gridpts, timesteps))
    A, q = ic(A, q)
    return A, q, x, t


# Obtain the pressure from Area using the model obtained from empirical data
@jit
def get_p(A, k1=2 * 10**7, k2=-22.53, k3=8.65 * 10**5):
    p = p0 + 4 / 3 * (k1 * np.exp(k2 * r0) + k3) * \
        (1 - np.sqrt(np.pi * r0**2 / A))
    return p


# Obtain the pressure from Area using the same model as above
@jit
def get_A(p, k1=2 * 10**7, k2=-22.53, k3=8.65 * 10**5):
    A = (np.pi * r0**2) / (1 - 0.75 * (p - p0) /
                           (k1 * np.exp(k2 * r0) + k3))**2
    return A


# Second order artificial dissipation
@jit
def dissipation_2(q_step, mu2):
    tmp = np.zeros_like(q_step)
    tmp[1:-1] = mu2 * (q_step[:-2] - 2 *
                       q_step[1:-1] + q_step[2:])
    return tmp


# Fourth order artificial dissipation
@jit
def dissipation_4(q_step, mu4):
    tmp = np.zeros_like(q_step)
    tmp[2:-2] = (q_step[:-4] - 4 * q_step[1:-3] + 6 *
                 q_step[2:-2] - 4 * q_step[3:-1] + q_step[4:])
    tmp = -mu4 * tmp
    return tmp


# @jit
# Forward time central space discretization. Performs one time step.
# bc is the function defining boundary condition and viscA and viscq are
# artificial viscosities for the corresponding differential equations
def FTCS(A, q, t_step, t, x, bc, viscA=[0.001, 0.0001], viscq=[0.000001, 0.0000001]):
    mu2, mu4 = viscA
    At = A[:, t_step]
    qt = q[:, t_step]
    pt = get_p(At)
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    A[1:-1, t_step + 1] = (At[1:-1] - dt / dx / 2 *
                           (qt[2:] - qt[:-2]))

    # Adding the artificial dissipation terms
    A[:, t_step + 1] += dissipation_2(At, mu2) +\
        dissipation_4(At, mu4)

    mu2, mu4 = viscq
    q[1:-1, t_step + 1] = q[1:-1, t_step] - dt * \
        (2 * np.sqrt(At[1:-1] * np.pi) * nu * qt[1:-1] / delta / At[1:-1]) \
        - dt / dx / 2 * ((qt**2 / At)[2:] - (qt**2 / At)[:-2])\
        - dt / dx / 2 * At[1:-1] / rho * (pt[2:] - pt[:-2])

    # Adding the artificial dissipation terms
    q[:, t_step + 1] += dissipation_2(qt, mu2) + dissipation_4(qt, mu4)

    # computing the inlet and outlet boundary conditions
    left_bc, right_bc = bc(At, qt, resistance, t[t_step])
    A[0, t_step + 1], q[0, t_step + 1] = left_bc
    A[-1, t_step + 1], q[-1, t_step + 1] = right_bc
    if t_step % 10000 == 0:
        print(t_step)


# Initial conditions. Assuming a constant area duct with zero flow rate
def ic(A, q):
    A[:, 0] = np.pi * r0**2
    q[:, 0] = 0
    return A, q


# Boundary conditions. The flow rate at the inflow is prescribed and the
# outflow bc is a function argument. The scheme requires 2 more conditions at
# the inlet and outlet and so the area at the left boundary and the flow rate
# at the right boundary have been extrapolated from within the domain.
@jit
def bc(A_prev, q_prev, outflow, time):
    q_l = 22 * np.sin(freq * 2 * np.pi * time)
    A_l = A_prev[1]
    left_bc = A_l, q_l
    q_r = q_prev[-2]
    A_r = outflow(q_r)
    right_bc = A_r, q_r
    return left_bc, right_bc


# Outflow proportional to pressure. Simplest bc
@jit
def resistance(q, model_params=[3444.15]):
    p = q * model_params[0] + p0
    return get_A(p)


@jit
def windkessel(q, model_params=[1722, 1722, 1722], omega=freq):
    R1, R2, C = model_params
    i = np.complex(0, 1)
    Z = (R1 + R2 + i * omega * C * R1 * R2)
    p = q * Z + p0
    return get_A(p)


# Windkessel model for outflow. Refer Olufsen's paper on
# "Structured tree outflow boundary condition for blood flow
# in larger systemic arteries"
@jit
def windkessel(q, model_params=[1, 1, 1]):
    return np.zeros_like(q)


def plot_step(A, q, t_step, x):
    plt.plot(x, A[:, t_step])
    plt.show()
    plt.plot(x, q[:, t_step])
    plt.show()


def plot_hysteresis(A, q, time=[50000, 70000], x_arr=[0, 0.25, 0.5, 1]):
    x_arr = np.array(x_arr) * len(A[:, 0])
    for pos in x_arr:
        pos = int(pos)
        if pos == 0:
            pos += 1
        p = get_p(A[pos - 1, time[0]:time[1]])
        plt.plot(p, q[pos - 1, time[0]:time[1]], label=str(pos))
    plt.legend()
    plt.show()


# Code for animation
def animate(gridpts, timesteps, spaceDomain, timeDomain, ic):
    A, q, x, t = init(gridpts, timesteps, spaceDomain, timeDomain, ic)
    lines = []
    fig, axarr = plt.subplots(2, sharex=True)
    lines.append(axarr[0].plot(x, q[:, 0])[0])
    axarr[0].set_ylabel('q')
    axarr[0].set_ylim([-25, 25])
    lines.append(axarr[1].plot(x, A[:, 0])[0])
    axarr[1].set_ylabel('A')
    axarr[1].set_ylim([2.75, 3.8])
    line_ani = animation.FuncAnimation(fig, update_line,
                                       frames=int(len(t) / 100) - 1,
                                       # frames=500,
                                       fargs=(A, q, lines, t, x, bc,),
                                       interval=1e-5 * (t[1] - t[0]),
                                       blit=True, repeat=False)
    plt.show()
    # line_ani.save('resistance.mp4', fps=10)
    return A, q, x, t


def update_line(i, A, q, lines, t, x, bc):
    for j in range(100):
        FTCS(A, q, 100 * i + j, t, x, bc)
    lines[0].set_data(x, q[:, i * 100 + 1])
    lines[1].set_data(x, A[:, i * 100 + 1])
    return lines


if __name__ == '__main__':
    A, q, x, t = animate(100, 1000000, [0, 100], [0, 10], ic)
    plot_hysteresis(A, q, time=[800000, 1000000 - 100])
    # A, q, x, t = init(200, 200000, [0, 100], [0, 10], ic)
    # dt = t[1] - t[0]
    # dx = x[1] - x[0]
    # for i in range(10000):
    #     FTCS(A, q, i, dt, dx, bc)
    #     if i % 100 == 0:
    #         plot_step(A, q, i, x)
