from numba import jit
import numpy as np


class FTCS:

    def __init__(self, dtdx, bc):
        self.dtdx = dtdx
        self.bc = bc

    @jit
    def central_diff(self, q_step, e_step):
        tmp = np.zeros_like(q_step)
        tmp[:, 1:-1] = -self.dtdx * 0.5 * (e_step[:, 2:] - e_step[:, :-2])
        return tmp

    @jit
    def dissipation_2(self, q_step, mu2):
        tmp = np.zeros_like(q_step)
        tmp[:, 1:-1] = mu2 * (q_step[:, :-2] - 2 *
                              q_step[:, 1:-1] + q_step[:, 2:])
        return tmp

    @jit
    def dissipation_4(self, q_step, mu4):
        tmp = np.zeros_like(q_step)
        tmp[:, 2:-2] = (q_step[:, :-4] - 4 * q_step[:, 1:-3] + 6 *
                        q_step[:, 2:-2] - 4 * q_step[:, 3:-1] + q_step[:, 4:])
        tmp = -mu4 * tmp
        return tmp

    @jit
    def step(self, q_step, e_step, t_step, dt, art_viscosity):
        mu2, mu4 = art_viscosity
        dq = self.central_diff(q_step, e_step) +\
            self.dissipation_2(q_step, mu2) +\
            self.dissipation_4(q_step, mu4)
        q_step1 = np.zeros_like(q_step)
        q_step1 = q_step + dq
        left_bc, right_bc = self.bc(t_step + 1, q_step1)
        q_step1[:, 0] = left_bc
        q_step1[:, -1] = right_bc
        return q_step1


class Lax_Fred:

    def __init__(self, dtdx, bc):
        self.dtdx = dtdx
        self.bc = bc

    @jit
    def step(self, q_step, e_step, t_step, dt, art_viscosity):
        q_step1 = np.ones_like(q_step)
        q_step1[:, 1:-1] = 0.5 * (q_step[:, 2:] + q_step[:, :-2]) - \
            self.dtdx * 0.5 * (e_step[:, 2:] - e_step[:, :-2])
        left_bc, right_bc = self.bc(t_step + 1, q_step1)
        q_step1[:, 0] = left_bc
        q_step1[:, -1] = right_bc
        return q_step1
