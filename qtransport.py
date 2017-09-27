#! coding: utf8
import numpy
from numpy import exp, pi
from scipy.integrate import quad

Q = 1.60217662e-19 #кл
H = 4.135667e-15 # эв * с
HBAR =1.055e-34 # дж * с (деленная на 2 pi)
J_TO_EV = 1.0 / Q

K_B = 1.38064852e-23 # дж / К
K_B_ev = K_B / Q # дж / К

EPSILON_0 = 8.85418782e-12 # фарад / м
EPS = 1e-4

# масса электрона 
M_E = 9.11e-31 # кг

MAX_SCC_STEPS = 1500

def fermidir(E, mue, T):
	"""
	E - eV
	mue - eV
	T - K
	"""
	KT = T * K_B_ev
	return 1.0 / (1 + exp((E - mue) / KT ))


def lorence(E, epsilon,  gamma):
	return gamma / (2 * pi) / ((E - epsilon) ** 2 + (gamma / 2) ** 2)

class Device(object):
	def __init__(self, mue, gamma_1, gamma_2, dos, SCC=None, temperature = 300, SCC_alpha=0.9):

		self.gamma_1, self.gamma_2 = gamma_1, gamma_2
		self.gamma = gamma_1 + gamma_2
		self.mue = mue

		self.dos = dos
		

		self.temperature = temperature
		if not isinstance(temperature, tuple):
			self.temperature = (temperature, temperature)


		self.E_min = -10.0 # eV HARD
		self.fermi_de = self.calc_delta_fermi()

		self.use_SCC = False
		if SCC is not None:
			self.use_SCC = True
			self.SCC_alpha = SCC_alpha
			self.C_S, self.C_G, self.C_D, self.N_0 = SCC
			C_E = self.C_G + self.C_S + self.C_D
			self.U_0 = J_TO_EV * Q ** 2 / C_E # eV !!!


	def calc_delta_fermi(self):
		i = 0
		de = 1e-5

		T1, T2 = self.temperature

		def fmax(e):
			return max(fermidir(e, 0, T1), fermidir(e, 0, T2))

		while fmax(i * de) > EPS:
			i += 1

		return i * de

	def n_electrons(self, U, V):
		gamma_1, gamma_2, gamma = self.gamma_1, self.gamma_2, self.gamma
		
		T1, T2 = self.temperature

		mue_1 = self.mue
		mue_2 = self.mue - Q * V * J_TO_EV # of course = mue - V (but only for energy in eV)

		def n(E):
			f1 = fermidir(E, mue_1, T1)
			f2 = fermidir(E, mue_2, T2)
			return self.dos(E - U) * (gamma_1 * f1 + gamma_2 * f2) / gamma

		return quad(n, self.E_min, max(mue_1, mue_2) + self.fermi_de)[0]

	def calc_SCC_U(self, V_D, V_G):
		C_S, C_G, C_D = self.C_S, self.C_G, self.C_D
		C_E = C_S + C_G + C_D

		U = 0.0
		dE = 1.0

		alpha = self.SCC_alpha
		steps = 0
		U_L = (C_G / C_E * (-Q * V_G) + C_D / C_E * (-Q * V_D)) * J_TO_EV

		while dE > EPS and steps < MAX_SCC_STEPS:
			steps += 1
			N = self.n_electrons(U_L + U, V_D)
			U_cur = self.U_0 * (N - self.N_0)
			U_next = alpha * U + (1 - alpha) * U_cur
			dE = abs(U_next - U)
			U = U_next

		if steps == MAX_SCC_STEPS: print 'Max SCC iterations exceed. Current error is', dE
		return U_next + U_L

	def T(self, E, U = 0.0):
		"""
		Transmission function.
		U - SCC_shift
		"""
		gamma_1, gamma_2, gamma = self.gamma_1, self.gamma_2, self.gamma
		return self.dos(E - U) * 2 * pi * gamma_1 * gamma_2 / gamma


	def IV_characteristic(self, V_D, V_G, verbose=False):
		"""
		V_D : numpy.array volts
		V_G : gate potential volts
		"""
		I = numpy.zeros_like(V_D)
		mue = self.mue
		mue_1 = mue
		T1, T2 = self.temperature

		fermi_de = self.fermi_de

		for i, v in enumerate(V_D):
			if verbose : 'Calculation point', i, 'voltage value', v
			mue_2 = mue - Q * v * J_TO_EV # of course = mue - V (but only for energy in eV)
			f1 = lambda e : fermidir(e, mue_1, T1)
			f2 = lambda e : fermidir(e, mue_2, T2)
			
			min_mue = min(mue_1, mue_2)
			max_mue = max(mue_1, mue_2)

			U = self.calc_SCC_U(v, V_G) if self.use_SCC else -Q * 0.5 * v * J_TO_EV

			I[i] = quad(lambda e : self.T(e, U) * (f1(e) - f2(e)), min_mue - fermi_de, max_mue + fermi_de)[0]

		I *= Q / H 
		return I

