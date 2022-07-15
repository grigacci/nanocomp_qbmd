"""
Core functions that perform the calculations used in e-mulate.
These functions do not interact with the interface.
"""

# Global variables - physical constants
E_CHARGE = 1.602176634e-19  # Electron charge - C
E_MASS = 9.1093837015e-31  # Electron mass - Kg
PLANCK = 6.62607015e-34  # Planck constant - J/Hz
HBAR = 1.054571817e-34  # Reduced Planck constant - J.s
# Energy potential of the boundary. May be set as zero, the barrier material's or infinite
BOUNDARY_POT = 1e20  # eV

import numpy as np
from numpy.lib import scimath as sm  # sqrt of negative number is complex
import time


def Shooting(inputE, effm_cte, pot, npe, dx):
    """
    Método shooting.
    Identificando estados ligados que se iniciam e terminam em zero. (A função de onda inicia e
    termina em zero)
    """
    # Creating the variables
    Energy = inputE * E_CHARGE
    V_q_array = pot * E_CHARGE

    Psi_qn = 0.0
    Psi_q = 1.0
    m_qn = effm_cte[0]

    if V_q_array[0] == Energy:  # evitando divisao por zero
        print(f"Ouch, almost divided by zero when energy = {Energy:.4e} eV")
        # Energy = Energy + 0.000001 * dE * e_charge
        Energy = Energy * (1.0 + 1.0e-8)

    m_q = (effm_cte * (1.0 + (Energy - V_q_array) / npe) + m_qn) / 2.0

    # constant that is used in the loop, to avoid repeating this multiplication.
    a_ct = 2.0 * dx * dx / HBAR / HBAR
    # Subtracting the array as a whole, to avoid a subtraction per loop
    V_q_array_E = V_q_array - Energy

    # subtraindo 2 porque estou calculando dois passos a menos do que a espessura desejada
    for i in range(len(pot) - 2):
        # pense numa estrutura com apenas 3 passos a serem calculados.
        # o primeiro passo eh dado Psi_qn e m_qn, ja tenho Psi_q e m_q e vou calcular apenas
        # uma vez o Psi_qp. O i=i+1 eh para comecar o calculo em Psi_q e m_q
        i += 1
        # Nas interfaces a matriz descontinuidade deve levar em conta as massas das
        # camadas, e nao a media das massas
        Psi_qp = m_q[i] * (
            ((a_ct * V_q_array_E[i]) + 1.0 / m_qn + 1.0 / m_q[i]) * Psi_q
            - Psi_qn / m_qn
        )
        Psi_qn = Psi_q
        Psi_q = Psi_qp
        m_qn = m_q[i]
    return Psi_qp


def T_1_1Re(inputE, x, effm_cte, pot, npe, m_eff_ct_barrier):
    """
    Parte Real
    esta funcao eh igual a funcao Transmission, ela serve para ser usada dentro da
    funcao de newton na procura dos zeros. soh aceita uma funcao com apenas um argumento
    de saida exatamente igual a funcao Transmission, mas para encontrar os zeros da
    funcao por newton, preciso de uma funcao com apenas uma saida.
    """
    Energy = inputE * E_CHARGE / 1000.0  # Going from meV to SI units
    # Going from eV to SI units (multiplying the potential by the charge of the electron)
    pot = pot * E_CHARGE

    if pot[0] == Energy:  # Avoiding a division by zero error
        print("Ouch, almost divided by zero...")
        # Energy = Energy + 0.000001 * dE * e_charge
        Energy = Energy * (1.0 + 1.0e-8)

    # Creating the transfer matrix as a complex identity (avoids casting type errors)
    Transfer = np.mat([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])

    # The first step should always be a discontinuity like an unity matriz
    effm_i = effm_cte[0] * (1.0 + (Energy - pot[0]) / npe[0])
    effm_0 = effm_cte[0] * (1.0 + (Energy - pot[0]) / npe[0])

    # Effective mass used for the boundary conditions
    effm_bound = m_eff_ct_barrier

    # Wavenumber in position zero and boundary condition
    wave_k_zero = sm.sqrt(2.0 * effm_i * (Energy - pot[0])) / HBAR + 0.0j
    wave_k_bound = sm.sqrt(2.0 * effm_bound * (Energy - BOUNDARY_POT)) / HBAR + 0.0j

    # Trying to calculate some things before the loop
    effm = effm_cte * (1.0 + (Energy - pot) / npe)
    wave_k = sm.sqrt(2.0 * effm * (Energy - pot)) / HBAR + 0.0j
    effm_previous = np.roll(effm, 1)
    wave_k_previous = np.roll(wave_k, 1)

    gamma = ((wave_k_previous * effm) / (effm_previous * wave_k)) + 0.0j

    # Creating the D matrices in one step (instead of iteratively)
    len_arrays = len(x)
    D_mats = np.empty((len_arrays, 2, 2), dtype=complex)
    D_mats[:, 0, 0] = (1.0 + gamma + 0.0j) * 0.5
    D_mats[:, 1, 0] = (1.0 - gamma + 0.0j) * 0.5
    D_mats[:, 0, 1] = D_mats[:, 1, 0]
    D_mats[:, 1, 1] = D_mats[:, 0, 0]

    # Array with the values of delta x, except that it's zero on the first step.
    deltax_array = x - np.roll(x, 1)
    deltax_array[0] = 0.0

    # Creating the Pinitial and Pfinal matrices in one step (instead of iteratively)
    # Initial depends on the previous wavenumbers and final depends on the current
    Pi_mats = np.empty((len_arrays, 2, 2), dtype=complex)
    Pi_mats[:, 0, 0] = np.exp(1.0j * wave_k_previous * deltax_array / 2.0)
    Pi_mats[:, 1, 0] = 0.0
    Pi_mats[:, 0, 1] = 0.0
    Pi_mats[:, 1, 1] = np.exp(-1.0j * wave_k_previous * deltax_array / 2.0)
    Pf_mats = np.empty((len_arrays, 2, 2), dtype=complex)
    Pf_mats[:, 0, 0] = np.exp(1.0j * wave_k * deltax_array / 2.0)
    Pf_mats[:, 1, 0] = 0.0
    Pf_mats[:, 0, 1] = 0.0
    Pf_mats[:, 1, 1] = np.exp(-1.0j * wave_k * deltax_array / 2.0)

    for i in range(len_arrays):
        """
        This loop calculates the transfer matrix by multiplying the P and D matrices of
        each layer. Without an electric field, most of these layers are identical, except
        for the ones at the interfaces between two layers and at the boundaries.
        The parenthesis are necessary, as the calculations were happening in the wrong order
        during testing
        """
        Transfer = ((Pi_mats[i] @ D_mats[i]) @ Pf_mats[i]) @ Transfer

    gamma_i = (wave_k[-1] * effm_bound) / (effm[-1] * wave_k_bound) + 0.0j

    D_i = (
        np.mat(
            [
                [1.0 + gamma_i + 0.0j, 1.0 - gamma_i + 0.0j],
                [1.0 - gamma_i + 0.0j, 1.0 + gamma_i + 0.0j],
            ]
        )
        * 0.5
    )

    gamma_f = (effm_0 * wave_k_bound) / (wave_k_zero * effm_bound) + 0.0j

    D_f = (
        np.mat(
            [
                [1.0 + gamma_f + 0.0j, 1.0 - gamma_f + 0.0j],
                [1.0 - gamma_f + 0.0j, 1.0 + gamma_f + 0.0j],
            ]
        )
        * 0.5
    )

    Transfer = D_i * Transfer * D_f
    # terminando condicao de contorno
    # o transfer eh real, forcando para funcionar a funcao de newton para encontrar os zeros

    out = np.real(Transfer[1, 1])
    # print(f"Input: {inputE:.16e} - Output: {out:.16e}")
    return out


def Numerov(inputE, x, effm_cte, pot):
    """
    Numerov method for solving the Schrödinger equation.
    Splits the structure in two parts, following Pedro's algorithm verbatim.
    """
    numerov_t_start = time.time()

    psi = np.zeros_like(x, dtype=complex)
    psi[1] = 0.001 + 0.0j

    dx = x[1] - x[0]

    for i in np.arange(1, len(x) - 2, 1):
        # Calculating a[i] = a_i
        m_i = (effm_cte[i] + effm_cte[i + 1]) / 2.0
        k_i = sm.sqrt(2.0 * m_i * (inputE - pot[i])) / HBAR
        a_i = 2.0 * (1.0 - (5.0 * (dx**2) * (k_i**2) / 12.0))
        # Calculating b[i-1] = b_i_ant
        m_i_ant = (effm_cte[i - 1] + effm_cte[i]) / 2.0
        k_i_ant = sm.sqrt(2.0 * m_i_ant * (inputE - pot[i - 1])) / HBAR
        b_i_ant = 1.0 + ((dx**2) * (k_i_ant**2) / 12.0)
        # Calculating c[i+i] = c_i_pos
        m_i_pos = (effm_cte[i + 1] + effm_cte[i + 2]) / 2.0
        k_i_pos = sm.sqrt(2.0 * m_i_pos * (inputE - pot[i + 1])) / HBAR
        c_i_pos = 1.0 + ((dx**2) * (k_i_pos**2) / 12.0)
        # Calculating psi_L
        psi[i + 1] = (a_i * psi[i]) - (b_i_ant * psi[i - 1]) / c_i_pos

    # Probability density from the Wavefunction
    conj = np.conjugate(psi)
    vec_probability = np.real(psi * np.conjugate(psi))

    norm = np.trapz(vec_probability, x)
    vec_probability = vec_probability / norm
    # Definition of the amplitude (100 meV) and energy
    vec_probability = 0.1 * vec_probability / max(vec_probability) + inputE

    # Normalization of the Wavefunction
    psi = psi / sm.sqrt(norm)

    Resultado = np.array([x, np.real(psi), np.imag(psi), np.real(vec_probability)])

    print(f"Total time: {time.time() - numerov_t_start:.3f} s")

    return Resultado


def NumerovSplit(inputE, x, effm_cte, pot, odd, split):
    """
    Numerov method for solving the Schrödinger equation.
    Splits the structure in two parts, following Pedro's algorythmn verbatim.
    """
    numerov_t_start = time.time()

    # Both psi's are 0 at the edges and 0.1 on the adjacent value
    # psi_L's indexes go from [0] -> [L_index]
    # psi_R's indexes go from [len(x)] -> [R_index]
    psi_L = np.zeros(split, dtype=complex)
    psi_R = np.zeros(len(x) - split, dtype=complex)
    psi_L[1] = 0.1
    psi_R[-2] = 0.1

    dx = x[1] - x[0]

    for i in range(1, split - 1):
        # Calculating a[i] = a_i
        m_i = (effm_cte[i] + effm_cte[i + 1]) / 2
        k_i = sm.sqrt(2 * m_i * (inputE - pot[i])) / HBAR
        a_i = 2 * (1 - (5 * (dx**2) * (k_i**2) / 12))
        # Calculating b[i-1] = b_i_ant
        m_i_ant = (effm_cte[i - 1] + effm_cte[i]) / 2
        k_i_ant = sm.sqrt(2 * m_i_ant * (inputE - pot[i - 1])) / HBAR
        b_i_ant = 1 + ((dx**2) * (k_i_ant**2) / 12)
        # Calculating c[i+i] = c_i_pos
        m_i_pos = (effm_cte[i + 1] + effm_cte[i + 2]) / 2
        k_i_pos = sm.sqrt(2 * m_i_pos * (inputE - pot[i + 1])) / HBAR
        c_i_pos = 1 + ((dx**2) * (k_i_pos**2) / 12)
        # Calculating psi_L
        psi_L[i + 1] = (a_i * psi_L[i]) - (b_i_ant * psi_L[i - 1]) / c_i_pos

    for i in range(1, len(x) - split - 1):
        # Using k to reverse the indexing on psi_R (full length = R_len)
        k = len(x) - split - 1 - i
        # Using j to reverse indexing on effm_cte and pot (full length = L_len + R_len)
        j = k + split - 1
        # Calculating a[i] = a_i
        m_i = (effm_cte[j] + effm_cte[j - i]) / 2
        k_i = sm.sqrt(2 * m_i * (inputE - pot[j])) / HBAR
        a_i = 1 - ((5 * (dx**2) * (k_i**2)) / 12)
        # Calculating b[i-1] = b_i_ant
        m_i_ant = (effm_cte[j - 1] + effm_cte[j - 2]) / 2
        k_i_ant = sm.sqrt(2 * m_i_ant * (inputE - pot[j - 1])) / HBAR
        b_i_ant = 1 + ((dx**2) * (k_i_ant**2) / 12)
        # Calculating c[i+1] = c_i_pos
        m_i_pos = (effm_cte[j + 1] + effm_cte[j]) / 2
        k_i_pos = sm.sqrt(2 * m_i_pos * (inputE - pot[j + 1])) / HBAR
        c_i_pos = 1 + ((dx**2) * (k_i_pos**2) / 12)
        # Calculating psi_R
        psi_R[k - 1] = ((a_i * psi_R[k]) - (c_i_pos * psi_R[k + i])) / b_i_ant

    # If the eigenstate is odd, invert the right side
    if odd:
        psi_R = -psi_R

    # Normalization
    psi_L = psi_L * psi_R[0] / psi_L[-1]
    # joining both sides
    psi_total = np.concatenate((psi_L, psi_R))

    # Probability density from the Wavefunction
    vec_probability = np.real(psi_total * np.conjugate(psi_total))

    norm = np.trapz(vec_probability, x)
    vec_probability = vec_probability / norm
    # Definition of the amplitude (100 meV) and energy
    vec_probability = 0.1 * vec_probability / max(vec_probability) + inputE

    # Normalization of the Wavefunction
    psi_total = psi_total / sm.sqrt(norm)

    Resultado = np.array(
        [x, np.real(psi_total), np.imag(psi_total), np.real(vec_probability)]
    )

    print(f"Total time: {time.time() - numerov_t_start:.3f} s")

    return Resultado


def NumerovInArrays(inputE, x, effm_cte, pot, odd, split):
    """
    Numerov method for solving the Schrödinger equation.
    Splits the structure in two parts, mirrors the right side. Calculates both as if they were left
    sides (indexing starting from 0). Appends the result from the original left side to the mirrored
    result of the right side. This strategy avoids having to develop two indexing methods depending
    on the direction of the calculation.
    """
    numerov_t_start = time.time()

    # Calculating the left side
    # Both psi's are 0 at i=0 and 0.1 at i=1
    # psi_R's indexes [0] -> [R_dim-1] correspond to psi [li] -> [n] as I calculate the psi_R in the
    # same way I calculate psi_L, but as if the structure was flipped.
    psi_L = np.zeros(split, dtype=complex)
    psi_R = np.zeros(len(x) - split, dtype=complex)
    psi_L[1] = 0.1
    psi_R[1] = 0.1

    dx = x[1] - x[0]

    # Effective mass considering the average between two layers. Left and right side are
    # calculated going in opposite directions. The roll, average and trim avoids a for loop.
    effm_roll_L = np.roll(effm_cte, -1)  # Rolling
    effm_avg_L = (effm_cte + effm_roll_L) / 2  # Averaging
    # Trimming to keep only the values of the left side, but adding one for i+1 in a, k and
    # bc
    effm_avg_L = effm_avg_L[0 : split + 1]
    # Right side
    effm_roll_R = np.roll(effm_cte, 1)  # Rolling
    effm_avg_R = (effm_cte + effm_roll_R) / 2  # Averaging
    effm_avg_R = effm_avg_R[split - 1 : :]  # Trimming (with 1 plus)
    effm_avg_R = effm_avg_R[::-1]  # Flipping

    # Sppliting the electronic potential (V) into two arrays
    v_L = pot[0 : split + 1]
    v_R = pot[split - 1 : :]
    v_R = v_R[::-1]  # Flipping

    # K depends on m_i, E and V_i
    k_L = sm.sqrt(2.0 * effm_avg_L * (inputE - v_L)) / HBAR
    k_R = sm.sqrt(2.0 * effm_avg_R * (inputE - v_R)) / HBAR

    # A depends on K and dx
    a_L = 1 - (5 * (dx**2.0) * (k_L**2.0) / 12)
    a_R = 1 - (5 * (dx**2.0) * (k_R**2.0) / 12)

    # B and C are the same, only the index changes
    bc_L = 1 + ((dx**2.0) * (k_L**2.0) / 12)
    bc_R = 1 + ((dx**2.0) * (k_R**2.0) / 12)

    for i in range(1, split - 1):
        psi_L[i + 1] = ((a_L[i] * psi_L[i]) - (bc_L[i - 1] * psi_L[i - 1])) / bc_L[
            i + 1
        ]

    for i in range(1, len(x) - split - 1):
        psi_R[i + 1] = ((a_R[i] * psi_R[i]) - (bc_R[i - 1] * psi_R[i - 1])) / bc_R[
            i + 1
        ]
    # If the eigenstate is odd, invert the right side
    if odd:
        psi_R = -psi_R

    # Normalization
    psi_L = psi_L * psi_R[-1] / psi_L[-1]
    # flipping back the right psi
    psi_R = psi_R[::-1]
    # joining both sides
    psi_total = np.concatenate((psi_L, psi_R))

    # Probability density from the Wavefunction
    vec_probability = np.real(psi_total * np.conjugate(psi_total))

    norm = np.trapz(vec_probability, x)
    vec_probability = vec_probability / norm
    vec_probability = 0.1 * vec_probability / max(vec_probability) + inputE

    # Normalization of the Wavefunction
    psi_total = psi_total / sm.sqrt(norm)

    Resultado = np.array(
        [x, np.real(psi_total), np.imag(psi_total), np.real(vec_probability)]
    )

    print(f"Total time: {time.time() - numerov_t_start:.3f} s")

    return Resultado


def Funcao_de_Onda(inputE, x, effm_cte, pot, npe):
    """
    Calculo de funcao de onda
    """
    # Timing
    # t_f_o_start = time.time()
    WFunction = np.mat([[-1.0 + 0.0j], [1.0 + 0.0j]])

    # Bound states can be always real - pag 123 intro to QM in one dimension
    VecWFunction = np.zeros_like(x, dtype=complex)
    VecProbability = np.zeros_like(x, dtype=complex)

    Energy = inputE * E_CHARGE

    if pot[0] * E_CHARGE == Energy:  # Avoiding a division by zero error
        print("Ouch, almost divided by zero...")
        # Energy = Energy + 0.000001 * dE * e_charge
        Energy = Energy * (1.0 + 1.0e-8)

    # Creating the transfer matrix as a complex identity (avoids casting type errors)
    # Transfer = np.mat([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])

    # Trying to calculate some things before the loop - Effective mass and wavenumber

    effm = effm_cte * (1 + (Energy - pot * E_CHARGE) / npe)
    wave_k = sm.sqrt(2 * effm * (Energy - pot * E_CHARGE)) / HBAR + 0j
    # These arrays are used to calculate gamma based on the previous step
    effm_previous = np.roll(effm, 1)
    wave_k_previous = np.roll(wave_k, 1)

    gamma = ((wave_k_previous * effm) / (effm_previous * wave_k)) + 0j

    # Creating the D matrices in one step (instead of iteratively)
    D_mats = np.empty((len(gamma), 2, 2), dtype=complex)
    D_mats[:, 0, 0] = (1 + gamma + 0j) * 0.5
    D_mats[:, 1, 1] = D_mats[:, 0, 0]
    D_mats[:, 1, 0] = (1 - gamma + 0j) * 0.5
    D_mats[:, 0, 1] = D_mats[:, 1, 0]

    # Array with the values of delta x, except that it's zero on the first step.
    deltax_array = x - np.roll(x, 1)
    deltax_array[0] = 0

    # Creating the Pinitial and Pfinal matrices in one step (instead of iteratively)
    # Initial depends on the previous wavenumbers and final depends on the current
    # IEEE Journal of Quantum electronics, vol. 45, no 9 september 2009
    Pi_mats = np.empty((len(gamma), 2, 2), dtype=complex)
    Pi_mats[:, 0, 0] = np.exp(1j * wave_k_previous * deltax_array / 2)
    Pi_mats[:, 1, 0] = 0
    Pi_mats[:, 0, 1] = 0
    Pi_mats[:, 1, 1] = np.exp(-1j * wave_k_previous * deltax_array / 2)

    Pf_mats = np.empty((len(gamma), 2, 2), dtype=complex)
    Pf_mats[:, 0, 0] = np.exp(1j * wave_k * deltax_array / 2)
    Pf_mats[:, 1, 0] = 0
    Pf_mats[:, 0, 1] = 0
    Pf_mats[:, 1, 1] = np.exp(-1j * wave_k * deltax_array / 2)

    for i in range(len(x)):
        # comecando o calculo de funcao de onda
        # IEEE Journal of Quantum electronics, vol. 45, no 9 september 2009
        WFunction = ((Pf_mats[i] @ D_mats[i]) @ Pi_mats[i]) @ WFunction
        # esta matriz esta invertida com o exemplo do artigo acima. Eu faco a onda incidir
        # pela direita e sair pela esquerda. O contrario do artigo
        VecWFunction[i] = WFunction[0, 0] + WFunction[1, 0]
        teste = 2

    # Timing
    # t_f_o_1 = time.time()
    # print(f"Time after transfer_function: {t_f_o_1 - t_f_o_start:.3f} s")
    VecProbability = np.real(VecWFunction * np.conjugate(VecWFunction))
    # VecEstrutura = np.asarray(DesignerMainWindow.VecEstrutura)
    # dx =  float(self.dx_text.text()) * 1E-9  # entrando o valor em nm
    # norm = np.trapz(VecProbability, X_Pot_Meff_ENP.T[0]) # todo o vetor
    norm = np.trapz(VecProbability, x)  # todo o vetor integrado em x
    VecProbability = VecProbability / norm
    # Densidade de probab. fica nivelado no valor de energia dela
    VecProbability = 0.1 * VecProbability / max(VecProbability) + inputE
    VecWFunction = VecWFunction / sm.sqrt(norm)

    Resultado = np.array(
        [x, np.real(VecWFunction), np.imag(VecWFunction), np.real(VecProbability)]
    )
    # retornando a posicao X, parte real, parte imaginaria da funcao de onda e a densidade
    # de probabilidade
    return Resultado


def Funcao_de_Onda_split(inputE, x, effm_cte, pot, npe, odd, split):
    """
    Wavefunction calculation. In order to avoid the exponential explosion that happens when
    x -> infinity (large barriers at the end of the structure), the calculation is split
    into two parts, one beginning from the left (xmin -> x+) and one from the right
    (xmax -> x-). The parts meet and should be equal at the interface of a well.
    Before performing the split+join, calculating everything from both directions to test.
    """
    WFunction_L = np.mat([[-1.0 + 0.0j], [1.0 + 0.0j]])
    WFunction_R = np.mat([[-1.0 + 0.0j], [1.0 + 0.0j]])

    # Bound states can be always real - pag 123 intro to QM in one dimension
    # With reduced dimensions, to merge the calculations
    vec_wfunction_L = np.zeros(split, dtype=complex)
    vec_wfunction_R = np.zeros(len(x) - split, dtype=complex)

    Energy = inputE * E_CHARGE
    if pot[0] * E_CHARGE == Energy:  # Avoiding a division by zero error
        print("Ouch, almost divided by zero...")
        # Energy = Energy + 0.000001 * dE * e_charge
        Energy = Energy * (1.0 + 1.0e-8)

    # Splitting and mirroring the relevant arrays, to avoid having to index everything
    effm_cte_L = effm_cte[0:split]
    effm_cte_R = effm_cte[split::][::-1]  # gets the remaing array and mirrors
    pot_L = pot[0:split]
    pot_R = pot[split::][::-1]
    npe_L = npe[0:split]
    npe_R = npe[split::][::-1]
    x_L = x[0:split]
    x_R = x[split::][::-1]

    # Trying to calculate some things before the loop - Effective mass and wavenumber
    effm_L = effm_cte_L * (1.0 + (Energy - pot_L * E_CHARGE) / npe_L)
    wave_k_L = sm.sqrt(2.0 * effm_L * (Energy - pot_L * E_CHARGE)) / HBAR + 0.0j
    # These arrays are used to calculate gamma based on the previous step
    effm_previous_L = np.roll(effm_L, 1)
    wave_k_previous_L = np.roll(wave_k_L, 1)
    # The first and last items of the previous arrays need to be manually set
    effm_previous_L[0] = effm_L[0]
    effm_previous_L[-1] = effm_L[-1]
    wave_k_previous_L[0] = wave_k_L[0]
    wave_k_previous_L[-1] = wave_k_L[-1]

    # Right side
    effm_R = effm_cte_R * (1.0 + (Energy - pot_R * E_CHARGE) / npe_R)
    wave_k_R = sm.sqrt(2.0 * effm_R * (Energy - pot_R * E_CHARGE)) / HBAR + 0.0j
    # Rolling these arrays to get the previous value is in the same direction, because they
    # are already reversed
    effm_previous_R = np.roll(effm_R, 1)
    wave_k_previous_R = np.roll(wave_k_R, 1)
    # The first and last items of the previous arrays need to be manually set
    effm_previous_R[0] = effm_R[0]
    effm_previous_R[-1] = effm_R[-1]
    wave_k_previous_R[0] = wave_k_R[0]
    wave_k_previous_R[-1] = wave_k_R[-1]

    # All the arrays have the full dimension, even though they will not be used in their
    # entireties. The for loop won't access every item.
    gamma_L = ((wave_k_previous_L * effm_L) / (effm_previous_L * wave_k_L)) + 0.0j
    gamma_R = ((wave_k_previous_R * effm_R) / (effm_previous_R * wave_k_R)) + 0.0j

    # Creating the D matrices in one step (instead of iteratively)
    D_mats_L = np.empty((len(gamma_L), 2, 2), dtype=complex)
    D_mats_L[:, 0, 0] = (1.0 + gamma_L + 0.0j) * 0.5
    D_mats_L[:, 1, 1] = D_mats_L[:, 0, 0]
    D_mats_L[:, 1, 0] = (1.0 - gamma_L + 0.0j) * 0.5
    D_mats_L[:, 0, 1] = D_mats_L[:, 1, 0]

    D_mats_R = np.empty((len(gamma_R), 2, 2), dtype=complex)
    D_mats_R[:, 0, 0] = (1.0 + gamma_R + 0.0j) * 0.5
    D_mats_R[:, 1, 1] = D_mats_R[:, 0, 0]
    D_mats_R[:, 1, 0] = (1.0 - gamma_R + 0.0j) * 0.5
    D_mats_R[:, 0, 1] = D_mats_R[:, 1, 0]

    # Array with the values of delta x, except that it's zero on the first step.
    deltax_array_L = x_L - np.roll(x_L, 1)
    deltax_array_L[0] = 0.0
    deltax_array_R = x_R - np.roll(x_R, 1)
    deltax_array_R[0] = 0.0

    # Creating the Pinitial and Pfinal matrices in one step (instead of iteratively)
    # Initial depends on the previous wavenumbers and final depends on the current
    # IEEE Journal of Quantum electronics, vol. 45, no 9 september 2009
    Pi_mats_L = np.empty((len(gamma_L), 2, 2), dtype=complex)
    Pi_mats_L[:, 0, 0] = np.exp(1.0j * wave_k_previous_L * deltax_array_L / 2.0)
    Pi_mats_L[:, 1, 0] = 0
    Pi_mats_L[:, 0, 1] = 0
    Pi_mats_L[:, 1, 1] = np.exp(-1.0j * wave_k_previous_L * deltax_array_L / 2.0)

    Pi_mats_R = np.empty((len(gamma_R), 2, 2), dtype=complex)
    Pi_mats_R[:, 0, 0] = np.exp(1.0j * wave_k_previous_R * deltax_array_R / 2.0)
    Pi_mats_R[:, 1, 0] = 0
    Pi_mats_R[:, 0, 1] = 0
    Pi_mats_R[:, 1, 1] = np.exp(-1.0j * wave_k_previous_R * deltax_array_R / 2.0)

    Pf_mats_L = np.empty((len(gamma_L), 2, 2), dtype=complex)
    Pf_mats_L[:, 0, 0] = np.exp(1.0j * wave_k_L * deltax_array_L / 2.0)
    Pf_mats_L[:, 1, 0] = 0
    Pf_mats_L[:, 0, 1] = 0
    Pf_mats_L[:, 1, 1] = np.exp(-1.0j * wave_k_L * deltax_array_L / 2.0)

    Pf_mats_R = np.empty((len(gamma_R), 2, 2), dtype=complex)
    Pf_mats_R[:, 0, 0] = np.exp(1.0j * wave_k_R * deltax_array_R / 2.0)
    Pf_mats_R[:, 1, 0] = 0
    Pf_mats_R[:, 0, 1] = 0
    Pf_mats_R[:, 1, 1] = np.exp(-1.0j * wave_k_R * deltax_array_R / 2.0)

    # Wavefunction calculation
    # IEEE Journal of Quantum electronics, vol. 45, no 9 september 2009
    for i in range(split):  # Left side
        WFunction_L = ((Pf_mats_L[i] @ D_mats_L[i]) @ Pi_mats_L[i]) @ WFunction_L
        vec_wfunction_L[i] = WFunction_L[0, 0] + WFunction_L[1, 0]

    for i in range(len(x) - split):  # Right side
        WFunction_R = ((Pf_mats_R[i] @ D_mats_R[i]) @ Pi_mats_R[i]) @ WFunction_R
        vec_wfunction_R[i] = WFunction_R[0, 0] + WFunction_R[1, 0]

    # Correcting the direction of the right side WFunction
    vec_wfunction_R = -vec_wfunction_R[::-1]

    # Correcting the amplitude of the left wavefunction based on the first point of the right
    vec_wfunction_L = vec_wfunction_L * (vec_wfunction_R[0] / vec_wfunction_L[-1])

    if odd:  # If the current wavefunction corresponds to an odd state, multiply by -1
        vec_wfunction = vec_wfunction_R * -1

    # Concatenating left and right sides
    vec_wfunction = np.concatenate((vec_wfunction_L, vec_wfunction_R))

    # Probability density from the Wavefunction
    vec_probability = np.real(vec_wfunction * np.conjugate(vec_wfunction))

    # Integrating using the trapezoidal method to obtain a normalization factor.
    norm = np.trapz(vec_probability, x)
    # Normalizing
    vec_probability = vec_probability / norm
    # Adjusting the amplitude and setting the minimum at the wavefunction's energy
    amplitude = 0.1  # eV
    vec_probability = amplitude * vec_probability / max(vec_probability) + inputE

    # Normalization of the Wavefunction
    vec_wfunction = vec_wfunction / sm.sqrt(norm)

    # Returns the x, real and imaginary parts of the wavefunction, and real part of the
    # probability density function
    result = np.array(
        [x, np.real(vec_wfunction), np.imag(vec_wfunction), np.real(vec_probability)]
    )
    return result


def Absorption(Energias, ResultadoWF, wf_0_index, E0, Ef, dE, broadening):
    """
    Calculates the structure's absorption spectrum.
    """
    Energia = []
    wave_functions = []
    DeltaE = []
    OscStrength = []
    Dipole = []
    energy_axis = np.arange(E0, Ef + dE, dE)

    # Removing the elements that are equal to zero. They appear in the solution due to
    # the positive second derivative of the transmission
    for i, e in enumerate(Energias):
        if e != 0:
            Energia.append(e)
            wave_functions.append(ResultadoWF[i])

    wf_0 = wave_functions[wf_0_index]
    E_wf_0 = Energia[wf_0_index]

    for i, wf_n in enumerate(wave_functions):
        if i != wf_0_index:
            # Calculation of the dipole moment for wave functions other than wf_i
            integ = (wf_0[1] - 1j * wf_0[2]) * wf_0[0] * (wf_n[1] + 1j * wf_n[2])
            DipoleMoment = np.abs(np.trapz(integ, wf_0[0]))
            Dipole += [DipoleMoment]
            DeltaE_float = (Energia[i] - E_wf_0) * E_CHARGE
            DeltaE += [Energia[i] - E_wf_0]  # in eV
            OscStrength += [
                2 * E_MASS * DeltaE_float * abs(DipoleMoment) ** 2 / (HBAR * HBAR)
            ]

    DeltaE = np.asarray(DeltaE)
    OscStrength = np.asarray(OscStrength)

    absorption = 0.0  # Initializing the absorption

    for i, osc_s in enumerate(OscStrength):
        # Eq. of the Lorentzian with a FWHM = broadening and amplitude = osc_strength
        absorption = absorption + (
            osc_s * 100 / (4 * ((energy_axis - DeltaE[i]) ** 2 + (broadening / 2) ** 2))
        )

    try:  # If possible, normalize the absorption curve
        abs_norm = absorption / np.max(absorption)
    # If the Lorentzian is only zeros, np.max raises an exception, avoids dividing by 0
    except:
        abs_norm = absorption

    return (energy_axis, abs_norm, DeltaE, Dipole, OscStrength)


def TransferMatrixPedro(inputE, x, effm_cte, pot, npe, m_eff_ct_barrier):
    """
    This is an (almost) direct copy of Pedro's routine, developed in fortran. It is
    slower due to many calculations being performed for each i, instead of using batch
    array operations.
    """
    Ener = inputE * E_CHARGE  # Going from meV to SI units
    # Going from eV to SI units (multiplying the potential by the electron charge)
    pot = pot * E_CHARGE

    # Effective mass, taking into account the non-parabolicity
    m = effm_cte * (1.0 + (Ener - pot) / npe)

    # Creating the transfer matrix as a complex identity (avoids casting type errors)
    # The first step should always be a discontinuity like an unity matrix
    T0 = np.mat([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])
    # Preallocating matrices that will be used in the loop
    P_inicial = np.empty((2, 2), dtype=complex)
    P_final = np.empty((2, 2), dtype=complex)
    Dn = np.empty((2, 2), dtype=complex)

    # k is calculated for the whole array in one step
    k = sm.sqrt(2.0 * m * (Ener - pot)) / HBAR + 0.0j

    # dx
    dx = x[1] - x[0]

    for i in range(len(x) - 1):

        # Matriz propagacao para i=1
        P_inicial[0, 0] = np.exp(-1.0j * k[i] * dx / 2)
        P_inicial[0, 1] = 0
        P_inicial[1, 0] = 0
        P_inicial[1, 1] = np.exp(1.0j * k[i] * dx / 2)

        P_final[0, 0] = np.exp(-1.0j * k[i + 1] * dx / 2)
        P_final[0, 1] = 0
        P_final[1, 0] = 0
        P_final[1, 1] = np.exp(1.0j * k[i + 1] * dx / 2)

        # Matriz descontinuidade para i=1
        beta = (k[i] * m[i + 1]) / (k[i + 1] * m[i])
        Dn[0, 0] = 0.5 * (1 + beta)
        Dn[0, 1] = 0.5 * (1 - beta)
        Dn[1, 0] = 0.5 * (1 - beta)
        Dn[1, 1] = 0.5 * (1 + beta)

        # AUX = P_inicial * Dn
        aux = P_inicial @ Dn
        # Ti = AUX * P_final
        Ti = aux @ P_final
        # T = T0 * Ti
        T = T0 @ Ti
        # Conta total:
        # T = T0 * ((P_inicial * Dn) * P_final)

        T0 = T

    # In order to calculate the transmission, it's necessary to calculate the k_ratio
    effm_bound = m_eff_ct_barrier
    k_boundary = sm.sqrt(2.0 * effm_bound * (Ener - BOUNDARY_POT)) / HBAR + 0.0j
    k_ratio = k_boundary / k[-1]

    return T, k_ratio


def TransferMatrix(inputE, x, effm_cte, pot, npe, m_eff_ct_barrier):
    """
    This function calculates the Transfer Matrix of a given structure.

    Inputs:
    inputE: Energy for which the transfer matrix will be calculated.
    x: Array containing the value of the structure's physical dimensions.
    effm_cte: Effective mass for each position.
    pot: Electronic potential for each position (in eV).
    npe: non-parabolicity parameter, for each position.
    m_eff_ct_barrier: Effective mass of the boundary barrier.

    Outputs:
    Transfer: used to calculate the structure's transmission, from Transfer[1, 1],
    doesn't take into account boundary conditions.

    k_ratio is used to calculate the transmission, it's the ratio between the k of the
    first and last layers. If there is no electric field and the structures starts and
    ends with the same material, this should be unity.

    Transfer_Boundary: is used to find the bound states (zero crossings), from
    Transfer_Boundary[0, 0]. It's the same as transfer, but multiplied by the initial
    and final D matrices, which define the boundary condition.

    """
    # Going from eV to SI units (multiplying the potential by the electron charge)
    Energy = inputE * E_CHARGE
    pot = pot * E_CHARGE

    if pot[0] == Energy:  # Avoiding a division by zero error
        print("Ouch, almost divided by zero...")
        # Energy = Energy + 0.000001 * dE * e_charge
        Energy = Energy * (1.0 + 1.0e-8)

    # Creating the transfer matrix as a complex identity (avoids casting type errors)
    # The first step should always be a discontinuity like an unity matrix
    Transfer = np.mat([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])

    # Calculation of the effective mass at the first points
    effm_i = effm_cte[0] * (1.0 + (Energy - pot[0]) / npe[0])
    effm_0 = effm_cte[0] * (1.0 + (Energy - pot[0]) / npe[0])

    # Effective mass used for the boundary conditions
    effm_bound = m_eff_ct_barrier

    # Wavenumber in position zero and boundary condition
    k_zero = sm.sqrt(2.0 * effm_i * (Energy - pot[0])) / HBAR + 0.0j
    k_boundary = sm.sqrt(2.0 * effm_bound * (Energy - BOUNDARY_POT)) / HBAR + 0.0j
    # Using ** 0.5 instead of sm.sqrt gives a lot of warnings
    # k_zero = ((2.0 * effm_i * (Energy - pot[0])) ** 0.5) / HBAR + 0.0j
    # k_boundary = ((2.0 * effm_bound * (Energy - BOUNDARY_POT)) ** 0.5) / HBAR + 0.0j
    # Using sqrt from cmath (imported as csqrt) results in the same as sm.sqrt
    # k_zero = csqrt(2.0 * effm_i * (Energy - pot[0])) / HBAR + 0.0j
    # k_boundary = csqrt(2.0 * effm_bound * (Energy - BOUNDARY_POT)) / HBAR + 0.0j

    # Trying to calculate some arrays before the loop. Calculating entire arrays in one
    # operation is faster than iterating and calculating each step at every iteration
    effm = effm_cte * (1.0 + (Energy - pot) / npe)
    k = sm.sqrt(2.0 * effm * (Energy - pot)) / HBAR + 0.0j

    # These arrays are equal to the "non-previous" ones, but shifted by one unit.
    # They're used to avoid operations like effm[i+1], allowing batch calculation of
    # gamma and the p matrices.
    effm_previous = np.roll(effm, 1)
    k_previous = np.roll(k, 1)

    # This factor may be also called beta, depending on the author
    gamma = ((k_previous * effm) / (effm_previous * k)) + 0.0j

    # Creating the D matrices in one step (instead of iteratively)
    len_arrays = len(x)
    D_mats = np.empty((len_arrays, 2, 2), dtype=complex)
    D_mats[:, 0, 0] = (1.0 + gamma + 0.0j) * 0.5
    D_mats[:, 1, 0] = (1.0 - gamma + 0.0j) * 0.5
    D_mats[:, 0, 1] = D_mats[:, 1, 0]
    D_mats[:, 1, 1] = D_mats[:, 0, 0]

    # Array with the values of delta x, except that it's zero on the first step.
    deltax_array = x - np.roll(x, 1)
    deltax_array[0] = 0.0

    # Creating the Pinitial and Pfinal matrices in one step (instead of iteratively)
    # Initial depends on the previous wavenumbers and final depends on the current
    Pi_mats = np.empty((len_arrays, 2, 2), dtype=complex)
    Pi_mats[:, 0, 0] = np.exp(1.0j * k_previous * deltax_array / 2.0)
    Pi_mats[:, 1, 0] = 0.0
    Pi_mats[:, 0, 1] = 0.0
    Pi_mats[:, 1, 1] = np.exp(-1.0j * k_previous * deltax_array / 2.0)

    Pf_mats = np.empty((len_arrays, 2, 2), dtype=complex)
    Pf_mats[:, 0, 0] = np.exp(1.0j * k * deltax_array / 2.0)
    Pf_mats[:, 1, 0] = 0.0
    Pf_mats[:, 0, 1] = 0.0
    Pf_mats[:, 1, 1] = np.exp(-1.0j * k * deltax_array / 2.0)

    for i in range(len_arrays):
        """
        This loop calculates the transfer matrix by multiplying the P and D matrices of
        each layer. Without an electric field, most of these layers are identical,
        except for the ones at the interfaces between two layers and at the boundaries.
        The parenthesis are necessary, as the calculations were happening in the wrong
        order during testing.
        """
        Transfer = ((Pi_mats[i] @ D_mats[i]) @ Pf_mats[i]) @ Transfer

    # D_i and D_f are matrices used to include boundary conditions. These are used to
    # find the wavefunction's energies (eigenstates) using newton's (or secant) method.

    gamma_i = (k[-1] * effm_bound) / (effm[-1] * k_boundary) + 0.0j

    D_i = (
        np.mat(
            [
                [1.0 + gamma_i + 0.0j, 1.0 - gamma_i + 0.0j],
                [1.0 - gamma_i + 0.0j, 1.0 + gamma_i + 0.0j],
            ]
        )
        * 0.5
    )

    gamma_f = (effm_0 * k_boundary) / (k_zero * effm_bound) + 0.0j

    D_f = (
        np.mat(
            [
                [1.0 + gamma_f + 0.0j, 1.0 - gamma_f + 0.0j],
                [1.0 - gamma_f + 0.0j, 1.0 + gamma_f + 0.0j],
            ]
        )
        * 0.5
    )

    Transfer_Boundary = (D_i @ Transfer) @ D_f

    # In order to calculate the transmission, it's necessary to calculate the k_ratio
    k_ratio = k[0] / k[-1]

    return Transfer, k_ratio, Transfer_Boundary


def Photocurrent(energies, x, pot, effm_cte, npe, psi, E0):
    """
    Photocurrent calculation, based on theory and routine developed by Pedro.
    Computing the photovoltaic photocurrent spectrum using the coherent carrier
    propagation in the continuum - doi: https://doi.org/10.1103/PhysRevB.60.R13993
    This version attempts to calculate from right to left, as defined by Pedro.
    """
    # Time and profiling
    t_start = time.time()

    len_arrays = len(x)
    # TODO Evaluate whether obtaining dx from this calculation increases rounding errors
    dx = x[1] - x[0]

    # Arrays used for current calculation in each direction
    J_left = np.zeros(len(energies), dtype=np.complex128)
    J_right = np.zeros(len(energies), dtype=np.complex128)

    # Amplitude of the external electric field [J]/[M]
    Fd = 1 * 1e5 * E_CHARGE

    # Converting the energies from eV to SI units
    energies = energies * E_CHARGE
    pot = pot * E_CHARGE
    E0 = E0 * E_CHARGE

    # Normalizing the wavefunction
    psi = psi[1] + 1.0j * psi[2]

    for i, en in enumerate(energies):
        # Preallocating arrays - Same length as x, one value for each position
        effm = np.zeros(len_arrays, dtype=np.float64)
        k = np.zeros(len_arrays, dtype=np.complex128)

        # In order to calculate the effective mass, the energy of the wavefunction 0 
        # must be added to the calculation
        # Effective mass calculation - array with the mass at each x point.
        effm = effm_cte * (1.0 + (E0 + en - pot) / npe)

        # K array
        k = sm.sqrt(2.0 * effm * (E0 + en - pot)) / HBAR

        # Arrays used for calculating gamma. Rolling +1 is equivalent to shifting the
        # entire array right and putting the last element on the first position. This
        # trick is used to perform calculations using the x[i] and x[i-1], without the
        # "-1". In this case, the first item in "_previous" arrays should not be used,
        # since it's actually the last one.
        # "_nxt" arrays are rolled the other way: psi_nxt[i] = psi[i+1]
        effm_nxt = np.roll(effm, -1)
        k_nxt = np.roll(k, -1)
        x_nxt = np.roll(x, -1)
        psi_nxt = np.roll(psi, -1)

        # Creating the Pinitial and Pfinal matrices in one step (instead of iteratively)
        # Initial depends on the previous wavenumbers and final depends on the current
        Pi_mats = np.empty((len_arrays, 2, 2), dtype=np.complex128)
        Pi_mats[:, 0, 0] = np.exp(-1.0j * k_nxt * x)
        Pi_mats[:, 1, 0] = 0.0
        Pi_mats[:, 0, 1] = 0.0
        Pi_mats[:, 1, 1] = np.exp(+1.0j * k_nxt * x)

        Pf_mats = np.empty((len_arrays, 2, 2), dtype=np.complex128)
        Pf_mats[:, 0, 0] = np.exp(1.0j * k * x)
        Pf_mats[:, 1, 0] = 0.0
        Pf_mats[:, 0, 1] = 0.0
        Pf_mats[:, 1, 1] = np.exp(-1.0j * k * x)

        # This factor may be also called beta, depending on the author
        beta = ((k_nxt * effm) / (effm_nxt * k)) + 0.0j

        # A constant used in the photocurrent calculation
        cte = (effm * Fd) / (2.0j * HBAR * HBAR * k) * dx
        # Arrays to calculate the electron-foton interaction on the interface
        beta_plus = -cte * (
            np.exp(1.0j * k * x) * x * psi - np.exp(1.0j * k * x_nxt) * x_nxt * psi_nxt
        )
        beta_minus = -cte * (
            np.exp(-1.0j * k * x) * x * psi
            - np.exp(-1.0j * k * x_nxt) * x_nxt * psi_nxt
        )
        f = np.mat([[0], [0]], dtype=np.complex128)

        # Creating the D matrix using beta
        D_mats = np.empty((len_arrays, 2, 2), dtype=np.complex128)
        D_mats[:, 0, 0] = (1.0 + beta) * 0.5
        D_mats[:, 1, 0] = (1.0 - beta) * 0.5
        D_mats[:, 0, 1] = D_mats[:, 1, 0]  # This works because the values are copied,
        D_mats[:, 1, 1] = D_mats[:, 0, 0]  # not only the reference to the value.

        # Creating the transfer matrix as identity, complex128 type
        # The first step should always be a discontinuity like an unity matrix
        Transfer = np.mat([[1, 0], [0, 1]], dtype=np.complex128)

        for j in range(len_arrays - 1):  # the first step is skipped, so -1
            """
            This loop calculates the transfer matrix by multiplying the P and D matrices
            of each layer. Without an electric field, most of these layers are
            identical, except for the ones at the interfaces between two layers and at
            the boundaries. The parenthesis are necessary to guarantee the matrix 
            multiplication occurs in the correct order.
            Some index math is required, because the calculation is performed from right
            to left, that is, max index (len_arrays) to 1, in -1 steps
            """
            ji = len_arrays - 2 - j
            Transfer = Transfer @ ((Pi_mats[ji] @ D_mats[ji]) @ Pf_mats[ji])
            f[0] = f[0] + (
                Transfer[0, 0] * beta_minus[ji] + Transfer[0, 1] * beta_plus[ji]
            )
            f[1] = f[1] + (
                Transfer[1, 0] * beta_minus[ji] + Transfer[1, 1] * beta_plus[ji]
            )

        ctecur_left = (HBAR * k[0]) / effm[0]
        ctecur_right = (HBAR * k[-1]) / effm[-1]

        J_left[i] = (
            ctecur_left
            * Fd
            * (-f[1] / Transfer[1, 1])
            * (np.conj(-f[1] / Transfer[1, 1]))
        )
        J_right[i] = (
            ctecur_right
            * Fd
            * (f[0] - f[1] * Transfer[0, 1] / Transfer[1, 1])
            * np.conj(f[0] - f[1] * Transfer[0, 1] / Transfer[1, 1])
        )

    # Time and profiling
    print(f"Total time: {time.time() - t_start:.3f} s")
    
    return np.real(J_right - J_left)  # /6.966529605e-23 #    Photocurrent
