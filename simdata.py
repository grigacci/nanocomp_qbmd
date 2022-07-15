import numpy as np
import simcore, os
from scipy.optimize import newton
import matplotlib.pyplot as plt

NM = 1.0E-9

class SimData:
    """
    This class stores the data regarding one structure
    """

    def __init__(
        self,
        title,
        array_data,
        material_data={
            "latpar": 5.8687e-10,
            "barrier": "InAlAs",
            "m_eff_ct_barrier": 7.61544325168e-32,
            "e_nonparab_barrier": 2.065014390417958e-19,
            "pot_barrier": 0.503,
            "well": "InGaAs",
            "m_eff_ct_well": 3.97169049968e-32,
            "pot_well": 0.0,
            "e_nonparab_well": 1.0769692275385522e-19,
        },
    ):
        """
        Creates a new simulation.
        """
        self.title = title
        # Arrays used in the calculation
        self.array_data = array_data
        self.estrutura = array_data["estrutura"]
        self.material = []
        self.feature = []
        self.massa_eff_const = array_data["massa_eff_const"]
        self.pot = array_data["pot"]
        self.E_nonparab = array_data["E_nonparab"]
        self.e_field = 0.0  # in KV/cm
        # self.x0 = 0.0  # Set to position 0
        # Defining material properties
        self.material_data = material_data
        self.latpar = material_data["latpar"]
        self.barrier = material_data["barrier"]
        self.m_eff_ct_barrier = material_data["m_eff_ct_barrier"]
        self.e_nonparab_barrier = material_data["e_nonparab_barrier"]
        self.pot_barrier = material_data["pot_barrier"]
        self.well = material_data["well"]
        self.m_eff_ct_well = material_data["m_eff_ct_well"]
        self.pot_well = material_data["pot_well"]
        self.e_nonparab_well = material_data["e_nonparab_well"]

        # Since it was just created, the simulation and absorption were not run
        self.sim_ran = False
        self.abs_ran = False
        self.tra_ran = False
        self.pc_ran = False

    def AddWell(self, thickness):
        """
        Adds a well with the desired thickness to the end of the structure.
        """
        self.estrutura = np.append(self.estrutura, thickness * NM)
        self.material.append(self.well)
        self.feature.append("Well")
        self.massa_eff_const = np.append(self.massa_eff_const, self.m_eff_ct_well)
        self.pot = np.append(self.pot, self.pot_well)
        self.E_nonparab = np.append(self.E_nonparab, self.e_nonparab_well)

    def AddBarrier(self, thickness):
        """
        Adds a barrier with the desired thickness to the end of the structure.
        """
        self.estrutura = np.append(self.estrutura, thickness * NM)
        self.material.append(self.barrier)
        self.feature.append("Barrier")
        self.massa_eff_const = np.append(self.massa_eff_const, self.m_eff_ct_barrier)
        self.pot = np.append(self.pot, self.pot_barrier)
        self.E_nonparab = np.append(self.E_nonparab, self.e_nonparab_barrier)

    def ReplaceWell(self, thickness, index):
        """
        Replaces the selected layer by a quantum well with the desired thickness.
        """
        self.estrutura[index] = thickness * NM
        self.material[index] = self.well
        self.feature[index] = "Well"
        self.massa_eff_const[index] = self.m_eff_ct_well
        self.pot[index] = self.pot_well
        self.E_nonparab[index] = self.e_nonparab_well

    def ReplaceBarrier(self, thickness, index):
        """
        Replaces the selected layer by a quantum barrier with the desired thickness.
        """
        self.estrutura[index] = thickness * NM
        self.material[index] = self.barrier
        self.feature[index] = "Barrier"
        self.massa_eff_const[index] = self.m_eff_ct_barrier
        self.pot[index] = self.pot_barrier
        self.E_nonparab[index] = self.e_nonparab_barrier

    def InsertWell(self, thickness, index):
        """
        Inserts a quantum well with the desired thickness, after the selected layer.
        """
        self.estrutura = np.insert(self.estrutura, [index + 1], thickness * NM)
        self.material.insert(index + 1, self.well)
        self.feature.insert(index + 1, "Well")
        self.massa_eff_const = np.insert(
            self.massa_eff_const, [index + 1], self.m_eff_ct_well
        )
        self.pot = np.insert(self.pot, [index + 1], self.pot_well)
        self.E_nonparab = np.insert(self.E_nonparab, [index + 1], self.e_nonparab_well)

    def InsertBarrier(self, thickness, index):
        """
        Inserts a quantum barrier with the desired thickness, after the selected layer.
        """
        self.estrutura = np.insert(self.estrutura, [index + 1], thickness * NM)
        self.material.insert(index + 1, self.barrier)
        self.feature.insert(index + 1, "Barrier")
        self.massa_eff_const = np.insert(
            self.massa_eff_const, [index + 1], self.m_eff_ct_barrier
        )
        self.pot = np.insert(self.pot, [index + 1], self.pot_barrier)
        self.E_nonparab = np.insert(
            self.E_nonparab, [index + 1], self.e_nonparab_barrier
        )

    def RemoveSelected(self, index):
        """
        Removes the selected layer.
        """
        self.estrutura = np.delete(self.estrutura, index)
        self.massa_eff_const = np.delete(self.massa_eff_const, index)
        self.pot = np.delete(self.pot, index)
        self.E_nonparab = np.delete(self.E_nonparab, index)
        self.material.pop(index)
        self.feature.pop(index)

    def RemoveAll(self):
        """
        Clears the structure.
        """
        self.estrutura = np.array([], dtype=np.float64)
        self.massa_eff_const = np.array([], dtype=np.float64)
        self.pot = np.array([], dtype=np.float64)
        self.E_nonparab = np.array([], dtype=np.float64)
        self.material = []
        self.feature = []

    def check_values(self, barrier_higher=True):
        self.results = []
        if barrier_higher:
            for i in range(len(self.OscStrength)):
                if self.Autoenergias[i + 1] > self.pot_barrier:
                    self.results.append(
                        [
                            self.Autoenergias[i + 1] - self.Autoenergias[0],
                            self.OscStrength[i],
                            [self.Autoenergias[i + 1], self.Autoenergias[0]],
                            i + 1,
                        ]
                    )
        else:
            for i in range(len(self.OscStrength)):
                self.results.append(
                    [
                        self.Autoenergias[i + 1] - self.Autoenergias[0],
                        self.OscStrength[i],
                        [self.Autoenergias[i + 1], self.Autoenergias[0]],
                        i + 1,
                    ]
                )

    def SaveStructure_best_osc(self, path_file="test.png", info="x", plot_width=80):
        self.x_graf = np.array([0])
        self.v_graf = np.array([self.pot[0]])

        # Going through the layers and creating the arrays iteratively
        for i, en in enumerate(self.pot):
            if (
                en == self.v_graf[-1]
            ):  # If this layer has the same energy as the previous one
                self.x_graf = np.append(
                    self.x_graf, [self.x_graf[-1] + self.estrutura[i]]
                )
                self.v_graf = np.append(self.v_graf, [en])
            else:
                self.x_graf = np.append(
                    self.x_graf, [self.x_graf[-1], self.x_graf[-1] + self.estrutura[i]]
                )
                self.v_graf = np.append(self.v_graf, [en, en])

        # Changing x and y to [nm] and [meV]
        self.x_graf = 1.0e9 * self.x_graf
        self.v_graf = 1.0e3 * self.v_graf

        # Centering the structure around 0
        self.x_graf = self.x_graf - (np.max(self.x_graf) - np.min(self.x_graf)) / 2.0

        # fig, axs = plt.subplots(2)
        # fig.suptitle('Vertically stacked subplots')
        # # plt.clf()
        # plt.xlabel('Length (nm)')
        # plt.ylabel('Energy (meV)')
        # plt.xlim([0, 80])
        # plt.ylim([-50, 700])
        # axs[0].plot(self.x_graf, self.v_graf) # x in [nm] and y in [meV]

        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
        # fig, (ax1, ax2) = plt.subplots(2,1)
        # fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [1, 2]})

        fig.suptitle(info)

        ax1.plot(self.x_graf, self.v_graf)
        ax1.set(xlabel="Length (nm)", ylabel="Energy (meV)")
        ax1.set(xlim=[-plot_width / 2, plot_width / 2], ylim=[-50, 900])

        # print(len(self.ResultadoWF))
        # print(self.best_index)
        # grafica as funções de onda

        # for num, result in enumerate(self.ResultadoWF, start=0):
        #     if num == 0:
        #         ax1.plot(result[0, :] * 1.0E9, result[3, :] * 1.0E3, color='green')
        #     elif num == self.best_index:
        #         ax1.plot(result[0, :] * 1.0E9, result[3, :] * 1.0E3, color='red')
        #     else:
        #         ax1.plot(result[0, :] * 1.0E9, result[3, :] * 1.0E3, color='#cccccc')

        # ax1.plot(self.ResultadoWF[0][0, :] * 1.0E9, self.ResultadoWF[0][3, :] * 1.0E3, color='green')
        # ax1.plot(self.ResultadoWF[self.best_index][0, :] * 1.0E9, self.ResultadoWF[self.best_index][3, :] * 1.0E3, color='red')
        osc_max = 0
        for res in self.results:
            if res[1] > osc_max:
                osc_max = res[1]

        for num, result in enumerate(self.ResultadoWF, start=1):
            ax1.plot(result[0, :] * 1.0e9, result[3, :] * 1.0e3, color="#cccccc")

        # print("self.ResultadoWF: ", len(self.ResultadoWF))
        # print("self.results: ", len(self.results))

        for jj in range(len(self.results)):
            if self.results[jj][1] / osc_max > 0.8:

                red_index = jj + 1
                ax1.plot(
                    self.ResultadoWF[red_index][0, :] * 1.0e9,
                    self.ResultadoWF[red_index][3, :] * 1.0e3,
                    color="red",
                )

        # ax1.plot(self.ResultadoWF[0][0, :] * 1.0E9, self.ResultadoWF[0][3, :] * 1.0E3, color='green')
        # ax1.plot(self.ResultadoWF[self.best_index][0, :] * 1.0E9, self.ResultadoWF[self.best_index][3, :] * 1.0E3, color='red')

        ax1.plot(
            self.ResultadoWF[0][0, :] * 1.0e9,
            self.ResultadoWF[0][3, :] * 1.0e3,
            color="green",
        )

        x2 = []
        for jj in range(len(self.abs_result)):
            x2.append(jj + 1)

        for res in self.results:
            # plt.plot(x, y, 'o')
            ax2.plot(res[1], res[0] * 1000, ".", color="blue")
            # print(res[1], res[0]*1000)
        # desenha linha na absorção na energia com Osc maior
        # ax2.axvline(x=(self.Autoenergias[self.best_index]-self.Autoenergias[0])*1000)

        # print(self.Autoenergias[self.best_index]-self.Autoenergias[0])

        # for energia in self.results:
        #     print(energia)
        #     if energia[1]>0.9:
        #         # ax2.plot([energia[0]*1000], [1], 'ro')
        #         ax2.axvline(x=energia[0]*1000)

        # plt.yscale('log')
        # ax2.set(xlim=[0, 1000], ylim=[0.000001, 2])

        # plt.yscale('log')
        # plt.xscale('log')

        E0 = self.Autoenergias[0] * 1000
        # print(E0)
        # ax2.set(xlim=[0.00001, 1.2], ylim=[-E0-50, 950-E0-50])
        # ax2.set(xlim=[-0.1, 1.1], ylim=[-E0-50, 950-E0-50])
        ax2.set(ylim=[-E0 - 50, 950 - E0 - 50])
        # ax2.set(ylim=[-E0-50, 950-E0-50])
        # ax2.set(xlim=[0, 1000], ylim=[0, 1])
        # ax2.set(xlabel='Energy (meV)', ylabel='Absorption (u.a.)')
        ax2.set(xlabel="OscStrength (u.a.), max: " + "{:.2f}".format(osc_max))

        folder_path = os.path.dirname(os.path.abspath(path_file))
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(path_file, dpi=150)

    def RunSim(self, method, split_i, E0, Ef, dE, dx, dx_unit, central_layer):
        """
        Main function that performs the simulation
        """
        # Correcting the unit of dx based on the unit chosen from the interface and the 
        # lattice parameter
        if dx_unit == "nm":
            dx = dx * NM  # Conversion from nm to m
        elif dx_unit == "ml":
            dx = dx * self.latpar / 2.0  # Conversion from ml to m
        else:
            dx = 1.0e-11
            print("dx_unit was defined incorrectly; possible options: nm or ml")

        # Creating the energy array
        VecEnergy = np.arange(E0, Ef + dE, dE)
        VecEnLen = len(VecEnergy)

        # Creating the x-axis array - the x-axis 0 is always centered in the target 
        # layer
        layers = len(self.estrutura)
        if central_layer <= 0:
            x0 = 0.0
        else:
            # In case the number of the layer exceeds the limit
            if (
                central_layer > layers
            ):  
                central_layer = layers
            # Thickness of target layers
            x0 = -np.sum(self.estrutura[0:central_layer] )  
        x_first_step = x0 + dx

        # Divides the length of each layer by dx to get the number of steps in each 
        # layer
        # Must round and convert to int so that it can be used as a array length
        steps_per_layer = np.around(np.abs(self.estrutura) / dx, decimals=0).astype(
            "int"
        )

        # Dimension of the arrays used to calculate each step of the structure. It's 
        # size is the number of layers plus the thickness of the layers divided by dx 
        # and rounded.
        length_arrays = layers + np.sum(steps_per_layer)
        # Pre-allocating
        x = np.zeros((length_arrays), dtype=np.float64)
        pot = np.zeros((length_arrays), dtype=np.float64)
        effm_cte = np.zeros((length_arrays), dtype=np.float64)

        # Well Non-Parabolicity from Leavitt
        # Nonparabw = 1.3E-18
        # eq. 7 (PRBvol 35, number 14, pg 7770, Nelson D.F. et all,  
        # doi.org/10.1103/PhysRevB.35.7770)
        npe = np.zeros((length_arrays), dtype=np.float64)

        pos = 0

        # Going through every layer
        for layer in range(layers):
            x[pos] = x_first_step - dx
            pot[pos] = self.pot[layer] - self.e_field * 1e5 * (x_first_step - dx)
            effm_cte[pos] = self.massa_eff_const[layer]
            npe[pos] = self.E_nonparab[layer]

            pos += 1

            # Going through the steps of each layer
            for step in range(steps_per_layer[layer]):
                x[pos] = x_first_step
                pot[pos] = self.pot[layer] - self.e_field * 1e5 * x_first_step
                effm_cte[pos] = self.massa_eff_const[layer]
                npe[pos] = self.E_nonparab[layer]

                # Manually taking into account the dimensions of the steps
                x_first_step += dx
                pos += 1

        # Calculating the Transfer Matrix
        # A transfer matrix (2,2) is calculated for each energy.
        self.TM = np.zeros((VecEnLen, 2, 2), dtype=np.complex128)
        self.k_ratio = np.zeros((VecEnLen), dtype=np.complex128)
        self.TMB = np.zeros((VecEnLen, 2, 2), dtype=np.complex128)

        # TODO Paralelizar o cálculo da matriz de transferência
        for i, energy in enumerate(VecEnergy):
            # Método rapido para identificar os auto estados (shooting)
            # VecResultado[i] = simcore.Shooting(energy, effm_cte, pot, npe, dx)
            # Método da matriz de transferência
            # VecResultado[i] = simcore.T_1_1Re(1000*energy, x, effm_cte, pot, npe,
            #                                   self.m_eff_ct_barrier)
            self.TM[i], self.k_ratio[i], self.TMB[i] = simcore.TransferMatrix(
                energy, x, effm_cte, pot, npe, self.m_eff_ct_barrier
            )
            # TODO ver ser é mais rápido rodar com Numerov
            #  VecResultado[i] = simcore.Numerov(energy, effm_cte, pot, npe, dx)

        T_11 = np.real(self.TM[:, 1, 1])

        # The eigen energies are found when TM_array(1, 1) crosses zero
        # finding eigenenergies from the real part of transmission and calculating
        # wavefunctions
        signs = np.sign((T_11))
        # replace zeros with -1 (if there is 0 on the original array, np.sign returns 0)
        signs[signs == 0] = -1

        # Bound states are found at the zero-crossings
        # Just crossing zero doesn't necessarily means it's a bound state, the second 
        # derivative must also be negative (aka, a peak)
        # Ref.: Elementary Quantum Mechanics in One Dimension Pg. 92
        bound_states_indexes = np.where(np.diff(signs))[0]
        bound_states_number = len(bound_states_indexes)

        # Length of the energy array where the solution of T11 is real.
        bound_states = np.zeros(bound_states_number, dtype=np.float64)

        # The calculated wave functions are represented in (4,len(x)) arrays, one for 
        # each bound state
        wave_functions = np.zeros((bound_states_number, 4, length_arrays))

        # If the chosen method requires a structure split, we need to calculate the 
        # split index.
        # The interface index must be between 1 and (layers-1). If the interface is set 
        # at 0 or if it's equal to the number of layers,  the split method doesn't work.
        # If interface is 0 or "layers", then a non-split method should be used
        interface = split_i
        if interface <= 0:
            interface = 1
        elif interface >= layers:
            interface = layers - 1
        # The index of the split interface enables the calculation of the number of 
        # points to the right and the left. "split" is the length of the left side and 
        # the index of the first item of the right side
        split = interface + np.sum(steps_per_layer[0:interface])

        # Helper function created to run the newton method on Re(T(1,1))
        def TM_Re11(inputE, x, effm_cte, pot, npe, m_eff_ct_barrier):
            _, _, TMB = simcore.TransferMatrix(
                inputE, x, effm_cte, pot, npe, m_eff_ct_barrier
            )
            return np.real(TMB[1, 1])

        for i, crossing in enumerate(bound_states_indexes):
            try:
                x0 = VecEnergy[crossing]
                # TODO Acertar tol e rtol
                bound_states[i] = newton(
                    TM_Re11,
                    VecEnergy[crossing],
                    fprime=None,
                    args=(x, effm_cte, pot, npe, self.m_eff_ct_barrier),
                    tol=1.0e-16,
                    rtol=1.0e-16,
                )

            except RuntimeError:
                print("Ignorando o erro de runtime. Energia = %f" % VecEnergy[crossing])

            # TODO Checar se isso ainda é necessário. Era usado com o método Shooting
            # if WF_EnergyRe[i] < 0:
            #     print(f"Invalid energy, negative: {WF_EnergyRe[i]:.3f} eV")
            # if WF_EnergyRe[i] > Ef:
            #     print(f"Invalid energy, larger than maximum: {WF_EnergyRe[i]:.3f} eV")

            # Needs to invert the right side signal for odd states
            odd = True if i % 2 > 0 else False
            if method == 0:  # Numerov - For
                wave_functions[i] = simcore.Numerov(bound_states[i], x, effm_cte, pot)

            elif method == 1:  # Numerov - Split
                wave_functions[i] = simcore.NumerovSplit(
                    bound_states[i], x, effm_cte, pot, odd, split_i
                )
            elif method == 2:  # Numerov - Arrays
                wave_functions[i] = simcore.NumerovInArrays(
                    bound_states[i], x, effm_cte, pot, odd, split_i
                )
            elif method == 3:  # TMM
                wave_functions[i] = simcore.Funcao_de_Onda(
                    bound_states[i], x, effm_cte, pot, npe
                )

            elif method == 4:  # TMM - Split
                wave_functions[i] = simcore.Funcao_de_Onda_split(
                    bound_states[i], x, effm_cte, pot, npe, odd, split
                )

        # Output arrays are stored
        # TODO Simplify this, use "self" before and change name of the ones in portuguese
        self.sim_x = x
        self.sim_pot = pot
        self.sim_effm_cte = effm_cte
        self.sim_npe = npe
        # self.sim_VecResultado = VecResultado
        # self.sim_VecResultado = T_11
        self.sim_VecEnergy = VecEnergy
        # self.sim_ResultadoWF = self.ResultadoWF = ResultadoWF
        self.sim_ResultadoWF = self.ResultadoWF = wave_functions
        # self.sim_Energias = self.Autoenergias = WF_EnergyRe
        self.sim_Energias = self.Autoenergias = bound_states

        # This works as a signal to allow the absorption to run
        self.sim_ran = True

    def CalcAbs(self, wf_0_index, E0, Ef, dE, broadening):
        abs = simcore.Absorption(
            self.sim_Energias, self.sim_ResultadoWF, wf_0_index, E0, Ef, dE, broadening
        )

        self.abs_energy_axis = abs[0]
        self.abs_result = abs[1]
        self.abs_delta_E = abs[2]
        self.abs_dipole = abs[3]
        self.abs_osc_strength = self.OscStrength = abs[4]

        self.abs_ran = True

    def Transmission(self):
        """
        Calculates the transmission for a given range of energies. The simulation must
        have been run before.
        """
        if not self.sim_ran:
            print("The simulation must be run before calculating the transmission")
            return

        try:
            TM_11 = self.TM[:, 1, 1]
            self.sim_Transmission = np.real(1 / (np.absolute(TM_11))) * self.k_ratio
            self.tra_ran = True
        except:
            print("Could not calculate transmission, run the simulation first")

    def RunPhotocurrent(self, dx, E0, Ef, dE):
        """
        Calculates the photocurrent for a given range of energies. The simulation must
        have been run before.
        """
        if not self.sim_ran:
            print("The simulation must be run before calculating the transmission")
            return
        # print(f"dx no Photocurrent = {dx:.15f}")
        # Creating the energy array
        energies = np.arange(E0, Ef + dE, dE)
        pc = simcore.Photocurrent(
            energies=energies,
            x=self.sim_x,
            pot=self.sim_pot,
            effm_cte=self.sim_effm_cte,
            npe=self.sim_npe,
            psi=self.sim_ResultadoWF[0],
            E0=self.sim_Energias[0],
        )
        self.sim_Photocurrent = energies, pc
