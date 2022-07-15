"""
e-mulate
This is the main script of the e-mulate software.
"""
# Qt5
from code import interact
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QMainWindow,
    QSplashScreen,
    QTableWidgetItem,
    QWidget,
)

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # enable highdpi scaling
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # use highdpi icons

# To deal with files, time, paths...
import configparser
from copy import deepcopy

# from multiprocessing import cpu_count
import numpy as np
import os

# from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import sys
import time
import webbrowser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Toolbar shown on the figures
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

# Changes to rcParams are system-wide, so at the end of the program, the defaults need
# to be reset
from matplotlib import rcParams

# Guarantees that every part of the figure is inside the canvas
rcParams.update({"figure.autolayout": True})

# File containing the main class
import simdata
import interface_config

# Adding the GUI files directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "GUI"))

NM = 1.0e-9


class MainWindow(QMainWindow):
    """
    Main window
    """

    def __init__(self, parent=None):
        """
        Initialization of the main window
        """
        super(MainWindow, self).__init__(parent)
        # Loads the ui
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(
            os.path.join(self.base_path, "GUI", "TM_tabs.ui"),
            self,
        )
        # Sets the window icon
        self.setWindowIcon(
            QIcon(
                os.path.join(
                    self.base_path,
                    "Imagens",
                    "favicon.ico",
                )
            )
        )

        # Reads the configuration file and create the corresponding variables
        self.mat = configparser.ConfigParser()
        self.mat.read(os.path.join(self.base_path, "materials.data"))

        # Creating the list containing all simulations
        self.sim_list = []
        # Stores the number of the current simulation
        self.current_number = 1

        self.ConnectSignals()
        # Calls an auxiliary script that contains interface configuration functions
        interface_config.run(self)
        self.CreatePlots()
        self.InitializeStructTable()
        self.InitializeDataTable()
        self.UpdateInterface()

    # GUI ##############################################################################
    def ConnectSignals(self):
        """
        Connect the signals from buttons to functions.
        Disables any button that should only be enabled after a new simulation is
        created.
        """
        # Lower bar
        # self.new_btn.clicked.connect(self.new_sim_window.show)
        self.new_btn.clicked.connect(self.OpenNewSimWindow)
        self.save_btn.clicked.connect(self.SaveSimulation)
        self.load_btn.clicked.connect(self.LoadSimulation)
        self.delete_btn.clicked.connect(self.DeleteSimulation)
        self.output_folder_btn.clicked.connect(self.ChooseOutputFolder)
        self.simulation_cbox.currentIndexChanged.connect(self.UpdateInterface)
        self.rename_btn.clicked.connect(self.RenameSimulation)
        self.copy_btn.clicked.connect(self.CopySimulation)

        # Design tab
        self.add_well_btn.clicked.connect(lambda: self.UpdateStructure("AddWell"))
        self.add_barrier_btn.clicked.connect(lambda: self.UpdateStructure("AddBarrier"))
        self.replace_well_btn.clicked.connect(
            lambda: self.UpdateStructure("ReplaceWell")
        )
        self.replace_barrier_btn.clicked.connect(
            lambda: self.UpdateStructure("ReplaceBarrier")
        )
        self.insert_well_btn.clicked.connect(lambda: self.UpdateStructure("InsertWell"))
        self.insert_barrier_btn.clicked.connect(
            lambda: self.UpdateStructure("InsertBarrier")
        )
        self.remove_selected_btn.clicked.connect(
            lambda: self.UpdateStructure("RemoveSelected")
        )
        self.remove_last_btn.clicked.connect(lambda: self.UpdateStructure("RemoveLast"))
        self.remove_all_btn.clicked.connect(lambda: self.UpdateStructure("RemoveAll"))
        self.well_ml_spb.valueChanged.connect(self.UpdateUnits)
        self.well_nm_spb.valueChanged.connect(self.UpdateUnits)
        self.barrier_ml_spb.valueChanged.connect(self.UpdateUnits)
        self.barrier_nm_spb.valueChanged.connect(self.UpdateUnits)

        # Simulation tab
        self.sim_Efield_btn.clicked.connect(self.Campo)
        # Electric field button and spinbox are disabled because they don't work
        self.sim_Efield_btn.setEnabled(False)
        self.sim_Efield_spb.setEnabled(False)
        self.sim_run_btn.clicked.connect(lambda: self.RunSimulation())
        self.sim_plot_results_btn.clicked.connect(lambda: self.PlotSimResults())
        self.sim_plot_structure_btn.clicked.connect(lambda: self.PlotStructure())
        self.sim_clear_plot_btn.clicked.connect(self.ClearSimPlot)
        self.sim_dx_ml_spb.valueChanged.connect(self.UpdateUnits)
        self.sim_dx_nm_spb.valueChanged.connect(self.UpdateUnits)
        self.sim_central_layer_spb.valueChanged.connect(lambda: self.PlotStructure())

        # Absorption tab
        self.abs_run_btn.clicked.connect(lambda: self.RunAbsorption())
        self.abs_plot_btn.clicked.connect(lambda: self.PlotAbsorption())
        self.abs_clear_plot_btn.clicked.connect(self.ClearAbsPlot)

        # Transmission tab
        self.tra_run_btn.clicked.connect(lambda: self.RunTransmission())
        self.tra_plot_btn.clicked.connect(lambda: self.PlotTransmission())
        self.tra_clear_plot_btn.clicked.connect(lambda: self.ClearTransPlot())

        # Photocurrent tab
        self.pc_run_btn.clicked.connect(lambda: self.RunPhotocurrent())
        self.pc_clear_plot_btn.clicked.connect(lambda: self.ClearPhotocurrentPlot())

        # GA tab
        # self.ga_btn.clicked.connect(self.Sobre)

        # Automation tab
        self.auto_run_btn.clicked.connect(self.RunAutomation)

        # Advanced options tab
        self.adv_nm_layers_chkbx.stateChanged.connect(self.UpdateUnits)
        self.adv_save_gui_config.clicked.connect(self.SaveGUIConfigFile)

        # Actions from menus
        # File Menu
        # self.action_new.triggered.connect(self.NewSimulation)
        self.action_new.triggered.connect(self.OpenNewSimWindow)
        self.action_load.triggered.connect(self.LoadSimulation)
        self.action_save.triggered.connect(self.SaveSimulation)
        self.action_exit.triggered.connect(exit)
        # Help Menu
        self.action_about.triggered.connect(self.Sobre)

    def UpdateUnits(self):
        """
        This function is used to update the layer thickness spinboxes, making sure that
        the value in nm corresponds an integer multiple of the lattice parameter,
        defined by the monolayer's spinboxes, and vice versa.
        This routine is divided in two parts (by the try-except), things that can be
        determined before creating a simulation and the ones that depend on the lattice
        parameter.
        """
        # If the user prefers to use nanometers instead of monolayers:
        if self.adv_nm_layers_chkbx.checkState():
            # Changes the suffix of the spinboxes
            self.auto_step_spb.setSuffix(" nm")
            self.auto_final_spb.setSuffix(" nm")
            self.auto_init_spb.setSuffix(" nm")
        else:  # If the user is using monolayers
            # Changes the suffix of the spinboxes
            self.auto_step_spb.setSuffix(" ML")
            self.auto_final_spb.setSuffix(" ML")
            self.auto_init_spb.setSuffix(" ML")

        # The lattice parameter is only defined after a simulation was created.
        try:  # Gets the current simulation and the value of the lattice parameter
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
            ml = sim.latpar / 2.0
        except:
            return

        # Identifies which spinbox was modified (the one which called this function)
        op = self.sender()

        # If the user prefers to use nanometers instead of monolayers:
        if self.adv_nm_layers_chkbx.checkState():
            # Disable the ml spinboxes, so that the user cannot interact with them
            self.barrier_ml_spb.setEnabled(False)
            self.well_ml_spb.setEnabled(False)
            self.sim_dx_ml_spb.setEnabled(False)
            # Enable the nm spinboxes
            self.barrier_nm_spb.setEnabled(True)
            self.well_nm_spb.setEnabled(True)
            self.sim_dx_nm_spb.setEnabled(True)

            # Set the value of the monolayers spinboxes, base on the nm values (converted to meters)
            # if op == self.barrier_nm_spb:
            self.barrier_ml_spb.setValue(
                np.round(self.barrier_nm_spb.value() * NM / ml, 3)
            )
            # if op == self.well_nm_spb:
            self.well_ml_spb.setValue(np.round(self.well_nm_spb.value() * NM / ml, 3))
            self.sim_dx_ml_spb.setValue(
                np.round(self.sim_dx_nm_spb.value() * NM / ml, 3)
            )

        # If the user is using monolayers
        else:
            # Enable the ml spinboxes, so that the user can interact with them
            self.barrier_ml_spb.setEnabled(True)
            self.well_ml_spb.setEnabled(True)
            self.sim_dx_ml_spb.setEnabled(True)
            # Disable the nm spinboxes
            self.barrier_nm_spb.setEnabled(False)
            self.well_nm_spb.setEnabled(False)
            self.sim_dx_nm_spb.setEnabled(False)
            # if op == self.barrier_ml_spb:
            # lp is in meters, converts to nm
            self.barrier_nm_spb.setValue(self.barrier_ml_spb.value() * ml / NM)
            # if op == self.well_ml_spb:
            self.well_nm_spb.setValue(self.well_ml_spb.value() * ml / NM)
            self.sim_dx_nm_spb.setValue(self.sim_dx_ml_spb.value() * ml / NM)

    def CreatedNewSimulation(self):
        """
        Called after a new simulation is created, just to update the interface
        """
        self.simulation_cbox.setCurrentIndex(len(self.sim_list) - 1)
        # Defines the output folder based on the simulation title
        sim = self.sim_list[
            -1
        ]  # The new simulation was just appended to the end of the list
        sim.output_folder = str(self.output_folder_line.text())
        self.PlotStructure()
        self.UpdateStructureTable()
        self.UpdateSimList()
        self.UpdateInterface()
        self.UpdateLayerCount()
        self.UpdateUnits()

    def UpdateSimList(self):
        """
        If simulations are added, deleted or loaded, needs to update the list
        """
        # Clears the simulation list
        self.simulation_cbox.clear()
        # Fills the simulation combobox with every simulation from the list
        for sim in self.sim_list:
            self.simulation_cbox.addItem(sim.title)
        self.simulation_cbox.setCurrentIndex(len(self.sim_list) - 1)

    def UpdateStructure(self, op):
        """
        Function that adds, inserts or removes layers from the surface, based on the user choice on
        the GUI.
        op is the operation the user wants to perform, defined in self.ConnectSignals
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]

        # Adding a new well
        if op == "AddWell":
            sim.AddWell(self.well_nm_spb.value())

        # Adding a new barrier
        elif op == "AddBarrier":
            sim.AddBarrier(self.barrier_nm_spb.value())

        # Replacing a well
        elif op == "ReplaceWell":
            index = self.struct_table.currentRow()  # Selected table line
            if index == -1:  # In case there is nothing to replace
                return
            sim.ReplaceWell(self.well_nm_spb.value(), index)

        # Replacing a barrier
        elif op == "ReplaceBarrier":
            index = self.struct_table.currentRow()  # Selected table line
            if index == -1:  # In case there is nothing to replace
                return
            sim.ReplaceBarrier(self.barrier_nm_spb.value(), index)

        # Inserting a well
        elif op == "InsertWell":
            index = self.struct_table.currentRow()  # Selected table line
            sim.InsertWell(self.well_nm_spb.value(), index)

        # Inserting a barrier
        elif op == "InsertBarrier":
            index = self.struct_table.currentRow()  # Selected table line
            sim.InsertBarrier(self.barrier_nm_spb.value(), index)

        elif op == "RemoveSelected":
            index = self.struct_table.currentRow()  # Linha da tabela selecionada
            if len(sim.estrutura) > 1:  # If there is a structure, delete the last item
                sim.RemoveSelected(index)
            else:
                op = "RemoveAll"  # Just to avoid repeating 5 lines of code

        elif op == "RemoveLast":
            if len(sim.estrutura) > 1:  # If there is a structure, delete the last item
                sim.RemoveSelected(-1)
            else:
                op = "RemoveAll"  # Just to avoid repeating 5 lines of code

        # This is not "elif" just so that the "else" from RemoveSelected and RemoveLast work
        if op == "RemoveAll":
            sim.RemoveAll()

        # Since the structure was modified, define this simulation as not ran
        sim.sim_ran = False
        sim.abs_ran = False

        # Updates table, graph and buttons
        self.UpdateStructureTable()
        self.PlotStructure(sim)
        self.UpdateInterface()
        self.UpdateLayerCount()

    def UpdateLayerCount(self):
        """
        Calculates the number of layers and updates the Spinbox on the advanced tab
        """

        try:  # If there is a simulation and this simulation has at least one layer
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
            layers = len(sim.estrutura)
            self.adv_total_layers_spb.setValue(layers)
        except:  # If there is no simulation or it doesn't have any layers yet
            self.adv_total_layers_spb.setValue(0)

    def UpdateInterface(self):
        """
        Updates the buttons, list of simulations available on the simulations combobox.
        This simulation is called almost everytime after user interaction.
        """
        # Check whether there are simulations
        # The selected simulation defines whether some options on the interface are available
        # If there are no simulations, disable most buttons, except "load" and "new"
        if len(self.sim_list) == 0:
            # Lower bar
            self.save_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)
            self.output_folder_btn.setEnabled(False)
            # Structure tab
            self.add_well_btn.setEnabled(False)
            self.add_barrier_btn.setEnabled(False)
            self.replace_well_btn.setEnabled(False)
            self.replace_barrier_btn.setEnabled(False)
            self.insert_well_btn.setEnabled(False)
            self.insert_barrier_btn.setEnabled(False)
            self.remove_selected_btn.setEnabled(False)
            self.remove_last_btn.setEnabled(False)
            self.remove_all_btn.setEnabled(False)
            # Layer thickness spinboxes
            self.barrier_ml_spb.setEnabled(False)
            self.well_ml_spb.setEnabled(False)
            self.barrier_nm_spb.setEnabled(False)
            self.well_nm_spb.setEnabled(False)
            # Simulation tab
            self.sim_Efield_btn.setEnabled(False)
            self.sim_run_btn.setEnabled(False)
            self.sim_plot_results_btn.setEnabled(False)
            self.sim_plot_structure_btn.setEnabled(False)
            # Absorption tab
            self.abs_run_btn.setEnabled(False)
            self.abs_plot_btn.setEnabled(False)
            # Transmission tab
            self.tra_plot_btn.setEnabled(False)
            # Photocurrent tab
            self.pc_run_btn.setEnabled(False)
            # Genetic algorithm tab
            self.ga_run_btn.setEnabled(False)
            # Automation tab
            self.auto_run_btn.setEnabled(False)

            self.ClearStructureTable()
            self.ClearSimPlot()
            self.ClearAbsPlot()
            self.ClearMaterialData()

        else:
            # current simulation
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
            # If there are available simulations, some buttons must to be enabled
            # Lower bar
            self.save_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            self.rename_btn.setEnabled(True)
            self.output_folder_btn.setEnabled(True)
            # Structure tab
            self.add_well_btn.setEnabled(True)
            self.add_barrier_btn.setEnabled(True)
            self.insert_well_btn.setEnabled(True)
            self.insert_barrier_btn.setEnabled(True)
            # Layer thickness spinboxes
            if self.adv_nm_layers_chkbx.checkState():
                self.barrier_nm_spb.setEnabled(True)
                self.well_nm_spb.setEnabled(True)
            else:
                self.barrier_ml_spb.setEnabled(True)
                self.well_ml_spb.setEnabled(True)

            if (
                len(sim.estrutura) > 0
            ):  # If the structure is has layers, a simulation may be run
                # Simulation tab
                self.sim_run_btn.setEnabled(True)
                self.sim_plot_structure_btn.setEnabled(True)
                # Structure tab
                self.replace_well_btn.setEnabled(True)
                self.replace_barrier_btn.setEnabled(True)
                self.remove_selected_btn.setEnabled(True)
                self.remove_last_btn.setEnabled(True)
                self.remove_all_btn.setEnabled(True)
                # Genetic algorithm tab
                self.ga_run_btn.setEnabled(True)
                # Automation tab
                self.auto_run_btn.setEnabled(True)

                if sim.sim_ran is False:  # If the simulation has not yet been executed
                    # Simulation tab
                    self.sim_plot_results_btn.setEnabled(False)
                    # Absorption tab
                    self.abs_run_btn.setEnabled(False)
                    # Transmission tab
                    self.tra_plot_btn.setEnabled(False)
                    # Photocurrent tab
                    self.pc_run_btn.setEnabled(False)
                else:
                    # Simulation tab
                    self.sim_plot_results_btn.setEnabled(True)
                    # Absorption tab
                    self.abs_run_btn.setEnabled(True)
                    # Transmission tab
                    self.tra_plot_btn.setEnabled(True)
                    # Photocurrent tab
                    self.pc_run_btn.setEnabled(True)

                if (
                    sim.abs_ran is False
                ):  # If the absorption has not yet been calculated
                    self.abs_plot_btn.setEnabled(False)
                else:
                    self.abs_plot_btn.setEnabled(True)
            else:
                # Simulation tab
                self.sim_run_btn.setEnabled(False)
                # Structure tab
                self.replace_well_btn.setEnabled(False)
                self.replace_barrier_btn.setEnabled(False)
                self.remove_selected_btn.setEnabled(False)
                self.remove_last_btn.setEnabled(False)
                self.remove_all_btn.setEnabled(False)
                # Simulation tab
                self.sim_Efield_btn.setEnabled(False)
                self.sim_plot_results_btn.setEnabled(False)
                self.sim_plot_structure_btn.setEnabled(False)
                # Absorption tab
                self.abs_run_btn.setEnabled(False)
                # Genetic algorithm tab
                self.ga_run_btn.setEnabled(False)
                # Automation tab
                self.auto_run_btn.setEnabled(False)
            self.UpdateStructureTable()
            self.FillMaterialData()

    # Information about the materials used on the simulation, shown on the structure tab
    def ClearMaterialData(self):
        """
        Removes the information about the materials from the Structure Tab. This
        function is called when there is no simulation.
        """
        self.lbl_lattice_parameter_val.setText("")
        self.lbl_barrier_material_val.setText("")
        self.lbl_barrier_effective_mass_val.setText("")
        self.lbl_barrier_electronic_potential_val.setText("")
        self.lbl_barrier_non_parabolicity_val.setText("")
        self.lbl_well_material_val.setText("")
        self.lbl_well_effective_mass_val.setText("")
        self.lbl_well_electronic_potential_val.setText("")
        self.lbl_well_non_parabolicity_val.setText("")

    def FillMaterialData(self):
        """
        Fills the structure tab with data for the selected materials.
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.lbl_lattice_parameter_val.setText(f"{1E10*sim.latpar:.3f} Ang")
        self.lbl_barrier_material_val.setText(f"{sim.barrier}")
        self.lbl_barrier_effective_mass_val.setText(f"{sim.m_eff_ct_barrier:.3e}")
        self.lbl_barrier_electronic_potential_val.setText(
            f"{1E3*sim.pot_barrier:.2f} meV"
        )
        self.lbl_barrier_non_parabolicity_val.setText(f"{sim.e_nonparab_barrier:.3e}")
        self.lbl_well_material_val.setText(f"{sim.well}")
        self.lbl_well_effective_mass_val.setText(f"{sim.m_eff_ct_well:.3e}")
        self.lbl_well_electronic_potential_val.setText(f"{1E3*sim.pot_well:.2f} meV")
        self.lbl_well_non_parabolicity_val.setText(f"{sim.e_nonparab_well:.3e}")

    # Tables ###########################################################################
    # Structure
    def InitializeStructTable(self):
        """
        Initializes the table presenting the structure
        """
        self.struct_table.setColumnCount(4)
        self.struct_table.setColumnWidth(0, 60)
        self.struct_table.setColumnWidth(1, 60)
        self.struct_table.setColumnWidth(2, 40)
        self.struct_table.setColumnWidth(3, 40)
        self.struct_table.move(0, 0)
        self.struct_table.setHorizontalHeaderLabels(["Material", "Feature", "ML", "nm"])
        self.struct_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.struct_table.setSelectionBehavior(QAbstractItemView.SelectRows)

    def UpdateStructureTable(self):
        """
        Function that updates the table at the structure tab. The table is erased every time and
        rewrites it all again usig information from sim.material and sim.estrutura.
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.struct_table.setRowCount(len(sim.material))
        for i, material in enumerate(sim.material):
            col0 = QTableWidgetItem(material)  # Creates the item
            col0.setTextAlignment(0x0084)  # Align h center, v baseline
            col1 = QTableWidgetItem(sim.feature[i])
            col1.setTextAlignment(0x0084)  # Align h center, v baseline
            col2 = QTableWidgetItem(f"{sim.estrutura[i] / ( sim.latpar / 2):.0f}")
            col2.setTextAlignment(0x0082)  # Align h right, v baseline
            col3 = QTableWidgetItem(f"{sim.estrutura[i] / NM:.3f}")
            col3.setTextAlignment(0x0082)  # Align h right, v baseline
            self.struct_table.setItem(i, 0, col0)
            self.struct_table.setItem(i, 1, col1)
            self.struct_table.setItem(i, 2, col2)
            self.struct_table.setItem(i, 3, col3)
            # Adjusts the line heigth
            self.struct_table.setRowHeight(i, 18)
            # Corrects the line index (without this correction, it starts from 1, instead of 0)
            self.struct_table.setVerticalHeaderItem(i, QTableWidgetItem(f"{i}"))
        # Adjusts the column width
        self.struct_table.setColumnWidth(0, 60)
        self.struct_table.setColumnWidth(1, 60)
        self.struct_table.setColumnWidth(2, 40)
        self.struct_table.setColumnWidth(3, 40)

    def ClearStructureTable(self):
        self.struct_table.setRowCount(0)

    # Simulation data
    def InitializeDataTable(self):
        """
        Initializes the table presenting the results
        """
        self.data_table.setColumnCount(1)
        self.data_table.setColumnWidth(0, 120)
        # self.struct_table.setColumnWidth(1, 60)
        # self.struct_table.setColumnWidth(2, 40)
        # self.struct_table.setColumnWidth(3, 40)
        self.data_table.move(0, 0)
        # Definition of the header labels
        h_lbls = ["Energy (meV)"]
        self.data_table.setHorizontalHeaderLabels(h_lbls)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)

    def UpdateDataTable(self):
        """
        Function that updates the table at the data tab. The table is erased every time and
        rewritten with data from the latest simulation
        """
        self.ClearDataTable()
        # Gets the current selected simulation
        sim = self.sim_list[self.simulation_cbox.currentIndex()]

        self.data_table.setRowCount(len(sim.sim_Energias))
        for i, energia in enumerate(sim.sim_Energias):

            col0 = QTableWidgetItem(f"{energia * 1000:.3f} meV")  # Creates the item
            col0.setTextAlignment(0x0084)  # Align h center, v baseline
            # col1 = QTableWidgetItem(sim.feature[i])
            # col1.setTextAlignment(0x0084)  # Align h center, v baseline
            # col2 = QTableWidgetItem(f"{sim.estrutura[i] / ( sim.latpar / 2):.0f}")
            # col2.setTextAlignment(0x0082)  # Align h right, v baseline
            # col3 = QTableWidgetItem(f"{sim.estrutura[i] / NM:.3f}")
            # col3.setTextAlignment(0x0082)  # Align h right, v baseline
            self.data_table.setItem(i, 0, col0)
            # self.struct_table.setItem(i, 1, col1)
            # self.struct_table.setItem(i, 2, col2)
            # self.struct_table.setItem(i, 3, col3)
            # Adjusts the line heigth
            self.data_table.setRowHeight(i, 18)
            # Corrects the line index (without this correction, it starts from 1, instead of 0)
            self.data_table.setVerticalHeaderItem(i, QTableWidgetItem(f"{i}"))
        # Adjusts the column width
        self.data_table.setColumnWidth(0, 120)
        # self.struct_table.setColumnWidth(1, 60)
        # self.struct_table.setColumnWidth(2, 40)
        # self.struct_table.setColumnWidth(3, 40)

    def ClearDataTable(self):
        self.data_table.setRowCount(0)

    # Plots ############################################################################
    def CreatePlots(self):
        """
        Initial configuration of the plots
        """
        # Simulation
        self.sim_fig = plt.figure()
        self.sim_canvas = FigureCanvas(self.sim_fig)
        self.sim_plot_layout.addWidget(self.sim_canvas)
        self.sim_nav = NavigationToolbar(self.sim_canvas, self.sim_tab)
        self.sim_plot_layout.addWidget(self.sim_nav)
        self.sim_subplot = self.sim_fig.add_subplot(111)
        self.sim_subplot.grid(True, axis="y")
        self.show()
        # Absorption
        self.abs_fig = plt.figure()
        self.abs_canvas = FigureCanvas(self.abs_fig)
        self.abs_plot_layout.addWidget(self.abs_canvas)
        self.abs_nav = NavigationToolbar(self.abs_canvas, self.abs_tab)
        self.abs_plot_layout.addWidget(self.abs_nav)
        self.abs_subplot = self.abs_fig.add_subplot(111)
        self.abs_subplot.grid(True, axis="both")
        self.show()
        # Transmission
        self.tra_fig = plt.figure()
        # self.tra_fig, self.tra_ax = plt.subplots(nrows=1, ncols=1)
        self.tra_canvas = FigureCanvas(self.tra_fig)
        self.tra_plot_layout.addWidget(self.tra_canvas)
        self.tra_nav = NavigationToolbar(self.tra_canvas, self.transmission_tab)
        self.tra_plot_layout.addWidget(self.tra_nav)
        self.tra_subplot = self.tra_fig.add_subplot(111)
        self.tra_subplot.grid(True, axis="both")
        self.show()
        # Photocurrent
        self.pc_fig = plt.figure()
        self.pc_canvas = FigureCanvas(self.pc_fig)
        self.pc_plot_layout.addWidget(self.pc_canvas)
        self.pc_nav = NavigationToolbar(self.pc_canvas, self.photocurrent_tab)
        self.pc_plot_layout.addWidget(self.pc_nav)
        self.pc_subplot = self.pc_fig.add_subplot(111)
        self.pc_subplot.grid(True, axis="both")
        self.show()
        # Genetic algorithm
        self.ga_fig = plt.figure()
        self.ga_canvas = FigureCanvas(self.ga_fig)
        self.ga_plot_layout.addWidget(self.ga_canvas)
        self.ga_nav = NavigationToolbar(self.ga_canvas, self.ga_tab)
        self.ga_plot_layout.addWidget(self.ga_nav)
        self.ga_subplot = self.ga_fig.add_subplot(111)
        self.ga_subplot.grid(True, axis="both")
        self.show()

    # Structure and Wave Function
    def PlotStructure(self, sim=None):
        """
        Function that interprets the structure data and create arrays to plot the graph.
        It is necessary to analyze the position and potential arrays in order to correctly plot the
        structure. In the interface between two materials the position is the same, but the energy
        is different. If the material is repeated, but the position is different, there is no need
        to create another point.
        """
        # If this function is called without a specified sim, it gets the selected one from the
        # interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.ClearSimPlot()
        if len(sim.estrutura) < 1:  # Only continue if there is a structure
            return
        # Creating the x and energy arrays  just for the plot
        sim.x_graf = np.array([0])
        sim.v_graf = np.array([sim.pot[0]])

        # Going through the layers and creating the arrays iteratively
        for i, en in enumerate(sim.pot):
            if (
                en == sim.v_graf[-1]
            ):  # If this layer has the same energy as the previous one
                sim.x_graf = np.append(sim.x_graf, [sim.x_graf[-1] + sim.estrutura[i]])
                sim.v_graf = np.append(sim.v_graf, [en])
            else:
                sim.x_graf = np.append(
                    sim.x_graf, [sim.x_graf[-1], sim.x_graf[-1] + sim.estrutura[i]]
                )
                sim.v_graf = np.append(sim.v_graf, [en, en])

        # Changing x and y to [nm] and [meV]
        sim.x_graf = 1.0e9 * sim.x_graf
        sim.v_graf = 1.0e3 * sim.v_graf

        # Centering the structure around 0
        # sim.x_graf = sim.x_graf - (np.max(sim.x_graf) - np.min(sim.x_graf)) / 2.0

        layers = len(sim.estrutura)
        central_layer = self.sim_central_layer_spb.value()
        """
        This part of the code was used to put the 0 of the x-axis in the center of a 
        layer, but this doesn't work for Pedro's photocurrent calculations, therefore
        the 0 will be at the interfaces.
        # Creating the x-axis array - the x-axis 0 is centered in the target layer
        if central_layer < 0:
            x0 = 0.0
        else:
            if (
                central_layer > layers - 1
            ):  # In case the number of the layer exceeds the limit
                central_layer = layers - 1
            left_x_central = np.sum(
                sim.estrutura[0:central_layer]
            )  # Thickness of layers before
            central_thickness = sim.estrutura[
                central_layer
            ]  # Thickness of target layers
            x0 = -left_x_central - central_thickness / 2.0
        x0 = x0 / NM
        sim.x_graf = sim.x_graf + x0
        """
        if central_layer <= 0:
            x0 = 0.0
        else:
            # In case the number of the layer exceeds the limit
            if central_layer > layers:
                central_layer = layers
            # Thickness of layers on the left side of the target interface
            x0 = -np.sum(sim.estrutura[0:central_layer])
        x0 = x0 / NM
        sim.x_graf = sim.x_graf + x0

        """
        Plots the structure using arrays that were created by the UpdateStructure. Doesn't erase
        the graph, in order to allow comparison between two structures
        """
        # sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.sim_subplot.plot(sim.x_graf, sim.v_graf)  # x in [nm] and y in [meV]
        self.sim_subplot.set_xlabel("Length (nm)")
        self.sim_subplot.set_ylabel("Energy (meV)")
        self.sim_subplot.set_xbound(sim.x_graf[0] - 0.2, sim.x_graf[-1] + 0.2)
        self.sim_subplot.set_ybound(np.min(sim.v_graf) - 50, np.max(sim.v_graf) + 100)
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()

    def PlotSimResults(self, sim=None):
        """
        Plost the results from ResultadoWF.
        """
        # If this function is called without a specified sim, it gets the selected one from the
        # interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        # Gets the selected simulation
        # sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.sim_subplot.grid(True, axis="y")
        # Plots the probability density
        for result in sim.sim_ResultadoWF:
            self.sim_subplot.plot(result[0, :] / NM, result[3, :] * 1.0e3)
            # self.subplot_sim.plot(result[:, 0] / 1.0E9, result[:, 3] * 1.0E0)
        self.sim_canvas.draw()

    def ClearSimPlot(self):
        """
        Clears the simulation plot.
        """
        self.sim_fig.clf()
        self.sim_fig.tight_layout()
        self.sim_subplot = self.sim_fig.add_subplot(111)
        self.sim_canvas.draw()

    # Absorption
    def PlotAbsorption(self, sim=None):
        """
        Plots the results from the absorption of the selected simulation.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.abs_subplot.plot(sim.abs_energy_axis, sim.abs_result)
        self.abs_subplot.set_xlabel("Energy (eV)")
        self.abs_subplot.set_ylabel("Absorption (u.a.)")
        self.abs_fig.tight_layout()
        self.abs_subplot.grid(True, axis="both")
        self.abs_canvas.draw()

    def ClearAbsPlot(self):
        """
        Clears the absorption plot.
        """
        self.abs_fig.clf()
        self.abs_fig.tight_layout()
        self.abs_subplot = self.abs_fig.add_subplot(111)
        self.abs_canvas.draw()

    # Transmission
    def PlotTransmission(self, sim=None):
        """
        Plots the transmission. The transmission is only calculated after simulation was
        run.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.tra_subplot.plot(sim.sim_VecEnergy * 1.0e3, sim.sim_Transmission)
        self.tra_subplot.set_xlabel("Energy (eV)")
        self.tra_subplot.set_ylabel("Transmission (u.a.)")
        self.tra_fig.tight_layout()
        self.tra_subplot.grid(True, axis="both")
        # self.tra_subplot.set_yscale("log")
        self.tra_canvas.draw()

    def ClearTransPlot(self):
        """
        Clears the Transmission plot.
        """
        self.tra_fig.clf()
        self.tra_fig.tight_layout()
        self.tra_subplot = self.tra_fig.add_subplot(111)
        self.tra_canvas.draw()

    # Photocurrent
    def PlotPhotocurrent(self, sim=None):
        """
        Plots the photocurrent. The photocurrent is only calculated after simulation was
        run.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.pc_subplot.plot(sim.sim_Photocurrent[0], sim.sim_Photocurrent[1])
        self.pc_subplot.set_xlabel("Energy (eV)")
        self.pc_subplot.set_ylabel("Photocurrent (u.a.)")
        self.pc_fig.tight_layout()
        self.pc_subplot.grid(True, axis="both")
        # self.pc_subplot.set_yscale("log")
        self.pc_canvas.draw()

    def ClearPhotocurrentPlot(self):
        """
        Clears the photocurrent plot.
        """
        self.pc_fig.clf()
        self.pc_fig.tight_layout()
        self.pc_subplot = self.pc_fig.add_subplot(111)
        self.pc_canvas.draw()

    # File inputs and outputs ##########################################################
    def ChooseOutputFolder(self):
        """
        Prompts the user to choose the simulation data output folder. Inside this folder, there will
        be a folder with the simulation title.
        """
        folder = QFileDialog.getExistingDirectory()
        if (
            folder
        ):  # If the output folder was properly selects, add the os separator to it
            os.path.join(folder, "")  # OS independent separator
        self.output_folder_line.setText(folder)

        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        sim.output_folder = self.output_folder_line.text()

    def SaveSimulation(self):
        """
        Saves the structure, simulation and absorption parameters.
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        try:
            save_dir = os.path.join(sim.output_folder, f"{sim.title}.qwsim")
            # print(f'save_dir: {save_dir}')
            file, _ = QFileDialog.getSaveFileName(
                self,
                caption="Save structure and simulation parameters",
                directory=save_dir,
                filter=self.tr("*.qwsim"),
            )

            output_file = open(file, "wb")
            pickle.dump(sim, output_file)
            output_file.close()
        except:
            print("Couldn't save the structure")
            # If the user doesn't choose a filename, closing the interface, do nothing
            return

    def LoadSimulation(self):
        """
        Loads structure and simulation data from a file
        """
        try:  # Returns a tuple
            file, _ = QFileDialog.getOpenFileName(
                self, "Loads the structure and simulation data", self.tr("*.qwsim")
            )
            input_file = open(file, "rb")
            sim = pickle.load(input_file)

            self.sim_list.append(sim)
            self.simulation_cbox.setCurrentIndex(len(self.sim_list) - 1)
            self.UpdateStructureTable()
            self.UpdateSimList()
            self.UpdateLayerCount()
            # If an structure was saved, plot it
            try:  # The try-except is to avoid errors in case sim.structure doesn't exist
                if len(sim.estrutura) > 0:
                    self.PlotStructure(sim)
            except:
                pass
            # If a simulation result was saved, plot it
            try:
                if sim.sim_ran:
                    self.PlotSimResults(sim)
            except:
                pass
            # If an absorption result was saved, plot it
            try:
                if sim.abs_ran:
                    self.PlotAbsorption(sim)
            except:
                pass
            self.UpdateInterface()

        except:
            # If an error happens upon opening the file, the function just returns False.
            print("Couldn't load the structure")
            return

    def CreateOutputFolder(self, sim):
        """
        Checks whether the selected output folder exists or must be created
        """
        # Defines the output folder based on the simulation title
        # output_folder = str(self.output_folder_line.text())
        output_folder = sim.output_folder
        # If the folder was not chosen, just get the file execution path
        if not output_folder:
            output_folder = os.path.dirname(os.path.abspath(__file__))
        # If the menu checkbox is checked (it is, by default) create a folder with the same name as
        # the simulation
        if self.adv_new_folder_chkbx.isChecked:
            output_folder = os.path.join(output_folder, sim.title)

        return output_folder

    def SaveSimOutput(self, sim):
        """
        Based on the interface's checkboxes, saves the simulation output
        """
        output_folder = self.CreateOutputFolder(sim)

        # To avoid repetition of the "all checkbox"
        save_all = self.sim_files_all_chkbx.isChecked()

        # After the end of the calculations, save the result in text files if desired
        if (
            save_all
            or self.sim_files_effm_chkbx.isChecked()
            or self.sim_files_energies_chkbx.isChecked()
            or self.sim_files_npe_chkbx.isChecked()
            or self.sim_files_pot_chkbx.isChecked()
            or self.sim_files_wf_chkbx.isChecked()
            or self.sim_files_x_chkbx.isChecked()
        ):
            data_folder = os.path.join(output_folder, "Data")

            # if the folder to save the data doesn't exist, create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

        if save_all or self.sim_files_x_chkbx.isChecked():
            np.savetxt(os.path.join(data_folder, "X Axis.txt"), sim.sim_x, newline="\n")

        if save_all or self.sim_files_pot_chkbx.isChecked():
            np.savetxt(
                os.path.join(data_folder, "Electrical Potential.txt"),
                sim.sim_pot,
                newline="\n",
            )

        if save_all or self.sim_files_effm_chkbx.isChecked():
            np.savetxt(
                os.path.join(data_folder, "Effective Mass.txt"),
                sim.sim_effm_cte,
                newline="\n",
            )

        if save_all or self.sim_files_npe_chkbx.isChecked():
            np.savetxt(
                os.path.join(data_folder, "Non-Parabolicity.txt"),
                sim.sim_npe,
                newline="\n",
            )

        if save_all or self.sim_files_energies_chkbx.isChecked():
            np.savetxt(os.path.join(data_folder, "Autoenergias.txt"), sim.sim_Energias)

        if save_all or self.sim_files_wf_chkbx.isChecked():
            PastaWave = os.path.join(data_folder, "Wave Functions")
            if not os.path.exists(PastaWave):
                os.makedirs(PastaWave)
            for a in range(len(sim.sim_Energias)):
                fname = f"WF_E{a}_{sim.sim_Energias[a]:02.6f}.txt"
                # The transposed is print, so that the WF is in a column, not row
                np.savetxt(os.path.join(PastaWave, fname), sim.sim_ResultadoWF[a].T)

    def SaveTransmissionOutput(self, sim):
        """
        Based on the interface's checkboxes, saves the Transmission output
        """
        output_folder = self.CreateOutputFolder(sim)

        # After the end of the calculations, save the result in text files if desired
        if (
            self.sim_files_all_chkbx.isChecked()
            or self.sim_files_trans_chkbx.isChecked()
        ):
            data_folder = os.path.join(output_folder, "Data")

            # if the folder to save the data doesn't exist, create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            output = np.column_stack((sim.sim_VecEnergy, sim.sim_Transmission))
            output_file = os.path.join(data_folder, "Transmission Spectrum.txt")
            np.savetxt(output_file, output)

    def SavePhotocurrentOutput(self, sim):
        """
        Based on the interface's checkboxes, saves the Transmission output
        """
        output_folder = self.CreateOutputFolder(sim)

        # After the end of the calculations, save the result in text files if desired
        if self.sim_files_all_chkbx.isChecked() or self.sim_files_pc_chkbox.isChecked():
            data_folder = os.path.join(output_folder, "Data")

            # if the folder to save the data doesn't exist, create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            output = sim.sim_Photocurrent
            output_file = os.path.join(data_folder, "Photocurrent Spectrum.txt")
            np.savetxt(output_file, output)

    def SaveGUIConfigFile(self):
        """
        Saves all the values and settings on the interface to the "interface.cfg" file.
        This is meant to be a easy way of saving the default settings, so that the user doesn't need
        to change the values every time the software is opened, nor needs to manually edit the
        config file.
        The values must be in string format.
        """
        # Reads the configuration defaults from the "interface.cfg" file:
        cfg = configparser.ConfigParser()
        file = os.path.join(self.base_path, "interface.cfg")
        cfg.read(file)

        # Structure tab
        # Saves only the values of nm or ml, depending on the checkbox state
        if self.adv_nm_layers_chkbx.checkState():
            cfg["stru"]["barrier_nm"] = f"{self.barrier_nm_spb.value()}"
            cfg["stru"]["well_nm"] = f"{self.well_nm_spb.value()}"
        else:
            cfg["stru"]["barrier_ml"] = f"{self.barrier_ml_spb.value()}"
            cfg["stru"]["well_ml"] = f"{self.well_ml_spb.value()}"

        # Simulation tab
        cfg["simu"]["E0"] = f"{self.sim_E0_spb.value()}"
        cfg["simu"]["dE"] = f"{self.sim_dE_spb.value()}"
        cfg["simu"]["Ef"] = f"{self.sim_Ef_spb.value()}"
        if self.adv_nm_layers_chkbx.checkState():
            cfg["simu"]["dx_nm"] = f"{self.sim_dx_nm_spb.value()}"
        else:
            cfg["simu"]["dx_ml"] = f"{self.sim_dx_ml_spb.value()}"
        cfg["simu"]["Efield"] = f"{self.sim_Efield_spb.value()}"
        cfg["simu"]["central_layer"] = f"{self.sim_central_layer_spb.value()}"
        # Checkboxes
        cfg["simu"]["output_all"] = (
            "True" if self.sim_files_all_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_effm"] = (
            "True" if self.sim_files_effm_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_energies"] = (
            "True" if self.sim_files_energies_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_npe"] = (
            "True" if self.sim_files_npe_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_pot"] = (
            "True" if self.sim_files_pot_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_trans"] = (
            "True" if self.sim_files_trans_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_wf"] = (
            "True" if self.sim_files_wf_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_x"] = (
            "True" if self.sim_files_x_chkbx.checkState() else "False"
        )

        # Absorption tab
        cfg["abso"]["initial_WF"] = f"{self.abs_init_wf_spb.value()}"
        cfg["abso"]["lorz_broad"] = f"{self.abs_broadening_spb.value()}"
        cfg["abso"]["E0"] = f"{self.abs_E0_spb.value()}"
        cfg["abso"]["dE"] = f"{self.abs_dE_spb.value()}"
        cfg["abso"]["Ef"] = f"{self.abs_Ef_spb.value()}"

        # Transmission tab

        # Photocurrent tab
        cfg["phot"]["E0"] = f"{self.pc_E0_spb.value()}"
        cfg["phot"]["dE"] = f"{self.pc_dE_spb.value()}"
        cfg["phot"]["Ef"] = f"{self.pc_Ef_spb.value()}"

        # Genetic Algorithm tab
        cfg["gene"]["iterations"] = f"{self.ga_iter_spb.value():d}"
        cfg["gene"]["population"] = f"{self.ga_pop_spb.value():d}"
        cfg["gene"]["target_E"] = f"{self.ga_tgt_en_spb.value()}"
        cfg["gene"]["target_E_margin"] = f"{self.ga_tgt_en_margin_spb.value()}"
        cfg["gene"]["goal"] = f"{self.ga_goal_cbox.currentIndex():d}"

        # Automation tab
        cfg["auto"]["target_layer"] = f"{self.auto_layer_spb.value():d}"
        cfg["auto"]["thickness_initial"] = f"{self.auto_init_spb.value()}"
        cfg["auto"]["thickness_step"] = f"{self.auto_step_spb.value()}"
        cfg["auto"]["thickness_final"] = f"{self.auto_final_spb.value()}"

        # Data tab

        # Advanced tab
        cfg["adva"]["create_new_folder"] = (
            "True" if self.adv_new_folder_chkbx.checkState() else "False"
        )
        cfg["adva"]["use_nanometers"] = (
            "True" if self.adv_nm_layers_chkbx.checkState() else "False"
        )
        cfg["adva"]["autorun_abs"] = (
            "True" if self.adv_autorun_abs_chkbx.checkState() else "False"
        )
        cfg["adva"]["autorun_abs_initial_WF"] = f"{self.adv_wf0_spb.value()}"
        cfg["adva"]["autorun_trans"] = (
            "True" if self.adv_autorun_trans_chkbx.checkState() else "False"
        )
        cfg["adva"]["interface_to_split"] = f"{self.adv_split_layer_spb.value()}"
        cfg["adva"]["method"] = f"{self.adv_method_cbox.currentIndex()}"

        # Saves the configuration file
        with open(file, "w") as configfile:
            cfg.write(configfile)

    # Run the calculations #############################################################
    # Absorption
    def RunAbsorption(self, sim=None):
        """
        Calculates the structure's absorption spectra, the dipole moment, oscilator strenght and
        delta energy between the absorption peaks and the reference wavefunction (wf_0).
        """
        # If this function is called without a specified sim, it gets the selected one from the
        # interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return
        # Checks whether there are at least two wavefunctions, in order to calculate the absorption.
        if len(sim.sim_Energias) < 2:
            print("Cannot calculate absorption if there are less than 2 wavefunctions.")
            return
        # Collecting the relevant data from the GUI
        wf_0_index = int(self.abs_init_wf_spb.value())
        E0 = self.abs_E0_spb.value()
        Ef = self.abs_Ef_spb.value()
        dE = self.abs_dE_spb.value()
        # Linewidth broadening of the lorentzian. From "Van Hove singularities in intersubband
        # transitions in multiquantum well photodetectors" doi.org/10.1016/j.infrared.2006.10.016
        broadening = self.abs_broadening_spb.value()

        # Gets the choosen simulation
        # sim = self.sim_list[self.simulation_cbox.currentIndex()]
        sim.CalcAbs(wf_0_index, E0, Ef, dE, broadening)

        # Saving the absorption output to files
        output_folder = (
            sim.output_folder
        )  # As defined when the sim was created or "change folder"
        # Defines the output folder based on the simulation title
        if self.adv_new_folder_chkbx.isChecked:
            output_folder = os.path.join(output_folder, sim.title)
        # Saves in a folder called "Absorption"
        output_folder = os.path.join(output_folder, "Absorption")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            data = np.column_stack(
                (sim.abs_delta_E, sim.abs_dipole, sim.abs_osc_strength)
            )
            filename = f"DeltaE_DipoloEletrico_ForcadeOscilador_E{wf_0_index:01d}.txt"
            np.savetxt(os.path.join(output_folder, filename), data)
        except:
            print("Could not save absorption results")

        try:
            data = np.column_stack((sim.abs_energy_axis, sim.abs_result))
            filename = f"Absorcao_E{wf_0_index:01d}.txt"
            np.savetxt(os.path.join(output_folder, filename), data)
        except:
            print("Could not save absorption results")

        # Plot the absorption and update the interface
        self.PlotAbsorption(sim)
        self.UpdateInterface()

    def RunAutomation(self):
        """
        Creates an automation changing the thickness of the target layer by the values
        in the range defined by the user in the GUI
        """
        # Gets the base simulation, which will be modified by the automated steps
        base_sim = self.sim_list[self.simulation_cbox.currentIndex()]

        # Obtaining the target range of thicknesses
        th_init = self.auto_init_spb.value()
        th_step = self.auto_step_spb.value()
        th_final = self.auto_final_spb.value()
        # List of thicknesses
        th_list = np.arange(th_init, th_final + th_step, th_step, dtype=float)

        # Defining whether the thickness is in nanometers or monolayers
        if (
            self.adv_nm_layers_chkbx.checkState()
        ):  # If user wants nanometers, use value as is
            th_unit = "nm"
        else:  # else, calculate dx as a multiple of monolayer = latpar/2
            # Correcting the units, if the value was entered in monolayers
            th_unit = "ml"
            th_list = th_list * base_sim.latpar / 2

        # Gets the index of the layer that must be modified
        tgt_layer = self.auto_layer_spb.value()
        # The target layer must exist. In case the structure doesn't have the defined layer, correct
        if tgt_layer < 0:
            tgt_layer = 0
        elif tgt_layer > len(base_sim.estrutura) - 1:
            tgt_layer = len(base_sim.estrutura) - 1

        slist = []
        # For each thickness
        for th in th_list:
            # Creates a copy of the original simulation
            new_sim = deepcopy(base_sim)
            # Puts it into the simulations list
            self.sim_list.append(new_sim)
            # Puts it into a list just for the multiprocessing
            slist.append(new_sim)
            # Changes the title based on the title of the base simulation and the modification
            new_sim.title = base_sim.title + f" {th:.3f}{th_unit}"
            # Changes the thickness of the target layer
            if new_sim.feature[tgt_layer] == "Well":
                new_sim.ReplaceWell(th, tgt_layer)
            else:
                new_sim.ReplaceBarrier(th, tgt_layer)
            self.RunSimulation(new_sim)

        # cores = cpu_count()
        # p = Pool(processes=cores)
        # p = Pool(processes=1)
        # p.map(self.RunASim, slist)
        # with Pool(processes=cores) as p:
        #     for i, _ in enumerate(p.imap_unordered(self.RunASim, slist)):
        #         print(i)

        self.UpdateStructureTable()
        self.UpdateSimList()

    # Simulation
    def RunSimulation(self, sim=None):
        """
        This function is a copy of "Run" but it takes the simulation as an argument
        instead of getting the selected simulation from the combobox.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        # Obtaining the target range of energies
        E0 = self.sim_E0_spb.value()
        Ef = self.sim_Ef_spb.value()
        dE = self.sim_dE_spb.value()

        # Reads the value of dx from the interface
        if self.adv_nm_layers_chkbx.checkState():
            # If user wants nanometers, use value as is
            dx = self.sim_dx_nm_spb.value()
            dx_unit = "nm"
        else:  # else, calculate dx as a multiple of monolayer = latpar/2
            dx = self.sim_dx_ml_spb.value()
            dx_unit = "ml"

        # From the interface, defines the interface of the wavefunction split
        split_i = self.adv_split_layer_spb.value()

        # From the structure tab, gets the index of the central layer
        central_layer = self.sim_central_layer_spb.value()

        # The method chosen by the user to perform the calculations
        # Methods avaliable:
        # 0 - "Numerov - For"
        # 1 - "Numerov - Split"
        # 2 - "Numerov - Arrays"
        # 3 - "TMM"
        # 4 - "TMM - Split"
        wf_method = self.adv_method_cbox.currentIndex()

        # Timing
        t_start_run = time.time()

        # Runs the calculations with information gathered from the interface and stored
        # at a "sim".
        # Sim is the selected simulation, containing all relevant data
        sim.RunSim(wf_method, split_i, E0, Ef, dE, dx, dx_unit, central_layer)
        # Timing
        print(f"Total time: {time.time() - t_start_run:.3f} s")

        self.SaveSimOutput(sim)
        self.PlotSimResults(sim)
        self.UpdateInterface()
        self.UpdateDataTable()
        # If the user wants, automatically calculate the absorption, transmission or
        # photocurrent after the main simulation
        if self.sim_autorun_abs_chkbx.checkState():
            self.RunAbsorption(sim)
        if self.sim_autorun_tra_chkbx.checkState():
            self.RunTransmission(sim)
        if self.sim_autorun_pc_chkbx.checkState():
            self.RunPhotocurrent(sim)

    # Transmission
    def RunTransmission(self, sim=None):
        """
        Performs the calculation of the transmission and displays it on the graph.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        sim.Transmission()

        self.PlotTransmission(sim)
        self.SaveTransmissionOutput(sim)

    def RunPhotocurrent(self, sim=None):
        """
        Performs the calculation of the Photocurrent and plots on the graph.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        # Reads the value of dx from the interface
        if self.adv_nm_layers_chkbx.checkState():
            # If user wants nanometers, use value as is
            dx = self.sim_dx_nm_spb.value()
            dx_unit = "nm"
        else:  # else, calculate dx as a multiple of monolayer = latpar/2
            dx = self.sim_dx_ml_spb.value()
            dx_unit = "ml"

        # Obtaining the target range of energies and converting to eV
        E0 = self.pc_E0_spb.value() * 1.0e-3
        Ef = self.pc_Ef_spb.value() * 1.0e-3
        dE = self.pc_dE_spb.value() * 1.0e-3

        sim.RunPhotocurrent(dx, E0, Ef, dE)

        self.PlotPhotocurrent(sim)
        self.SavePhotocurrentOutput(sim)

    # Simulation instance functions
    def CreateSimulation(self, title, materials):
        """
        Function that creates a new simulation with the relevant information
        """
        # Creating the arrays necessary to the simulation
        array_data = dict()
        array_data["estrutura"] = np.array([], dtype=np.float64)
        array_data["massa_eff_const"] = np.array([], dtype=np.float64)
        array_data["pot"] = np.array([], dtype=np.float64)
        array_data["E_nonparab"] = np.array([], dtype=np.float64)
        # Loads the available materias from the file
        self.mat = configparser.ConfigParser()
        self.mat.read(os.path.join(self.base_path, "materials.data"))
        # Defining material properties from the materials.data file
        material_data = dict()
        material_data["latpar"] = self.mat[materials].getfloat("latpar")
        material_data["barrier"] = self.mat[materials]["barrier"]
        material_data["m_eff_ct_barrier"] = self.mat[materials].getfloat(
            "m_eff_ct_barrier"
        )
        material_data["e_nonparab_barrier"] = self.mat[materials].getfloat(
            "e_nonparab_barrier"
        )
        material_data["pot_barrier"] = self.mat[materials].getfloat("pot_barrier")
        material_data["well"] = self.mat[materials]["well"]
        material_data["m_eff_ct_well"] = self.mat[materials].getfloat("m_eff_ct_well")
        material_data["pot_well"] = self.mat[materials].getfloat("pot_well")
        material_data["e_nonparab_well"] = self.mat[materials].getfloat(
            "e_nonparab_well"
        )

        # Creates the simulation and puts it into a list of simulations
        self.sim_list.append(simdata.SimData(title, array_data, material_data))
        self.current_number += 1
        self.new_sim_window.close()
        self.CreatedNewSimulation()

    def DeleteSimulation(self):
        """
        Deletes the simulation selected in the combobox.
        """
        # Gets the index of the selected simulation
        current_index = self.simulation_cbox.currentIndex()
        list_len = len(self.sim_list)

        # Removes it from the simulations list, if not empty
        if list_len == 0:
            return
        self.sim_list.pop(current_index)

        # if it was the last item in the combobox, select the previous
        if current_index == list_len - 1:
            self.simulation_cbox.setCurrentIndex(current_index - 1)
        # else if just selects the next item
        else:
            self.simulation_cbox.setCurrentIndex(current_index)

        # Calls the function that updates the interface with the simulation selected from the
        # combobox
        # Updates the combobox to reflect the change made to the simulation list
        self.UpdateInterface()
        self.UpdateSimList()
        self.UpdateLayerCount()
        # self.ChangedSimulation()

    def RenameSimulation(self):
        # Gets the current selected simulation
        try:
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
        except:
            print("There is no simulation to choose from, create one first.")
            return
        # Creates a new window pasing the selected simulation
        self.rename_window = RenameSimWindow(sim)
        # Shows the window
        self.rename_window.show()
        # Connects the signal to the function used to close the window
        self.rename_window.signal_renamed.connect(self.CloseRenameWindow)

    def CopySimulation(self):
        """
        Copies the current simulation creating a new instance with the same attributes and opens the
        title window so that the user can change the title.
        """
        # Gets the current selected simulation
        try:
            current_sim = self.sim_list[self.simulation_cbox.currentIndex()]
        except:
            print("There is no simulation to choose from, create one first.")
            return

        # Copies the current simulation and adds it to the list
        new_sim = deepcopy(current_sim)
        self.sim_list.append(new_sim)

        # Creates a new window
        self.rename_window = RenameSimWindow(new_sim)
        # Shows the window
        self.rename_window.show()
        # Connects the signal to the function used to close the window
        # This window will be closed by another function, which is called when the user
        # confirms the name of the new simulation. This is done by a signal emitted from
        # the TitleWindow.
        self.rename_window.signal_renamed.connect(self.CloseRenameWindow)

    # Functions related to other windows ###############################################
    def CloseRenameWindow(self):
        """
        Closes the window that was opened to change the simulation name.
        """
        self.UpdateSimList()
        self.rename_window.close()

    def OpenNewSimWindow(self):
        """
        Creates and opens the window that will create the title for the new simulation.
        """
        self.new_sim_window = NewSimWindow(self.current_number)
        # Shows the window
        self.new_sim_window.show()
        # Connects the signal to the function used to close the window
        self.new_sim_window.signal_updated_current_number.connect(
            self.UpdateCurrentNumber
        )
        # This window will be closed by another function, which is called when the user
        # confirms the name of the new simulation. This is done by a signal emitted from
        # the TitleWindow.
        self.new_sim_window.signal_new_title.connect(self.CreateSimulation)

    def UpdateCurrentNumber(self, cnum):
        """
        Updates the current number, in order to keep track of how many simulations were run.
        """
        self.current_number = int(cnum)

    def Sobre(self):
        """
        Opens the "About" window
        """
        # chamando a nova classe SobreWindow que cria uma nova janela
        # se nao colocar o self, garbage collection will remove that object as soon as setupUi
        # method finishes.
        self.Sobre = SobreWindow()
        # mostrando na tela a classe criada para a segunda janela
        self.Sobre.show()

    # Legacy functions #################################################################
    def Campo(self):
        """
        Função que aplica o campo elétrico à estrutura no momento que o usuário aperta o botão
        apply.
        """
        l = self.sim.self.sim.x_graf
        # trazendo o grafico para o centro da estrutura e colocando em nm
        l = l - l[-1] / 2  # / 1E-9
        self.sim.v_graf = (
            self.sim.v_graf - self.sim_Efield_spb.value() * 1.0e-4 * l
        )  # 1E5 * NM
        self.sim.e_field = self.sim.e_field + self.sim_Efield_spb.value()
        self.label_campo.setText(str(self.sim.e_field) + " kV/cm")
        # So altero o vetor posicao ao clicar a primeira vez (quando x0 = 0). Depois disso, o x ja
        # esta definido
        if self.sim.x0 == 0:
            self.sim.x0 = self.sim.x0 - np.sum(self.sim.estrutura) / 2
            # este valor vai ser usado no calculo quando usarmos a funcao run.

        v = self.sim.v_graf
        self.sim.self.sim.x_graf = l

        self.UpdateSimGraph()


class SobreWindow(QMainWindow):
    """
    About window
    """

    def __init__(self, parent=None):
        super(SobreWindow, self).__init__(parent)
        # uic.loadUi(os.path.join(os.getcwd(), "GUI", "Sobre_Gui.ui"), self)
        uic.loadUi(os.path.join(os.path.dirname(__file__), "GUI", "Sobre_Gui.ui"), self)
        self.webpage_btn.clicked.connect(self.webpage)
        self.Ok_btn.clicked.connect(self.close)

    def webpage(self):
        webbrowser.open("http://www.if.ufrj.br/~gpenello/")


class NewSimWindow(QWidget):
    """
    Window that allows the user to create a new simulation.
    """

    signal_new_title = pyqtSignal(str, str)
    signal_updated_current_number = pyqtSignal(int)

    def __init__(self, current_number, parent=None):
        super(NewSimWindow, self).__init__(parent)
        uic.loadUi(
            os.path.join(os.path.dirname(__file__), "GUI", "New_Simulation.ui"), self
        )

        self.ConnectSignals()
        # Load the information from the configuration file to the UI and updates other values
        self.InterfaceSetup(current_number)
        # Creates a timer to update the texts
        self.CreateInterfaceTimer()

    def InterfaceSetup(self, current_number):
        """
        Fills the interface with the updated values.
        """
        # Set the options in the comboboxes
        options = ["No", "Before Title", "After Title"]
        for option in options:
            self.title_number_cbox.addItem(option)
            self.title_date_cbox.addItem(option)
            self.title_time_cbox.addItem(option)
            self.title_materials_cbox.addItem(option)

        # Loads the available materias from the file
        self.mat = configparser.ConfigParser()
        self.mat.read(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "materials.data")
        )

        for pair in self.mat.sections():
            self.materials_cbox.addItem(pair)

        # Updates the current simulation number
        self.number_spb.setValue(current_number)

    def ConnectSignals(self):
        """
        Definition of the interaction between button clicks, signals and actions.
        """
        # Set the focus to the title line, so that the user can type right away
        self.title_line.setFocus()

        # The window might be closed in three ways:
        # pressing enter while the title_line is selected
        self.title_line.returnPressed.connect(self.SetTitle)
        # or pressing enter while it's selected
        self.create_simulation_btn.setDefault(True)
        # or by clicking the create_simulation button
        self.create_simulation_btn.clicked.connect(self.SetTitle)

        # When the user changes the simulation number, the main window gets a signal
        # If the value in the number spinbox was modified, update the self.current_number
        self.number_spb.valueChanged.connect(self.UpdatedCurrentNumber)

    def UpdatedCurrentNumber(self):
        self.current_number = self.number_spb.value()
        self.signal_updated_current_number.emit(self.number_spb.value())

    def CreateInterfaceTimer(self):
        """
        Creates a QTimer that regularly updates the interface, in order to keep track of the current
        time.
        """
        self.UpdateTexts()  # Runs the function that updates the title once just to keep the ui tidy
        self.txt_timer = QTimer()
        txt_update_interval = 200  # ms
        self.txt_timer.start(txt_update_interval)
        self.txt_timer.timeout.connect(self.UpdateTexts)

    def SetTitle(self):
        """
        When the user clicks the button or press Enter, this function will get the title, the
        current selection of materials and emit a signal with this information, so that the main
        window can create a new simulation.
        """
        title = self.ComposeTitle()
        materials = self.materials_cbox.currentText()
        self.signal_new_title.emit(title, materials)

    def UpdateTexts(self):
        """
        Keeps the info shown on the interface updated
        """
        self.number_spb.setValue(self.current_number)
        now = time.localtime()
        self.date_text.setText(time.strftime("%Y-%m-%d", now))
        self.time_text.setText(time.strftime("%Hh%Mm%Ss", now))
        self.materials_text.setText(self.materials_cbox.currentText())
        self.name_preview_text.setText(self.ComposeTitle())

    def ComposeTitle(self):
        """
        Function that reads the options defined on the interface and returns the title based on that
        The items are inserted in the order the comboboxes appear:
        Number Date Time Material Title Number Date Time Material
        """
        title = self.title_line.text()
        now = time.localtime()

        if self.title_materials_cbox.currentIndex() == 1:
            title = self.materials_cbox.currentText() + " " + title
        if self.title_time_cbox.currentIndex() == 1:
            title = time.strftime("%Hh%Mm%Ss", now) + " " + title
        if self.title_date_cbox.currentIndex() == 1:
            title = time.strftime("%Y-%m-%d", now) + " " + title
        if self.title_number_cbox.currentIndex() == 1:
            title = str(self.number_spb.value()) + " " + title

        # Just in case there are trailing spaces due to the title being empty
        title = title.rstrip()

        if self.title_materials_cbox.currentIndex() == 2:
            title += " " + self.materials_cbox.currentText()
        if self.title_time_cbox.currentIndex() == 2:
            title += " " + time.strftime("%Hh%Mm%Ss", now)
        if self.title_date_cbox.currentIndex() == 2:
            title += " " + time.strftime("%Y-%m-%d", now)
        if self.title_number_cbox.currentIndex() == 2:
            title += " " + str(self.number_spb.value())

        # Just in case there are leading spaces due to the title being empty
        title = title.lstrip()

        return title


class RenameSimWindow(QWidget):
    """
    Window that is shown when the user renames or copies a simulation
    """

    signal_renamed = pyqtSignal()

    def __init__(self, sim, parent=None):
        super(RenameSimWindow, self).__init__(parent)
        # uic.loadUi(os.path.join(os.getcwd(), "GUI", "Sobre_Gui.ui"), self)
        uic.loadUi(
            os.path.join(os.path.dirname(__file__), "GUI", "RenameSimulation.ui"), self
        )
        self.rename_btn.clicked.connect(lambda: self.Rename(sim))
        # Set the focus to the title line, so that the user can type right away
        self.title_line.setFocus()
        # The window might be closed in three ways:
        # pressing enter while the title_line is selected
        self.title_line.returnPressed.connect(lambda: self.Rename(sim))
        # or pressing enter while it's selected
        self.rename_btn.setDefault(True)
        # or by clicking the create_simulation button
        self.rename_btn.clicked.connect(lambda: self.Rename(sim))

        self.title_line.setText(sim.title)

    def Rename(self, sim):
        """
        Function called when the user clicks on the "rename" button. Renames the title of sim.
        """
        # Gets the string from the text box
        title = self.title_line.text()
        # Just in case there are leading spaces due to the title being empty
        sim.title = title.lstrip()
        self.signal_renamed.emit()


if __name__ == "__main__":
    # Create the GUI application
    app = QApplication(sys.argv)

    # Creating splash screen
    splash_pix = QPixmap(os.path.join("Imagens", "SplashScreen5.png"))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    time.sleep(0.2)
    splash.close()
    # app.processEvents()

    # instantiate the main window
    mw = MainWindow()
    # show it
    mw.show()
    # Revert changes to matplotlib rc params
    plt.rcParams.update(plt.rcParamsDefault)
    # start the Qt main loop execution, exiting from this script
    # with the same return code of Qt application
    sys.exit(app.exec_())
