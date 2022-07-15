"""
Auxiliary script to offload interface configuration from e-mulate.py
Fills the information on the GUI with units, default values, materials...
It's possible to also configure more complex properties of the interface, such
as minimum steps, maximum and minimum values, prefix, suffix... some of these
values are currently defined in the qt creator.
Whenever a new property is added to the interface and this function is modified
to load its default value, it is necessary to manually define its default value
in interface.cfg. If this function is called before a default value is present
in interface.cfg, it crashes because there is no value to read from the file.
        
"""

import configparser
import os


def run(mw):
    """
    Base function responsible to call the specific functions for each tab or part.
    input parameters:
    mw -> instance of the main window, which contains all the qt widgets that must be
          adjusted using this script.
    """
    cfg = read_config_file(mw.base_path)

    base_window_cfg(mw, cfg)
    structure_tab_cfg(mw, cfg)


def read_config_file(path):
    """
    Reads the configuration defaults from the "interface.cfg" file.
    """
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(path, "interface.cfg"))
    return cfg


def base_window_cfg(mw, cfg):
    """
    Configuration of the items outside the main window tabs.
    """
    # Lower bar
    mw.output_folder_line.setText(mw.base_path)


def structure_tab_cfg(mw, cfg):
    """
    Configuration of the "Structure" tab on the main window.
    """
    mw.barrier_nm_spb.setEnabled(False)
    mw.barrier_ml_spb.setEnabled(False)
    mw.well_nm_spb.setEnabled(False)
    mw.well_ml_spb.setEnabled(False)
    mw.barrier_nm_spb.setValue(cfg["stru"].getfloat("barrier_nm"))
    mw.barrier_ml_spb.setValue(cfg["stru"].getfloat("barrier_ml"))
    mw.well_nm_spb.setValue(cfg["stru"].getfloat("well_nm"))
    mw.well_ml_spb.setValue(cfg["stru"].getfloat("well_ml"))
    # sets the value of the spinboxes' steps
    mw.barrier_nm_spb.setSingleStep(0.1)
    mw.barrier_ml_spb.setSingleStep(1)
    mw.well_nm_spb.setSingleStep(0.1)
    mw.well_ml_spb.setSingleStep(1)


def simulation_tab_cfg(mw, cfg):
    """
    Configuration of the "Simulation" tab on the main window.
    """
    mw.sim_E0_spb.setValue(cfg["simu"].getfloat("E0"))
    mw.sim_dE_spb.setValue(cfg["simu"].getfloat("dE"))
    mw.sim_Ef_spb.setValue(cfg["simu"].getfloat("Ef"))
    mw.sim_dx_nm_spb.setValue(cfg["simu"].getfloat("dx_nm"))
    mw.sim_dx_ml_spb.setValue(cfg["simu"].getfloat("dx_ml"))
    mw.sim_Efield_spb.setValue(cfg["simu"].getfloat("Efield"))
    mw.sim_central_layer_spb.setValue(cfg["simu"].getint("central_layer"))
    # mw.sim_title_line.setText("Simulation")
    # Checkboxes
    mw.sim_files_all_chkbx.setChecked(cfg["simu"].getboolean("output_all"))
    mw.sim_files_effm_chkbx.setChecked(cfg["simu"].getboolean("output_effm"))
    mw.sim_files_energies_chkbx.setChecked(cfg["simu"].getboolean("output_energies"))
    mw.sim_files_npe_chkbx.setChecked(cfg["simu"].getboolean("output_npe"))
    mw.sim_files_pot_chkbx.setChecked(cfg["simu"].getboolean("output_pot"))
    mw.sim_files_trans_chkbx.setChecked(cfg["simu"].getboolean("output_trans"))
    mw.sim_files_wf_chkbx.setChecked(cfg["simu"].getboolean("output_wf"))
    mw.sim_files_x_chkbx.setChecked(cfg["simu"].getboolean("output_x"))
    mw.sim_autorun_abs_chkbx.setChecked(cfg["simu"].getboolean("autorun_abs"))
    mw.sim_autorun_tra_chkbx.setChecked(cfg["simu"].getboolean("autorun_tra"))
    mw.sim_autorun_pc_chkbx.setChecked(cfg["simu"].getboolean("autorun_pc"))
    # Progress bar
    progress(mw, 55)


def absorption_tab_cfg(mw, cfg):
    """
    Configuration of the "Absorption" tab on the main window.
    """
    mw.abs_init_wf_spb.setValue(cfg["abso"].getfloat("initial_WF"))
    mw.abs_broadening_spb.setValue(cfg["abso"].getfloat("lorz_broad"))
    mw.abs_E0_spb.setValue(cfg["abso"].getfloat("E0"))
    mw.abs_dE_spb.setValue(cfg["abso"].getfloat("dE"))
    mw.abs_Ef_spb.setValue(cfg["abso"].getfloat("Ef"))


def photocurrent_tab_cfg(mw, cfg):
    """
    Configuration of the "Photocurrent" tab on the main window.
    """
    mw.pc_E0_spb.setValue(cfg["phot"].getfloat("E0"))
    mw.pc_Ef_spb.setValue(cfg["phot"].getfloat("Ef"))
    mw.pc_dE_spb.setValue(cfg["phot"].getfloat("dE"))


def genetic_algorithm_tab_cfg(mw, cfg):
    """
    Configuration of the "Genetic Algorithm" tab on the main window.
    """
    mw.ga_iter_spb.setValue(cfg["gene"].getint("iterations"))
    mw.ga_pop_spb.setValue(cfg["gene"].getint("population"))
    mw.ga_tgt_en_spb.setValue(cfg["gene"].getfloat("target_E"))
    mw.ga_tgt_en_margin_spb.setValue(cfg["gene"].getfloat("target_E_margin"))
    priorities = ["Energy", "Oscillator Strength"]
    mw.ga_goal_cbox.clear()  # Clearing the combobox
    for p in priorities:
        mw.ga_goal_cbox.addItem(p)  # Filling the combobox
    mw.ga_goal_cbox.setCurrentIndex(cfg["gene"].getint("goal"))


def automation_tab_cfg(mw, cfg):
    """
    Configuration of the "Automation" tab on the main window.
    """
    mw.auto_layer_spb.setValue(cfg["auto"].getint("target_layer"))
    mw.auto_init_spb.setValue(cfg["auto"].getfloat("thickness_initial"))
    mw.auto_step_spb.setValue(cfg["auto"].getfloat("thickness_step"))
    mw.auto_final_spb.setValue(cfg["auto"].getfloat("thickness_final"))


def advanced_tab_cfg(mw, cfg):
    """
    Configuration of the "Advanced" tab on the main window.
    """
    mw.adv_new_folder_chkbx.setChecked(cfg["adva"].getboolean("create_new_folder"))
    mw.adv_nm_layers_chkbx.setChecked(cfg["adva"].getboolean("use_nanometers"))
    mw.adv_split_layer_spb.setValue(cfg["adva"].getint("interface_to_split"))
    methods = ["Numerov", "Numerov - For", "Numerov - Arrays", "TMM", "TMM - Split"]
    mw.adv_method_cbox.clear()  # Clearing the combobox
    for m in methods:
        mw.adv_method_cbox.addItem(m)  # Filling the combobox
    mw.adv_method_cbox.setCurrentIndex(cfg["adva"].getint("method"))


def progress(mw, val=None):
    """
    Adjusts the value of the progress bar.
    mw -> Main window class
    val -> target value of the progress bar
    """
    if val == None:
        val = 0
    # Progress bar
    mw.sim_progressBar.setValue(val)
