#!/usr/bin/env python3
"""
Main script to parse Dynare + trends, solve core, then solve/filter augmented model.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from dynare_parser import DynareParser
from augmented_statespace import SimpleModelSolver, AugmentedStateSpace

def main():
    os.chdir(os.path.dirname(__file__))

    # 1) Parse and generate all core & trend code
    parser = DynareParser("qpm_simpl1_with_trends.dyn")
    ok = parser.parse(out_folder="model_files")
    if not ok:
        sys.exit("Parsing failed.")
    # generate trend jac and meas blocks
    parser.generate_trend_state_space(out_folder="model_files")

    # 2) Solve core model
    solver = SimpleModelSolver(
        "model_files/model_json.json",
        "model_files/jacobian_matrices.py",
        "model_files/model_structure.py"
    )
    solver.solve_model()
    if not solver.is_solved:
        sys.exit("Core solution failed.")

    # 3) Plot core IRFs
    for s in solver.shock_names:
        solver.plot_irf(s, shock_size=1.0, variables_to_plot=[
            "RES_RS","RES_L_GDP_GAP","RES_DLA_CPI","RS","L_GDP_GAP","DLA_CPI","RR_GAP"
        ], periods=40)

    # 4) Build augmented model
    aug = AugmentedStateSpace(solver, parser, out_folder="model_files")

    # 5) Plot augmented IRFs
    print("Aug shocks:", aug.aug_shock_labels)
    # core shock
    idx0 = aug.aug_shock_labels.index(solver.shock_names[0])
    aug.plot_irf(idx0, variables_to_plot=["L_GDP_GAP","L_GDP_OBS", "DLA_CPI", "RS"], periods=40)
    # trend shock
    if "SHK_G_TREND" in aug.aug_shock_labels:
        idx_tr = aug.aug_shock_labels.index("SHK_L_GDP_TREND")
        aug.plot_irf(idx_tr, variables_to_plot=["G_TREND","L_GDP_OBS", "DLA_CPI", "RS"], periods=40)

if __name__=="__main__":
    main()