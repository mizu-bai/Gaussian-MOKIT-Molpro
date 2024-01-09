#!/usr/bin/env python3
import json
import os
import re
import sys
from dataclasses import dataclass
from io import StringIO
from typing import List

import numpy as np
import pandas as pd

# constants
ANG2BOHR = 1.8897259886

ELEMENTS = [
    "Bq",
    "H ",                                                                                                                                                                                     "He",
    "Li", "Be",                                                                                                                                                 "B ", "C ", "N ", "O ", "F ", "Ne",
    "Na", "Mg",                                                                                                                                                 "Al", "Si", "P ", "S ", "Cl", "Ar",
    "K ", "Ca", "Sc", "Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",                                                                                     "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y ", "Zr", "Nb", "Mo", "Te", "Ru", "Rh", "Pd", "Ag", "Cd",                                                                                     "In", "Sn", "Sb", "Te", "I ", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U ", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


@dataclass
class Config:
    g_nproc: int
    g_mem: str
    g_level: str
    m_nproc: int
    m_mem: str
    m_cmds: List[str]


def _load_settings():
    with open("gmm.json") as fo:
        settings = json.load(fo)

    config = Config(
        g_nproc=settings["Gaussian"]["nproc"],
        g_mem=settings["Gaussian"]["memory"],
        g_level=settings["Gaussian"]["level"],
        m_nproc=settings["Molpro"]["nproc"],
        m_mem=settings["Molpro"]["memory"],
        m_cmds=settings["Molpro"]["commands"]
    )

    return config


def _D2E(filename):
    with open(filename, "r") as f:
        contents = f.read()

    contents = re.sub("D", "E", contents)
    tmp_io = StringIO(contents)

    return tmp_io


if __name__ == "__main__":
    # load config json
    config = _load_settings()

    # prepare Gaussian ROHF calculation input file templete
    rohf_contents = (
        f"%nproc={config.g_nproc}\n"
        f"%mem={config.g_mem}\n"
        f"%chk=mol_rohf.chk\n"
        f"{config.g_level}\n"
        f"\n"
        "ROHF TASK\n"
        f"\n"
    )

    # parse Gaussian args
    (layer, InputFile, OutputFile, MsgFile, FChkFile, MatElFile) = sys.argv[1:]

    with open(InputFile, "r") as f:
        # parse atom
        (atoms, derivs, charge, spin) = [int(x) for x in f.readline().split()]

        # append to rohf contents
        rohf_contents += f"{charge} {spin}\n"

        for i in range(atoms):
            arr = f.readline().split()
            tmp = [ELEMENTS[int(arr[0])]]
            tmp += [f"{(float(x) / ANG2BOHR):13.8f}" for x in arr[1:4]]
            rohf_contents += "    ".join(tmp)
            rohf_contents += "\n"

        rohf_contents += "\n\n"

    curr_dir = os.getcwd()
    curr_dir = os.path.abspath(curr_dir)

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    os.chdir("tmp")

    # write rohf input file
    with open("mol_rohf.gjf", "w") as f:
        f.write(rohf_contents)

    print(">>> Starting ROHF Calculation...")

    os.system("g16 mol_rohf.gjf")
    os.system("formchk mol_rohf.chk")

    print(">>> ROHF Calculation Done!")

    print(">>> Preparing Molpro input file...")

    os.system("fch2com mol_rohf.fchk")
    os.system("mv mol_rohf.com mol_post.com")

    # add addtional molpro calculation procedures
    with open("mol_post.com", "a") as f:
        for cmd in config.m_cmds:
            f.write(f"{cmd}\n")

        f.writelines("{table,energy;save,energy.csv,new;}\n")

        if derivs == 1:
            f.writelines([
                "{force;varsav;}\n",
                "{table,gradx,grady,gradz;save,grad.csv,new;}\n",
            ])

        f.writelines("{table,dmx,dmy,dmz;save,dipole.csv,new;}\n")

    print(">>> Starting Molpro Calculation...")

    os.system(f"molpro -s -n {config.m_nproc} -m {config.m_mem} mol_post.com")

    print(">>> Molpro Calculation Done!")

    # extract molpro output file
    print(">>> Extracting Molpro output file ...")

    output_contents = []

    df_energy = pd.read_csv(_D2E("energy.csv"))
    energy = df_energy.values.tolist()[0][0]

    df_dipole = pd.read_csv(_D2E("dipole.csv"))
    dipole = df_dipole.values.tolist()[0]

    output_contents = []

    output_contents.append(
        f"{energy:20.12E}"
        f"{dipole[0]:20.12E}{dipole[1]:20.12E}{dipole[2]:20.12E}"
    )

    if derivs == 1:
        df_grad = pd.read_csv(_D2E("grad.csv"))
        grad = df_grad.values.tolist()
        for grad_on_atom in grad:
            output_contents.append(
                "".join([f"{g:20.12E}" for g in grad_on_atom])
            )

    output_contents = "\n".join(output_contents)

    # write output
    with open(OutputFile, "w") as f:
        f.write(output_contents)

    print(">>> Done!")
