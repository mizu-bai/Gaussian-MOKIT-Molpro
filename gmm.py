#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import pandas as pd

# parse settings file
with open("gmm.json") as fo:
    settings = json.load(fo)

G_NPROC    = settings["Gaussian"]["nproc"]
G_MEMORY   = settings["Gaussian"]["memory"]
G_LEVEL    = settings["Gaussian"]["level"]

M_NPROC    = settings["Molpro"]["nproc"]
M_MEMORY   = settings["Molpro"]["memory"]
M_COMMANDS = settings["Molpro"]["commands"]

if not os.path.exists("tmp"):
    os.mkdir("tmp")

os.chdir("tmp")

# Gaussian ROHF calculation input file templete
ROHF_INP = f"""%nproc={G_NPROC}
%mem={G_MEMORY}
%chk=mol_rohf.chk
{G_LEVEL}

ROHF TASK

"""

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

(layer, InputFile, OutputFile, MsgFile, FChkFile, MatElFile) = sys.argv[1:]

atoms = []
coords = []
with open(InputFile, "r") as fi:
    (atoms, derivs, charge, spin) = [int(x) for x in fi.readline().split()]
    ROHF_INP += f"{charge} {spin}\n"
    for i in range(0, atoms):
        arr = fi.readline().split()
        atom = ELEMENTS[int(arr[0])]
        coord = [f"{(float(x) / ANG2BOHR)}" for x in arr[1:4]]
        ROHF_INP += f" {atom}"
        ROHF_INP += " " * 4
        ROHF_INP += (" " * 4).join(coord)
        ROHF_INP += "\n"

ROHF_INP += "\n"

with open("mol_rohf.gjf", "w") as fo:
    fo.write(ROHF_INP)

print(">>> Starting ROHF Calculation...")
os.system("g16 mol_rohf.gjf")
os.system("formchk mol_rohf.chk")
print(">>> ROHF Calculation Done!")

print(">>> Preparing Molpro input file...")
os.system("fch2com mol_rohf.fchk")
os.system("mv mol_rohf.com mol_post.com")

# add addtional molpro calculation procedures
for command in M_COMMANDS:
    os.system(f"echo '{command}' >> mol_post.com")

os.system("echo '{table,energy;save,energy.csv,new;}' >> mol_post.com")

if derivs == 1:
    os.system("echo '{force;varsav;}' >> mol_post.com")
    os.system("echo '{table,gradx,grady,gradz;save,grad.csv,new;}' >> mol_post.com")

# call molpro to do post hartree fock calculation
print(">>> Starting Molpro Calculation...")
os.system(f"molpro -s -n {M_NPROC} -m {M_MEMORY} mol_post.com")
print(">>> Molpro Calculation Done!")

# extract molpro output file
print(">>> Extracting Molpro output file ...")

with open(OutputFile, "w") as fo:
    os.system("sed -i 's/D/E/g' energy.csv")
    df_energy = pd.read_csv("energy.csv")
    energy = df_energy.values.tolist()[0][0]
    fo.writelines(f"{energy:20.12E}{0:20.12E}{0:20.12E}{0:20.12E}\n")
    if derivs == 1:
        os.system("sed -i 's/D/E/g' grad.csv")
        df_grad = pd.read_csv("grad.csv")
        grad = df_grad.values.tolist()
        np.savetxt(fo, grad, fmt="%20.12E", delimiter="")

print(">>> Done!")
