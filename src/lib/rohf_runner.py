from abc import ABC, abstractmethod
import os
import shutil
import shlex
from typing import List, Literal

from dataclasses import dataclass
import subprocess
import numpy as np


_BASE_NAME = "mol_rohf"


@dataclass
class XYZMol:
    num_atoms: int
    title: str
    atom_name: List[str]
    xyz: np.ndarray  # Angstrom

    def __str__(self) -> str:
        xyz_contents = []
        xyz_contents.append(f"{self.num_atoms}")
        xyz_contents.append(f"{self.title}")

        for i in range(self.num_atoms):
            xyz_contents.append(
                f"{self.atom_name[i]:4s}"
                f"{self.xyz[i][0]:13.8f}"
                f"{self.xyz[i][1]:13.8f}"
                f"{self.xyz[i][2]:13.8f}"
            )

        return "\n".join(xyz_contents)

    @staticmethod
    def from_file(
        xyz_file: str,
    ) -> 'XYZMol':
        contents = open(xyz_file, "r").readlines()
        contents = [line.rstrip() for line in contents]

        num_atoms = int(contents.pop(0))
        title = contents.pop(0).rstrip()
        atom_name = []
        xyz = []

        for _ in range(num_atoms):
            line = contents.pop(0)
            arr = line.split()
            atom_name.append(arr[0])
            atom_xyz = [float(x) for x in arr[1:4]]
            xyz.append(atom_xyz)

        xyz = np.array(xyz)

        xyz_mol = XYZMol(
            num_atoms=num_atoms,
            title=title,
            atom_name=atom_name,
            xyz=xyz,
        )

        return xyz_mol


@dataclass
class ROHFRunner(ABC):
    num_proc: int   # number of process
    mem: str        # memory [size][unit]
    basis_set: str  # name of basis set
    charge: int     # charge
    spin: int       # spin, **2S+1**
    xyz: str        # path to xyz file
    workdir: str    # working directory

    @abstractmethod
    def run(self) -> None:
        pass


class GauROHFRunner(ROHFRunner):
    gau_exe: str  # path to Gaussian executable

    def __init__(self) -> None:
        if shutil.which("g16") and shutil.which("g09"):
            raise RuntimeError("Both g16 and g09 are found in $PATH!")

        if shutil.which("g16"):
            self.gau_exe = "g16"
        elif shutil.which("g09"):
            self.gau_exe = "g09"
        else:
            raise RuntimeError("No g16 or g09 found in $PATH!")

    def _prepare_gjf(self):
        xyz = os.path.abspath(self.xyz)
        xyz_mol = XYZMol.from_file(xyz)

        gjf = os.path.abspath(os.path.join(self.workdir, f"{_BASE_NAME}.gjf"))

        gjf_contents = \
            f"""%nproc={self.num_proc}
%mem={self.mem}
%chk={_BASE_NAME}.chk
#p rohf {self.basis_set} nosymm int=nobasistransform

Gaussian ROHF Runner

{self.charge} {self.spin}
"""

        with open(gjf, "w") as f_gjf:
            print(gjf_contents, end="", file=f_gjf)

            for (atom_symbol, atom_xyz) in zip(xyz_mol.atom_name, xyz_mol.xyz):
                print(
                    f"{atom_symbol:4s}"
                    f"{atom_xyz[0]:13.8f}"
                    f"{atom_xyz[1]:13.8f}"
                    f"{atom_xyz[2]:13.8f}",
                    file=f_gjf,
                )

            print("\n\n", file=f_gjf)

    def _exe(self):
        log_file = os.path.join(self.workdir, f"{_BASE_NAME}.log")
        with open(log_file, "w") as f_log:
            subprocess.run(
                args=shlex.split(f"{self.gau_exe} {_BASE_NAME}.gjf"),
                stdout=f_log,
                stderr=f_log,
                cwd=self.workdir,
            )

        subprocess.run(
            args=shlex.split(f"formchk {_BASE_NAME}.chk"),
            cwd=self.workdir,
        )

    def run(self) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        self._prepare_gjf()
        self._exe()


class PySCFROHFRunner(ROHFRunner):
    def _prepare_input(self):
        _mem = ""
        if "MB" in self.mem:
            _mem = self.mem
        elif "GB" in self.mem:
            _mem = f"{int(self.mem.replace('GB', '')) * 1024}"

        py = os.path.abspath(os.path.join(self.workdir, f"{_BASE_NAME}.py"))

        py_contents = \
            f"""from pyscf import gto, scf, lib
from mokit.lib.py2fch_direct import fchk

lib.num_threads({self.num_proc})

mol = gto.M()
mol.atom = '{os.path.abspath(self.xyz)}'
mol.basis = '{self.basis_set}'
mol.charge = {self.charge}
mol.spin = {self.spin - 1}
mol.verbose = 4
mol.build(parse_arg=False)

mf = scf.ROHF(mol)
mf.max_memory = {_mem}  # MB
mf.max_cycle = 200

old_e = mf.kernel()

if not mf.converged:
    mf = mf.newton()
    old_e = mf.kernel()

new_e = old_e + 2.0e-05

for i in range(10):
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.newton()
    if abs(new_e - old_e) < 1.0e-05:
        break  # cannot find lower solution

    old_e = new_e
else:
    raise RuntimeError("PySCF stable=opt failed after 10 attempts.")

# output fchk
fchk(mf, __file__.replace('.py', '.fchk'), density=True)
"""
        with open(py, "w") as f_py:
            print(py_contents, file=f_py)

    def _exe(self):
        log_file = os.path.join(self.workdir, f"{_BASE_NAME}.log")
        with open(log_file, "w") as f_log:
            subprocess.run(
                args=shlex.split(
                    f"python3 -u {_BASE_NAME}.py"
                ),
                stdout=f_log,
                stderr=f_log,
                cwd=self.workdir,
            )

    def run(self) -> None:
        self._prepare_input()
        self._exe()


class ROHFRunnerFactory():
    _runner_dict = {
        "gau": GauROHFRunner,
        "pyscf": PySCFROHFRunner,
    }

    @classmethod
    def get(
        cls,
        runner_key: Literal["gau", "pyscf"]
    ) -> ROHFRunner:
        return cls._runner_dict[runner_key]


if __name__ == "__main__":
    gau_runner = GauROHFRunner(
        num_proc=8,
        mem="8GB",
        basis_set="aug-cc-pVTZ",
        charge=0,
        spin=1,
        xyz="test.xyz",
        workdir="tmp",
    )

    gau_runner.run()

    pyscf_runner = PySCFROHFRunner(
        num_proc=8,
        mem="8GB",
        basis_set="aug-cc-pVTZ",
        charge=0,
        spin=1,
        xyz="test.xyz",
        workdir="tmp",
    )

    pyscf_runner.run()
