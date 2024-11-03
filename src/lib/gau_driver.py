import sys
from typing import Optional

import numpy as np

from .data import ANG2BOHR, PERIODIC_TABLE

from typing import List


class GauDriver:
    # stdio
    layer: str
    input_file: str
    output_file: str
    msg_file: str
    fchk_file: str
    mat_el_file: str

    # Gaussian input data
    natom: int
    derivs: int
    charge: int
    multiplicity: int

    # external infomation
    do_gradient: bool = False
    do_hessian: bool = False

    # molecular infomation
    sysbols: List[str]
    atoms_nuc: List[int]
    coords: np.ndarray

    @staticmethod
    def from_stdio() -> "GauDriver":
        (layer, InputFile, OutputFile, MsgFile,
            FChkFile, MatElFile) = sys.argv[1:]
        return GauDriver(layer, InputFile, OutputFile, MsgFile,
                         FChkFile, MatElFile)

    def __init__(
        self,
        layer: str,
        input_file: str,
        output_file: str,
        msg_file: str,
        fchk_file: str,
        mat_el_file: str,
    ) -> None:
        self.layer = layer
        self.input_file = input_file
        self.output_file = output_file
        self.msg_file = msg_file
        self.fchk_file = fchk_file
        self.mat_el_file = mat_el_file

        self.coords: List[List[float]] = []
        self.atoms_nuc: List[int] = []
        self.symbols: List[str] = []

        with open(self.input_file, "r") as f:
            (natom, derivs, charge, spin) = \
                [int(x) for x in f.readline().split()]

            self.natom = natom
            self.derivs = derivs
            self.charge = charge
            self.multiplicity = spin
            for _ in range(self.natom):
                arr = f.readline().split()
                nuc = int(arr[0])
                self.atoms_nuc.append(nuc)
                self.symbols.append(PERIODIC_TABLE[nuc])
                coord = [(float(x) / ANG2BOHR) for x in arr[1:4]]
                self.coords.append(coord)

        if self.derivs in [1, 2]:
            self.do_gradient = True
        if self.derivs == 2:
            self.do_hessian = True

        self.coords = np.array(self.coords)

    def write(
        self,
        energy: float,
        gradients: Optional[np.ndarray],
        force_constants: Optional[np.ndarray],
    ) -> None:
        contents = []

        # energy and dipole-monent
        line = f"{energy:20.12E}" + f"{0:20.12E}" * 3
        contents.append(line)

        # gradient on atom
        if self.derivs in [1, 2]:
            for gradient_on_atom in gradients:
                line = "".join([f"{g:20.12E}" for g in gradient_on_atom])
                contents.append(line)

        # polarizability, dipole derivatives, & force constants
        if self.derivs == 2:
            empty_line = f"{0:20.12E}" * 3

            # polarizability
            for _ in range(2):
                contents.append(empty_line)

            # dipole derivatives
            for _ in range(3 * self.natom):
                contents.append(empty_line)

            # force constants
            for component in force_constants.reshape((-1, 3)):
                line = "".join([f"{c:20.12E}" for c in component])
                contents.append(line)

        contents = "\n".join(contents)

        with open(self.output_file, "w") as f:
            f.write(contents)

    def xyz(self, comment: str = "") -> str:
        xyz_str = f"{self.natom}\n{comment}\n"
        for sym, coord in zip(self.symbols, self.coords):
            xyz_str += f"  {sym} {coord[0]:>23.17e} {coord[1]:>23.17e} {coord[2]:>23.17e}\n"
        return xyz_str

    def pyscf_atom(self) -> str:
        atom_str = ""
        for nuc, coord in zip(self.atoms_nuc, self.coords):
            atom_str += f"{nuc} {coord[0]:>23.17e} {coord[1]:>23.17e} {coord[2]:>23.17e};"
        return atom_str
