#!/usr/bin/env python3
import json
import os
import re
from dataclasses import dataclass
from io import StringIO
from typing import List

import pandas as pd

import subprocess
from lib import GauDriver, ROHFRunnerFactory
import shlex
import shutil


@dataclass
class Config:
    num_proc: int
    mem: str
    basis_set: str
    rohf_runner: str
    post_hf: List[str]

    @classmethod
    def from_json(cls) -> "Config":
        with open("gmm.json") as f:
            json_data = json.load(f)

        return cls(**json_data)


def _D2E(filename):
    with open(filename, "r") as f:
        contents = f.read()

    contents = re.sub("D", "E", contents)
    tmp_io = StringIO(contents)

    return tmp_io


if __name__ == "__main__":
    # load config json
    config = Config.from_json()

    # parse Gaussian args
    gau_driver = GauDriver.from_stdio()

    # prepare workdir
    workdir = os.path.abspath("tmp")

    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    os.makedirs(workdir)

    # prepare xyz file
    xyz_file = os.path.abspath(os.path.join(workdir, "mol.xyz"))
    with open(xyz_file, "w") as f:
        print(gau_driver.xyz(), file=f)

    # perform ROHF calculation
    ROHFRunner = ROHFRunnerFactory.get(config.rohf_runner)

    rohf_runner = ROHFRunner(
        num_proc=config.num_proc,
        mem=config.mem,
        basis_set=config.basis_set,
        charge=gau_driver.charge,
        spin=gau_driver.multiplicity,
        xyz=xyz_file,
        workdir=workdir,
    )

    rohf_runner.run()

    # prepare molpro input file
    subprocess.run(
        args=shlex.split("fch2com mol_rohf.fchk"),
        cwd=workdir,
    )

    os.rename(
        os.path.join(workdir, "mol_rohf.com"),
        os.path.join(workdir, "mol_post.com"),
    )

    with open(os.path.join(workdir, "mol_post.com"), "a") as f_com:
        for cmd in config.post_hf:
            print(cmd, file=f_com)

        if gau_driver.derivs == 1:
            print("{force;varsav;}", file=f_com)

        print("{table,energy;save,energy.csv,new;}", file=f_com)

        if gau_driver.derivs == 1:
            print(
                "{table,gradx,grady,gradz;save,grad.csv,new;}",
                file=f_com,
            )

    # perform molpro calculation
    mem_mw = ""

    if "GB" in config.mem:
        mem_mb = int(config.mem.replace("GB", "")) * 1024
    elif "MB" in config.mem:
        mem_mb = int(config.mem.replace("MB", ""))
    else:
        raise ValueError(f"Unsupported memory setting {config.mem}")

    mem_mw = int(mem_mb / (8 * config.num_proc))

    subprocess.run(
        args=shlex.split(
            f"molpro -s -n {config.num_proc} -t 1 -m {mem_mw}m mol_post.com"
        ),
        stderr=subprocess.STDOUT,
        cwd=workdir,
    )

    # parse energy and gradients from molpro output

    with open(gau_driver.output_file, "w") as f:
        energy_csv = os.path.join(workdir, "energy.csv")
        energy = pd.read_csv(_D2E(energy_csv)).values[0, 0]

        grad = None

        if gau_driver.derivs == 1:
            grad_csv = os.path.join(workdir, "grad.csv")
            grad = pd.read_csv(_D2E(grad_csv)).values

        gau_driver.write(
            energy=energy,
            gradients=grad,
            force_constants=None,
        )
