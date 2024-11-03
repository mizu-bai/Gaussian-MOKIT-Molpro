from pyscf import gto, scf, lib
from mokit.lib.py2fch_direct import fchk

lib.num_threads(4)

mol = gto.M()
mol.atom = '/home/jh-li/Developer/Gaussian-MOKIT-Molpro/example/HCN/tmp/mol.xyz'
mol.basis = 'cc-pVDZ'
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build(parse_arg=False)

mf = scf.ROHF(mol)
mf.max_memory = 8192  # MB
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

