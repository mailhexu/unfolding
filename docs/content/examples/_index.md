---
title: "Examples"
weight: 20
---

Each card links to a walkthrough with input settings and runnable
reproduction scripts. Some fixtures are generated locally instead of
bundled (large GPAW restarts and VASP's licensed POTCAR/WAVECAR).

## Phonons

{{< gallery >}}

{{< card title="Phonopy Cu phonons" link="/examples/phonopy-cu/" src="/images/phonopy_unfolded_band_structure.png" text="Unfold a 3×3×3 fcc Cu phonon calculation from FORCE_CONSTANTS." >}}

{{< card title="ABINIT DDB phonons" link="/examples/abinit-ddb/" src="/images/cu_fcc_unfolded.png" text="Phonon unfolding from ABINIT DDB files (Cu and CaTiO₃)." >}}

{{< /gallery >}}

## Magnons

{{< gallery >}}

{{< card title="TB2J magnons SrMnO3" link="/examples/tb2j-magnon-srmmo3/" src="/images/srmmo3_magnon_unfolded.png" text="Downfold TB2J magnon bands from the G-AFM cell onto the pseudo-cubic primitive cell." >}}

{{< /gallery >}}

## Electrons — LCAO

{{< gallery >}}

{{< card title="SIESTA Si bands" link="/examples/siesta-si/" src="/images/si_unfolded.png" text="Unfold an 8-atom Si supercell Hamiltonian onto the primitive path." >}}

{{< card title="SIESTA WFSX route" link="/examples/siesta-wfsx/" src="/images/si_wfsx_unfolded.png" text="Use SIESTA's own wavefunctions instead of diagonalizing the Hamiltonian." >}}

{{< card title="SIESTA Si:P dopant" link="/examples/siesta-p-doped/" src="/images/si_p_doped_unfolded.png" text="A substitutional defect turns selected weights fractional." >}}

{{< card title="SIESTA spinors" link="/examples/siesta-spinor/" src="/images/si_spinor_unfolded.png" text="Non-collinear (nspin=4) supercells unfold through the same pipeline." >}}

{{< card title="OpenMX Si and Si:P" link="/examples/openmx-si/" src="/images/openmx_si_unfolded.png" text="Unfold OpenMX LCAO supercell bands from scfout Hamiltonian and overlap data." >}}

{{< card title="GPAW LCAO Si and Si:P" link="/examples/gpaw-si/" src="/images/gpaw_si_unfolded.png" text="Unfold GPAW LCAO supercells using their finite-range orbital overlaps." >}}

{{< card title="ABACUS LCAO Si and Si:P" link="/examples/abacus-si/" src="/images/abacus_si_unfolded.png" text="Unfold ABACUS numerical-orbital Hamiltonians with overlap-aware weights." >}}

{{< /gallery >}}

## Electrons — plane waves

{{< gallery >}}

{{< card title="ABINIT WFK Si and Si:P" link="/examples/abinit-wfk/" src="/images/si8_abinit_unfolded.png" text="Planewave unfolding straight from an ETSF netCDF WFK; pristine Si validated against the primitive cell." >}}

{{< card title="ABINIT WFK AFM NiO" link="/examples/abinit-nio/" src="/images/nio_afm_unfolded.png" text="Type-II antiferromagnet unfolded per spin channel onto the 2-atom primitive cell." >}}

{{< card title="ABINIT PAW Si and Si:P" link="/examples/abinit-paw/" src="/images/abinit_paw_si_path.png" text="JTH PAW overlap augmentation on the dense GXWGLWX path; pristine binary, dopant fractional." >}}

{{< card title="VASP PAW bcc Fe" link="/examples/vasp-paw/" src="/images/vasp_fe_path.png" text="Licensed WAVECAR/POTCAR, 16-fold supercell along the dense primitive path." >}}

{{< card title="GPAW plane-wave Si and Si:P" link="/examples/gpaw-si-pw/" src="/images/gpaw_si_pw_unfolded.png" text="Unfold GPAW pseudo plane-wave coefficients, without PAW augmentation." >}}

{{< /gallery >}}

## Electrons — Wannier

{{< gallery >}}

{{< card title="Wannier90 synthetic t2g" link="/examples/wannier-sto/" src="/images/sto_nodefect.png" text="Unfold the bundled 2×2×2 tight-binding supercell and a one-site perturbation." >}}

{{< /gallery >}}
