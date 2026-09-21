---
title: "Examples"
weight: 20
---

# Example gallery

Each card links to a full walkthrough (inputs, code, and how to read the
figure) on its own page.

<style>
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.2rem;
  margin-top: 1rem;
}
.gallery-card {
  border: 1px solid rgba(134,134,134,.25);
  border-radius: .4rem;
  padding: .8rem;
  display: flex;
  flex-direction: column;
  gap: .5rem;
  background: rgba(134,134,134,.05);
}
.gallery-card h3 { margin: 0; font-size: 1.05rem; }
.gallery-card p { margin: 0; font-size: .85rem; opacity: .8; }
.gallery-card figure { margin: 0; }
.gallery-card figcaption { font-size: .75rem; opacity: .7; }
</style>

<div class="gallery-grid" markdown="1">

<div class="gallery-card" markdown="1">

### [Phonopy Cu phonons](phonopy-cu/)

Unfold a 3×3×3 fcc Cu phonon calculation from FORCE_CONSTANTS.

{{< figure src="/images/phonopy_unfolded_band_structure.png" link="phonopy-cu/" title="Cu phonons unfolded onto the primitive fcc path" >}}

</div>

<div class="gallery-card" markdown="1">

### [SIESTA Si bands](siesta-si/)

Unfold an 8-atom Si supercell Hamiltonian onto the primitive path.

{{< figure src="/images/si_unfolded.png" link="siesta-si/" title="Si unfolded bands with primitive reference overlay" >}}

</div>

<div class="gallery-card" markdown="1">

### [SIESTA WFSX route](siesta-wfsx/)

Use SIESTA's own wavefunctions instead of diagonalizing the Hamiltonian.

{{< figure src="/images/si_wfsx_unfolded.png" link="siesta-wfsx/" title="Same Si spectrum from stored WFSX coefficients" >}}

</div>

<div class="gallery-card" markdown="1">

### [SIESTA Si:P dopant](siesta-p-doped/)

A substitutional defect turns selected weights fractional.

{{< figure src="/images/si_p_doped_unfolded.png" link="siesta-p-doped/" title="Host bands stay at weight 1; impurity states drop below" >}}

</div>

<div class="gallery-card" markdown="1">

### [SIESTA spinors](siesta-spinor/)

Non-collinear (nspin=4) supercells unfold through the same pipeline.

{{< figure src="/images/si_spinor_unfolded.png" link="siesta-spinor/" title="Spinor bands with Kramers degeneracy" >}}

</div>

<div class="gallery-card" markdown="1">

### [ABINIT WFK Si:P](abinit-wfk/)

Planewave unfolding straight from an ETSF netCDF WFK.

{{< figure src="/images/si7p_abinit_unfolded.png" link="abinit-wfk/" title="Si:P unfolded with fractional donor-window weights highlighted" >}}

</div>

<div class="gallery-card" markdown="1">

### [ABINIT DDB phonons](abinit-ddb/)

Phonon unfolding from ABINIT DDB files (Cu and CaTiO₃).

{{< figure src="/images/cu_fcc_unfolded.png" link="abinit-ddb/" title="Cu phonons from a DDB, binary pristine weights" >}}

</div>

<div class="gallery-card" markdown="1">

### [Wannier90 SrTiO₃](wannier-sto/)

Unfold Wannier90 tight-binding supercells (pristine and defect).

{{< figure src="/images/sto_nodefect.png" link="wannier-sto/" title="Pristine SrTiO₃ unfolded from a Wannier90 model" >}}

</div>

</div>
