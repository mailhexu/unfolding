This directory holds the ABINIT WFK outputs consumed by reproduce.py and
unfold.toml. No WFK ships with this bundle (the dense nsppol-2 path WFK
is large): generate it from inputs/nio_afm.abi with ABINIT >= 9 as
printed by `python reproduce.py`, then leave nio_afm_patho_DS2_WFK.nc
here. The corner deck (nio_afm_corners.abi) yields a small WFK that is
enough for a coarse figure.
