This directory holds the ABINIT WFK outputs consumed by reproduce.py and
unfold.toml. No WFK ships with this bundle (each dense path WFK is
hundreds of MB): generate them from inputs/ with ABINIT >= 9 as printed
by `python reproduce.py`, then leave the *_DS2_WFK.nc files here.

Corner WFKs (Gamma/X/W/L only, a few MB each) are enough to render
coarse figures; the dense 305-point decks reproduce the published maps.
