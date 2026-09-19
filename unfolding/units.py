"""Unit-conversion constants, derived from CODATA-2018 exact SI definitions.

Vendored explicitly (rather than read from ase.units private names) so the
values cannot silently drift with an ase refactor:

    c = 299792458.0 m/s          (exact, SI definition of the metre)
    h = 6.62607015e-34 J s       (exact, SI definition of the kilogram)
    e = 1.602176634e-19 C        (exact, SI definition of the ampere)

The historical hardcodes (33.356, 8065.6) are reproduced to their stated
precision; these constants carry the full derived precision instead.
"""

_C = 299792458.0            # m/s
_H = 6.62607015e-34         # J s
_E = 1.602176634e-19        # C

# 1 THz = 1e12 s^-1;  wavenumber = f / (c in cm/s)
THZ_TO_CM = 1.0e12 / (_C * 100.0)

# 1 eV = _E J;  wavenumber = E / (h c in cm^-1)
EV_TO_CM = _E / (_H * _C * 100.0)

__all__ = ["THZ_TO_CM", "EV_TO_CM"]
