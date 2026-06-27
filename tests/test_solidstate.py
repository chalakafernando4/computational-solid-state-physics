import numpy as np

from solidstate import (
    allowed_reflection,
    body_centered_cubic,
    brillouin_zone_2d,
    cubic_d_spacing,
    diamond_cubic,
    diatomic_chain_dispersion,
    effective_density_of_states,
    face_centered_cubic,
    intrinsic_carrier_concentration,
    monoatomic_chain_dispersion,
    polygon_area,
    simple_cubic,
    square_reciprocal_basis,
    square_lattice_dispersion,
)


def test_reciprocal_vectors_satisfy_duality() -> None:
    lattice = face_centered_cubic(a=2.0)
    product = lattice.primitive_vectors @ lattice.reciprocal_vectors.T
    np.testing.assert_allclose(product, 2.0 * np.pi * np.eye(3), atol=1.0e-12)


def test_cubic_primitive_cell_volumes() -> None:
    a = 3.0
    assert np.isclose(simple_cubic(a).volume, a**3)
    assert np.isclose(body_centered_cubic(a).volume, a**3 / 2.0)
    assert np.isclose(face_centered_cubic(a).volume, a**3 / 4.0)


def test_reflection_selection_rules() -> None:
    assert allowed_reflection("bcc", 1, 1, 0)
    assert not allowed_reflection("bcc", 1, 0, 0)
    assert allowed_reflection("fcc", 1, 1, 1)
    assert allowed_reflection("fcc", 2, 0, 0)
    assert not allowed_reflection("fcc", 2, 1, 0)


def test_phonon_zero_modes() -> None:
    k = np.array([0.0])
    assert np.isclose(monoatomic_chain_dispersion(k)[0], 0.0)
    acoustic, optical = diatomic_chain_dispersion(k, mass_1=1.0, mass_2=2.0)
    assert np.isclose(acoustic[0], 0.0)
    assert optical[0] > 0.0


def test_square_lattice_band_at_gamma() -> None:
    energy = square_lattice_dispersion(np.array([0.0]), np.array([0.0]), hopping=1.0)
    assert np.isclose(energy[0], -4.0)


def test_d_spacing_and_semiconductor_helpers() -> None:
    assert np.isclose(cubic_d_spacing(2.0, 1, 0, 0), 2.0)
    nc = effective_density_of_states(300.0, effective_mass_ratio=1.08)
    nv = effective_density_of_states(300.0, effective_mass_ratio=0.56)
    ni = intrinsic_carrier_concentration(1.12, 300.0, nc, nv)
    assert nc > 0.0
    assert nv > 0.0
    assert ni > 0.0


def test_square_brillouin_zone_area() -> None:
    basis = square_reciprocal_basis(a=1.0)
    polygon = brillouin_zone_2d(basis)
    assert len(polygon) == 4
    assert np.isclose(polygon_area(polygon), (2.0 * np.pi) ** 2)


def test_diamond_cubic_basis_points() -> None:
    lattice = diamond_cubic(a=1.0)
    unit_points = lattice.points(0, 0, include_basis=True)
    assert unit_points.shape == (2, 3)
