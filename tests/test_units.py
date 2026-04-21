import pytest
from astropy import units as u
from pyEDITH.units import (
    lambda_d_to_radians,
    radians_to_lambda_d,
    lambda_d_to_arcsec,
    arcsec_to_lambda_d,
    arcsec_to_au,
    LAMBDA_D,
    LENGTH,
    ARCSEC,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def standard_wavelength():
    """Fixture providing standard wavelength of 500 nm for testing."""
    return 500 * u.nm


@pytest.fixture
def standard_diameter():
    """Fixture providing standard telescope diameter of 10 m for testing."""
    return 10 * u.m


@pytest.fixture
def standard_distance():
    """Fixture providing standard distance of 10 parsecs for testing."""
    return 10 * u.pc


# ============================================================================
# Tests for lambda_d_to_radians
# ============================================================================


def test_lambda_d_to_radians_conversion(standard_wavelength, standard_diameter):
    """Test conversion from λ/D to radians with standard parameters."""
    value_lod = 2 * LAMBDA_D

    result = lambda_d_to_radians(value_lod, standard_wavelength, standard_diameter)
    expected = 1e-7 * u.rad

    assert pytest.approx(result.value) == expected.value


def test_lambda_d_to_radians_correct_units(standard_wavelength, standard_diameter):
    """Test that lambda_d_to_radians returns angle in radians."""
    value_lod = 2 * LAMBDA_D

    result = lambda_d_to_radians(value_lod, standard_wavelength, standard_diameter)

    assert result.unit == u.rad


# ============================================================================
# Tests for radians_to_lambda_d
# ============================================================================


def test_radians_to_lambda_d_conversion(standard_wavelength, standard_diameter):
    """Test conversion from radians to λ/D with standard parameters."""
    angle = 1e-7 * u.rad

    result = radians_to_lambda_d(angle, standard_wavelength, standard_diameter)
    expected = 2 * LAMBDA_D

    assert pytest.approx(result.value) == expected.value


def test_radians_to_lambda_d_correct_units(standard_wavelength, standard_diameter):
    """Test that radians_to_lambda_d returns dimensionless λ/D units."""
    angle = 1e-7 * u.rad

    result = radians_to_lambda_d(angle, standard_wavelength, standard_diameter)

    assert result.unit == LAMBDA_D


# ============================================================================
# Tests for lambda_d_to_arcsec
# ============================================================================


def test_lambda_d_to_arcsec_conversion(standard_wavelength, standard_diameter):
    """Test conversion from λ/D to arcseconds with standard parameters."""
    value_lod = 2 * LAMBDA_D

    result = lambda_d_to_arcsec(value_lod, standard_wavelength, standard_diameter)
    expected = 0.02062648 * ARCSEC

    assert pytest.approx(result.value) == expected.value


def test_lambda_d_to_arcsec_correct_units(standard_wavelength, standard_diameter):
    """Test that lambda_d_to_arcsec returns angle in arcseconds."""
    value_lod = 2 * LAMBDA_D

    result = lambda_d_to_arcsec(value_lod, standard_wavelength, standard_diameter)

    assert result.unit == ARCSEC


# ============================================================================
# Tests for arcsec_to_lambda_d
# ============================================================================


def test_arcsec_to_lambda_d_conversion(standard_wavelength, standard_diameter):
    """Test conversion from arcseconds to λ/D with standard parameters."""
    angle = 0.02062648 * ARCSEC

    result = arcsec_to_lambda_d(angle, standard_wavelength, standard_diameter)
    expected = 2 * LAMBDA_D

    assert pytest.approx(result.value) == expected.value


def test_arcsec_to_lambda_d_correct_units(standard_wavelength, standard_diameter):
    """Test that arcsec_to_lambda_d returns dimensionless λ/D units."""
    angle = 0.02062648 * ARCSEC

    result = arcsec_to_lambda_d(angle, standard_wavelength, standard_diameter)

    assert result.unit == LAMBDA_D


# ============================================================================
# Tests for arcsec_to_au
# ============================================================================


def test_arcsec_to_au_conversion_at_parsec(standard_distance):
    """Test conversion from arcseconds to AU using definition of parsec."""
    angle = 0.1 * ARCSEC

    result = arcsec_to_au(angle, standard_distance)
    expected = 1 * u.au

    assert pytest.approx(result.value) == expected.value


def test_arcsec_to_au_correct_units(standard_distance):
    """Test that arcsec_to_au returns distance in AU."""
    angle = 0.1 * ARCSEC

    result = arcsec_to_au(angle, standard_distance)

    assert result.unit == u.au


# ============================================================================
# Tests for round-trip conversions
# ============================================================================


def test_lambda_d_radians_round_trip(standard_wavelength, standard_diameter):
    """Test that λ/D → radians → λ/D conversion is reversible."""
    original = 2 * LAMBDA_D

    radians = lambda_d_to_radians(original, standard_wavelength, standard_diameter)
    result = radians_to_lambda_d(radians, standard_wavelength, standard_diameter)

    assert pytest.approx(result.value) == original.value


def test_lambda_d_arcsec_round_trip(standard_wavelength, standard_diameter):
    """Test that λ/D → arcsec → λ/D conversion is reversible."""
    original = 2 * LAMBDA_D

    arcsec = lambda_d_to_arcsec(original, standard_wavelength, standard_diameter)
    result = arcsec_to_lambda_d(arcsec, standard_wavelength, standard_diameter)

    assert pytest.approx(result.value) == original.value


# ============================================================================
# Tests for edge cases
# ============================================================================


def test_lambda_d_to_radians_zero_value(standard_wavelength, standard_diameter):
    """Test conversion of zero λ/D value to radians."""
    value_lod = 0 * LAMBDA_D

    result = lambda_d_to_radians(value_lod, standard_wavelength, standard_diameter)

    assert result.value == 0
    assert result.unit == u.rad


def test_lambda_d_to_arcsec_zero_value(standard_wavelength, standard_diameter):
    """Test conversion of zero λ/D value to arcseconds."""
    value_lod = 0 * LAMBDA_D

    result = lambda_d_to_arcsec(value_lod, standard_wavelength, standard_diameter)

    assert result.value == 0
    assert result.unit == ARCSEC


def test_arcsec_to_au_zero_angle(standard_distance):
    """Test conversion of zero arcsecond angle to AU."""
    angle = 0 * ARCSEC

    result = arcsec_to_au(angle, standard_distance)

    assert result.value == 0
    assert result.unit == u.au


# ============================================================================
# Tests for different wavelengths
# ============================================================================


def test_lambda_d_to_radians_different_wavelength(standard_diameter):
    """Test that conversion scales correctly with wavelength changes."""
    wavelength1 = 500 * u.nm
    wavelength2 = 1000 * u.nm  # Double the wavelength
    value_lod = 2 * LAMBDA_D

    result1 = lambda_d_to_radians(value_lod, wavelength1, standard_diameter)
    result2 = lambda_d_to_radians(value_lod, wavelength2, standard_diameter)

    # Doubling wavelength should double the angle
    assert pytest.approx(result2.value) == 2 * result1.value


# ============================================================================
# Tests for different telescope diameters
# ============================================================================


def test_lambda_d_to_radians_different_diameter(standard_wavelength):
    """Test that conversion scales correctly with diameter changes."""
    diameter1 = 10 * u.m
    diameter2 = 20 * u.m  # Double the diameter
    value_lod = 2 * LAMBDA_D

    result1 = lambda_d_to_radians(value_lod, standard_wavelength, diameter1)
    result2 = lambda_d_to_radians(value_lod, standard_wavelength, diameter2)

    # Doubling diameter should halve the angle
    assert pytest.approx(result2.value) == 0.5 * result1.value


# ============================================================================
# Tests for different distances
# ============================================================================


def test_arcsec_to_au_different_distance():
    """Test that conversion scales correctly with distance changes."""
    angle = 0.1 * ARCSEC
    distance1 = 10 * u.pc
    distance2 = 20 * u.pc  # Double the distance

    result1 = arcsec_to_au(angle, distance1)
    result2 = arcsec_to_au(angle, distance2)

    # Doubling distance should double the physical separation
    assert pytest.approx(result2.value) == 2 * result1.value
