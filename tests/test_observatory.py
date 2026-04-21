import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from astropy import units as u

from pyEDITH.observatory import Observatory, ObservatoryMediator
from pyEDITH.components.telescopes import Telescope
from pyEDITH.components.detectors import Detector
from pyEDITH.components.coronagraphs import Coronagraph
from pyEDITH.observation import Observation
from pyEDITH.astrophysical_scene import AstrophysicalScene
from pyEDITH.units import (
    PHOTON_FLUX_DENSITY,
    DIMENSIONLESS,
    LENGTH,
    LAMBDA_D,
    ARCSEC,
    WAVELENGTH,
    TEMPERATURE,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    TIME,
    FRAME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    ELECTRON,
    PHOTON_COUNT,
    INV_SQUARE_ARCSEC,
    PIXEL,
    ZODI,
)


# ============================================================================
# Mock Component Classes
# ============================================================================


class MockTelescope(Telescope):
    """Mock telescope for testing."""

    def load_configuration(self, parameters, mediator):
        self.path = None
        self.keyword = "ToyModel"
        self.diameter = 7.87 * u.m
        self.unobscured_area = 0.879
        self.toverhead_fixed = 8381.3 * u.s
        self.toverhead_multi = 1.1 * DIMENSIONLESS
        self.telescope_optical_throughput = u.Quantity([0.823], DIMENSIONLESS)
        self.temperature = 290.0 * u.K
        self.T_contamination = 0.95 * DIMENSIONLESS
        self.Area = 42.75906827 * u.m**2


class MockDetector(Detector):
    """Mock detector for testing."""

    def load_configuration(self, parameters, mediator):
        self.path = None
        self.keyword = "ToyModel"
        self.pixscale_mas = 6.55224925 * u.mas
        self.npix_multiplier = u.Quantity([1.0], DIMENSIONLESS)
        self.DC = u.Quantity([3.0e-05], DARK_CURRENT)
        self.RN = u.Quantity([0.0], READ_NOISE)
        self.tread = u.Quantity([1000.0], READ_TIME)
        self.CIC = u.Quantity([0.0013], CLOCK_INDUCED_CHARGE)
        self.QE = u.Quantity([0.897], QUANTUM_EFFICIENCY)
        self.dQE = u.Quantity([0.75], DIMENSIONLESS)


class MockCoronagraph(Coronagraph):
    """Mock coronagraph for testing."""

    def load_configuration(self, parameters, mediator):
        self.path = None
        self.keyword = "ToyModel"
        self.pixscale = 30.0 * LAMBDA_D
        self.minimum_IWA = 1.0 * LAMBDA_D
        self.maximum_OWA = 60.0 * LAMBDA_D
        self.contrast = 1.05e-13 * DIMENSIONLESS
        self.noisefloor_factor = 0.03 * DIMENSIONLESS
        self.bandwidth = 0.2
        self.psf_trunc_ratio = 0.3 * DIMENSIONLESS
        self.photometric_aperture_radius = 0.7 * LAMBDA_D
        self.Tcore = 0.2968371 * DIMENSIONLESS
        self.TLyot = 0.65 * DIMENSIONLESS
        self.nrolls = 1
        self.nchannels = 2
        self.coronagraph_optical_throughput = u.Quantity([0.44], DIMENSIONLESS)
        self.coronagraph_spectral_resolution = 1.0 * DIMENSIONLESS
        self.npsfratios = 1
        self.npix = 4
        self.xcenter = 2.0 * PIXEL
        self.ycenter = 2.0 * PIXEL
        self.r = u.Quantity(
            [
                [63.63961031, 47.4341649, 47.4341649, 63.63961031],
                [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                [63.63961031, 47.4341649, 47.4341649, 63.63961031],
            ],
            LAMBDA_D,
        )
        self.omega_lod = u.Quantity(
            [
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
            ],
            LAMBDA_D**2,
        )
        self.skytrans = u.Quantity(
            [
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
            ],
            DIMENSIONLESS,
        )
        self.photometric_aperture_throughput = u.Quantity(
            [
                [[0.0], [0.2968371], [0.2968371], [0.0]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.0], [0.2968371], [0.2968371], [0.0]],
            ],
            DIMENSIONLESS,
        )
        self.PSFpeak = u.Quantity(0.01625, DIMENSIONLESS)
        self.Istar = u.Quantity(
            [
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            ],
            DIMENSIONLESS,
        )
        self.noisefloor = u.Quantity(
            [
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            ],
            DIMENSIONLESS,
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_observatory():
    """Fixture providing a mock observatory with components."""
    obs = Observatory()
    obs.telescope = MockTelescope()
    obs.detector = MockDetector()
    obs.coronagraph = MockCoronagraph()
    return obs


@pytest.fixture
def mock_observation():
    """Fixture providing a mock observation."""
    obs = Observation()
    obs.observing_mode = "IMAGER"
    obs.td_limit = 1.0e20 * u.s
    obs.wavelength = u.Quantity([0.5], u.micron)
    obs.SNR = u.Quantity([7], DIMENSIONLESS)
    obs.CRb_multiplier = 2
    obs.nlambd = 1
    obs.tp = 0.0 * u.s
    obs.exptime = u.Quantity([0.0], u.s)
    obs.fullsnr = u.Quantity([0.0], DIMENSIONLESS)
    return obs


@pytest.fixture
def mock_scene():
    """Fixture providing a mock astrophysical scene."""
    scene = AstrophysicalScene()
    scene.F0V = 10374.9964895 * u.photon / u.nm / u.s / u.cm**2
    scene.dist = 14.8 * u.pc
    scene.F0 = u.Quantity([12638.83670769], u.photon / u.nm / u.s / u.cm**2)
    scene.vmag = 5.84 * u.mag
    scene.mag = u.Quantity([6.189576], u.mag)
    scene.deltamag = u.Quantity([25.5], u.mag)
    scene.min_deltamag = 25 * u.mag
    scene.Fs_over_F0 = u.Quantity([0.00334326], DIMENSIONLESS)
    scene.Fp_over_Fs = u.Quantity([6.30957344e-11], DIMENSIONLESS)
    scene.Fp_min_over_Fs = 1.0e-10 * DIMENSIONLESS
    scene.stellar_angular_diameter_arcsec = 0.01 * ARCSEC
    scene.nzodis = 3 * ZODI
    scene.ra = 236.00757737 * u.deg
    scene.dec = 2.51516683 * u.deg
    scene.separation = 0.0628 * u.arcsec
    scene.xp = 0.0628 * u.arcsec
    scene.yp = 0.0 * u.arcsec
    scene.M_V = 4.98869142 * u.mag
    scene.Fzodi_list = (u.Quantity([6.11055505e-10], 1 / u.arcsec**2),)
    scene.Fexozodi_list = (u.Quantity([2.97724302e-09], 1 / u.arcsec**2),)
    scene.Fbinary_list = u.Quantity([0], DIMENSIONLESS)
    return scene


@pytest.fixture
def configured_mock_observatory(mock_observatory):
    """Fixture providing a mock observatory with loaded components."""
    mock_observatory.telescope.load_configuration({}, {})
    mock_observatory.coronagraph.load_configuration({}, {})
    mock_observatory.detector.load_configuration({}, {})
    return mock_observatory


# ============================================================================
# Tests for Observatory initialization
# ============================================================================


def test_observatory_init():
    """Test that Observatory initializes with None components."""
    obs = Observatory()

    assert obs.telescope is None
    assert obs.detector is None
    assert obs.coronagraph is None


# ============================================================================
# Tests for Observatory.validate_configuration
# ============================================================================


def test_observatory_validate_configuration_valid(configured_mock_observatory):
    """Test that validation passes with valid configuration."""
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    # Should not raise any exception
    configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_missing_attribute(
    configured_mock_observatory,
):
    """Test that missing attribute raises AttributeError."""
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS
    delattr(configured_mock_observatory, "optics_throughput")

    with pytest.raises(
        AttributeError, match="Observatory is missing attribute: optics_throughput"
    ):
        configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_not_quantity(configured_mock_observatory):
    """Test that non-Quantity attribute raises TypeError."""
    configured_mock_observatory.optics_throughput = 0.8
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    with pytest.raises(
        TypeError, match="Observatory attribute optics_throughput should be a Quantity"
    ):
        configured_mock_observatory.validate_configuration()


def test_observatory_validate_configuration_incorrect_units(
    configured_mock_observatory,
):
    """Test that incorrect units raise ValueError."""
    configured_mock_observatory.optics_throughput = [0.8] * u.meter
    configured_mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    configured_mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS

    with pytest.raises(
        ValueError, match="Observatory attribute optics_throughput has incorrect units"
    ):
        configured_mock_observatory.validate_configuration()


# ============================================================================
# Tests for Observatory.calculate_optics_throughput
# ============================================================================


def test_calculate_optics_throughput_with_t_optical(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test calculation of optics throughput with explicit T_optical parameter."""
    parameters = {"T_optical": [0.8], "observing_mode": "IMAGER"}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    assert configured_mock_observatory.optics_throughput.value == [0.8]


def test_calculate_optics_throughput_ifs_mode(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test calculation of optics throughput in IFS mode with IFS efficiency."""
    parameters = {"T_optical": [0.8], "observing_mode": "IFS", "IFS_eff": [0.9]}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    assert configured_mock_observatory.optics_throughput.value == [0.8 * 0.9]


def test_calculate_optics_throughput_from_components(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test calculation of optics throughput from component throughputs."""
    parameters = {"observing_mode": "IMAGER"}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    configured_mock_observatory.calculate_optics_throughput(parameters, mediator)

    expected = [
        configured_mock_observatory.telescope.telescope_optical_throughput.value[0]
        * configured_mock_observatory.coronagraph.coronagraph_optical_throughput.value[
            0
        ]
    ]
    assert configured_mock_observatory.optics_throughput.value == expected


# ============================================================================
# Tests for Observatory.calculate_warmemissivity_coldtransmission
# ============================================================================


def test_calculate_warmemissivity_coldtransmission_explicit(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test calculation with explicit epswarmTrcold parameter."""
    parameters = {"epswarmTrcold": 0.3}
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    configured_mock_observatory.calculate_warmemissivity_coldtransmission(
        parameters, mediator
    )

    assert configured_mock_observatory.epswarmTrcold.value == 0.3


def test_calculate_warmemissivity_coldtransmission_calculated(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test calculation derived from optics throughput."""
    parameters = {}
    configured_mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    configured_mock_observatory.calculate_warmemissivity_coldtransmission(
        parameters, mediator
    )

    assert configured_mock_observatory.epswarmTrcold.value == 1 - 0.8


# ============================================================================
# Tests for Observatory.calculate_total_throughput
# ============================================================================


def test_calculate_total_throughput(mock_observatory):
    """Test calculation of total system throughput."""
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.detector.dQE = [0.9] * DIMENSIONLESS
    mock_observatory.detector.QE = [0.9] * QUANTUM_EFFICIENCY
    mock_observatory.telescope.T_contamination = 0.95 * DIMENSIONLESS

    mock_observatory.calculate_total_throughput()

    expected = 0.8 * 0.9 * 0.9 * 0.95
    assert np.isclose(mock_observatory.total_throughput.value, expected)
    assert mock_observatory.total_throughput.unit == QUANTUM_EFFICIENCY


# ============================================================================
# Tests for Observatory.load_configuration
# ============================================================================


def test_observatory_load_configuration(mock_observatory, mock_observation, mock_scene):
    """Test loading complete observatory configuration."""
    parameters = {"observing_mode": "IMAGER", "T_optical": [0.8]}

    mock_observatory.load_configuration(parameters, mock_observation, mock_scene)

    assert mock_observatory.observing_mode == "IMAGER"
    assert mock_observatory.optics_throughput.value == [0.8]
    assert hasattr(mock_observatory, "epswarmTrcold")
    assert hasattr(mock_observatory, "total_throughput")


# ============================================================================
# Tests for ObservatoryMediator.get_telescope_parameter
# ============================================================================


def test_observatory_mediator_get_telescope_parameter(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test getting telescope parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    result = mediator.get_telescope_parameter("diameter")

    assert result == configured_mock_observatory.telescope.diameter


def test_observatory_mediator_get_telescope_parameter_nonexistent(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test getting non-existent telescope parameter returns None."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    result = mediator.get_telescope_parameter("nonexistent")

    assert result is None


# ============================================================================
# Tests for ObservatoryMediator.get_coronagraph_parameter
# ============================================================================


def test_observatory_mediator_get_coronagraph_parameter(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test getting coronagraph parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    result = mediator.get_coronagraph_parameter("contrast")

    assert result == configured_mock_observatory.coronagraph.contrast


# ============================================================================
# Tests for ObservatoryMediator.get_detector_parameter
# ============================================================================


def test_observatory_mediator_get_detector_parameter(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test getting detector parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    result = mediator.get_detector_parameter("pixscale_mas")

    assert result == configured_mock_observatory.detector.pixscale_mas


# ============================================================================
# Tests for ObservatoryMediator.get_observation_parameter
# ============================================================================


def test_observatory_mediator_get_observation_parameter(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test getting observation parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    result = mediator.get_observation_parameter("wavelength")

    assert result == mock_observation.wavelength


# ============================================================================
# Tests for ObservatoryMediator.get_scene_parameter
# ============================================================================


def test_observatory_mediator_get_scene_parameter(
    configured_mock_observatory, mock_observation, mock_scene
):
    """Test getting scene parameter through mediator."""
    mediator = ObservatoryMediator(
        configured_mock_observatory, mock_observation, mock_scene
    )

    result = mediator.get_scene_parameter("vmag")

    assert result == mock_scene.vmag
