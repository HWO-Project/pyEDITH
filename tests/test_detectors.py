import pytest
import numpy as np
from astropy import units as u
from astropy import constants as const
from unittest.mock import patch, MagicMock
from pyEDITH.components.detectors import ToyModelDetector, EACDetector
from pyEDITH.units import (
    MAS,
    DIMENSIONLESS,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    WAVELENGTH,
    LENGTH,
    ARCSEC,
    SECOND,
    FRAME,
)


# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator:
    """Mock mediator for testing detector configurations."""

    def __init__(self, observing_mode="IMAGER"):
        self.observing_mode = observing_mode

    def get_scene_parameter(self, param):
        if param == "stellar_radius":
            return 1 * const.R_sun
        return 1.0

    def get_telescope_parameter(self, param):
        if param == "diameter":
            return 8.0 * LENGTH
        return 1.0

    def get_observation_parameter(self, param):
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.2]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.5]) * WAVELENGTH
        return 1.0

    def get_coronagraph_parameter(self, param):
        if param == "bandwidth":
            return 0.2
        return 1.0


@pytest.fixture
def mock_instrument():
    """Fixture providing a mock instrument object for EAC detector testing."""
    mock = MagicMock()
    mock.lam = [0.5, 1.5] * u.um

    array_length = 2
    default_array = np.linspace(0.359, 0.988, array_length)

    mock.__dict__.update(
        {
            "verbose": False,
            "lam": mock.lam,
            "OP_full": [
                "PM",
                "SM",
                "TCA",
                "wave_beamsplitter",
                "pol_beamsplitter",
                "FSM",
                "OAPs_forward",
                "DM1",
                "DM2",
                "Fold",
                "OAPs_back",
                "Apodizer",
                "Focal_Plane_Mask",
                "Lyot_Stop",
                "Field_Stop",
                "filters",
                "Detector",
            ],
            "OP_tele": ["PM", "SM"],
            "OP_inst": [
                "TCA",
                "wave_beamsplitter",
                "pol_beamsplitter",
                "FSM",
                "OAPs_forward",
                "DM1",
                "DM2",
                "Fold",
                "OAPs_back",
                "Apodizer",
                "Focal_Plane_Mask",
                "Lyot_Stop",
                "Field_Stop",
                "filters",
            ],
            "OP_det": ["Detector"],
            "TCA": default_array,
            "wb_tran": np.concatenate([np.zeros(5), np.ones(5)]),
            "wb_refl": np.concatenate([np.ones(5), np.zeros(5)]),
            "wave_beamsplitter": np.ones(array_length),
            "pol_beamsplitter": np.ones(array_length),
            "FSM": default_array,
            "OAPs_forward": default_array,
            "DM1": default_array,
            "DM2": default_array,
            "Fold": default_array,
            "OAPs_back": default_array,
            "Apodizer": np.full(array_length, 0.95),
            "Focal_Plane_Mask": np.linspace(0.91, 0.89, array_length),
            "Lyot_Stop": default_array,
            "Field_Stop": np.linspace(0.91, 0.89, array_length),
            "filters": np.ones(array_length),
            "total_inst_refl": np.full(array_length, 0.7),
        }
    )

    return mock


@pytest.fixture
def mock_detector():
    """Fixture factory for creating mock detector objects for different observing modes."""

    def _create_mock(detector_type):
        mock = MagicMock()

        mock.lam = np.array([0.2, 0.8, 1.1, 1.6]) * u.um
        mock.verbose = False

        qe_vis = np.array([0.9, 0.9, np.nan, np.nan])
        qe_nir = np.array([np.nan, np.nan, 0.85, 0.85])

        common_dict = {
            "lam": mock.lam,
            "verbose": False,
            "qe_vis": qe_vis,
            "dc_vis": 3e-05,
            "cic_vis": None,
            "qe_nir": qe_nir,
            "dc_nir": 0.0001,
            "cic_nir": None,
        }

        if detector_type == "IMAGER":
            mock.__dict__.update(
                {
                    **common_dict,
                    "rn_vis": 0.1,
                    "rn_nir": 0.3,
                }
            )
        elif detector_type == "IFS":
            mock.__dict__.update(
                {
                    **common_dict,
                    "rn_vis": 0.0,
                    "rn_nir": 0.4,
                }
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        return mock

    return _create_mock


@pytest.fixture
def toy_detector_parameters():
    """Fixture providing standard parameters for ToyModelDetector testing."""
    return {
        "pixscale_mas": 10,
        "npix_multiplier": [2],
        "DC": [4e-5],
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
    }


# ============================================================================
# Tests for ToyModelDetector initialization
# ============================================================================


def test_toy_model_detector_init():
    """Test that ToyModelDetector initializes with None values."""
    detector = ToyModelDetector()

    assert detector.path is None
    assert detector.keyword is None


# ============================================================================
# Tests for ToyModelDetector.load_configuration - IMAGER mode
# ============================================================================


def test_toy_model_detector_load_configuration_imager_user_params(
    toy_detector_parameters,
):
    """Test loading ToyModelDetector configuration with user parameters in IMAGER mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IMAGER")

    detector.load_configuration(toy_detector_parameters, mediator)

    assert detector.pixscale_mas == 10 * MAS
    assert np.all(detector.npix_multiplier == [2] * DIMENSIONLESS)
    assert np.all(detector.DC == [4e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [1.0] * READ_NOISE)
    assert np.all(detector.tread == [1100] * READ_TIME)
    assert np.all(detector.CIC == [1.5e-3] * CLOCK_INDUCED_CHARGE)


def test_toy_model_detector_load_configuration_imager_default_qe():
    """Test that default QE and dQE values are used in IMAGER mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IMAGER")
    parameters = {
        "pixscale_mas": 10,
        "npix_multiplier": [2],
        "DC": [4e-5],
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
    }

    detector.load_configuration(parameters, mediator)

    assert np.all(detector.QE == [0.9] * QUANTUM_EFFICIENCY)
    assert np.all(detector.dQE == [0.75] * DIMENSIONLESS)


def test_toy_model_detector_load_configuration_imager_defaults():
    """Test that default pixscale is calculated correctly in IMAGER mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IMAGER")

    detector.load_configuration({}, mediator)

    assert np.isclose(detector.pixscale_mas, 6.4457752 * MAS)


# ============================================================================
# Tests for ToyModelDetector.load_configuration - IFS mode
# ============================================================================


def test_toy_model_detector_load_configuration_ifs_user_params(toy_detector_parameters):
    """Test loading ToyModelDetector configuration with user parameters in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")

    detector.load_configuration(toy_detector_parameters, mediator)

    assert detector.pixscale_mas == 10 * MAS
    assert np.all(detector.npix_multiplier == [2, 2, 2] * DIMENSIONLESS)
    assert np.all(detector.DC == [4e-5, 4e-5, 4e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [1.0, 1.0, 1.0] * READ_NOISE)
    assert np.all(detector.tread == [1100, 1100, 1100] * READ_TIME)
    assert np.all(detector.CIC == [1.5e-3, 1.5e-3, 1.5e-3] * CLOCK_INDUCED_CHARGE)


def test_toy_model_detector_load_configuration_ifs_default_qe():
    """Test that default QE and dQE values are broadcast in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")
    parameters = {
        "pixscale_mas": 10,
        "npix_multiplier": [2],
        "DC": [4e-5],
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
    }

    detector.load_configuration(parameters, mediator)

    assert np.all(detector.QE == [0.9, 0.9, 0.9] * QUANTUM_EFFICIENCY)
    assert np.all(detector.dQE == [0.75, 0.75, 0.75] * DIMENSIONLESS)


def test_toy_model_detector_load_configuration_ifs_defaults():
    """Test that default pixscale is calculated correctly in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")

    detector.load_configuration({}, mediator)

    assert np.isclose(detector.pixscale_mas, 6.4457752 * MAS)


# ============================================================================
# Tests for EACDetector.load_configuration - IMAGER mode
# ============================================================================


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_imager_basic(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test basic EACDetector configuration loading in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IMAGER")

    detector = EACDetector()
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    detector.load_configuration(parameters, mediator)

    assert detector.pixscale_mas is not None
    assert np.all(detector.npix_multiplier == 1 * DIMENSIONLESS)


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_imager_units(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test that all detector parameters have correct units in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IMAGER")

    detector = EACDetector()
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    detector.load_configuration(parameters, mediator)

    assert detector.DC.unit == DARK_CURRENT
    assert detector.RN.unit == READ_NOISE
    assert detector.tread.unit == READ_TIME
    assert detector.CIC.unit == CLOCK_INDUCED_CHARGE
    assert detector.QE.unit == QUANTUM_EFFICIENCY
    assert detector.dQE.unit == DIMENSIONLESS


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_imager_shapes(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test that detector parameter arrays have correct shapes in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IMAGER")

    detector = EACDetector()
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    detector.load_configuration(parameters, mediator)

    expected_shape = (1,)
    assert detector.DC.shape == expected_shape
    assert detector.RN.shape == expected_shape
    assert detector.QE.shape == expected_shape


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_imager_values(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test that detector parameters have correct values in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IMAGER")

    detector = EACDetector()
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    detector.load_configuration(parameters, mediator)

    assert np.allclose(detector.DC.value, 3e-05)
    assert np.allclose(detector.RN.value, 0.1)


# ============================================================================
# Tests for EACDetector.load_configuration - IFS mode
# ============================================================================


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_ifs_basic(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test basic EACDetector configuration loading in IFS mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IFS")

    detector = EACDetector()
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    detector.load_configuration(parameters, mediator)

    assert detector.pixscale_mas is not None
    assert np.all(detector.npix_multiplier == 1 * DIMENSIONLESS)


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_ifs_units(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test that all detector parameters have correct units in IFS mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IFS")

    detector = EACDetector()
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    detector.load_configuration(parameters, mediator)

    assert detector.DC.unit == DARK_CURRENT
    assert detector.RN.unit == READ_NOISE
    assert detector.tread.unit == READ_TIME
    assert detector.CIC.unit == CLOCK_INDUCED_CHARGE
    assert detector.QE.unit == QUANTUM_EFFICIENCY
    assert detector.dQE.unit == DIMENSIONLESS


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_ifs_shapes(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test that detector parameter arrays have correct shapes in IFS mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IFS")

    detector = EACDetector()
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    detector.load_configuration(parameters, mediator)

    expected_shape = (3,)
    assert detector.DC.shape == expected_shape
    assert detector.RN.shape == expected_shape
    assert detector.QE.shape == expected_shape
    assert detector.CIC.shape == expected_shape


@patch("eacy.load_detector")
@patch("eacy.load_instrument")
def test_eac_detector_load_configuration_ifs_wavelength_dependent_values(
    mock_load_instrument, mock_load_detector, mock_instrument, mock_detector
):
    """Test that detector parameters vary correctly with wavelength in IFS mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_detector.return_value = mock_detector("IFS")

    detector = EACDetector()
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    detector.load_configuration(parameters, mediator)

    # VIS wavelengths (< 1 μm)
    assert np.allclose(detector.DC[:2].value, 3e-05)
    assert np.allclose(detector.RN[:2].value, 0.0)

    # NIR wavelengths (>= 1 μm)
    assert np.allclose(detector.DC[2:].value, 0.0001)
    assert np.allclose(detector.RN[2:].value, 0.4)


# ============================================================================
# Tests for EACDetector.load_configuration - Error handling
# ============================================================================


def test_eac_detector_load_configuration_unsupported_mode():
    """Test that unsupported observing mode raises KeyError."""
    detector = EACDetector()
    parameters = {"observing_mode": "test"}
    mediator = MockMediator("test")

    with pytest.raises(KeyError, match="Unsupported observing mode: test"):
        detector.load_configuration(parameters, mediator)


def test_eac_detector_load_configuration_invalid_mode():
    """Test that invalid observing mode raises KeyError."""
    detector = EACDetector()
    parameters = {"observing_mode": "INVALID"}
    mediator = MockMediator("IMAGER")

    with pytest.raises(KeyError, match="Unsupported observing mode: INVALID"):
        detector.load_configuration(parameters, mediator)


# ============================================================================
# Tests for EACDetector validation inputs
# ============================================================================


@pytest.mark.parametrize("observing_mode", ["IMAGER", "IFS"])
def test_eac_detector_etc_validation_inputs(observing_mode):
    """Test that ETC validation inputs are correctly loaded."""
    detector = EACDetector()
    mediator = MockMediator(observing_mode)

    parameters = {
        "observing_mode": observing_mode,
        "t_photon_count_input": 0.7,
        "det_npix_input": 200,
    }

    detector.load_configuration(parameters, mediator)

    assert hasattr(detector, "t_photon_count_input")
    assert hasattr(detector, "det_npix_input")
    assert detector.t_photon_count_input == 0.7 * SECOND / FRAME
    assert detector.det_npix_input == 200 * DIMENSIONLESS


# ============================================================================
# Tests for Detector.validate_configuration
# ============================================================================


def test_detector_validate_configuration_all_valid(toy_detector_parameters):
    """Test that validation passes with all correct attributes."""
    detector = ToyModelDetector()
    parameters = {
        **toy_detector_parameters,
        "QE": [0.95],
        "dQE": [0.8],
    }
    mediator = MockMediator()

    detector.load_configuration(parameters, mediator)

    # Should not raise
    detector.validate_configuration()


def test_detector_validate_configuration_missing_pixscale(toy_detector_parameters):
    """Test that missing pixscale_mas attribute raises AttributeError."""
    detector = ToyModelDetector()
    mediator = MockMediator()

    detector.load_configuration(toy_detector_parameters, mediator)
    delattr(detector, "pixscale_mas")

    with pytest.raises(
        AttributeError, match="Detector is missing attribute: pixscale_mas"
    ):
        detector.validate_configuration()


def test_detector_validate_configuration_pixscale_not_quantity(toy_detector_parameters):
    """Test that non-Quantity pixscale_mas raises TypeError."""
    detector = ToyModelDetector()
    mediator = MockMediator()

    detector.load_configuration(toy_detector_parameters, mediator)
    detector.pixscale_mas = 10  # Not a Quantity

    with pytest.raises(
        TypeError, match="Detector attribute pixscale_mas should be a Quantity"
    ):
        detector.validate_configuration()


def test_detector_validate_configuration_incorrect_pixscale_units(
    toy_detector_parameters,
):
    """Test that pixscale_mas with incorrect units raises ValueError."""
    detector = ToyModelDetector()
    mediator = MockMediator()

    detector.load_configuration(toy_detector_parameters, mediator)
    detector.pixscale_mas = 10 * u.arcsec  # Wrong unit

    with pytest.raises(
        ValueError, match="Detector attribute pixscale_mas has incorrect units"
    ):
        detector.validate_configuration()


# ============================================================================
# Tests for parameter broadcasting in IFS mode
# ============================================================================


def test_toy_model_detector_scalar_to_array_broadcasting():
    """Test that scalar detector parameters are correctly broadcast to arrays in IFS mode."""
    detector = ToyModelDetector()
    mediator = MockMediator("IFS")

    parameters = {
        "DC": [4e-5],  # Single value
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
    }

    detector.load_configuration(parameters, mediator)

    # Should be broadcast to match wavelength array length (3)
    assert len(detector.DC) == 3
    assert len(detector.RN) == 3
    assert len(detector.tread) == 3
    assert len(detector.CIC) == 3
