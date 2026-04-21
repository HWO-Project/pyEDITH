import pytest
import os
import json
from unittest.mock import patch, mock_open

from pyEDITH.observatory_builder import ObservatoryBuilder
from pyEDITH.observatory import Observatory
from pyEDITH.components import telescopes, coronagraphs, detectors


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_registry():
    """Fixture providing mock component registry."""
    return {
        "telescopes": {
            "EAC1": {"class": "EACTelescope", "path": ""},
            "ToyModel": {"class": "ToyModelTelescope", "path": None},
        },
        "coronagraphs": {
            "AAVC": {"class": "CoronagraphYIP", "path": "AAVC_coronagraph"},
            "LUVOIR": {"class": "CoronagraphYIP", "path": "usort_offaxis_ovc"},
            "ToyModel": {"class": "ToyModelCoronagraph", "path": None},
        },
        "detectors": {
            "EAC1": {"class": "EACDetector", "path": ""},
            "ToyModel": {"class": "ToyModelDetector", "path": None},
        },
    }


@pytest.fixture
def valid_custom_config():
    """Fixture providing valid custom observatory configuration."""
    return {
        "telescope": "EAC1",
        "coronagraph": "AAVC",
        "detector": "ToyModel",
    }


@pytest.fixture
def new_preset_config():
    """Fixture providing configuration for a new preset."""
    return {
        "telescope": "NewTelescope",
        "coronagraph": "NewCoronagraph",
        "detector": "NewDetector",
    }


# ============================================================================
# Tests for ObservatoryBuilder.load_registry
# ============================================================================


def test_load_registry(mock_registry):
    """Test loading component registry from JSON."""
    with patch("json.load", return_value=mock_registry):
        registry = ObservatoryBuilder.load_registry()

    assert "ToyModel" in registry["telescopes"]
    assert "ToyModel" in registry["coronagraphs"]
    assert "ToyModel" in registry["detectors"]


# ============================================================================
# Tests for ObservatoryBuilder.build_component_path
# ============================================================================


def test_build_component_path_telescope():
    """Test building telescope component path with environment variables set."""
    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        result = ObservatoryBuilder.build_component_path("telescopes", "")

        assert result == "/sci_eng/"


def test_build_component_path_coronagraph():
    """Test building coronagraph component path with full path."""
    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        result = ObservatoryBuilder.build_component_path(
            "coronagraphs", "AAVC_coronagraph"
        )

        assert result == "/yip_coro/AAVC_coronagraph"


def test_build_component_path_telescope_env_not_set():
    """Test that missing SCI_ENG_DIR environment variable raises EnvironmentError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            EnvironmentError, match="SCI_ENG_DIR environment variable not set"
        ):
            ObservatoryBuilder.build_component_path("telescopes", "EAC1")


def test_build_component_path_coronagraph_env_not_set():
    """Test that missing YIP_CORO_DIR environment variable raises EnvironmentError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            EnvironmentError, match="YIP_CORO_DIR environment variable not set"
        ):
            ObservatoryBuilder.build_component_path("coronagraphs", "AAVC")


def test_build_component_path_invalid_type():
    """Test that invalid component type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown component type: unknown"):
        ObservatoryBuilder.build_component_path("unknown", "EAC1")


# ============================================================================
# Tests for ObservatoryBuilder.create_observatory - Presets
# ============================================================================


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.load_registry")
def test_create_observatory_with_preset(mock_load_registry, mock_registry):
    """Test creating observatory from preset configuration."""
    mock_load_registry.return_value = mock_registry

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        observatory = ObservatoryBuilder.create_observatory("EAC1")

    assert isinstance(observatory, Observatory)
    assert observatory.telescope.path == "/sci_eng/"
    assert observatory.coronagraph.path == "/yip_coro/usort_offaxis_ovc"
    assert observatory.detector.path == "/sci_eng/"


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.load_registry")
def test_create_observatory_with_custom_config(
    mock_load_registry, mock_registry, valid_custom_config
):
    """Test creating observatory from custom configuration."""
    mock_load_registry.return_value = mock_registry

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        observatory = ObservatoryBuilder.create_observatory(valid_custom_config)

    assert isinstance(observatory, Observatory)
    assert observatory.telescope.path == "/sci_eng/"
    assert observatory.coronagraph.path == "/yip_coro/AAVC_coronagraph"
    assert observatory.detector.path is None


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.load_registry")
def test_create_observatory_invalid_preset(mock_load_registry, mock_registry):
    """Test that invalid preset name raises ValueError."""
    mock_load_registry.return_value = mock_registry

    with pytest.raises(ValueError, match="Unknown preset observatory: UNKNOWN"):
        ObservatoryBuilder.create_observatory("UNKNOWN")


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.load_registry")
def test_create_observatory_invalid_input_type(mock_load_registry, mock_registry):
    """Test that invalid input type raises ValueError."""
    mock_load_registry.return_value = mock_registry

    with pytest.raises(ValueError):
        ObservatoryBuilder.create_observatory(123)


# ============================================================================
# Tests for ObservatoryBuilder._create_component
# ============================================================================


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component_telescope(mock_build_path, mock_registry):
    """Test creating telescope component."""
    mock_build_path.return_value = "/mock/telescopes/"

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        telescope = ObservatoryBuilder._create_component(
            "telescopes", "EAC1", mock_registry
        )

    assert isinstance(telescope, telescopes.EACTelescope)
    assert telescope.path == "/mock/telescopes/"
    assert telescope.keyword == "EAC1"


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component_coronagraph(mock_build_path, mock_registry):
    """Test creating coronagraph component."""
    mock_build_path.return_value = "/mock/coronagraphs/AAVC_coronagraph"

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        coronagraph = ObservatoryBuilder._create_component(
            "coronagraphs", "AAVC", mock_registry
        )

    assert isinstance(coronagraph, coronagraphs.CoronagraphYIP)
    assert coronagraph.path == "/mock/coronagraphs/AAVC_coronagraph"
    assert coronagraph.keyword == "AAVC"


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component_detector(mock_build_path, mock_registry):
    """Test creating detector component."""
    mock_build_path.return_value = "/mock/detectors/"

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        detector = ObservatoryBuilder._create_component(
            "detectors", "EAC1", mock_registry
        )

    assert isinstance(detector, detectors.EACDetector)
    assert detector.path == "/mock/detectors/"
    assert detector.keyword == "EAC1"


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component_toymodel_no_path(mock_build_path, mock_registry):
    """Test creating ToyModel component with no path."""
    mock_build_path.return_value = None

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        toy_telescope = ObservatoryBuilder._create_component(
            "telescopes", "ToyModel", mock_registry
        )

    assert isinstance(toy_telescope, telescopes.ToyModelTelescope)
    assert toy_telescope.path is None
    assert toy_telescope.keyword == "ToyModel"


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component_unknown_keyword(mock_build_path, mock_registry):
    """Test that unknown component keyword raises ValueError."""
    with pytest.raises(ValueError, match="Unknown telescopes keyword: Unknown"):
        ObservatoryBuilder._create_component("telescopes", "Unknown", mock_registry)


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component_unknown_class(mock_build_path, mock_registry):
    """Test that unknown component class raises ValueError."""
    mock_registry["telescopes"]["Invalid"] = {"class": "InvalidClass", "path": ""}

    with pytest.raises(ValueError, match="Unknown component class: InvalidClass"):
        ObservatoryBuilder._create_component("telescopes", "Invalid", mock_registry)


# ============================================================================
# Tests for ObservatoryBuilder.configure_observatory
# ============================================================================


def test_configure_observatory():
    """Test configuring observatory with parameters."""
    observatory = Observatory()
    config = {"some_config": "value"}
    observation = "mock_observation"
    scene = "mock_scene"

    with patch.object(Observatory, "load_configuration") as mock_load_config:
        configured_observatory = ObservatoryBuilder.configure_observatory(
            observatory, config, observation, scene
        )

    mock_load_config.assert_called_once_with(config, observation, scene)
    assert configured_observatory == observatory


# ============================================================================
# Tests for ObservatoryBuilder preset management - Add
# ============================================================================


def test_add_preset(new_preset_config):
    """Test adding a new preset."""
    ObservatoryBuilder.add_preset("NewPreset", new_preset_config)

    assert "NewPreset" in ObservatoryBuilder.PRESETS
    assert ObservatoryBuilder.PRESETS["NewPreset"] == new_preset_config


def test_add_preset_already_exists(new_preset_config):
    """Test that adding existing preset raises ValueError."""
    with pytest.raises(ValueError, match="Preset 'EAC1' already exists"):
        ObservatoryBuilder.add_preset("EAC1", new_preset_config)


# ============================================================================
# Tests for ObservatoryBuilder preset management - Remove
# ============================================================================


def test_remove_preset():
    """Test removing an existing preset."""
    ObservatoryBuilder.remove_preset("EAC1")

    assert "EAC1" not in ObservatoryBuilder.PRESETS


def test_remove_preset_not_exists():
    """Test that removing non-existent preset raises ValueError."""
    with pytest.raises(ValueError, match="Preset 'NonExistentPreset' does not exist"):
        ObservatoryBuilder.remove_preset("NonExistentPreset")


# ============================================================================
# Tests for ObservatoryBuilder preset management - List and Get
# ============================================================================


def test_list_presets():
    """Test listing all available presets."""
    presets = ObservatoryBuilder.list_presets()

    assert isinstance(presets, list)
    assert "ToyModel" in presets


def test_get_preset_config():
    """Test retrieving configuration for a specific preset."""
    config = ObservatoryBuilder.get_preset_config("ToyModel")

    assert isinstance(config, dict)
    assert "telescope" in config
    assert config["telescope"] == "ToyModel"


def test_get_preset_config_not_exists():
    """Test that getting non-existent preset config raises ValueError."""
    with pytest.raises(ValueError, match="Preset 'NonExistentPreset' does not exist"):
        ObservatoryBuilder.get_preset_config("NonExistentPreset")


# ============================================================================
# Tests for ObservatoryBuilder.validate_config
# ============================================================================


def test_validate_config_valid():
    """Test validation of valid configuration."""
    valid_config = {"telescope": "EAC1", "coronagraph": "AAVC", "detector": "EAC1"}

    # Should not raise any exception
    ObservatoryBuilder.validate_config(valid_config)


def test_validate_config_missing_key():
    """Test that missing required key raises ValueError."""
    invalid_config = {"telescope": "EAC1", "coronagraph": "AAVC"}

    with pytest.raises(
        ValueError, match="Missing required configuration key: detector"
    ):
        ObservatoryBuilder.validate_config(invalid_config)


def test_validate_config_invalid_value_type():
    """Test that non-string configuration value raises ValueError."""
    invalid_config = {"telescope": "EAC1", "coronagraph": "AAVC", "detector": 123}

    with pytest.raises(
        ValueError, match="Configuration value for detector must be a string"
    ):
        ObservatoryBuilder.validate_config(invalid_config)


# ============================================================================
# Tests for ObservatoryBuilder.modify_preset
# ============================================================================


def test_modify_preset():
    """Test modifying an existing preset."""
    ObservatoryBuilder.modify_preset("ToyModel", coronagraph="AAVC")

    assert ObservatoryBuilder.PRESETS["ToyModel"]["coronagraph"] == "AAVC"


def test_modify_preset_not_exists():
    """Test that modifying non-existent preset raises ValueError."""
    with pytest.raises(ValueError, match="Preset 'NonExistentPreset' does not exist"):
        ObservatoryBuilder.modify_preset("NonExistentPreset", telescope="NewTelescope")


def test_modify_preset_invalid_key():
    """Test that modifying with invalid key raises ValueError."""
    with pytest.raises(ValueError, match="Invalid configuration key: invalid_key"):
        ObservatoryBuilder.modify_preset("ToyModel", invalid_key="Value")
