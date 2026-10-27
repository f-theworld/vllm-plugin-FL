import pytest

from vllm_fl.utils import get_device_control_env_var

_OVERRIDE = "FLAGOS_DEVICE_CONTROL_ENV_VAR"
_PLACEHOLDER = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"


def test_nvidia_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var("nvidia") == "CUDA_VISIBLE_DEVICES"


@pytest.mark.parametrize(
    "vendor",
    [
        "ascend",
        "metax",
        "mthreads",
        "sunrise",
        "hygon",
        "thead",
        "not_a_vendor",
    ],
)
def test_unvalidated_vendors_preserve_vllm_base_noop(monkeypatch, vendor):
    monkeypatch.delenv(_OVERRIDE, raising=False)
    assert get_device_control_env_var(vendor) == _PLACEHOLDER


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(_OVERRIDE, "CUSTOM_VISIBLE_DEVICES")
    assert get_device_control_env_var("ascend") == "CUSTOM_VISIBLE_DEVICES"
    assert get_device_control_env_var("not_a_vendor") == "CUSTOM_VISIBLE_DEVICES"


def test_blank_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv(_OVERRIDE, "   ")
    assert get_device_control_env_var("ascend") == _PLACEHOLDER


@pytest.mark.parametrize("visible_devices, expected", [("4", [4]), ("5,4", [5, 4])])
def test_iluvatar_visible_device_mapping(monkeypatch, visible_devices, expected):
    from vllm_fl.platform import PlatformFL

    monkeypatch.delenv(_OVERRIDE, raising=False)
    device_control_env_var = get_device_control_env_var("iluvatar")
    assert device_control_env_var == "CUDA_VISIBLE_DEVICES"
    monkeypatch.setattr(PlatformFL, "device_control_env_var", device_control_env_var)
    monkeypatch.setenv(device_control_env_var, visible_devices)
    for visible_device_id, physical_device_id in enumerate(expected):
        assert (
            PlatformFL.visible_device_id_to_physical_device_id(visible_device_id)
            == physical_device_id
        )


def test_platform_picks_up_the_dispatched_value():
    from vllm_fl.platform import PlatformFL

    expected = get_device_control_env_var(PlatformFL.vendor_name)
    assert PlatformFL.device_control_env_var == expected
