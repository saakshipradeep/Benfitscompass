import pytest
from app import REQUIREMENTS, NO_FAULTS, FAULT_DESCRIPTIONS, run_suite


@pytest.mark.parametrize("req", REQUIREMENTS, ids=lambda r: r.rid)
def test_requirement(req):
    ok, detail = req.test(NO_FAULTS)
    assert ok, f"{req.rid} failed: {detail}"


@pytest.mark.parametrize("fault", list(FAULT_DESCRIPTIONS))
def test_suite_detects_seeded_defect(fault):
    result = run_suite(frozenset({fault}))
    assert (result["Result"] == "FAIL").any(), f"Seeded defect '{fault}' went undetected"
