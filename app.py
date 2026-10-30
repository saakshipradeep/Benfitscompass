"""
BenefitsCompass - Health Plan Decision-Support Tool with a built-in QA layer
=============================================================================
A simulated benefits decision-support calculator that helps an employee choose
between three illustrative health plans (HDHP + HSA, PPO, HMO) based on their
expected usage, and then PROVES the calculator works through:

  1. A requirements register (19 requirements, each mapped to an executable test)
  2. A live QA test log (pass / fail per requirement)
  3. Fault injection (defects are deliberately seeded into the engine to show the
     test suite actually catches calculation errors)
  4. "Regret analysis" - the worst-case exposure of each plan choice

All plan designs, premiums and charges are SYNTHETIC and for demonstration only.
This is not financial, tax or insurance advice.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Run the QA suite headless:
    pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

# =============================================================================
# 1. REFERENCE DATA (synthetic / illustrative)
# =============================================================================

TIERS: List[str] = ["Employee Only", "Employee + Spouse/Partner", "Family"]

# Illustrative HSA contribution limits (IRS limits change yearly - verify before real use)
HSA_LIMIT: Dict[str, float] = {"individual": 4400.0, "family": 8750.0}

# "Allowed amounts" = what the plan's negotiated rate says each service costs
ALLOWED: Dict[str, float] = {
    "pcp": 150.0,         # primary care visit
    "specialist": 300.0,  # specialist visit
    "rx": 70.0,           # prescription fill (generic/brand blend)
    "er": 1800.0,         # emergency room visit
}

PLAN_NAMES = ["HDHP + HSA", "PPO", "HMO"]

# Seeded defects used by the QA effectiveness (fault-injection) test
NO_FAULTS: FrozenSet[str] = frozenset()
FAULT_DESCRIPTIONS: Dict[str, str] = {
    "no_oop_cap": "Out-of-pocket maximum is not enforced",
    "ignore_deductible": "Deductible is skipped; coinsurance applied from dollar one",
    "hsa_overoffset": "Employer HSA seed credited in full even when unused",
    "inverted_coinsurance": "Member pays (1 - coinsurance) instead of coinsurance",
}


@dataclass(frozen=True)
class Plan:
    name: str
    monthly_premium: Dict[str, float]      # employee share, by tier
    deductible: Dict[str, float]           # by coverage level: individual / family
    oop_max: Dict[str, float]              # by coverage level
    coinsurance: float                     # member share after deductible
    pcp_copay: Optional[float]             # None => subject to deductible + coinsurance
    specialist_copay: Optional[float]
    rx_copay: Optional[float]
    er_copay: Optional[float]
    employer_hsa: Dict[str, float]         # by tier ($0 for non-HSA plans)
    hsa_eligible: bool
    summary: str


PLANS: List[Plan] = [
    Plan(
        name="HDHP + HSA",
        monthly_premium={TIERS[0]: 250.0, TIERS[1]: 620.0, TIERS[2]: 900.0},
        deductible={"individual": 3500.0, "family": 7000.0},
        oop_max={"individual": 6000.0, "family": 12000.0},
        coinsurance=0.20,
        pcp_copay=None, specialist_copay=None, rx_copay=None, er_copay=None,
        employer_hsa={TIERS[0]: 500.0, TIERS[1]: 1000.0, TIERS[2]: 1000.0},
        hsa_eligible=True,
        summary="Lowest premium. You pay full cost until the deductible. Employer seeds an HSA; "
                "contributions are pre-tax and roll over.",
    ),
    Plan(
        name="PPO",
        monthly_premium={TIERS[0]: 330.0, TIERS[1]: 800.0, TIERS[2]: 1150.0},
        deductible={"individual": 500.0, "family": 1000.0},
        oop_max={"individual": 4000.0, "family": 8000.0},
        coinsurance=0.20,
        pcp_copay=25.0, specialist_copay=40.0, rx_copay=10.0, er_copay=250.0,
        employer_hsa={t: 0.0 for t in TIERS},
        hsa_eligible=False,
        summary="Highest premium, lowest deductible and lowest out-of-pocket maximum. "
                "Broadest provider choice.",
    ),
    Plan(
        name="HMO",
        monthly_premium={TIERS[0]: 290.0, TIERS[1]: 720.0, TIERS[2]: 1020.0},
        deductible={"individual": 1500.0, "family": 3000.0},
        oop_max={"individual": 6000.0, "family": 12000.0},
        coinsurance=0.20,
        pcp_copay=25.0, specialist_copay=50.0, rx_copay=12.0, er_copay=None,
        employer_hsa={t: 0.0 for t in TIERS},
        hsa_eligible=False,
        summary="Mid premium with predictable copays for routine care. In-network only; "
                "deductible applies to hospital and ER care.",
    ),
]


def plan_by_name(name: str) -> Plan:
    return next(p for p in PLANS if p.name == name)


# =============================================================================
# 2. CALCULATION ENGINE
# =============================================================================

@dataclass(frozen=True)
class Usage:
    tier: str = TIERS[0]
    pcp_visits: int = 0
    specialist_visits: int = 0
    rx_fills: int = 0
    er_visits: int = 0
    major_event_cost: float = 0.0   # allowed charges for surgery / hospital stay / delivery
    employee_hsa: float = 0.0       # employee's intended HSA contribution (HDHP only)
    marginal_tax_rate: float = 0.22 # combined marginal tax rate used for HSA tax savings


@dataclass(frozen=True)
class Result:
    plan: str
    annual_premium: float
    total_allowed: float
    member_oop: float
    hit_oop_max: bool
    employer_hsa_applied: float
    employee_hsa_contribution: float
    employee_hsa_used: float
    tax_savings: float
    hsa_rollover: float
    net_cost: float


def coverage_level(tier: str) -> str:
    return "individual" if tier == TIERS[0] else "family"


def validate_usage(u: Usage) -> None:
    if u.tier not in TIERS:
        raise ValueError(f"Unknown coverage tier: {u.tier}")
    for label, val in [
        ("PCP visits", u.pcp_visits), ("Specialist visits", u.specialist_visits),
        ("Rx fills", u.rx_fills), ("ER visits", u.er_visits),
        ("Major event cost", u.major_event_cost), ("HSA contribution", u.employee_hsa),
    ]:
        if val < 0:
            raise ValueError(f"{label} cannot be negative")
    if not 0.0 <= u.marginal_tax_rate <= 0.60:
        raise ValueError("Marginal tax rate must be between 0% and 60%")


def calculate(plan: Plan, usage: Usage, faults: FrozenSet[str] = NO_FAULTS) -> Result:
    """Annual net cost of one plan for one usage profile.

    Order of operations
      1. Premiums (annual, employee share)
      2. Copay-based services: member pays min(copay, allowed) per service
      3. Deductible-based services (+ any major event): member pays deductible first,
         then coinsurance on the remainder
      4. Total member cost-sharing is capped at the out-of-pocket maximum
      5. HSA: employer seed offsets actual spending; employee pre-tax dollars used
         create tax savings. Unused HSA money is NOT counted as savings - it is
         reported as a rollover balance.
    """
    validate_usage(usage)
    level = coverage_level(usage.tier)
    deductible = plan.deductible[level]
    oop_cap = plan.oop_max[level]

    lines = [
        (usage.pcp_visits, ALLOWED["pcp"], plan.pcp_copay),
        (usage.specialist_visits, ALLOWED["specialist"], plan.specialist_copay),
        (usage.rx_fills, ALLOWED["rx"], plan.rx_copay),
        (usage.er_visits, ALLOWED["er"], plan.er_copay),
    ]

    total_allowed = float(usage.major_event_cost)
    deductible_pool = float(usage.major_event_cost)
    copays = 0.0
    for count, allowed, copay in lines:
        total_allowed += count * allowed
        if copay is None:
            deductible_pool += count * allowed
        else:
            copays += count * min(copay, allowed)

    ded_applied = 0.0 if "ignore_deductible" in faults else min(deductible_pool, deductible)
    coins_rate = (1.0 - plan.coinsurance) if "inverted_coinsurance" in faults else plan.coinsurance
    coins_paid = (deductible_pool - ded_applied) * coins_rate

    raw_oop = copays + ded_applied + coins_paid
    member_oop = raw_oop if "no_oop_cap" in faults else min(raw_oop, oop_cap)
    member_oop = min(member_oop, total_allowed)

    annual_premium = plan.monthly_premium[usage.tier] * 12

    employer_seed = employee_contrib = employer_applied = employee_used = 0.0
    tax_savings = rollover = 0.0
    if plan.hsa_eligible:
        employer_seed = plan.employer_hsa[usage.tier]
        room = max(0.0, HSA_LIMIT[level] - employer_seed)
        employee_contrib = min(usage.employee_hsa, room)
        employer_applied = employer_seed if "hsa_overoffset" in faults else min(employer_seed, member_oop)
        employee_used = min(employee_contrib, max(0.0, member_oop - employer_applied))
        tax_savings = employee_used * usage.marginal_tax_rate
        rollover = employer_seed + employee_contrib - employer_applied - employee_used

    net_cost = annual_premium + member_oop - employer_applied - tax_savings

    return Result(
        plan=plan.name,
        annual_premium=round(annual_premium, 2),
        total_allowed=round(total_allowed, 2),
        member_oop=round(member_oop, 2),
        hit_oop_max=member_oop >= oop_cap - 0.005,
        employer_hsa_applied=round(employer_applied, 2),
        employee_hsa_contribution=round(employee_contrib, 2),
        employee_hsa_used=round(employee_used, 2),
        tax_savings=round(tax_savings, 2),
        hsa_rollover=round(rollover, 2),
        net_cost=round(net_cost, 2),
    )


def compare(usage: Usage, faults: FrozenSet[str] = NO_FAULTS) -> pd.DataFrame:
    rows = [calculate(p, usage, faults).__dict__ for p in PLANS]
    return pd.DataFrame(rows).set_index("plan")


# =============================================================================
# 3. SCENARIOS, REGRET ANALYSIS, BREAK-EVEN
# =============================================================================

BAD_YEAR_EXTRA = 40_000.0
CATASTROPHIC_EXTRA = 150_000.0


def build_scenarios(base: Usage) -> Dict[str, Usage]:
    """Four possible 'futures' for the same household."""
    return {
        "Healthy year (1 sick visit)": replace(
            base, pcp_visits=1, specialist_visits=0, rx_fills=0, er_visits=0, major_event_cost=0.0),
        "Your expected year": base,
        f"Bad year (+${BAD_YEAR_EXTRA:,.0f} hospitalization)": replace(
            base, major_event_cost=base.major_event_cost + BAD_YEAR_EXTRA),
        f"Catastrophic year (+${CATASTROPHIC_EXTRA:,.0f})": replace(
            base, major_event_cost=base.major_event_cost + CATASTROPHIC_EXTRA),
    }


def scenario_costs(base: Usage, faults: FrozenSet[str] = NO_FAULTS) -> pd.DataFrame:
    """Rows = scenarios, columns = plans, values = net annual cost."""
    data = {name: compare(u, faults)["net_cost"] for name, u in build_scenarios(base).items()}
    return pd.DataFrame(data).T[[p.name for p in PLANS]]


def regret_matrix(base: Usage, faults: FrozenSet[str] = NO_FAULTS) -> pd.DataFrame:
    """Regret = what you pay minus what the best plan would have cost in that same scenario."""
    costs = scenario_costs(base, faults)
    return costs.sub(costs.min(axis=1), axis=0).round(2)


def recommend(base: Usage, faults: FrozenSet[str] = NO_FAULTS) -> Dict[str, object]:
    costs = scenario_costs(base, faults)
    regret = regret_matrix(base, faults)
    expected = costs.loc["Your expected year"]
    max_regret = regret.max(axis=0)
    return {
        "lowest_expected_plan": expected.idxmin(),
        "lowest_expected_cost": float(expected.min()),
        "minimax_regret_plan": max_regret.idxmin(),
        "minimax_regret_value": float(max_regret.min()),
        "max_regret_by_plan": max_regret,
    }


def breakeven_curve(base: Usage, max_event: float = 100_000.0, step: float = 2_500.0,
                    faults: FrozenSet[str] = NO_FAULTS) -> pd.DataFrame:
    """Net annual cost of each plan as the size of a major medical event grows."""
    xs = [i * step for i in range(int(max_event / step) + 1)]
    rows = {x: compare(replace(base, major_event_cost=x), faults)["net_cost"] for x in xs}
    out = pd.DataFrame(rows).T
    out.index.name = "Major medical event (allowed charges, $)"
    return out


def cheapest_plan_ranges(curve: pd.DataFrame) -> pd.DataFrame:
    """Collapse the curve into 'plan X is cheapest from $a to $b'."""
    winners = curve.idxmin(axis=1)
    ranges, start, current, prev = [], None, None, None
    for x, w in winners.items():
        if w != current:
            if current is not None:
                ranges.append((current, start, prev))
            current, start = w, x
        prev = x
    ranges.append((current, start, prev))
    return pd.DataFrame(
        [{"Cheapest plan": w, "From ($)": f"{a:,.0f}", "To ($)": f"{b:,.0f}"} for w, a, b in ranges]
    )


# =============================================================================
# 4. REQUIREMENTS REGISTER + EXECUTABLE TESTS (the QA layer)
# =============================================================================

@dataclass
class Requirement:
    rid: str
    text: str
    category: str
    test: Callable[[FrozenSet[str]], Tuple[bool, str]]


REQUIREMENTS: List[Requirement] = []


def requirement(rid: str, category: str, text: str):
    def deco(fn: Callable[[FrozenSet[str]], Tuple[bool, str]]):
        REQUIREMENTS.append(Requirement(rid, text, category, fn))
        return fn
    return deco


def _grid():
    for tier in TIERS:
        for pcp in (0, 4, 12):
            for spec in (0, 6):
                for rx in (0, 24):
                    for er in (0, 2):
                        for major in (0, 2_000, 8_000, 30_000, 200_000):
                            yield Usage(tier=tier, pcp_visits=pcp, specialist_visits=spec, rx_fills=rx,
                                        er_visits=er, major_event_cost=major, employee_hsa=1_000)


def _close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol


HDHP, PPO, HMO = PLANS


@requirement("R01", "Cost-sharing", "Member out-of-pocket cost never exceeds the plan's out-of-pocket maximum.")
def _r01(f):
    for p in PLANS:
        for u in _grid():
            r = calculate(p, u, f)
            cap = p.oop_max[coverage_level(u.tier)]
            if r.member_oop > cap + 0.01:
                return False, f"{p.name}/{u.tier}: OOP {r.member_oop:,.0f} > cap {cap:,.0f}"
    return True, "OOP <= cap across full test grid"


@requirement("R02", "Cost-sharing", "Member out-of-pocket cost never exceeds total allowed charges.")
def _r02(f):
    for p in PLANS:
        for u in _grid():
            r = calculate(p, u, f)
            if r.member_oop > r.total_allowed + 0.01:
                return False, f"{p.name}: OOP {r.member_oop:,.0f} > charges {r.total_allowed:,.0f}"
    return True, "OOP <= allowed charges across full test grid"


@requirement("R03", "Baseline", "With zero utilization the member pays only premiums (no cost-sharing).")
def _r03(f):
    for p in PLANS:
        for t in TIERS:
            r = calculate(p, Usage(tier=t), f)
            if r.member_oop != 0 or not _close(r.net_cost, r.annual_premium):
                return False, f"{p.name}/{t}: oop={r.member_oop}, net={r.net_cost}, premium={r.annual_premium}"
    return True, "OOP = $0 and net = premium for all plans and tiers"


@requirement("R04", "Premiums", "Annual premium equals 12 x the monthly employee premium for the selected tier.")
def _r04(f):
    for p in PLANS:
        for t in TIERS:
            r = calculate(p, Usage(tier=t), f)
            if not _close(r.annual_premium, 12 * p.monthly_premium[t]):
                return False, f"{p.name}/{t}: {r.annual_premium} != {12 * p.monthly_premium[t]}"
    return True, "Premium = 12 x monthly for all plans and tiers"


@requirement("R05", "HSA", "Employer HSA credit applied to cost never exceeds the member's actual out-of-pocket spending.")
def _r05(f):
    for u in _grid():
        r = calculate(HDHP, u, f)
        if r.employer_hsa_applied > r.member_oop + 0.01:
            return False, f"{u.tier}: HSA credit {r.employer_hsa_applied:,.0f} > OOP {r.member_oop:,.0f}"
    return True, "HSA credit <= OOP across full test grid"


@requirement("R06", "Deductible", "Below the deductible, the member pays 100% of deductible-based services.")
def _r06(f):
    r = calculate(HDHP, Usage(major_event_cost=1_000), f)
    return _close(r.member_oop, 1_000), f"expected $1,000, got ${r.member_oop:,.2f}"


@requirement("R07", "Deductible", "Above the deductible, the member pays the deductible plus coinsurance on the remainder.")
def _r07(f):
    ded = HDHP.deductible["individual"]
    expected = ded + HDHP.coinsurance * 1_000
    r = calculate(HDHP, Usage(major_event_cost=ded + 1_000), f)
    return _close(r.member_oop, expected), f"expected ${expected:,.2f}, got ${r.member_oop:,.2f}"


@requirement("R08", "Copays", "Copay-based services cost the member the flat copay per visit and ignore the deductible.")
def _r08(f):
    r = calculate(PPO, Usage(pcp_visits=3), f)
    expected = 3 * PPO.pcp_copay
    return _close(r.member_oop, expected), f"expected ${expected:,.2f}, got ${r.member_oop:,.2f}"


@requirement("R09", "Copays", "A copay can never exceed the allowed charge of the service.")
def _r09(f):
    odd_plan = replace(PPO, pcp_copay=500.0)
    r = calculate(odd_plan, Usage(pcp_visits=1), f)
    return _close(r.member_oop, ALLOWED["pcp"]), f"expected ${ALLOWED['pcp']:,.2f}, got ${r.member_oop:,.2f}"


@requirement("R10", "HSA", "Non-HSA plans never produce HSA credits, tax savings or rollover balances.")
def _r10(f):
    u = Usage(major_event_cost=5_000, employee_hsa=5_000)
    for p in (PPO, HMO):
        r = calculate(p, u, f)
        if r.employer_hsa_applied or r.tax_savings or r.hsa_rollover or r.employee_hsa_contribution:
            return False, f"{p.name} produced HSA values"
    return True, "PPO and HMO show $0 HSA activity"


@requirement("R11", "HSA", "Employee HSA contribution is capped at the annual limit minus the employer contribution.")
def _r11(f):
    r = calculate(HDHP, Usage(tier=TIERS[2], employee_hsa=999_999), f)
    expected = HSA_LIMIT["family"] - HDHP.employer_hsa[TIERS[2]]
    return _close(r.employee_hsa_contribution, expected), \
        f"expected ${expected:,.2f}, got ${r.employee_hsa_contribution:,.2f}"


@requirement("R12", "HSA", "Tax savings equal the employee HSA dollars actually spent x marginal tax rate (zero at a 0% rate).")
def _r12(f):
    u = Usage(major_event_cost=2_000, employee_hsa=1_000, marginal_tax_rate=0.24)
    r = calculate(HDHP, u, f)
    r0 = calculate(HDHP, replace(u, marginal_tax_rate=0.0), f)
    ok = _close(r.tax_savings, 240.0) and r0.tax_savings == 0
    return ok, f"expected $240 / $0, got ${r.tax_savings:,.2f} / ${r0.tax_savings:,.2f}"


@requirement("R13", "Integrity", "Increasing utilization never reduces member out-of-pocket cost (monotonicity).")
def _r13(f):
    for p in PLANS:
        for t in TIERS:
            prev = -1.0
            for major in range(0, 100_001, 2_500):
                oop = calculate(p, Usage(tier=t, pcp_visits=4, major_event_cost=major), f).member_oop
                if oop < prev - 0.01:
                    return False, f"{p.name}/{t}: OOP fell from {prev:,.0f} to {oop:,.0f} at ${major:,}"
                prev = oop
            prev = -1.0
            for visits in range(0, 40):
                oop = calculate(p, Usage(tier=t, pcp_visits=visits), f).member_oop
                if oop < prev - 0.01:
                    return False, f"{p.name}/{t}: OOP fell as visits rose to {visits}"
                prev = oop
    return True, "OOP is non-decreasing in major-event cost and visit count"


@requirement("R14", "Coverage tiers", "Employee-only coverage uses the individual deductible; all other tiers use the family deductible.")
def _r14(f):
    ind = calculate(HDHP, Usage(tier=TIERS[0], major_event_cost=5_000), f).member_oop
    fam = calculate(HDHP, Usage(tier=TIERS[2], major_event_cost=5_000), f).member_oop
    exp_ind = HDHP.deductible["individual"] + HDHP.coinsurance * (5_000 - HDHP.deductible["individual"])
    ok = _close(ind, exp_ind) and _close(fam, 5_000)
    return ok, f"individual {ind:,.0f} (exp {exp_ind:,.0f}), family {fam:,.0f} (exp 5,000)"


@requirement("R15", "Regret analysis", "Regret is never negative and is exactly $0 for the best plan in every scenario.")
def _r15(f):
    for t in TIERS:
        reg = regret_matrix(Usage(tier=t, pcp_visits=4, specialist_visits=3, rx_fills=12), f)
        if (reg < -0.01).any().any() or not (reg.min(axis=1).abs() < 0.01).all():
            return False, f"{t}: invalid regret matrix"
    return True, "Regret >= 0 and row minimum = 0"


@requirement("R16", "Regret analysis", "The minimax-regret recommendation has the smallest worst-case regret of all plans.")
def _r16(f):
    for t in TIERS:
        rec = recommend(Usage(tier=t, pcp_visits=4, specialist_visits=3, rx_fills=12), f)
        worst = rec["max_regret_by_plan"]
        if worst[rec["minimax_regret_plan"]] > worst.min() + 0.01:
            return False, f"{t}: recommended plan is not the minimax plan"
    return True, "Recommended plan minimizes maximum regret"


@requirement("R17", "Cost-sharing", "In a catastrophic year every plan, in every tier, caps at its out-of-pocket maximum.")
def _r17(f):
    for p in PLANS:
        for t in TIERS:
            r = calculate(p, Usage(tier=t, major_event_cost=CATASTROPHIC_EXTRA), f)
            cap = p.oop_max[coverage_level(t)]
            if not _close(r.member_oop, cap):
                return False, f"{p.name}/{t}: OOP {r.member_oop:,.0f} != cap {cap:,.0f}"
    return True, "OOP equals the cap for all plans and tiers"


@requirement("R18", "Input validation", "Negative inputs and out-of-range tax rates are rejected with an error.")
def _r18(f):
    bad_inputs = [Usage(pcp_visits=-1), Usage(major_event_cost=-5), Usage(employee_hsa=-1),
                  Usage(marginal_tax_rate=0.9), Usage(tier="Unknown")]
    for u in bad_inputs:
        try:
            calculate(HDHP, u, f)
        except ValueError:
            continue
        return False, f"Accepted invalid input: {u}"
    return True, "All 5 invalid inputs rejected"


@requirement("R19", "HSA", "Unused HSA money is reported as a rollover balance, not counted as a saving.")
def _r19(f):
    r = calculate(HDHP, Usage(employee_hsa=1_000), f)
    expected = HDHP.employer_hsa[TIERS[0]] + 1_000
    ok = r.employer_hsa_applied == 0 and _close(r.hsa_rollover, expected) and _close(r.net_cost, r.annual_premium)
    return ok, f"credit={r.employer_hsa_applied}, rollover={r.hsa_rollover} (exp {expected}), net={r.net_cost}"


def run_suite(faults: FrozenSet[str] = NO_FAULTS) -> pd.DataFrame:
    rows = []
    for req in REQUIREMENTS:
        try:
            ok, detail = req.test(faults)
        except Exception as exc:  # a crash is a failed test, not a crashed app
            ok, detail = False, f"Exception: {exc}"
        rows.append({"ID": req.rid, "Category": req.category, "Requirement": req.text,
                     "Result": "PASS" if ok else "FAIL", "Evidence": detail})
    return pd.DataFrame(rows)


def fault_injection_report() -> pd.DataFrame:
    """Seed each defect, re-run the suite, and record which requirements caught it."""
    rows = []
    for key, desc in FAULT_DESCRIPTIONS.items():
        res = run_suite(frozenset({key}))
        failed = res.loc[res["Result"] == "FAIL", "ID"].tolist()
        rows.append({"Seeded defect": desc, "Detected?": "YES" if failed else "NO",
                     "Caught by": ", ".join(failed) if failed else "-"})
    return pd.DataFrame(rows)


# =============================================================================
# 5. STREAMLIT USER INTERFACE
# =============================================================================

def usd(x: float) -> str:
    return f"${x:,.0f}"


EVENT_PRESETS = {
    "None": 0.0,
    "Pregnancy / delivery (~$15,000)": 15_000.0,
    "Knee or shoulder surgery (~$22,000)": 22_000.0,
    "Appendectomy (~$18,000)": 18_000.0,
    "Custom amount": -1.0,
}


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="BenefitsCompass", page_icon="🧭", layout="wide")
    st.title("🧭 BenefitsCompass")
    st.caption("Health plan decision support with built-in regret analysis and QA  |  "
               "All plan data is synthetic and for demonstration only.")

    # ---------------- Sidebar: household inputs ----------------
    with st.sidebar:
        st.header("Your household")
        tier = st.selectbox("Coverage tier", TIERS)
        st.subheader("Expected yearly usage")
        pcp = st.number_input("Primary care visits", 0, 60, 4)
        spec = st.number_input("Specialist visits", 0, 60, 3)
        rx = st.number_input("Prescription fills", 0, 120, 12)
        er = st.number_input("ER visits", 0, 10, 0)
        preset = st.selectbox("Planned major event", list(EVENT_PRESETS))
        if EVENT_PRESETS[preset] < 0:
            major = st.number_input("Allowed charges for the event ($)", 0, 500_000, 10_000, step=1_000)
        else:
            major = EVENT_PRESETS[preset]
        st.subheader("HSA & taxes")
        hsa = st.number_input("Your planned HSA contribution ($)", 0, 10_000, 1_000, step=250,
                              help="Only applies to the HDHP. Capped at the IRS limit minus the employer seed.")
        tax = st.slider("Marginal tax rate (fed + state + FICA)", 0, 50, 22, format="%d%%") / 100

    usage = Usage(tier=tier, pcp_visits=int(pcp), specialist_visits=int(spec), rx_fills=int(rx),
                  er_visits=int(er), major_event_cost=float(major), employee_hsa=float(hsa),
                  marginal_tax_rate=float(tax))

    tab_rec, tab_regret, tab_curve, tab_plans, tab_qa, tab_brief = st.tabs(
        ["Recommendation", "Regret analysis", "Break-even curve", "Plan details", "QA & requirements", "Client brief"])

    # ---------------- Tab 1: recommendation ----------------
    with tab_rec:
        results = compare(usage)
        rec = recommend(usage)
        c1, c2 = st.columns(2)
        c1.metric("Lowest expected cost", rec["lowest_expected_plan"], usd(rec["lowest_expected_cost"]))
        c2.metric("Safest choice (minimax regret)", rec["minimax_regret_plan"],
                  f"worst-case regret {usd(rec['minimax_regret_value'])}", delta_color="off")
        if rec["lowest_expected_plan"] == rec["minimax_regret_plan"]:
            st.success(f"**{rec['lowest_expected_plan']}** is both the cheapest plan for your expected year "
                       "and the plan you would regret least if the year goes badly.")
        else:
            st.info(f"**{rec['lowest_expected_plan']}** is cheapest if your year goes as expected, but "
                    f"**{rec['minimax_regret_plan']}** protects you best if it does not. "
                    "That trade-off is the real decision.")

        st.subheader("Net annual cost by plan")
        chart_df = results[["annual_premium", "member_oop"]].rename(
            columns={"annual_premium": "Premiums", "member_oop": "Out-of-pocket"})
        st.bar_chart(chart_df)

        table = pd.DataFrame({
            "Annual premium": results["annual_premium"],
            "Out-of-pocket": results["member_oop"],
            "Employer HSA applied": -results["employer_hsa_applied"],
            "HSA tax savings": -results["tax_savings"],
            "NET ANNUAL COST": results["net_cost"],
            "HSA rollover balance": results["hsa_rollover"],
            "Hit OOP max?": results["hit_oop_max"].map({True: "Yes", False: "No"}),
        })
        st.dataframe(table.style.format({c: usd for c in table.columns if c != "Hit OOP max?"}))
        st.download_button("Download comparison (CSV)", table.to_csv().encode("utf-8"),
                           file_name="benefitscompass_comparison.csv", mime="text/csv")
        st.caption("HSA rollover is money that stays yours next year - it is shown separately and is "
                   "not counted as a saving.")

    # ---------------- Tab 2: regret analysis ----------------
    with tab_regret:
        st.subheader("What would you regret?")
        st.write("Regret = what you pay with a plan minus what the best plan would have cost in the same "
                 "scenario. A plan with a small worst-case regret is the safest choice under uncertainty.")
        costs = scenario_costs(usage)
        regret = regret_matrix(usage)
        st.markdown("**Net annual cost in each scenario**")
        st.dataframe(costs.style.format(usd).highlight_min(axis=1, color="#c6f6d5"))
        st.markdown("**Regret (green = best plan in that scenario)**")
        st.dataframe(regret.style.format(usd).highlight_min(axis=1, color="#c6f6d5"))
        worst = regret.max(axis=0)
        st.markdown("**Worst-case regret by plan**")
        st.bar_chart(worst.rename("Worst-case regret ($)"))
        for p in PLANS:
            worst_scn = regret[p.name].idxmax()
            st.write(f"- **{p.name}**: worst case is the *{worst_scn}* - you would pay "
                     f"**{usd(regret[p.name].max())}** more than the best plan.")

    # ---------------- Tab 3: break-even curve ----------------
    with tab_curve:
        st.subheader("How the cheapest plan changes as a medical event gets bigger")
        curve = breakeven_curve(replace(usage, major_event_cost=0.0))
        st.line_chart(curve)
        st.markdown("**Cheapest plan by size of major medical event** (all other usage held at your inputs)")
        st.dataframe(cheapest_plan_ranges(curve), hide_index=True)

    # ---------------- Tab 4: plan details ----------------
    with tab_plans:
        level = coverage_level(tier)
        st.subheader(f"Plan designs for: {tier}")
        details = pd.DataFrame({
            p.name: {
                "Monthly premium": usd(p.monthly_premium[tier]),
                "Deductible": usd(p.deductible[level]),
                "Out-of-pocket max": usd(p.oop_max[level]),
                "Coinsurance (you pay)": f"{p.coinsurance:.0%}",
                "Primary care": f"${p.pcp_copay:,.0f} copay" if p.pcp_copay is not None else "Deductible + coinsurance",
                "Specialist": f"${p.specialist_copay:,.0f} copay" if p.specialist_copay is not None else "Deductible + coinsurance",
                "Prescription": f"${p.rx_copay:,.0f} copay" if p.rx_copay is not None else "Deductible + coinsurance",
                "ER visit": f"${p.er_copay:,.0f} copay" if p.er_copay is not None else "Deductible + coinsurance",
                "Employer HSA seed": usd(p.employer_hsa[tier]) if p.hsa_eligible else "n/a",
            } for p in PLANS
        })
        st.dataframe(details)
        for p in PLANS:
            st.write(f"- **{p.name}**: {p.summary}")
        st.caption("Assumed allowed amounts: " + ", ".join(f"{k} ${v:,.0f}" for k, v in ALLOWED.items()))

    # ---------------- Tab 5: QA ----------------
    with tab_qa:
        st.subheader("Requirements register and live QA test log")
        st.write("Every requirement below maps to an executable test that runs against the calculation "
                 "engine each time this tab loads.")
        suite = run_suite()
        passed = int((suite["Result"] == "PASS").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Requirements", len(suite))
        m2.metric("Passing", passed)
        m3.metric("Failing", len(suite) - passed)
        st.dataframe(suite, hide_index=True)

        st.subheader("QA effectiveness: fault injection")
        st.write("A passing test log only means something if the tests can fail. Click below to seed known "
                 "defects into the engine one at a time and confirm the suite catches each one.")
        if st.button("Run fault-injection test"):
            report = fault_injection_report()
            caught = int((report["Detected?"] == "YES").sum())
            st.dataframe(report, hide_index=True)
            st.success(f"Suite detected {caught} of {len(report)} seeded defects.")

        st.subheader("Defect log (template)")
        st.write("Record real defects you find while building or changing the engine:")
        st.dataframe(pd.DataFrame([{
            "Defect ID": "D-001", "Found by": "(requirement ID)", "Symptom": "(what looked wrong)",
            "Root cause": "(why)", "Fix": "(what changed)", "Retest": "(PASS/FAIL)"}]), hide_index=True)

    # ---------------- Tab 6: client brief ----------------
    with tab_brief:
        st.subheader("One-page client brief (for an HR director)")
        st.markdown(
            """
**What it is.** A decision-support calculator embedded in open enrollment. Employees enter how they expect
to use care; the tool shows total annual cost for each plan - premiums plus out-of-pocket costs, minus HSA
benefits - and highlights the plan they would regret least if their year goes badly.

**Why it matters.** Many employees choose on premium alone. A plan with a low premium can cost far more in a
bad year, and a plan with a high premium can be wasted money in a healthy one. Showing both outcomes side by
side reduces the number of employees who pick the wrong plan and the number of "why did I pay so much?" calls
after enrollment.

**What employees see.** Net annual cost per plan, a regret table for four possible years, and a break-even
chart showing at what level of medical spending each plan becomes the cheapest.

**What HR would receive (engagement data the platform can report).**
- Share of employees who completed the decision tool before electing a plan
- Plan election mix versus the plan the tool recommended
- Share of employees who selected a plan that was not lowest-cost for their own stated usage
- HSA election and contribution rates
- Most common questions and drop-off points in the flow

**How accuracy is controlled.** 19 documented requirements map to automated tests. The suite is itself
validated by seeding known defects into the engine and confirming each one is caught.

**Implementation notes.** Plan designs live in a single configuration block, so loading a client's real plan
data (premiums, deductibles, copays, HSA seed) does not require changing calculation logic. Every change is
re-tested against the full requirements register before release.
            """
        )
        st.caption("Demonstration only: synthetic plan data. Not financial, tax or insurance advice.")


if __name__ == "__main__":
    main()
