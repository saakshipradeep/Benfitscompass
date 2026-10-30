# 🧭 BenefitsCompass

**Health plan decision-support tool with built-in regret analysis and a QA layer.**
Python · Streamlit · Pandas

BenefitsCompass helps an employee choose between three illustrative health plans (HDHP + HSA, PPO, HMO)
based on how they expect to use care. It then proves the calculator is correct with a documented
requirements register and an automated test suite.

> All plan designs, premiums and charges are **synthetic**. This is a portfolio demonstration, not financial,
> tax or insurance advice.

## What it does
- **Total annual cost** per plan: premiums + deductible + copays + coinsurance (capped at the out-of-pocket
  max), minus employer HSA credit and HSA tax savings.
- **Regret analysis:** compares each plan across four possible years (healthy, expected, bad, catastrophic)
  and recommends both the cheapest plan and the safest plan (minimax regret).
- **Break-even curve:** shows how the cheapest plan changes as a medical event gets larger.
- **Requirements register + QA log:** 19 requirements, each mapped to an executable test, run live in the app.
- **Fault injection:** seeds 4 known defects into the engine and confirms the test suite catches each one.
- **Client brief:** one-page explanation written for an HR director.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q        # run the QA suite headless
```

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Plan data, calculation engine, regret analysis, requirements/tests, Streamlit UI |
| `test_benefits_engine.py` | Runs every requirement (and the fault-injection check) under pytest |
| `requirements.txt` | Dependencies |

## Design decisions
- Plan terms live in one configuration block, so real client data can be loaded without touching logic.
- Unused HSA money is shown as a **rollover balance**, never counted as a saving.
- The UI imports Streamlit inside `main()`, so the engine and tests run without a browser.

## Defect log
Record real defects found while building or changing the engine (add your own as you test):

| ID | Found by | Symptom | Root cause | Fix | Retest |
|---|---|---|---|---|---|
| D-001 | | | | | |
