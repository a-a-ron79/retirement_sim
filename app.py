import streamlit as st

st.markdown('''
### How This Model Works
- **Purpose:** This simulator estimates retirement outcomes under uncertainty, using Monte Carlo methods to model investment returns and expenses across a lifetime.
- **Income:** Before retirement, earned income reduces how much must be withdrawn from investments. If income exceeds expenses, the surplus stays invested. After retirement, income stops unless you’ve added Social Security or another retirement source.
- **Taxes:** Two effective tax rates are applied — one while working and one during retirement. Taxes reduce annual income accordingly. Additionally, if investment withdrawals are required during retirement, a 50% portion of your retirement tax rate is applied to inflate the withdrawal, approximating partial taxation.
- **Mid-Year Convention:** Each year’s investment growth is split evenly — half applied before and half after income and spending — to create more realistic annual compounding behavior.
- **Inflation:** Two separate inflation rates are used — one for your home country (affecting retirement income like Social Security) and one for your target country (affecting local living expenses).
- **Investments:** Returns are simulated for equities (lognormal), bonds (normal), and cash (normal but floored at 0%). Correlations are included using a Cholesky decomposition.
- **Projection:** Each simulation represents one possible path for your portfolio balance from current age to death, incorporating income, taxes, inflation, and returns.
- **Results:** Key statistics (median, percentiles, min/max, and success rate) summarize performance across all simulations.
''')

import streamlit as st
st.title("Monte Carlo Retirement Simulator — Geographic Arbitrage Edition with Asset Allocation (Mid-Year Convention)")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

MAX_SIMS = 20000

# --- Inputs ---
init = float(st.text_input("Initial portfolio ($)", value="1000000"))
spend_base = float(st.text_input("Annual spending in target country ($)", value="30000"))
current_age = int(st.text_input("Current age", value="40"))
retire_age = int(st.text_input("Retirement age", value="65"))
death_age = int(st.text_input("Age at death", value="100"))
gross_income = float(st.text_input("Annual earned income before retirement ($)", value="0"))

work_tax_rate = float(st.text_input("Effective tax rate while working (%)", value="20.0")) / 100
retire_tax_rate = float(st.text_input("Effective tax rate during retirement (%)", value="10.0")) / 100

home_inflation_rate = float(st.text_input("Home country annual inflation rate (%)", value="2.0"))
target_inflation_rate = float(st.text_input("Target country annual inflation rate (%)", value="4.0"))

start_ssi_age = int(st.text_input("Age to start receiving retirement income (e.g., SSI)", value="67"))
ssi_amount_today = float(st.text_input("Annual retirement income in today's dollars ($)", value="26000"))
include_ssi_taxable = st.checkbox("Include SSI in taxable income?", value=False)

# --- Monte Carlo setup ---
mean_equity, std_equity = 0.09, 0.15
mean_bonds, std_bonds = 0.03, 0.06
mean_cash, std_cash = 0.02, 0.005

weights_equity, weights_bonds, weights_cash = 0.6, 0.3, 0.1

mu_eq_ln = np.log(1 + mean_equity) - 0.5 * (std_equity ** 2)
sigma_eq_ln = np.sqrt(np.log(1 + (std_equity ** 2)))

corr = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])
chol = np.linalg.cholesky(corr)

sims = int(st.text_input("Number of simulations (max 20,000)", value="1000"))
if sims > MAX_SIMS:
    sims = MAX_SIMS

total_years = death_age - current_age

home_inf = home_inflation_rate / 100
target_inf = target_inflation_rate / 100

# --- Simulation ---
results, final_balances = [], []

for _ in range(sims):
    balance = init
    path = []

    for year in range(total_years):
        age = current_age + year
        spend = spend_base * ((1 + target_inf) ** year)

        income, taxable_income = 0, 0
        if gross_income > 0 and age < retire_age:
            income += gross_income
            taxable_income += gross_income

        if age >= start_ssi_age:
            ssi_income = ssi_amount_today * ((1 + home_inf) ** (age - current_age))
            income += ssi_income
            if include_ssi_taxable:
                taxable_income += ssi_income

        if age < retire_age:
            tax = taxable_income * work_tax_rate
        else:
            tax = taxable_income * retire_tax_rate

        income_after_tax = income - tax

        # Withdrawal tax inflation adjustment
        if age >= retire_age and (income_after_tax < spend):
            withdraw_needed = spend - income_after_tax
            tax_adj = 1 + (retire_tax_rate * 0.5)
            spend = income_after_tax + withdraw_needed * tax_adj

        # Mid-year convention: apply half growth before, half after cash flows
        rand = np.random.normal(0, 1, 3)
        correlated = chol @ rand
        eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
        bd_ret = mean_bonds + correlated[1] * std_bonds
        cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)

        portfolio_growth = weights_equity * eq_ret + weights_bonds * bd_ret + weights_cash * cs_ret

        # Apply mid-year timing: half growth before and half after net cash flow
        mid_growth = (1 + portfolio_growth) ** 0.5 - 1
        balance = balance * (1 + mid_growth) + (income_after_tax - spend)
        balance = balance * (1 + mid_growth)

        path.append(balance)

    results.append(path)
    final_balances.append(path[-1])

results = np.array(results)

# --- Results ---
median_final = np.median(final_balances)
p10, p90 = np.percentile(final_balances, [10, 90])
min_final, max_final = np.min(final_balances), np.max(final_balances)
success_rate = np.mean(np.array(final_balances) > 0) * 100

# --- Plot ---
def currency_formatter(x, pos):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:,.0f}"

plt.figure(figsize=(10, 6))
for path in results:
    plt.plot(path, color='gray', alpha=0.2)
plt.title("Monte Carlo Retirement Projection (Mid-Year Convention)")
plt.xlabel("Years from Current Age")
plt.ylabel("Portfolio Value ($K / $M)")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
st.pyplot(plt)

# --- Summary ---
st.subheader("Simulation Results Summary")
st.write(f"**Median Portfolio at Death:** ${median_final:,.0f}")
st.write(f"**10th Percentile:** ${p10:,.0f}")
st.write(f"**90th Percentile:** ${p90:,.0f}")
st.write(f"**Minimum Portfolio at Death:** ${min_final:,.0f}")
st.write(f"**Maximum Portfolio at Death:** ${max_final:,.0f}")
st.write(f"**Success Rate (Final > 0):** {success_rate:.1f}%")