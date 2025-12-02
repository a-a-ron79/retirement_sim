import streamlit as st
st.title("Monte Carlo Retirement Simulator — Geographic Arbitrage Edition (Mid-Year Convention)")

st.markdown('''
### How This Model Works
- **Purpose:** This simulator estimates retirement outcomes under uncertainty, using Monte Carlo methods to model investment returns and expenses across a lifetime.
- **Home vs Target Country:** You can define a *home country* phase and a *target country* phase. During the home country phase (before move age), your income and spending reflect home-country conditions. After the move age, the simulation switches to target-country assumptions.
- **Income & Savings:** Any annual surplus (income after taxes minus spending) is added to the investment portfolio, increasing the portfolio before compounding.
- **Lump Sum Events:** You can include a single lump sum to be received in a future year (for example, inheritance or property sale). This amount is **not inflated** and is added directly to your portfolio in the year received.
- **Taxes:** Two effective tax rates are applied — one while working and one during retirement. Taxes reduce annual income accordingly. Additionally, if investment withdrawals are required during retirement, a 50% portion of your retirement tax rate is applied to inflate withdrawals, approximating partial taxation.
- **Mid-Year Convention:** Each year’s investment growth is split evenly — half applied before and half after income and spending — to create more realistic annual compounding behavior.
- **Inflation:** Two separate inflation rates are used — one for your home country (affecting home income and spending) and one for your target country (affecting retirement income and local expenses).
- **Investments:** You can adjust expected mean returns and volatility for equities, bonds, and cash. Correlations are included using a Cholesky decomposition.
''')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

MAX_SIMS = 20000

# --- Inputs ---
init = float(st.text_input("Initial portfolio ($)", value="1000000"))

# Home vs Target setup
current_age = int(st.text_input("Current age", value="40"))
move_age = int(st.text_input("Age when moving to target country", value="55"))
retire_age = int(st.text_input("Retirement age", value="65"))
death_age = int(st.text_input("Age at death", value="100"))

# Guard clause for age logic
if current_age >= move_age:
    st.info('Current age is beyond or equal to move age — simulation will start directly in the target country phase.')
    move_age = current_age

# Home country income and spending
home_income = float(st.text_input("Annual earned income in home country ($)", value="80000"))
home_spend_base = float(st.text_input("Annual spending in home country ($)", value="60000"))
home_inflation_rate = float(st.text_input("Home country annual inflation rate (%)", value="2.0"))

# Target country income and spending
spend_base = float(st.text_input("Annual spending in target country ($)", value="30000"))
target_inflation_rate = float(st.text_input("Target country annual inflation rate (%)", value="4.0"))

gross_income = float(st.text_input("Annual earned income before retirement in target country ($)", value="0"))

# Investment Return Assumptions
mean_equity = float(st.text_input("Mean annual return for equities (%)", value="8.5")) / 100
std_equity = float(st.text_input("Volatility for equities (%)", value="17.0")) / 100
mean_bonds = float(st.text_input("Mean annual return for bonds (%)", value="5.0")) / 100
std_bonds = float(st.text_input("Volatility for bonds (%)", value="6.0")) / 100
mean_cash = float(st.text_input("Mean annual return for cash (%)", value="2.5")) / 100
std_cash = float(st.text_input("Volatility for cash (%)", value="1.5")) / 100

# Portfolio Allocation Inputs
weights_equity = float(st.text_input("Equity allocation (%)", value="65")) / 100
weights_bonds = float(st.text_input("Bond allocation (%)", value="25")) / 100
weights_cash = float(st.text_input("Cash allocation (%)", value="10")) / 100

# Normalize if needed
weight_sum = weights_equity + weights_bonds + weights_cash
if weight_sum != 1:
    weights_equity /= weight_sum
    weights_bonds /= weight_sum
    weights_cash /= weight_sum

# Tax rates
work_tax_rate = float(st.text_input("Effective tax rate while working (%)", value="20.0")) / 100
retire_tax_rate = float(st.text_input("Effective tax rate during retirement (%)", value="15.0")) / 100

# SSI / Pension
start_ssi_age = int(st.text_input("Age to start receiving retirement income (e.g., SSI)", value="67"))
ssi_amount_today = float(st.text_input("Annual retirement income in today's dollars ($)", value="26000"))
include_ssi_taxable = st.checkbox("Include SSI in taxable income?", value=False)

# Lump Sum Event
receive_lump_age = int(st.text_input("Age when lump sum is received", value="70"))
lump_amount_today = float(st.text_input("Lump sum amount ($)", value="100000"))

# --- Monte Carlo setup ---
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

        # Determine which phase applies
        if age < move_age:
            income = home_income * ((1 + home_inf) ** year)
            spend = home_spend_base * ((1 + home_inf) ** year)
            tax_rate = work_tax_rate if age < retire_age else retire_tax_rate
        else:
            income = gross_income if (age < retire_age and gross_income > 0) else 0
            spend = spend_base * ((1 + target_inf) ** (age - move_age))
            tax_rate = retire_tax_rate if age >= retire_age else work_tax_rate

        # SSI income
        taxable_income = 0
        if age >= start_ssi_age:
            ssi_income = ssi_amount_today * ((1 + home_inf) ** (age - current_age))
            income += ssi_income
            if include_ssi_taxable:
                taxable_income += ssi_income

        # Lump sum income event
        if age == receive_lump_age:
            lump_sum = lump_amount_today  # no inflation adjustment
            balance += lump_sum

        # Taxes
        taxable_income += income
        tax = taxable_income * tax_rate
        income_after_tax = income - tax

        # Withdrawal tax inflation adjustment
        if age >= retire_age and (income_after_tax < spend):
            withdraw_needed = spend - income_after_tax
            tax_adj = 1 + (retire_tax_rate * 0.5)
            spend = income_after_tax + withdraw_needed * tax_adj

        # Mid-year convention: half growth before/after
        rand = np.random.normal(0, 1, 3)
        correlated = chol @ rand
        eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
        bd_ret = mean_bonds + correlated[1] * std_bonds
        cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)
        portfolio_growth = weights_equity * eq_ret + weights_bonds * bd_ret + weights_cash * cs_ret

        mid_growth = (1 + portfolio_growth) ** 0.5 - 1

        # Add surplus (savings) before growth
        if income_after_tax > spend:
            balance += (income_after_tax - spend)

        # Apply growth mid-year convention
        balance = balance * (1 + mid_growth)

        if income_after_tax < spend:
            balance -= (spend - income_after_tax)

        balance = balance * (1 + mid_growth)

        path.append(balance)

    results.append(path)
    final_balances.append(path[-1])

results = np.array(results)

# --- Results ---
median_final = np.median(final_balances)
p10, p20, p80, p90 = np.percentile(final_balances, [10, 20, 80, 90])
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
colors = cm.get_cmap('Spectral', sims)
for i, path in enumerate(results):
    plt.plot(path, color=colors(i / sims), alpha=0.3)

median_path = np.median(results, axis=0)
plt.plot(median_path, color='black', linewidth=2.5, label='Median Path')
plt.legend()

plt.title("Monte Carlo Retirement Projection — Home to Target Country Transition")
plt.xlabel("Years from Current Age")
plt.ylabel("Portfolio Value ($K / $M)")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
st.pyplot(plt)

# --- Summary ---
st.subheader("Simulation Results Summary")
st.write(f"**Median Portfolio at Death:** ${median_final:,.0f}")
st.write(f"**10th Percentile:** ${p10:,.0f}")
st.write(f"**20th Percentile:** ${p20:,.0f}")
st.write(f"**80th Percentile:** ${p80:,.0f}")
st.write(f"**90th Percentile:** ${p90:,.0f}")
st.write(f"**Minimum Portfolio at Death:** ${min_final:,.0f}")
st.write(f"**Maximum Portfolio at Death:** ${max_final:,.0f}")
st.write(f"**Success Rate (Final > 0):** {success_rate:.1f}%")
