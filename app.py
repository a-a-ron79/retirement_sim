import streamlit as st

st.markdown('''
### How This Model Works
- **Income:** Before retirement, earned income reduces how much must be withdrawn from investments. If income exceeds expenses, the surplus stays invested. Once retired, this income stops unless Social Security or another retirement income is selected.
- **Taxes:** Two effective tax rates are applied—one while working and one during retirement. Taxes are subtracted each year based on taxable income (earned income and, if selected, taxable SSI). This keeps the simulation simple but realistic.
- **Expenses:** Annual spending is entered in today's dollars and grows each year according to the **target country inflation rate**.
- **Social Security (SSI):** Begins at the age you specify and is adjusted annually using the **home country inflation rate**. SSI is applied toward expenses first; any remaining surplus is reinvested into the portfolio.
- **Lump Sum Events:** You can enter a one-time lump sum (for example, inheritance, property sale, or pension payout) received at a specific age. The amount is added to the portfolio balance in that year.
- **Investments:** Portfolio returns are simulated across equities, bonds, and cash:
  - **Equities** use *lognormal returns* to model realistic compounding and prevent impossible negative returns beyond −100%.
  - **Bonds** use *normal returns* that allow both gains and moderate losses.
  - **Cash** uses *normal returns clipped at 0%*, so cash never loses nominal value but still fluctuates slightly.
  - All three asset classes are modeled with realistic correlations using a Cholesky decomposition.
- **Inflation:** Two separate inflation rates are used — one for your home country (to adjust SSI or similar income) and one for your target country (to adjust living expenses).
- **Projection:** Each simulation represents a potential future path of your portfolio value from your current age through your expected age at death, considering income, spending, lump sums, and returns.
- **Results:** The simulator outputs key metrics:
  - Median, 10th percentile, and 90th percentile ending portfolio values.
  - Minimum and maximum ending values observed across simulations.
  - Success rate — the percentage of runs where your final portfolio exceeds your target threshold.**
''')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

st.title("Monte Carlo Retirement Simulator — Geographic Arbitrage Edition with Asset Allocation")

MAX_SIMS = 20000

# User Inputs
init = float(st.text_input("Initial portfolio ($)", value="1000000"))
spend_base = float(st.text_input("Annual spending in target country ($)", value="50000"))
current_age = int(st.text_input("Current age", value="45"))
retire_age = int(st.text_input("Retirement age", value="65"))
death_age = int(st.text_input("Age at death", value="100"))

# Work income parameters moved here
earn_income = st.checkbox("Earn income before retirement?")
gross_income = 0.0
if earn_income:
    gross_income = float(st.text_input("Gross annual pre-tax income ($)", value="50000"))

# Tax rates
work_tax_rate = float(st.text_input("Effective tax rate while working (%)", value="20.0")) / 100
retire_tax_rate = float(st.text_input("Effective tax rate during retirement (%)", value="10.0")) / 100

# Validation for age logic
if retire_age > death_age:
    st.warning("⚠️ Retirement age cannot exceed age at death. Adjusting retirement age to match death age.")
    retire_age = death_age

# If already retired (current_age >= retire_age), disable work income logic
already_retired = current_age >= retire_age
if already_retired:
    st.info("This scenario models someone already in retirement. Earned income will not be applied.")

# Inflation parameters
home_inflation_rate = float(st.text_input("Home country annual inflation rate (%)", value="2.0"))
target_inflation_rate = float(st.text_input("Target country annual inflation rate (%)", value="4.0"))

# Social Security / Pension parameters
start_ssi_age = int(st.text_input("Age to start receiving retirement income (e.g., SSI)", value="67"))
ssi_amount_today = float(st.text_input("Annual retirement income in today's dollars ($)", value="26000"))
include_ssi_taxable = st.checkbox("Include SSI in taxable income?", value=False)

# Lump sum input
st.subheader("Lump Sum Event")
receive_lump_sum = st.checkbox("Receive a lump sum in the future?")
lump_sum_amount = 0.0
lump_sum_age = None
if receive_lump_sum:
    lump_sum_amount = float(st.text_input("Lump sum amount ($)", value="100000"))
    lump_sum_age = int(st.text_input("Age when lump sum is received", value="70"))

# Asset allocation inputs
st.subheader("Portfolio Allocation")
weights_equity = float(st.text_input("% in Equities", value="60")) / 100
weights_bonds = float(st.text_input("% in Bonds", value="30")) / 100
weights_cash = float(st.text_input("% in Cash", value="10")) / 100

if abs(weights_equity + weights_bonds + weights_cash - 1.0) > 0.001:
    st.warning("⚠️ Allocation must sum to 100%. Values will be normalized.")
    total = weights_equity + weights_bonds + weights_cash
    weights_equity /= total
    weights_bonds /= total
    weights_cash /= total

# Expected returns and volatilities per asset class
mean_equity = float(st.text_input("Equity mean annual return (%)", value="10.0")) / 100
std_equity = float(st.text_input("Equity volatility (%)", value="15.0")) / 100
mean_bonds = float(st.text_input("Bond mean annual return (%)", value="3.0")) / 100
std_bonds = float(st.text_input("Bond volatility (%)", value="6.0")) / 100
mean_cash = float(st.text_input("Cash mean annual return (%)", value="2.0")) / 100
std_cash = float(st.text_input("Cash volatility (%)", value="1.0")) / 100

# Corrected conversion from arithmetic to lognormal parameters
sigma_eq_ln = np.sqrt(np.log(1 + (std_equity**2) / ((1 + mean_equity)**2)))
mu_eq_ln = np.log(1 + mean_equity) - 0.5 * sigma_eq_ln**2

# Correlation matrix (approximate typical assumptions)
corr = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])

# Covariance matrix from volatilities and correlations
stds = np.array([1.0, 1.0, 1.0])  # standardized for Cholesky
cov_matrix = np.outer(stds, stds) * corr

# Cholesky decomposition for correlated normal shocks
chol = np.linalg.cholesky(cov_matrix)

# Simulation control
sims_input = st.text_input("Number of simulations (max 20,000)", value="1000", help="For performance, simulations are capped at 20,000.")
try:
    sims_val = int(sims_input)
except ValueError:
    sims_val = 1000
    st.warning("Invalid simulations input. Using default of 1,000.")

if sims_val <= 0:
    st.warning("Number of simulations must be positive. Using 1.")
    sims_val = 1
elif sims_val > MAX_SIMS:
    st.warning(f"Simulations capped at {MAX_SIMS} for performance.")
    sims_val = MAX_SIMS
sims = sims_val
st.caption(f"Running {sims:,} simulations (cap = {MAX_SIMS:,}).")

threshold = float(st.text_input("Success threshold ($)", value="0"))

# Derived years
total_years = death_age - current_age

# Inflation decimals
home_inflation_decimal = home_inflation_rate / 100
target_inflation_decimal = target_inflation_rate / 100

# Monte Carlo simulation
results = []
final_balances = []

for _ in range(sims):
    balance = init
    path = []
    for year in range(total_years):
        age = current_age + year

        # Determine spending and income by life stage
        spend = spend_base * ((1 + target_inflation_decimal) ** year)
        income = 0

        if earn_income and age < retire_age:
            income += gross_income

        # Add SSI/retirement income after start_ssi_age, adjusted for home inflation
        if age >= start_ssi_age:
            ssi_income = ssi_amount_today * ((1 + home_inflation_decimal) ** (age - current_age))
            income += ssi_income

        # Add lump sum event if applicable
        if receive_lump_sum and age == lump_sum_age:
            balance += lump_sum_amount

        # Determine tax rate and taxable income
        effective_tax_rate = work_tax_rate if age < retire_age else retire_tax_rate
        taxable_income = 0
        if age < retire_age and earn_income:
            taxable_income += gross_income
        if include_ssi_taxable and age >= start_ssi_age:
            taxable_income += ssi_income

        tax = taxable_income * effective_tax_rate

        # Simulate correlated standard normal shocks
        rand = np.random.normal(0, 1, 3)
        correlated = chol @ rand

        # Transform to returns: equity lognormal, bonds normal, cash normal but nonnegative
        eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
        bd_ret = mean_bonds + correlated[1] * std_bonds
        cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)

        # Weighted portfolio return
        portfolio_growth = weights_equity * eq_ret + weights_bonds * bd_ret + weights_cash * cs_ret

        # Update portfolio with taxes applied
        balance = (balance + income - spend - tax) * (1 + portfolio_growth)
        path.append(balance)

    results.append(path)
    final_balances.append(path[-1])

results = np.array(results)

# Summary stats
median_final = np.median(final_balances)
p10 = np.percentile(final_balances, 10)
p90 = np.percentile(final_balances, 90)
min_final = np.min(final_balances)
max_final = np.max(final_balances)
success_rate = np.mean(np.array(final_balances) > threshold) * 100

# Plot with varied line colors and abbreviated Y-axis labels
def currency_formatter(x, pos):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"${x/1_000:.0f}K"
    else:
        return f"${x:,.0f}"

colors = cm.viridis(np.linspace(0, 1, sims))

plt.figure(figsize=(10, 6))
for i, path in enumerate(results):
    plt.plot(path, color=colors[i], alpha=0.3)

plt.title("Monte Carlo Retirement Projection (Value at Death)")
plt.xlabel("Years from Current Age")
plt.ylabel("Portfolio Value ($K / $M)")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
st.pyplot(plt)

# Display summary stats
st.subheader("Simulation Results Summary")
st.write(f"**Median Portfolio at Death:** ${median_final:,.0f}")
st.write(f"**10th Percentile:** ${p10:,.0f}")
st.write(f"**90th Percentile:** ${p90:,.0f}")
st.write(f"**Minimum Portfolio at Death:** ${min_final:,.0f}")
st.write(f"**Maximum Portfolio at Death:** ${max_final:,.0f}")
st.write(f"**Success Rate (Final > ${threshold:,.0f}):** {success_rate:.1f}%")











