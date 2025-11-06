import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Monte Carlo Retirement Simulator")

# User Inputs
init = st.number_input("Initial portfolio ($)", value=1_000_000)
spend_base = st.number_input("Annual spending ($)", value=60_000)
mean = st.number_input("Mean annual return (%)", value=5.0)
std = st.number_input("Volatility (%)", value=12.0)
years = st.slider("Years in retirement", 10, 50, 30)
sims = st.slider("Number of simulations", 100, 5000, 1000, step=100)
threshold = st.number_input("Success threshold ($)", value=0)

# Distribution selection
dist_type = st.selectbox("Return distribution", ["Normal", "Lognormal"])

# Inflation toggle
use_inflation = st.checkbox("Adjust spending for inflation?")
inflation_rate = 0.0
if use_inflation:
    inflation_rate = st.number_input("Annual inflation rate (%)", value=2.0)

# Monte Carlo simulation
results = []
final_balances = []

# Convert to decimals
mean_decimal = mean / 100
std_decimal = std / 100
inflation_decimal = inflation_rate / 100

# Adjust lognormal parameters if selected
if dist_type == "Lognormal":
    mu = np.log(1 + mean_decimal) - 0.5 * (std_decimal**2)
    sigma = np.sqrt(np.log(1 + (std_decimal**2)))

for _ in range(sims):
    balance = init
    path = []
    for year in range(years):
        # Spending with inflation
        spend = spend_base * ((1 + inflation_decimal) ** year) if use_inflation else spend_base

        # Draw return
        if dist_type == "Normal":
            growth = np.random.normal(mean_decimal, std_decimal)
        else:
            growth = np.random.lognormal(mu, sigma) - 1  # convert back to % return

        balance = (balance - spend) * (1 + growth)
        path.append(balance)
    results.append(path)
    final_balances.append(path[-1])

results = np.array(results)

# Summary stats
median_final = np.median(final_balances)
p10 = np.percentile(final_balances, 10)
p90 = np.percentile(final_balances, 90)
success_rate = np.mean(np.array(final_balances) > threshold) * 100

# Plot
plt.figure(figsize=(10, 6))
plt.plot(results.T, color='gray', alpha=0.05)
plt.title("Monte Carlo Retirement Projection")
plt.xlabel("Years")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
st.pyplot(plt)

# Display summary stats
st.subheader("Simulation Results Summary")
st.write(f"**Median Final Portfolio:** ${median_final:,.0f}")
st.write(f"**10th Percentile:** ${p10:,.0f}")
st.write(f"**90th Percentile:** ${p90:,.0f}")
st.write(f"**Success Rate (Final > ${threshold:,.0f}):** {success_rate:.1f}%")

