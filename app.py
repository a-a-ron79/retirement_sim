import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Monte Carlo Retirement Simulator")

# Inputs
init = st.number_input("Initial portfolio ($)", value=1_000_000)
spend = st.number_input("Annual spending ($)", value=60_000)
mean = st.number_input("Mean annual return (%)", value=5.0)
std = st.number_input("Volatility (%)", value=12.0)
years = st.slider("Years in retirement", 10, 50, 30)

# Monte Carlo simulation
sims = 1000
results = []
for _ in range(sims):
    balance = init
    path = []
    for _ in range(years):
        growth = np.random.normal(mean/100, std/100)
        balance = (balance - spend) * (1 + growth)
        path.append(balance)
    results.append(path)

plt.figure()
plt.plot(np.array(results).T, color='gray', alpha=0.05)
plt.title("Monte Carlo Retirement Projection")
plt.xlabel("Years")
plt.ylabel("Portfolio Value ($)")
st.pyplot(plt)