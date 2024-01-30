# Task Summary

**Objective:**

Construct and backtest a simple operational strategy to maximize revenue from a hypothetical battery by buying and selling electricity during the hold-out period at the nodes aeci_lmp, mich_lmp, minn_lmp.

**Dataset:**

The provided `model_ready.parquet` file contains a time series dataset with energy-related feature columns, a `row_type` column for train/hold-out separation, and three target columns representing electricity prices at different grid nodes.

**Battery Specifications:**

- Maximum total charge level: 10 MWh
- Initial charge level: Fully charged
- Instantaneous charge/discharge
- Efficiency factor: 0.80 for both charge and discharge
- No simultaneous charging and discharging

**Trading Rules:**

- Trading fees: $1 per MWh for both buy and sell transactions
- Buy/sell orders must be submitted one hour prior to execution
- Only one buy or sell order per grid node per time slice
- Participation in any or all three grid nodes concurrently

**Additional Assumptions:**

- Battery cannot discharge more energy than available
- Battery cannot store more energy than maximum capacity
- No simultaneous charging and discharging

**Benchmark Solution:**

- Always sell at hour-X, buy at hour-Y

**Deliverables:**

- Description of the operational strategy
- Backtesting results and performance comparison against the benchmark
- Validation of constraint adherence during all operating time slices
