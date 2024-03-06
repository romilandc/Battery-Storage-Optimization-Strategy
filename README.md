# Task Summary

We're constructing a simple operational trading strategy to maximize revenue from a hypothetical battery by Buying and selling electricity during the hold-out period at the nodes aeci_lmp, mich_lmp, minn_lmp.

The provided `model_ready.parquet` file contains a time series dataset with energy-related feature columns, a `row_type` column for train/hold-out separation, and three target columns representing electricity prices at different grid nodes.  Prices in the holdout dataset are assumed to be 'forecasted' prices (in a real world operation these would be replaced with actual forecasted prices at these nodes).

##Battery Assumptions

- Maximum total charge level: 10 MWh
- Initial charge level: Fully charged
- Instantaneous charge/discharge
- Efficiency factor: 0.80 for both charge and discharge
- No simultaneous charging and discharging
- Battery cannot discharge more energy than available
- Battery cannot store more energy than maximum capacity
- No simultaneous charging and discharging

##Trading Assumptions

- Trading fees: $1 per MWh for both buy and sell transactions
- Buy/sell orders must be submitted one hour prior to execution
- Only one buy or sell order per grid node per time slice
- Participation in any or all three grid nodes concurrently

