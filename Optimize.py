import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo import *

# Battery parameters
mcp = 10 # Max Charge Power MWh
mdp = 10 # Max Discharge Power MWh
e = 0.80 # round trip efficiency
fee = 1  # trade fees on both Buy and Sell trades in $/MWhr

# Read the data
df = pd.read_parquet('model_ready.parquet')
df['Datetime'] = df.index
df.reset_index(drop=True, inplace=True)

# Split the data into train and hold_out segments
train_data = df[df['row_type'] == 'train']
hold_out = df[df['row_type'] == 'hold_out']

# Spot prices for each time slice
aeci_lmp = hold_out["da_energy_aeci_lmpexpost_ac"]
mich_lmp = hold_out["da_energy_michigan_hub_lmpexpost_ac"]
minn_lmp = hold_out["da_energy_minn_hub_lmpexpost_ac"]

price = [aeci_lmp, mich_lmp, minn_lmp]

df_results = pd.DataFrame()

def battery_model(mcp, mdp, e, fee):

    model = pyo.ConcreteModel()
    
    # Time period for hold_out segment
    T = len(hold_out)
    
    # Sets
    model.t = pyo.Set(initialize=pyo.RangeSet(1,T))
    model.n = pyo.Set(initialize=[1,2,3])
    
    # Parameters
    model.min_cap = 0 #no negative discharging
    model.max_cap = mcp #don’t charge over 10Mwh
    model.fee = fee #trade fee in $/MWh
    
    # Variables
    model.buy = pyo.Var(model.t, model.n, bounds=(0,mcp), initialize=0) # Buy from grid
    model.sell = pyo.Var(model.t, model.n, bounds=(0,mdp),initialize=0) # Sell to grid
    model.C = pyo.Var(model.t, bounds=(0, mcp), initialize=0) # Battery state
    
    ###Constraints###
    #Battery state - set initial state to max capacity and accounts for efficiency      
    def storage_state(model, t):
        #set first hour at max charge
        if t == model.t.first():
            return model.C[t] == model.max_cap
        else:
            return model.C[t] == (model.C[t-1] + sum(model.buy[t, n]*e for n in model.n) - sum(model.sell[t, n]/e for n in model.n))
    
    model.storage_state = pyo.Constraint(model.t, rule=storage_state)

    #make sure battery does not charge above the limit
    def over_charge(model, t, n):
        return model.buy[t, n] <= (model.max_cap - model.C[t])/e
    
    model.over_charge = pyo.Constraint(model.t, model.n, rule=over_charge)
    
    def over_discharge(model, t, n):
        return model.sell[t, n] <= model.C[t]*e
    
    model.over_discharge = pyo.Constraint(model.t, model.n, rule=over_discharge)

    #Battery cannot store more energy than its maximum capacity
    def charge_less_than_capacity(model, t):
        return model.C[t] <= model.max_cap
    
    model.charge_constraint = pyo.Constraint(model.t, rule=charge_less_than_capacity)
    
    #Battery cannot be zero
    def zero_bound_capacity(model, t):
        return model.C[t] >= model.min_cap
    
    model.zero_bound_capacity = pyo.Constraint(model.t, rule=zero_bound_capacity)
    
    #Positive Buy/Sell values only
    def buy_positive(model, t, n):
        return model.buy[t,n] >= 0

    model.buy_positive = pyo.Constraint(model.t, model.n, rule=buy_positive)
    
    def sell_positive(model, t, n):
        return model.sell[t,n] >= 0

    model.sell_positive = pyo.Constraint(model.t, model.n, rule=sell_positive)
    
    ##OBJECTIVE FUNCTION

    def objective(model):
        profit = sum((model.sell[t,n] * (price[n-1][t-1] - fee)) - (model.buy[t,n] * (price[n-1][t-1] + fee)) for t in model.t for n in model.n)
        return profit
   
    model.objective = pyo.Objective(rule=objective,sense=pyo.maximize)
    
    #Solve the model
    #enter path to the glpsol.exe to your glpk package
    solverpath_exe='C://Users//~//anaconda3//pkgs//glpk-5.0-h8ffe710_0//Library//bin//glpsol.exe'
    
    solver = pyo.SolverFactory('glpk', executable=solverpath_exe)
    results = solver.solve(model, tee=False)
    
    profit = model.objective.expr()
    #print(f"Total profit: ${profit:.2f}")

    #RESULTS#    
    def dataframe(model, aeci_lmp, mich_lmp, minn_lmp):
        
        charge_hist = [pyo.value(x) for x in model.C.values()]
        buy_hist = [pyo.value(x) for x in model.buy.values()]
        sell_hist = [pyo.value(x) for x in model.sell.values()]
    
        ###code to separate the buys and sells by node
        buy_by_node = {node: [] for node in range(3)}
        sell_by_node = {node: [] for node in range(3)}
        
        #loop through all buy and sell orders and append to each node in dict
        for i in range(len(buy_hist)):
            node = i % 3
            buy_by_node[node].append(buy_hist[i])
            
        for i in range(len(sell_hist)):
            node = i % 3
            sell_by_node[node].append(sell_hist[i])   

        #Calculate profit by hour
        AECI_profit = [sell_by_node[0][i]*(price[0][i] - fee) - buy_by_node[0][i]*(price[0][i] + fee) for i in range(len(sell_by_node[0]))]
        Mich_profit = [sell_by_node[1][i]*(price[1][i] - fee) - buy_by_node[1][i]*(price[1][i] + fee) for i in range(len(sell_by_node[1]))]
        Minn_profit = [sell_by_node[2][i]*(price[2][i] - fee) - buy_by_node[2][i]*(price[2][i] + fee) for i in range(len(sell_by_node[2]))]
    
        #create df
        df_results = pd.DataFrame({'Datetime': hold_out['Datetime'],
                           'hour': [dt.hour for dt in hold_out['Datetime']],
                           'Charge State(KWh)': charge_hist,
                           'AECI Buys(KW)': buy_by_node[0],
                           'AECI Sells(KW)': sell_by_node[0],
                           'AECI LMP($)': price[0],
                           'AECI Profit': AECI_profit,
                           'Mich Buys(KW)': buy_by_node[1],
                           'Mich Sells(KW)': sell_by_node[1],
                           'Mich_LMP($)': price[1],
                           'Mich Profit': Mich_profit,
                           'Minn Buys(KW)': buy_by_node[2],
                           'Minn Sells(KW)': sell_by_node[2],
                           'Minn LMP($)': price[2],
                           'Minn Profit': Minn_profit})
    
        #Calculate cumulative profit
        df_results['Cumulative Profit'] = (df_results['AECI Profit']+df_results['Mich Profit'] + df_results['Minn Profit']).cumsum()
        print(df_results['Cumulative Profit'].iloc[-1])
    
    return(profit, df_results)

battery_model(mcp,mdp,e,fee)


# Plot the strategy and cumulative profit over time
import matplotlib.pyplot as plt

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line plot for cumulative profit
ax1.plot(df_results['Datetime'], df_results['Cumulative Profit'], label='Cumulative Profit')
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Cumulative Profit ($)', color='k')
ax1.tick_params(axis='y', labelcolor='k')

# Twin the axes for buy/sell plots
ax2 = ax1.twinx()

# Line plots for buy/sell of each node
ax2.plot(df_results['Datetime'], df_results['AECI Buys(KW)'], label='AECI Buys', color='g')
ax2.plot(df_results['Datetime'], df_results['AECI Sells(KW)'], label='AECI Sells', color='r')
ax2.plot(df_results['Datetime'], df_results['Mich Buys(KW)'], label='Mich Buys', color='c')
ax2.plot(df_results['Datetime'], df_results['Mich Sells(KW)'], label='Mich Sells', color='m')
ax2.plot(df_results['Datetime'], df_results['Minn Buys(KW)'], label='Minn Buys', color='y')
ax2.plot(df_results['Datetime'], df_results['Minn Sells(KW)'], label='Minn Sells', color='k')
ax2.set_ylabel('Buy/Sell (KW)', color='k')
ax2.tick_params(axis='y', labelcolor='k')

# Set labels and title
plt.title('Battery Operations and Profit Over Time')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
