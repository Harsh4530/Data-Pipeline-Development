
# 📦 Task-4: Optimization Model
# ✅ Linear Programming using PuLP

from pulp import LpMaximize, LpProblem, LpVariable, value

# ======================================
# 🔷 Step 1: Define the problem

problem = LpProblem("Maximize_Profit", LpMaximize)

# Decision variables: number of units of P1 and P2
x1 = LpVariable("P1_units", lowBound=0, cat='Continuous')
x2 = LpVariable("P2_units", lowBound=0, cat='Continuous')

# ======================================
# 🔷 Step 2: Objective function

# Profit = 20*x1 + 30*x2
problem += 20*x1 + 30*x2, "Total_Profit"

# ======================================
# 🔷 Step 3: Constraints

# Labor: 1*x1 + 2*x2 <= 40
problem += 1*x1 + 2*x2 <= 40, "Labor_Constraint"

# Material: 3*x1 + 2*x2 <= 120
problem += 3*x1 + 2*x2 <= 120, "Material_Constraint"

# ======================================
# 🔷 Step 4: Solve the Problem
problem.solve()


# 🔷 Step 5: Output the results
# ======================================
print("✅ Optimization Results:")
print(f"Produce {x1.varValue:.2f} units of Product P1")
print(f"Produce {x2.varValue:.2f} units of Product P2")
print(f"Maximum Profit = ${value(problem.objective):.2f}")