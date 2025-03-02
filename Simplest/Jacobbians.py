#%%
import sympy as sy
import numpy as np
import re

def derive_jacobians_economic_model(equations_dict, variables, shocks, parameters):
    """
    Derives symbolic Jacobian matrices for an economic model with the form F(X, X_p, W, θ) = 0
    where X are variables, X_p are the same variables with "_p" suffix, and W are shocks.
    
    Computes:
    - A = -∂F/∂X_p (negative Jacobian with respect to variables with "_p" suffix)
    - B = ∂F/∂X (Jacobian with respect to variables without "_p" suffix)
    - C = ∂F/∂W (Jacobian with respect to shock variables)
    
    Args:
        equations_dict: Dictionary of equations in the format {eq_name: equation_string}
        variables: List of variable names (without "_p" suffix)
        shocks: List of shock variable names
        parameters: List of parameter names (theta vector)
        
    Returns:
        A Python script as a string that defines a function to efficiently evaluate Jacobians
    """
    # Create the list of variables with "_p" suffix
    variables_p = [var + "_p" for var in variables]
    
    # Create symbolic variables
    var_symbols = {var: sy.symbols(var) for var in variables}
    var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
    shock_symbols = {shock: sy.symbols(shock) for shock in shocks}
    param_symbols = {param: sy.symbols(param) for param in parameters}
    
    # Combine all symbols in a dictionary
    all_symbols = {**var_symbols, **var_p_symbols, **shock_symbols, **param_symbols}
    
    # Parse equations into sympy expressions
    equations = []
    for eq_name, eq_str in equations_dict.items():
        # Convert string to sympy expression
        eq_expr = eq_str
        for name, symbol in all_symbols.items():
            eq_expr = eq_expr.replace(name, str(symbol))
        # Parse the equation of form "lhs - (rhs)"
        match = re.search(r'(.*) - \((.*)\)', eq_expr)
        if match:
            lhs = match.group(1)
            rhs = match.group(2)
            # Create expression of form lhs - rhs = 0
            expr = sy.sympify(lhs) - sy.sympify(rhs)
            equations.append(expr)
    
    # Create system as sympy Matrix
    F = sy.Matrix(equations)
    
    # Compute Jacobians
    X_symbols = [var_symbols[var] for var in variables]
    X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
    W_symbols = [shock_symbols[shock] for shock in shocks]
    
    # B = ∂F/∂X (with respect to variables without "_p")
    B_symbolic = F.jacobian(X_symbols)
    
    # A = -∂F/∂X_p (negative Jacobian with respect to variables with "_p")
    A_symbolic = -F.jacobian(X_p_symbols)
    
    # C = ∂F/∂W (with respect to shock variables)
    C_symbolic = F.jacobian(W_symbols)
    
    # Generate Python function that evaluates these Jacobians efficiently
    function_code = [
        "import numpy as np",
        "",
        "def evaluate_jacobians(theta):",
        "    \"\"\"",
        "    Evaluates Jacobian matrices for an economic model based on parameter values.",
        "    ",
        "    Args:",
        "        theta: List or array of parameter values in the order of:",
        f"              {parameters}",
        "        ",
        "    Returns:",
        "        A: Matrix -∂F/∂X_p (negative Jacobian with respect to variables with '_p' suffix)",
        "        B: Matrix ∂F/∂X (Jacobian with respect to variables without '_p' suffix)",
        "        C: Matrix ∂F/∂W (Jacobian with respect to shock variables)",
        "    \"\"\"",
        "    # Unpack parameters from theta",
    ]
    
    # Add parameter unpacking
    for i, param in enumerate(parameters):
        function_code.append(f"    {param}_ = theta[{i}]")
    
    # Initialize A, B, and C matrices
    function_code.extend([
        "",
        f"    A = np.zeros(({len(equations)}, {len(variables)}))",
        f"    B = np.zeros(({len(equations)}, {len(variables)}))",
        f"    C = np.zeros(({len(equations)}, {len(shocks)}))",
        ""
    ])
    
    # Add A matrix elements
    for i in range(A_symbolic.rows):
        for j in range(A_symbolic.cols):
            if A_symbolic[i, j] != 0:
                expr = str(A_symbolic[i, j])
                for param in parameters:
                    expr = expr.replace(param, f"{param}_")
                function_code.append(f"    A[{i}, {j}] = {expr}")
    
    # Add B matrix elements
    function_code.append("")
    for i in range(B_symbolic.rows):
        for j in range(B_symbolic.cols):
            if B_symbolic[i, j] != 0:
                expr = str(B_symbolic[i, j])
                for param in parameters:
                    expr = expr.replace(param, f"{param}_")
                function_code.append(f"    B[{i}, {j}] = {expr}")
    
    # Add C matrix elements
    function_code.append("")
    for i in range(C_symbolic.rows):
        for j in range(C_symbolic.cols):
            if C_symbolic[i, j] != 0:
                expr = str(C_symbolic[i, j])
                for param in parameters:
                    expr = expr.replace(param, f"{param}_")
                function_code.append(f"    C[{i}, {j}] = {expr}")
    
    # Return A, B, and C
    function_code.extend([
        "",
        "    return A, B, C"
    ])
    
    return "\n".join(function_code)

# Example for the macro model
if __name__ == "__main__":
    # Convert the equations from the list of dictionaries to a single dictionary
    equations = {
        "eq1": "(1-b1)*L_GDP_GAP_p + b1*L_GDP_GAP_lag - b4*RR_GAP_p + SHK_L_GDP_GAP - (L_GDP_GAP)",
        "eq2": "a1*DLA_CPI_lag + (1-a1)*DLA_CPI_p + a2*L_GDP_GAP + SHK_DLA_CPI - (DLA_CPI)",
        "eq3": "g1*RS_lag + (1-g1)*(DLA_CPI_p + g2*DLA_CPI_lead3 + g3*L_GDP_GAP) + SHK_RS - (RS)",
        "eq4": "RS - DLA_CPI_p - (RR_GAP)",
        "eq5": "L_GDP_GAP_lag_p - (L_GDP_GAP_lag)",
        "eq6": "DLA_CPI_lag_p - (DLA_CPI_lag)",
        "eq7": "RS_lag_p - (RS_lag)",
        "eq8": "DLA_CPI_p - (DLA_CPI_lead1)",
        "eq9": "DLA_CPI_lead1_p - (DLA_CPI_lead2)",
        "eq10": "DLA_CPI_lead2_p - (DLA_CPI_lead3)"
    }

    variables = ["L_GDP_GAP", "DLA_CPI", "RS", "RR_GAP", "L_GDP_GAP_lag", 
                "DLA_CPI_lag", "RS_lag", "DLA_CPI_lead1", "DLA_CPI_lead2", "DLA_CPI_lead3"]
    
    shocks = ["SHK_L_GDP_GAP", "SHK_DLA_CPI", "SHK_RS"]
    
    parameters = ["b1", "b4", "a1", "a2", "g1", "g2", "g3"]
    
    # Generate the evaluation function
    evaluator_code = derive_jacobians_economic_model(equations, variables, shocks, parameters)
    
    print("=" * 60)
    print("GENERATED FUNCTION CODE:")
    print("=" * 60)
    print(evaluator_code)
    
    # Save to a file
    with open("complete_economic_model_jacobian_evaluator.py", "w") as f:
        f.write(evaluator_code)
    
    print("\nThe function has been saved to 'complete_economic_model_jacobian_evaluator.py'")
    
    