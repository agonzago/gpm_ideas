#%%
import re
import numpy as np
import os
import json
import sympy as sy
import re
import numpy as np
import os
import json
import sympy as sy

class DynareParser:
    def __init__(self):
        # Core model components
        self.variables = []
        self.endogenous = []
        self.exogenous = []
        self.parameters = []
        self.param_values = {}
        self.equations = []
        self.shocks = []
        
        # Equation categories
        self.endogenous_equations = []
        self.exogenous_equations = {}
        
        # Exogenous process information
        self.exogenous_structure = {}
        self.phi_matrix = None
        self.shock_selection_matrix = None
        
        # Transformation results
        self.transformed_variables = []
        self.transformed_equations = []
        self.max_lead = 0
        self.max_lag = 0
    
    def parse_file(self, file_content):
        """
        Parse a Dynare file content into structured data
        
        Args:
            file_content (str): Content of the Dynare file
        """
        # Extract var section (endogenous variables)
        var_match = re.search(r'var\s+(.*?);', file_content, re.DOTALL)
        if var_match:
            var_section = var_match.group(1)
            var_section = re.sub(r'//.*?$', '', var_section, flags=re.MULTILINE)
            self.variables = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', var_section)]
        
        # Extract varexo section (exogenous shocks)
        varexo_match = re.search(r'varexo\s+(.*?);', file_content, re.DOTALL)
        if varexo_match:
            varexo_section = varexo_match.group(1)
            varexo_section = re.sub(r'//.*?$', '', varexo_section, flags=re.MULTILINE)
            self.shocks = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', varexo_section)]
        
        # Extract parameters
        param_match = re.search(r'parameters\s+(.*?);', file_content, re.DOTALL)
        if param_match:
            param_section = param_match.group(1)
            param_section = re.sub(r'//.*?$', '', param_section, flags=re.MULTILINE)
            self.parameters = [p.strip() for p in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', param_section)]
        
        # Extract parameter values
        for param in self.parameters:
            param_value_match = re.search(rf'{param}\s*=\s*([0-9.]+)\s*;', file_content)
            if param_value_match:
                self.param_values[param] = float(param_value_match.group(1))
        
        # Extract model equations
        model_match = re.search(r'model;(.*?)end;', file_content, re.DOTALL)
        if model_match:
            model_section = model_match.group(1)
            cleaned_lines = []
            for line in model_section.split(';'):
                line = re.sub(r'//.*?$', '', line, flags=re.MULTILINE).strip()
                if line:
                    cleaned_lines.append(line)
            self.equations = cleaned_lines
        
        # Classify variables and equations
        self.classify_variables()
        self.classify_equations()
        
        # Find max lead and lag in equations
        self.find_max_lead_lag()
    
    def classify_variables(self):
        """
        Classify variables as endogenous or exogenous based on naming patterns
        """
        self.endogenous = []
        self.exogenous = []
        
        for var in self.variables:
            # Variables starting with RES_ are typically exogenous
            if var.startswith('RES_'):
                self.exogenous.append(var)
            else:
                self.endogenous.append(var)
    
    def classify_equations(self):
        """
        Classify equations as endogenous or exogenous
        and extract information about exogenous processes
        """
        self.endogenous_equations = []
        self.exogenous_equations = {}
        
        # First pass: identify explicit exogenous process equations
        explicit_exo_eqs = []
        
        for eq in self.equations:
            eq_clean = re.sub(r'//.*?$', '', eq, flags=re.MULTILINE).strip()
            
            # Try to find equations that directly define exogenous processes
            exo_match = None
            for exo_var in self.exogenous:
                if re.search(rf'\b{exo_var}\b\s*=', eq_clean):
                    exo_match = exo_var
                    break
            
            if exo_match:
                explicit_exo_eqs.append(eq_clean)
                self.exogenous_equations[exo_match] = eq_clean
            else:
                # Check if any shock appears directly in this equation
                shock_in_eq = False
                for shock in self.shocks:
                    if re.search(rf'\b{shock}\b', eq_clean):
                        shock_in_eq = True
                        shock_match = shock
                        break
                
                if shock_in_eq:
                    # Create an implicit exogenous process
                    # Try to identify the main endogenous variable
                    endo_match = re.search(r'([A-Za-z0-9_]+)\s*=', eq_clean)
                    if endo_match:
                        endo_var = endo_match.group(1)
                        implicit_exo = f"RES_{endo_var}"
                        
                        # Create a new exogenous variable if it doesn't exist
                        if implicit_exo not in self.exogenous:
                            self.exogenous.append(implicit_exo)
                            self.variables.append(implicit_exo)
                        
                        # Create an equation for the implicit exogenous process
                        impl_eq = f"{implicit_exo} = {shock_match}"
                        self.exogenous_equations[implicit_exo] = impl_eq
                        
                        # Modify the original equation to use the implicit exogenous variable
                        mod_eq = eq_clean.replace(shock_match, implicit_exo)
                        self.endogenous_equations.append(mod_eq)
                    else:
                        # Couldn't identify main variable, keep as endogenous
                        self.endogenous_equations.append(eq_clean)
                else:
                    # Regular endogenous equation
                    self.endogenous_equations.append(eq_clean)
    
    def find_max_lead_lag(self):
        """
        Find maximum lead and lag in all endogenous equations
        """
        self.max_lead = 0
        self.max_lag = 0
        
        # Check all endogenous equations
        for eq in self.endogenous_equations:
            # Search for leads like varname(+n)
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq)
            for var, lead in lead_matches:
                self.max_lead = max(self.max_lead, int(lead))
            
            # Search for lags like varname(-n)
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
            for var, lag in lag_matches:
                self.max_lag = max(self.max_lag, int(lag))
                
        # Also check exogenous equations for lags
        for eq in self.exogenous_equations.values():
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
            for var, lag in lag_matches:
                self.max_lag = max(self.max_lag, int(lag))
    
    def transform_endogenous_model(self):
        """
        Transform the endogenous part of the Dynare model into a system with only t and t+1 variables
        """
        # Create the transformed variables list starting with original endogenous variables
        self.transformed_variables = self.endogenous.copy()
        
        # Track which variables have lags and leads and their maximum lag/lead
        var_max_lag = {}
        var_max_lead = {}
        
        # First pass - identify what needs transforming
        for eq in self.endogenous_equations:
            # Clean the equation of any comments before processing
            eq_clean = re.sub(r'//.*$', '', eq).strip()
            
            # Find all variables with leads
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq_clean)
            for var, lead in lead_matches:
                if var in self.endogenous:  # Only transform endogenous variables
                    lead_val = int(lead)
                    if lead_val >= 1:
                        var_max_lead[var] = max(var_max_lead.get(var, 0), lead_val)
            
            # Find all variables with lags
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq_clean)
            for var, lag in lag_matches:
                if var in self.endogenous:  # Only transform endogenous variables
                    lag_val = int(lag)
                    if lag_val >= 1:
                        var_max_lag[var] = max(var_max_lag.get(var, 0), lag_val)
        
        # Add lag variables to transformed variables list for endogenous variables
        for var, max_lag in var_max_lag.items():
            for lag in range(1, max_lag + 1):
                lag_suffix = str(lag) if lag > 1 else ""
                self.transformed_variables.append(f"{var}_lag{lag_suffix}")
        
        # Add lead variables beyond +1 to transformed variables list
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:
                for lead in range(1, max_lead + 1):
                    self.transformed_variables.append(f"{var}_lead{lead}")
        
        # Transform endogenous equations
        self.transformed_equations = []
        for i, eq in enumerate(self.endogenous_equations):
            # Remove comments
            eq_clean = re.sub(r'//.*$', '', eq).strip()
            transformed_eq = eq_clean
            
            # Replace leads with corresponding variables
            for lead in range(self.max_lead, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+' + str(lead) + r'\)'
                if lead == 1:
                    # For +1, use _p suffix (next period)
                    transformed_eq = re.sub(pattern, r'\1_p', transformed_eq)
                else:
                    # For +2 and higher, use _lead suffix
                    transformed_eq = re.sub(pattern, r'\1_lead' + str(lead), transformed_eq)
            
            # Replace lags with corresponding variables
            for lag in range(self.max_lag, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-' + str(lag) + r'\)'
                lag_suffix = str(lag) if lag > 1 else ""
                transformed_eq = re.sub(pattern, r'\1_lag' + lag_suffix, transformed_eq)
            
            # Split the equation on the equals sign and rearrange to standard form
            if "=" in transformed_eq:
                lhs, rhs = transformed_eq.split("=", 1)
                transformed_eq = f"{rhs.strip()} - {lhs.strip()}"
            
            self.transformed_equations.append({f"eq{i+1}": transformed_eq.strip()})
        
        # Add transition equations for lags
        eq_num = len(self.transformed_equations) + 1
        for var, max_lag in var_max_lag.items():
            # First lag: var_lag_p = var
            self.transformed_equations.append({f"eq{eq_num}": f"{var}_lag_p - {var}"})
            eq_num += 1
            
            # Additional lags: var_lagN_p = var_lag(N-1)
            for lag in range(2, max_lag + 1):
                prev_lag_suffix = str(lag-1) if lag-1 > 1 else ""
                self.transformed_equations.append({f"eq{eq_num}": f"{var}_lag{lag}_p - {var}_lag{prev_lag_suffix}"})
                eq_num += 1
        
        # Add transition equations for leads
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:
                # First lead equation: var_p = var_lead1
                self.transformed_equations.append({f"eq{eq_num}": f"{var}_p - {var}_lead1"})
                eq_num += 1
                
                # Additional lead equations: var_leadN_p = var_lead(N+1)
                for lead in range(1, max_lead):
                    self.transformed_equations.append({f"eq{eq_num}": f"{var}_lead{lead}_p - {var}_lead{lead+1}"})
                    eq_num += 1
    
    def prepare_exogenous_var(self):
        """
        Prepare exogenous processes in VAR form for JSON output and Phi matrix computation
        """
        # Skip if no exogenous variables
        if not self.exogenous:
            return
        
        # For each exogenous variable, ensure we have a standardized equation
        for exo in self.exogenous:
            if exo in self.exogenous_equations:
                # Clean up the equation
                eq_str = self.exogenous_equations[exo]
                eq_clean = re.sub(r'//.*', '', eq_str).strip()
                
                # Ensure it's in the form "variable = expression"
                if "=" in eq_clean:
                    lhs, rhs = eq_clean.split("=", 1)
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    
                    # Standardize the left side to be the variable name
                    if lhs != exo:
                        self.exogenous_equations[exo] = f"{exo} = {rhs}"
            else:
                # Create a default equation for exogenous variables without one
                # Try to match with a shock by name
                shock_name = None
                for shock in self.shocks:
                    base_name = exo[4:] if exo.startswith('RES_') else exo
                    if shock.endswith(base_name) or base_name in shock:
                        shock_name = shock
                        break
                
                if shock_name:
                    self.exogenous_equations[exo] = f"{exo} = {shock_name}"
                else:
                    # If no matching shock found, use a generic equation
                    self.exogenous_equations[exo] = f"{exo} = 0"

    def generate_jacobian_evaluator(self, output_file=None):
        """
        Generate a Python function that evaluates the Jacobian matrices for the model
        
        Args:
            output_file (str, optional): Path to save the generated Python code
                
        Returns:
            str: The generated Python code for the Jacobian evaluator
        """
        # Basic model components
        variables = self.transformed_variables
        exogenous = self.exogenous
        parameters = self.parameters
        
        # Create variables with "_p" suffix for t+1 variables
        variables_p = [var + "_p" for var in variables]
        
        # Create symbolic variables
        var_symbols = {var: sy.symbols(var) for var in variables}
        var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
        exo_symbols = {exo: sy.symbols(exo) for exo in exogenous}
        param_symbols = {param: sy.symbols(param) for param in parameters}
        
        # Combine all symbols
        all_symbols = {**var_symbols, **var_p_symbols, **exo_symbols, **param_symbols}
        
        # Parse endogenous equations into sympy expressions
        equations = []
        for eq_dict in self.transformed_equations:
            for eq_name, eq_str in eq_dict.items():
                # Convert string to sympy expression
                eq_expr = eq_str
                for name, symbol in all_symbols.items():
                    # Use regex to match whole words only
                    pattern = r'\b' + re.escape(name) + r'\b'
                    eq_expr = re.sub(pattern, str(symbol), eq_expr)
                
                # Try to parse the expression
                try:
                    expr = sy.sympify(eq_expr)
                    equations.append(expr)
                except Exception as e:
                    print(f"Failed to parse equation {eq_name}: {eq_str}")
                    print(f"Error: {str(e)}")
        
        # Create system as sympy Matrix
        F = sy.Matrix(equations)
        
        # Compute Jacobians for endogenous system
        X_symbols = [var_symbols[var] for var in variables]
        X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
        Z_symbols = [exo_symbols[exo] for exo in exogenous]
        
        # A = ∂F/∂X_p (Jacobian with respect to future variables)
        A_symbolic = F.jacobian(X_p_symbols)
        
        # B = -∂F/∂X (negative Jacobian with respect to current variables)
        B_symbolic = -F.jacobian(X_symbols)
        
        # C = -∂F/∂Z (negative Jacobian with respect to exogenous processes)
        C_symbolic = -F.jacobian(Z_symbols)
        
        # Compute Phi matrix for exogenous processes
        # We need to handle the exogenous system with lags
        # First, identify all exogenous variables including their lags
        exo_with_lags = []
        for exo in exogenous:
            exo_with_lags.append(exo)
            for lag in range(1, self.max_lag + 1):
                if any(f"{exo}_lag{lag}" in eq for eq in self.exogenous_equations.values()):
                    exo_with_lags.append(f"{exo}_lag{lag}")
        
        # Create symbolic variables for exogenous system
        exo_lag_symbols = {var: sy.symbols(var) for var in exo_with_lags}
        
        # Parse exogenous equations
        exo_equations = []
        for exo, eq_str in self.exogenous_equations.items():
            if "=" in eq_str:
                # Clean and split the equation
                eq_clean = re.sub(r'//.*', '', eq_str).strip()
                lhs, rhs = eq_clean.split("=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Convert to symbolic form
                eq_sym = rhs
                
                # Replace variables with symbols
                for var, symbol in exo_lag_symbols.items():
                    pattern = r'\b' + re.escape(var) + r'\b'
                    eq_sym = re.sub(pattern, str(symbol), eq_sym)
                
                # Replace parameters with symbols
                for param, symbol in param_symbols.items():
                    pattern = r'\b' + re.escape(str(param)) + r'\b'
                    eq_sym = re.sub(pattern, str(symbol), eq_sym)
                
                try:
                    # Parse the expression
                    expr = sy.sympify(eq_sym) - exo_lag_symbols[exo]
                    exo_equations.append(expr)
                except Exception as e:
                    print(f"Error parsing exogenous equation for {exo}: {eq_clean}")
                    print(f"Error: {str(e)}")
        
        # Create exogenous system as sympy Matrix
        G = sy.Matrix(exo_equations)
        
        # Compute Phi = -∂G/∂(exo_with_lags) Jacobian for VAR system
        Phi_symbolic = G.jacobian([exo_lag_symbols[var] for var in exo_with_lags])
        
        # Generate code for the Jacobian evaluation function
        function_code = [
            "import numpy as np",
            "",
            "def evaluate_jacobians(theta):",
            "    \"\"\"",
            "    Evaluates Jacobian matrices for the Klein method and VAR representation",
            "    ",
            "    Args:",
            "        theta: List or array of parameter values in the order of:",
            f"              {parameters}",
            "        ",
            "    Returns:",
            "        a: Matrix ∂F/∂X_p (Jacobian with respect to future variables)",
            "        b: Matrix -∂F/∂X (negative Jacobian with respect to current variables)",
            "        c: Matrix -∂F/∂Z (negative Jacobian with respect to exogenous processes)",
            "        phi: Matrix for VAR representation of exogenous processes",
            "    \"\"\"",
            "    # Unpack parameters from theta"
        ]
        
        # Add parameter unpacking
        for i, param in enumerate(parameters):
            function_code.append(f"    {param} = theta[{i}]")
        
        # Initialize matrices
        function_code.extend([
            "",
            f"    a = np.zeros(({len(equations)}, {len(variables)}))",
            f"    b = np.zeros(({len(equations)}, {len(variables)}))",
            f"    c = np.zeros(({len(equations)}, {len(exogenous)}))",
            f"    phi = np.zeros(({len(exo_equations)}, {len(exo_with_lags)}))",
            ""
        ])
        
        # Add A matrix elements
        for i in range(A_symbolic.rows):
            for j in range(A_symbolic.cols):
                if A_symbolic[i, j] != 0:
                    expr = str(A_symbolic[i, j])
                    for param in parameters:
                        expr = expr.replace(param, param)
                    function_code.append(f"    a[{i}, {j}] = {expr}")
        
        # Add B matrix elements
        function_code.append("")
        for i in range(B_symbolic.rows):
            for j in range(B_symbolic.cols):
                if B_symbolic[i, j] != 0:
                    expr = str(B_symbolic[i, j])
                    for param in parameters:
                        expr = expr.replace(param, param)
                    function_code.append(f"    b[{i}, {j}] = {expr}")
        
        # Add C matrix elements
        function_code.append("")
        for i in range(C_symbolic.rows):
            for j in range(C_symbolic.cols):
                if C_symbolic[i, j] != 0:
                    expr = str(C_symbolic[i, j])
                    for param in parameters:
                        expr = expr.replace(param, param)
                    function_code.append(f"    c[{i}, {j}] = {expr}")
        
        # Add Phi matrix elements
        function_code.append("")
        for i in range(Phi_symbolic.rows):
            for j in range(Phi_symbolic.cols):
                if Phi_symbolic[i, j] != 0:
                    expr = str(Phi_symbolic[i, j])
                    for param in parameters:
                        expr = expr.replace(param, param)
                    function_code.append(f"    phi[{i}, {j}] = {expr}")
        
        # Return all matrices
        function_code.extend([
            "",
            "    return a, b, c, phi"
        ])
        
        # Join all lines to form the complete function code
        complete_code = "\n".join(function_code)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(complete_code)
        
        return complete_code

    def prepare_exogenous_var(self):
        """
        Prepare exogenous processes in VAR form for JSON output and Phi matrix computation
        
        This function transforms the exogenous equations to handle lags properly:
        1. Each variable with a lag is transformed to use _lag{n} notation
        2. Equations are standardized to the form "variable = expression"
        """
        # Skip if no exogenous variables
        if not self.exogenous:
            return
        
        # First, transform equations to use _lag notation instead of (-n) notation
        transformed_equations = {}
        
        for exo, eq_str in self.exogenous_equations.items():
            # Clean up the equation
            eq_clean = re.sub(r'//.*', '', eq_str).strip()
            
            # Replace lags in the form variable(-n) with variable_lag{n}
            transformed_eq = eq_clean
            for lag in range(self.max_lag, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-' + str(lag) + r'\)'
                lag_suffix = str(lag) if lag > 1 else ""
                transformed_eq = re.sub(pattern, r'\1_lag' + lag_suffix, transformed_eq)
            
            # Ensure the equation is in the form "variable = expression"
            if "=" in transformed_eq:
                lhs, rhs = transformed_eq.split("=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Standardize the left side to be the variable name
                if lhs != exo:
                    transformed_eq = f"{exo} = {rhs}"
            
            transformed_equations[exo] = transformed_eq
        
        # Update exogenous equations with transformed versions
        self.exogenous_equations = transformed_equations
        
        # Create default equations for any exogenous variables without one
        for exo in self.exogenous:
            if exo not in self.exogenous_equations:
                # Try to match with a shock by name
                shock_name = None
                for shock in self.shocks:
                    base_name = exo[4:] if exo.startswith('RES_') else exo
                    if shock.endswith(base_name) or base_name in shock:
                        shock_name = shock
                        break
                
                if shock_name:
                    self.exogenous_equations[exo] = f"{exo} = {shock_name}"
                else:
                    # If no matching shock found, use a generic equation
                    self.exogenous_equations[exo] = f"{exo} = 0"

    def save_model_to_json(self, output_file):
        """
        Save the parsed and transformed model to a JSON file
        
        Args:
            output_file (str): Path to the output JSON file
        """
        # Ensure exogenous equations are in the right format
        self.prepare_exogenous_var()
        
        # Convert transformed equations from list of dicts to a single dict
        endogenous_eqs = {}
        for eq_dict in self.transformed_equations:
            for key, value in eq_dict.items():
                endogenous_eqs[key] = value
        
        # Create the model data structure
        model_data = {
            "endogenous_variables": self.transformed_variables,
            "exogenous_variables": self.exogenous,
            "parameters": self.parameters,
            "parameter_values": self.param_values,
            "shocks": self.shocks,
            "endogenous_equations": endogenous_eqs,  # This should include the transformed equations
            "original_equations": self.equations,    # Also include the original equations
            "exogenous_equations": self.exogenous_equations,
            "max_lead": self.max_lead,
            "max_lag": self.max_lag
        }
        
        # If phi_matrix is computed, include it
        if hasattr(self, 'phi_matrix') and self.phi_matrix is not None:
            model_data["phi_matrix"] = self.phi_matrix.tolist()
        
        # If shock_selection_matrix is computed, include it
        if hasattr(self, 'shock_selection_matrix') and self.shock_selection_matrix is not None:
            model_data["shock_selection_matrix"] = self.shock_selection_matrix.tolist()
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        return model_data
                    

import sys
import json
import numpy as np
from parser_2 import DynareParser

def test_jacobian():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
    print(f"Processing DSGE model from {dynare_file}...")
    
    #simple_model = sys.path.join("Simplest", "qpm_simpl1.dyn")
    # Create parser instance
    parser = DynareParser()
    
    # Parse the model string
    parser.parse_file(dynare_file)
    
    # Transform the model
    parser.transform_endogenous_model()
    
    # Prepare exogenous processes
    parser.prepare_exogenous_var()
    
    # Save the model to JSON
    model_data = parser.save_model_to_json("test_model.json")
    
    # Print the model data to verify
    print("Model data:")
    print("Endogenous variables:", model_data["endogenous_variables"])
    print("Exogenous variables:", model_data["exogenous_variables"])
    print("Parameters:", model_data["parameters"])
    print("\nEndogenous equations:")
    for eq_name, eq in model_data["endogenous_equations"].items():
        print(f"{eq_name}: {eq}")
    
    # Generate Jacobian evaluator code
    jacobian_code = parser.generate_jacobian_evaluator("test_jacobian_evaluator.py")
    
    # Write the code to a file
    with open("test_jacobian_evaluator.py", "w") as f:
        f.write(jacobian_code)
    
    # Import the module (dynamically for test purposes)
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_jacobian_evaluator", "test_jacobian_evaluator.py")
    jacobian_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jacobian_module)
    
    # Get the parameter values
    theta = []
    for param in model_data["parameters"]:
        theta.append(model_data["parameter_values"].get(param, 0.0))
    
    # Evaluate the Jacobians
    try:
        a, b, c, phi = jacobian_module.evaluate_jacobians(theta)
        
        print("\nJacobian evaluation successful!")
        print("\nA matrix (with respect to future variables):")
        print(a)
        print("\nB matrix (with respect to current variables):")
        print(b)
        print("\nC matrix (with respect to exogenous processes):")
        print(c)
        print("\nPhi matrix (VAR representation):")
        print(phi)
        
        return True
    except Exception as e:
        print("\nJacobian evaluation failed!")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_jacobian()
