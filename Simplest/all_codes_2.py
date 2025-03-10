#%%
import numpy as np
import sympy as sy
import scipy.linalg as la
import re
import os
import json
import matplotlib.pyplot as plt
import sys

import numpy as np
import sympy as sy
import scipy.linalg as la
import re
import os
import json
import matplotlib.pyplot as plt
import sys

import re
import os
import json

class DynareParser:
    def __init__(self):
        # Variable storage
        self.endogenous = []
        self.exogenous = []
        self.shocks = []
        self.parameters = {}
        
        # State tracking
        self.endogenous_states = []
        self.exogenous_states = []
        self.non_predetermined = []
        
        # Equation storage
        self.endogenous_equations = []
        self.exogenous_equations = {}
        self.transformed_equations = {}
        
        # Lag/lead tracking
        self.var_max_lag = {}
        self.var_max_lead = {}
        self.max_lag = 0
        self.max_lead = 0

    def transform_endogenous_model(self):
        """Complete transformation of endogenous model equations"""
        print("Transforming endogenous model...")
        
        # Reset state tracking
        self.endogenous_states = []
        self.exogenous_states = self.exogenous.copy()
        self.non_predetermined = self.endogenous.copy()
        self.transformed_equations = {}

        # First pass - detect max lags/leads
        for eq in self.endogenous_equations:
            # Detect leads
            leads = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq)
            for var, lead in leads:
                lead = int(lead)
                if var in self.endogenous and lead > self.var_max_lead.get(var, 0):
                    self.var_max_lead[var] = lead

            # Detect lags
            lags = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
            for var, lag in lags:
                lag = int(lag)
                if var in self.endogenous and lag > self.var_max_lag.get(var, 0):
                    self.var_max_lag[var] = lag

        # Process endogenous lags
        eq_num = 1
        for var in self.endogenous:
            max_lag = self.var_max_lag.get(var, 0)
            for lag in range(1, max_lag + 1):
                lag_var = f"{var}_lag{lag}" if lag > 1 else f"{var}_lag"
                self.endogenous_states.append(lag_var)
                
                # Create auxiliary equation
                aux_eq = f"{lag_var}_p = {var}" if lag == 1 else f"{lag_var}_p = {var}_lag{lag-1}"
                self.transformed_equations[f"aux{eq_num}"] = aux_eq
                eq_num += 1

        # Process exogenous lags
        for var in self.exogenous:
            max_lag = self.var_max_lag.get(var, 0)
            for lag in range(1, max_lag + 1):
                lag_var = f"{var}_lag{lag}" if lag > 1 else f"{var}_lag"
                self.exogenous_states.append(lag_var)
                self.transformed_equations[f"aux{eq_num}"] = f"{lag_var}_p = {var}"
                eq_num += 1

        # Transform equations
        for eq in self.endogenous_equations:
            transformed_eq = eq
            
            # Substitute leads
            for lead in range(self.max_lead, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+' + str(lead) + r'\)'
                replacement = r'\1_p' if lead == 1 else r'\1_lead' + str(lead)
                transformed_eq = re.sub(pattern, replacement, transformed_eq)
            
            # Substitute lags
            for lag in range(self.max_lag, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-' + str(lag) + r'\)'
                replacement = r'\1_lag' + (str(lag) if lag > 1 else r'\1_lag')
                transformed_eq = re.sub(pattern, replacement, transformed_eq)
            
            self.transformed_equations[f"eq{eq_num}"] = transformed_eq
            eq_num += 1

        print(f"Created {len(self.transformed_equations)} transformed equations")

    def save_model_to_json(self, output_file):
        """Complete JSON saving implementation"""
        print(f"Saving model to JSON file: {output_file}")
        
        model_data = {
            "variables": {
                "endogenous_states": self.endogenous_states,
                "exogenous_states": self.exogenous_states,
                "non_predetermined": self.non_predetermined,
                "shocks": self.shocks
            },
            "parameters": self.parameters,
            "equations": self.transformed_equations,
            "metadata": {
                "max_lag": self.max_lag,
                "max_lead": self.max_lead
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)
            
        return model_data


    def parse_file(self, file_path_or_content):
        """
        Parse a Dynare file into structured data
        
        Args:
            file_path_or_content (str): Either the path to a Dynare file or the content directly
        """
        # Check if input is a file path or content
        if os.path.isfile(file_path_or_content):
            print(f"Reading Dynare file: {file_path_or_content}")
            with open(file_path_or_content, 'r', encoding='utf-8') as f:
                file_content = f.read()
        else:
            file_content = file_path_or_content
            
        # Make sure line endings are standardized
        file_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Extract var section (endogenous variables)
        print("Extracting variable declarations...")
        var_pattern = r'var\s+([\s\S]*?);'
        var_match = re.search(var_pattern, file_content)
        if var_match:
            var_section = var_match.group(1)
            # Remove comments
            var_section = re.sub(r'\/\/.*?(?=\n|$)', '', var_section)
            # Extract variable names
            self.variables = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', var_section)]
            print(f"Found {len(self.variables)} variables: {self.variables}")
        else:
            print("WARNING: No variable section found!")
        
        # Extract varexo section (exogenous shocks)
        print("Extracting exogenous shock declarations...")
        varexo_pattern = r'varexo\s+([\s\S]*?);'
        varexo_match = re.search(varexo_pattern, file_content)
        if varexo_match:
            varexo_section = varexo_match.group(1)
            # Remove comments
            varexo_section = re.sub(r'\/\/.*?(?=\n|$)', '', varexo_section)
            # Extract shock names
            self.shocks = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', varexo_section)]
            print(f"Found {len(self.shocks)} shocks: {self.shocks}")
        else:
            print("WARNING: No varexo section found!")
        
        # Extract parameters
        print("Extracting parameters...")
        param_pattern = r'parameters\s+([\s\S]*?);'
        param_match = re.search(param_pattern, file_content)
        if param_match:
            param_section = param_match.group(1)
            # Remove comments
            param_section = re.sub(r'\/\/.*?(?=\n|$)', '', param_section)
            # Extract parameter names
            self.parameters = [p.strip() for p in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', param_section)]
            print(f"Found {len(self.parameters)} parameters: {self.parameters}")
        else:
            print("WARNING: No parameters section found!")
        
        # Extract parameter values
        print("Extracting parameter values...")
        for param in self.parameters:
            param_value_pattern = rf'{param}\s*=\s*([0-9.]+)\s*;'
            param_value_match = re.search(param_value_pattern, file_content)
            if param_value_match:
                self.param_values[param] = float(param_value_match.group(1))
        print(f"Found values for {len(self.param_values)} parameters")
        
        # Extract model equations
        print("Extracting model equations...")
        model_pattern = r'model\s*;([\s\S]*?)end\s*;'
        model_match = re.search(model_pattern, file_content)
        if model_match:
            model_section = model_match.group(1)
            # Split by semicolons and clean each equation
            raw_equations = model_section.split(';')
            cleaned_lines = []
            for line in raw_equations:
                # Remove comments
                line = re.sub(r'\/\/.*?(?=\n|$)', '', line)
                # Remove leading/trailing whitespace
                line = line.strip()
                if line:  # Only keep non-empty lines
                    cleaned_lines.append(line)
            self.equations = cleaned_lines
            print(f"Found {len(self.equations)} equations")
        else:
            print("WARNING: No model section found!")
        
        # Classify variables
        self.classify_variables()
        
        # Classify equations
        self.classify_equations()
        
        # Find max lead and lag in equations
        self.find_max_lead_lag()
    
    def classify_variables(self):
        """
        Classify variables as endogenous or exogenous based on naming patterns
        """
        print("Classifying variables...")
        self.endogenous = []
        self.exogenous = []
        
        for var in self.variables:
            # Variables starting with RES_ are typically exogenous
            if var.startswith('RES_'):
                self.exogenous.append(var)
            else:
                self.endogenous.append(var)
        
        print(f"Classified {len(self.endogenous)} endogenous and {len(self.exogenous)} exogenous variables")
    
    def classify_equations(self):
        """
        Classify equations as endogenous or exogenous
        """
        print("Classifying equations...")
        self.endogenous_equations = []
        self.exogenous_equations = {}
        
        for eq in self.equations:
            # Check if equation defines an exogenous process
            exo_match = None
            for exo_var in self.exogenous:
                if re.search(rf'\b{exo_var}\b\s*=', eq):
                    exo_match = exo_var
                    break
            
            if exo_match:
                self.exogenous_equations[exo_match] = eq
                print(f"Found exogenous equation: {eq}")
            else:
                # Regular endogenous equation
                self.endogenous_equations.append(eq)
                print(f"Added endogenous equation: {eq}")
        
        print(f"Classified {len(self.endogenous_equations)} endogenous and {len(self.exogenous_equations)} exogenous equations")
    
    def find_max_lead_lag(self):
        """
        Find maximum lead and lag in all equations
        """
        print("Finding maximum lead and lag in equations...")
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
        
        print(f"Maximum lead: {self.max_lead}, Maximum lag: {self.max_lag}")
    
    # def transform_exogenous_equation(self, exo_var, eq):
    #     """Transform an exogenous equation to the Klein format with future on LHS"""
    #     # First identify and replace all lags with proper _lag notation
    #     transformed_eq = eq
    #     for lag in range(self.max_lag, 0, -1):
    #         pattern = r'\b' + re.escape(exo_var) + r'\(\-' + str(lag) + r'\)'
    #         lag_suffix = str(lag) if lag > 1 else ""
    #         transformed_eq = re.sub(pattern, exo_var + "_lag" + lag_suffix, transformed_eq)
        
    #     # Split the equation
    #     if "=" in transformed_eq:
    #         lhs, rhs = transformed_eq.split("=", 1)
    #         lhs = lhs.strip()
    #         rhs = rhs.strip()
            
    #         # Process AR(p) process - starting with the highest lag
    #         # This approach handles each lag term individually
    #         lag_terms = {}
    #         for lag in range(self.max_lag, 0, -1):
    #             lag_suffix = str(lag) if lag > 1 else ""
    #             lag_var = exo_var + "_lag" + lag_suffix
                
    #             if lag_var in rhs:
    #                 # For lag 1, replace with current variable
    #                 if lag == 1:
    #                     rhs = rhs.replace(lag_var, exo_var)
    #                 # For higher lags, replace with one lag less
    #                 else:
    #                     new_lag_suffix = str(lag-1) if lag-1 > 1 else ""
    #                     new_lag_var = exo_var + "_lag" + new_lag_suffix
    #                     rhs = rhs.replace(lag_var, new_lag_var)
            
    #         # Return the equation with future on LHS
    #         return f"{exo_var}_p - ({rhs})"
        
    #     return transformed_eq

    def transform_exogenous_equation(self, exo_var, eq):
        """Transform exogenous equation to Klein format with future on LHS"""
        max_lag = self.var_max_lag.get(exo_var, 0)  # Per-variable max lag
        
        # Step 1: Replace original lag notation with _lag suffixes
        transformed_eq = eq
        for lag in range(max_lag, 0, -1):
            pattern = r'\b' + re.escape(exo_var) + r'\(\-' + str(lag) + r'\)'
            replacement = f"{exo_var}_lag{lag}" if lag > 1 else f"{exo_var}_lag"
            transformed_eq = re.sub(pattern, replacement, transformed_eq)
        
        # Step 2: Convert to forward-looking equation
        if "=" in transformed_eq:
            lhs, rhs = transformed_eq.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()

            # Step 3: Decrement all lags in RHS (t-2 → t-1, t-1 → t)
            adjusted_rhs = rhs
            for lag in range(max_lag, 0, -1):
                old_suffix = f"_lag{lag}" if lag > 1 else "_lag"
                new_suffix = f"_lag{lag-1}" if lag-1 > 1 else "_lag" if lag-1 == 1 else ""
                adjusted_rhs = re.sub(re.escape(old_suffix), new_suffix, adjusted_rhs)

            # Step 4: Handle current period variable
            adjusted_rhs = re.sub(r'\b' + re.escape(exo_var) + r'\b', 
                                exo_var + "_lag" if max_lag >=1 else exo_var, 
                                adjusted_rhs)

            # Step 5: Create new equation with future on LHS
            new_lhs = f"{exo_var}_p"
            shock_name = self.shock_name  # Get from class attributes
            adjusted_rhs = re.sub(r'\b' + re.escape(shock_name) + r'\b', 
                                f"{shock_name}_p", adjusted_rhs)
            
            transformed_eq = f"{new_lhs} = {adjusted_rhs}"

        return transformed_eq


    def transform_endogenous_model(self):
        """Transform endogenous equations to only contain t and t+1 variables"""
        print("Transforming endogenous model...")
        
        # Initialize tracking dictionaries
        self.var_max_lag = {var: 0 for var in self.endogenous}
        self.var_max_lead = {var: 0 for var in self.endogenous}
        self.endogenous_states = []
        self.non_predetermined = self.endogenous.copy()
        self.transformed_equations = {}

        # First pass - identify max lags/leads per variable
        for eq in self.endogenous_equations:
            # Find leads
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq)
            for var, lead in lead_matches:
                if var in self.endogenous:
                    lead_val = int(lead)
                    if lead_val > self.var_max_lead.get(var, 0):
                        self.var_max_lead[var] = lead_val

            # Find lags
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
            for var, lag in lag_matches:
                if var in self.endogenous:
                    lag_val = int(lag)
                    if lag_val > self.var_max_lag.get(var, 0):
                        self.var_max_lag[var] = lag_val

        # Second pass - create auxiliary variables and equations
        eq_num = 1
        
        # Process endogenous states (lagged variables)
        for var, max_lag in self.var_max_lag.items():
            if max_lag > 0:
                # Create lag variables (var_lag, var_lag2, etc)
                for lag in range(1, max_lag + 1):
                    lag_var = f"{var}_lag{lag}" if lag > 1 else f"{var}_lag"
                    self.endogenous_states.append(lag_var)
                    
                    # Create auxiliary equation
                    if lag == 1:
                        aux_eq = f"{lag_var}_p = {var}"
                    else:
                        prev_lag = lag - 1
                        prev_var = f"{var}_lag{prev_lag}" if prev_lag > 1 else f"{var}_lag"
                        aux_eq = f"{lag_var}_p = {prev_var}"
                    
                    self.transformed_equations[f"aux{eq_num}"] = aux_eq
                    eq_num += 1

        # Process lead variables
        for var, max_lead in self.var_max_lead.items():
            if max_lead > 1:
                # Create lead variables (var_lead2, var_lead3, etc)
                for lead in range(2, max_lead + 1):
                    lead_var = f"{var}_lead{lead}"
                    self.non_predetermined.append(lead_var)
                    
                    # Create auxiliary equation
                    if lead == 2:
                        aux_eq = f"{lead_var} = {var}_p"
                    else:
                        prev_lead = lead - 1
                        prev_var = f"{var}_lead{prev_lead}"
                        aux_eq = f"{lead_var} = {prev_var}_p"
                    
                    self.transformed_equations[f"aux{eq_num}"] = aux_eq
                    eq_num += 1

        # Third pass - transform original equations
        for eq in self.endogenous_equations:
            transformed_eq = eq
            
            # Replace leads
            for lead in range(self.max_lead, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+' + str(lead) + r'\)'
                if lead == 1:
                    transformed_eq = re.sub(pattern, r'\1_p', transformed_eq)
                else:
                    transformed_eq = re.sub(pattern, r'\1_lead' + str(lead), transformed_eq)

            # Replace lags
            for lag in range(self.max_lag, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-' + str(lag) + r'\)'
                if lag == 1:
                    transformed_eq = re.sub(pattern, r'\1_lag', transformed_eq)
                else:
                    transformed_eq = re.sub(pattern, r'\1_lag' + str(lag), transformed_eq)

            self.transformed_equations[f"eq{eq_num}"] = transformed_eq
            eq_num += 1

        print(f"Created {len(self.transformed_equations)} transformed equations")
        print(f"Endogenous states: {self.endogenous_states}")
        print(f"Non-predetermined: {self.non_predetermined}")

    
    def save_model_to_json(self, output_file):
        """
        Save the parsed and transformed model to a JSON file matching the specified format
        
        Args:
            output_file (str): Path to the output JSON file
        
        Returns:
            dict: The model data dictionary
        """
        print(f"Saving model to JSON file: {output_file}")
        
        # Calculate total state count (endogenous + exogenous states)
        n_states = len(self.endogenous_states) + len(self.exogenous_states)
        
        # Create the model data structure
        model_data = {
            "variables": {
                "endogenous_states": self.endogenous_states,
                "exogenous_states": self.exogenous_states,
                "non_predetermined": self.non_predetermined,
                "shocks": self.shocks
            },
            
            "dimensions": {
                "n_states": n_states,
                "n_endogenous": len(self.non_predetermined) + len(self.endogenous_states),
                "n_shocks": len(self.shocks),
                "max_lead": self.max_lead,
                "max_lag": self.max_lag
            },
            
            "parameters": self.param_values,
            
            "equations": self.transformed_equations
        }
        
        # Save to file if output_file is provided
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            print(f"Model saved to {output_file}")
        
        return model_data
    
    def generate_jacobian_evaluator(self, output_file=None):
        """
        Generate a Python function that evaluates the Jacobian matrices for the model
        with variables ordered to match Klein method expectations
        
        Args:
            output_file (str, optional): Path to save the generated Python code
                
        Returns:
            str: The generated Python code for the Jacobian evaluator
        """
        print("Generating Jacobian evaluator...")
        
        # REVERSED ORDER: [non_predetermined, endogenous_states, exogenous_states]
        # This places non-predetermined variables first, followed by predetermined variables
        # which matches the expectation of Klein's method with 'ouc' sorting
        variables = self.exogenous_states + self.endogenous_states + self.non_predetermined
        
        # Shock variables
        shocks = self.shocks
        
        # Parameters to use
        parameters = self.parameters
        
        # Create variables with "_p" suffix for t+1 variables
        variables_p = [var + "_p" for var in variables]
        
        # Create symbolic variables for all components
        import sympy as sy
        var_symbols = {var: sy.symbols(var) for var in variables}
        var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
        shock_symbols = {shock: sy.symbols(shock) for shock in shocks}
        param_symbols = {param: sy.symbols(param) for param in parameters}
        
        # Combine all symbols for equation parsing
        all_symbols = {**var_symbols, **var_p_symbols, **shock_symbols, **param_symbols}
        
        # Convert equations to sympy expressions
        equations = []
        success_count = 0
        error_count = 0
        
        for eq_name, eq_str in self.transformed_equations.items():
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
                success_count += 1
            except Exception as e:
                print(f"Failed to parse equation {eq_name}: {eq_str}")
                print(f"Error: {str(e)}")
                # Try to recover by using a placeholder
                equations.append(sy.sympify("0"))
                error_count += 1
        
        print(f"Parsed {success_count} equations successfully, {error_count} with errors")
        
        # Create system as sympy Matrix
        F = sy.Matrix(equations)
        
        # Compute Jacobians with proper variable ordering
        X_symbols = [var_symbols[var] for var in variables]
        X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
        S_symbols = [shock_symbols[shock] for shock in shocks]
        
        # A = ∂F/∂X_p (Jacobian with respect to future variables)
        print("Computing A matrix...")
        A_symbolic = F.jacobian(X_p_symbols)
        
        # B = -∂F/∂X (negative Jacobian with respect to current variables)
        print("Computing B matrix...")
        B_symbolic = -F.jacobian(X_symbols)
        
        # C = -∂F/∂S (negative Jacobian with respect to shock variables)
        print("Computing C matrix...")
        C_symbolic = -F.jacobian(S_symbols)
        
        # Generate code for the Jacobian evaluation function
        function_code = [
            "import numpy as np",
            "",
            "def evaluate_jacobians(theta):",
            "    \"\"\"",
            "    Evaluates Jacobian matrices for the Klein method",
            "    ",
            "    Args:",
            "        theta: List or array of parameter values in the order of:",
            f"              {parameters}",
            "        ",
            "    Returns:",
            "        a: Matrix ∂F/∂X_p (Jacobian with respect to future variables)",
            "        b: Matrix -∂F/∂X (negative Jacobian with respect to current variables)",
            "        c: Matrix -∂F/∂S (negative Jacobian with respect to shock variables)",
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
            f"    c = np.zeros(({len(equations)}, {len(shocks)}))"
        ])
        
        # Add A matrix elements
        function_code.append("")
        function_code.append("    # A matrix elements")
        for i in range(A_symbolic.rows):
            for j in range(A_symbolic.cols):
                if A_symbolic[i, j] != 0:
                    expr = str(A_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        # Replace symbol with parameter name
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    a[{i}, {j}] = {expr}")
        
        # Add B matrix elements
        function_code.append("")
        function_code.append("    # B matrix elements")
        for i in range(B_symbolic.rows):
            for j in range(B_symbolic.cols):
                if B_symbolic[i, j] != 0:
                    expr = str(B_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    b[{i}, {j}] = {expr}")
        
        # Add C matrix elements
        function_code.append("")
        function_code.append("    # C matrix elements")
        for i in range(C_symbolic.rows):
            for j in range(C_symbolic.cols):
                if C_symbolic[i, j] != 0:
                    expr = str(C_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    c[{i}, {j}] = {expr}")
        
        # Return matrices
        function_code.append("")
        function_code.append("    return a, b, c")
        
        # Join all lines to form the complete function code
        complete_code = "\n".join(function_code)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(complete_code)
            print(f"Jacobian evaluator saved to {output_file}")
        
        return complete_code



def dsolab(a, b, n, k):
    """
    Python implementation that mimics the Fortran DSOLAB subroutine.
    """
    # Initialize return code
    retco = 0
    
    # Call scipy's ordqz which is equivalent to DGGES in LAPACK
    try:
        # We need to use the adapter function that matches scipy's expected signature
        s, t, alphar, alphai, beta, vsr = la.ordqz(a, b, sort="iuc", output='real')
    except Exception as e:
        print(f"Error in ordqz: {e}")
        return None, None, 1, None
    
    # Check if we have the correct number of stable eigenvalues
    # In scipy, the eigenvalues are sorted so the first k eigenvalues should match our criteria
    # We need to manually count them since the sorting may not be perfect
    stable_count = 0
    # for i in range(n):
    #     if (alphar[i]**2 + alphai[i]**2) >= (beta[i]**2):
    #         stable_count += 1
    
    # if stable_count != k:
    #     print(f'ERROR: WRONG NUMBER OF STABLE EIGENVALUES (got {stable_count}, expected {k})')
    #     return None, None, 1, None
    
    # Extract the components of the vsr matrix (equivalent to VSR in Fortran)
    z11 = vsr[:k, :k]
    z21 = vsr[k:n, :k]
    
    # Extract components of s and t matrices
    s11 = s[:k, :k]
    t11 = t[:k, :k]
    
    # Compute Z11 inverse
    try:
        z11i = la.inv(z11)
    except Exception as e:
        print(f"Error inverting Z11: {e}")
        return None, None, 1, None
    
    # Compute S11 inverse
    try:
        s11i = la.inv(s11)
    except Exception as e:
        print(f"Error inverting S11: {e}")
        return None, None, 1, None
    
    # Compute F = Z21 * (Z11)^-1
    f = np.matmul(z21, z11i)
    
    # Compute P = Z11 * (S11)^-1 * T11 * (Z11)^-1
    temp1 = np.matmul(s11i, t11)      # (S11)^-1 * T11
    temp2 = np.matmul(z11, temp1)     # Z11 * (S11)^-1 * T11
    p = np.matmul(temp2, z11i)        # Z11 * (S11)^-1 * T11 * (Z11)^-1
    
    # Clean small values
    p[np.abs(p) <= 1e-10] = 0.0
    f[np.abs(f) <= 1e-10] = 0.0
    
    # Return eigenvalue information along with other results
    eig_info = (s, t, alphar, alphai, beta)
    
    return f, p, retco, eig_info

def klein(a=None, b=None, n_states=None, eigenvalue_warnings=True):
    '''
    Solves linear dynamic models using the approach from Klein (2000),
    implemented to mimic the Fortran DSOLAB subroutine.
    
    Args:
        a: Coefficient matrix on future-dated variables
        b: Coefficient matrix on current-dated variables
        n_states: Number of state variables
        eigenvalue_warnings: Whether to print eigenvalue warnings
        
    Returns:
        f: Solution matrix coeffients on s(t) for u(t)
        p: Solution matrix coeffients on s(t) for s(t+1)
        stab: Stability indicator
        eig: Generalized eigenvalues
    '''
    n_total = np.shape(a)[0]
    
    # Call the dsolab-like implementation, now also getting eigenvalue info
    f, p, retco, eig_info = dsolab(a, b, n_total, n_states)
    
    # If dsolab failed, set error code
    if retco != 0:
        stab = -1  # Indicating failure
        eig = None
        return f, p, stab, eig
    
    # Unpack eigenvalue information returned from dsolab
    s, t, alphar, alphai, beta = eig_info
    
    # Compute the generalized eigenvalues
    eig = np.zeros(n_total, dtype=np.complex128)
    for k in range(n_total):
        if np.abs(s[k, k]) > 0:
            if alphai[k] != 0:
                eig[k] = complex(alphar[k] / beta[k], alphai[k] / beta[k])
            else:
                eig[k] = complex(t[k, k] / s[k, k], 0)
        else:
            eig[k] = np.inf
    
    # Determine stability
    # stab = 0
    # if n_states > 0:
    #     if np.abs(t[n_states-1, n_states-1]) > np.abs(s[n_states-1, n_states-1]):
    #         if eigenvalue_warnings:
    #             print('Warning: Too few stable eigenvalues. Check model equations or parameter values.')
    #         stab = -1
            
    # if n_states < n_total:
    #     if np.abs(t[n_states, n_states]) < np.abs(s[n_states, n_states]):
    #         if eigenvalue_warnings:
    #             print('Warning: Too many stable eigenvalues. Check model equations or parameter values.')
    #         stab = 1
    
    return f, p, stab, eig

def map_variables_to_indices(model_data):
    """
    Create a mapping from variable names to their indices in the state and control vectors.
    
    Args:
        model_data: Dictionary containing model information
        
    Returns:
        var_map: Dictionary mapping variable names to their indices and types
    """
    var_map = {}
    
    # Map predetermined variables to state indices
    for i, var in enumerate(model_data['variables']['predetermined']):
        var_map[var] = {'type': 'state', 'index': i}
    
    # Map non-predetermined variables to control indices
    for i, var in enumerate(model_data['variables']['non_predetermined']):
        var_map[var] = {'type': 'control', 'index': i}
    
    # Map exogenous variables to exogenous indices
    for i, var in enumerate(model_data['variables']['exogenous']):
        var_map[var] = {'type': 'exogenous', 'index': i}
    
    # Map shock variables
    for i, var in enumerate(model_data['variables']['shocks']):
        var_map[var] = {'type': 'shock', 'index': i}
    
    return var_map


def plot_irfs_by_name(irfs, var_map, shock_name, variables=None, figsize=(12, 8)):
    """
    Plot impulse response functions for specific variables.
    
    Args:
        irfs: IRF dictionary from compute_irfs
        var_map: Variable mapping from map_variables_to_indices
        shock_name: Name of the shock to plot
        variables: List of variable names to plot (None = all)
        figsize: Figure size
    """
    if shock_name not in irfs:
        raise ValueError(f"Shock {shock_name} not found in IRFs")
    
    shock_irfs = irfs[shock_name]
    
    # If variables is None, plot all variables
    if variables is None:
        variables = [v for v in var_map.keys() if var_map[v]['type'] != 'shock']
    
    # Number of variables to plot
    n_vars = len(variables)
    
    # Create grid of subplots
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each variable
    for i, var in enumerate(variables):
        if var in var_map:
            var_info = var_map[var]
            var_type = var_info['type']
            var_idx = var_info['index']
            
            # Get the appropriate IRF data
            if var_type == 'state':
                irf_data = shock_irfs['states'][var_idx, :]
            elif var_type == 'control':
                irf_data = shock_irfs['controls'][var_idx, :]
            elif var_type == 'exogenous':
                irf_data = shock_irfs['exogenous'][var_idx, :]
            else:
                continue
            
            # Plot the IRF
            x = np.arange(len(irf_data))
            axes[i].plot(x, irf_data)
            axes[i].set_title(var)
            axes[i].grid(True)
            axes[i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Turn off any unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"IRFs for {shock_name}")
    plt.tight_layout()
    
    return fig


def setup_and_solve_model(json_file, jacobian_file):
    """
    Load a model from JSON, solve it, and prepare it for IRF analysis.
    
    Args:
        json_file: Path to the model JSON file
        jacobian_file: Path to the Jacobian evaluator Python file
        
    Returns:
        model_solution: Dictionary with solution matrices and model information
    """
    # Load model data from JSON
    with open(json_file, 'r') as f:
        model_data = json.load(f)
    
    # Extract variable lists
    endogenous_states = model_data['variables']['endogenous_states']
    exogenous_states = model_data['variables']['exogenous_states']
    non_predetermined = model_data['variables']['non_predetermined']
    shocks = model_data['variables']['shocks']
    
    # Calculate dimensions
    n_endogenous_states = len(endogenous_states)
    n_exogenous_states = len(exogenous_states)
    n_non_predetermined = len(non_predetermined)
    n_shocks = len(shocks)
    
    # Total number of predetermined states (endogenous + exogenous)
    n_states = n_endogenous_states + n_exogenous_states
    
    # Create variable mapping
    var_map = {}
    # Map endogenous states
    for i, var in enumerate(endogenous_states):
        var_map[var] = {'type': 'endogenous_state', 'index': i}
    
    # Map exogenous states
    for i, var in enumerate(exogenous_states):
        var_map[var] = {'type': 'exogenous_state', 'index': i}
    
    # Map non-predetermined variables
    for i, var in enumerate(non_predetermined):
        var_map[var] = {'type': 'non_predetermined', 'index': i}
    
    # Map shocks
    for i, var in enumerate(shocks):
        var_map[var] = {'type': 'shock', 'index': i}
    
    # Load Jacobian evaluator function
    import importlib.util
    spec = importlib.util.spec_from_file_location("jacobian_evaluator", jacobian_file)
    jacobian_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jacobian_module)
    evaluate_jacobians = jacobian_module.evaluate_jacobians
    
    # Get parameter values
    param_values = list(model_data['parameters'].values())
    
    # Evaluate Jacobians
    a, b, c = evaluate_jacobians(param_values)
    
    # Solve the model using Klein's method with complete matrices
    # Use total number of predetermined states (endogenous + exogenous)
    f, p, stab, eig = klein(a=a, b=b, n_states=n_states)
    
    # Check stability
    if stab != 0:
        print(f"Warning: Model stability indicator = {stab}")
        if stab == 1:
            print("Too many stable eigenvalues - model may have multiple equilibria")
        elif stab == -1:
            print("Too few stable eigenvalues - model may have no stable solution")
    
    # Return solution and model information
    return {
        'model_data': model_data,
        'var_map': var_map,
        'matrices': {
            'a': a,
            'b': b,
            'c': c,
            'f': f,
            'p': p
        },
        'dimensions': {
            'n_predetermined': n_states,  # Total predetermined states
            'n_non_predetermined': n_non_predetermined,
            'n_exogenous': n_exogenous_states,
            'n_endogenous_states': n_endogenous_states,
            'n_shocks': n_shocks,
            'n_states': n_states
        },
        'variables': {
            'endogenous_states': endogenous_states,
            'exogenous_states': exogenous_states,
            'non_predetermined': non_predetermined,
            'shocks': shocks
        },
        'stability': {
            'status': stab,
            'eigenvalues': eig
        }
    }
    


# def compute_and_plot_irfs(model_solution, shock_indices=None, variables=None, periods=40):
#     """
#     Compute and plot IRFs for a solved DSGE model.
    
#     Args:
#         model_solution: Solution dictionary from setup_and_solve_model
#         shock_indices: Indices of shocks to simulate (None = all)
#         variables: List of variable names to plot (None = all)
#         periods: Number of periods for IRF
        
#     Returns:
#         irfs: Dictionary of IRFs
#         figures: List of figure objects
#     """
#     # Extract needed components
#     f = model_solution['matrices']['f']
#     n = model_solution['matrices']['n']
#     H = model_solution['matrices']['H']
#     G = model_solution['matrices']['G']
    
#     n_predetermined = model_solution['dimensions']['n_predetermined']
#     n_exogenous = model_solution['dimensions']['n_exogenous']
#     n_non_predetermined = model_solution['dimensions']['n_non_predetermined']
    
#     var_map = model_solution['var_map']
#     model_data = model_solution['model_data']
    
#     # If shock_indices is None, use all shocks
#     if shock_indices is None:
#         shock_indices = list(range(model_solution['dimensions']['n_shocks']))
#     elif isinstance(shock_indices, int):
#         shock_indices = [shock_indices]
    
#     # Compute IRFs
#     irfs = compute_irfs(f, n, H, G, n_predetermined, n_exogenous, 
#                      n_non_predetermined, periods, None)
    
#     # Generate plots
#     figures = []
#     for i, shock_idx in enumerate(shock_indices):
#         shock_name = f"shock_{shock_idx}"
#         if model_data['variables']['shocks'] and i < len(model_data['variables']['shocks']):
#             shock_label = model_data['variables']['shocks'][i]
#         else:
#             shock_label = shock_name
            
#         fig = plot_irfs_by_name(irfs, var_map, shock_name, variables)
#         figures.append(fig)
    
#     return irfs, figures


def run_dsge_workflow(dynare_file, output_prefix=None):
    """
    Run the complete DSGE model workflow from a Dynare file.
    
    Args:
        dynare_file: Path to the Dynare model file
        output_prefix: Prefix for output files (if None, derive from dynare_file)
        
    Returns:
        parser: The DynareParser instance
        model_solution: The solved model
        irfs: Impulse response functions
    """
    # Set output prefix if not provided
    if output_prefix is None:
        output_prefix = os.path.splitext(dynare_file)[0] + "_parsed"
    
    # Define output files
    json_file = f"{output_prefix}.json"
    jacobian_file = f"{output_prefix}_jacobian.py"
    
    # Create parser instance
    parser = DynareParser()
    
    # Parse the file content
    parser.parse_file(dynare_file)
    
    # Transform the model
    parser.transform_endogenous_model()
    
    # Save the model to JSON
    model_data = parser.save_model_to_json(json_file)
    print(f"Parsed model saved to {json_file}")
    
    # Generate Jacobian evaluator
    parser.generate_jacobian_evaluator(jacobian_file)
    print(f"Jacobian evaluator saved to {jacobian_file}")
    
    # Solve the model
    model_solution = setup_and_solve_model(json_file, jacobian_file)
    print("Model solved successfully")
    
    print("\nModel Solution:")
    #print(f"Stability: {model_solution['stability']['status']}")
    #print(f"Eigenvalues: {np.abs(model_solution['stability']['eigenvalues'])}")
    # # Compute IRFs
    #irfs, figures = compute_and_plot_irfs(model_solution)
    # print("IRFs computed and plotted")
    
    return parser, model_solution #, irfs


if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
    print(f"Processing DSGE model from {dynare_file}...")
    
    # Run the complete workflow
    parser, solution= run_dsge_workflow(dynare_file)
    
    # Display model solution information
    print("\nModel Solution Information:")
    print(f"Stability: {solution['stability']['status']} (0=stable, 1=indeterminate, -1=unstable)")
    print(f"Number of states: {solution['dimensions']['n_predetermined']}")
    print(f"Number of controls: {solution['dimensions']['n_non_predetermined']}")
    
    # Plot IRFs for each shock
    #plt.show()
