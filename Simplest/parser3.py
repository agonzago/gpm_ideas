#%%
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
        
        # Classify variables and equations
        self.classify_variables()
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
        and extract information about exogenous processes
        """
        print("Classifying equations...")
        self.endogenous_equations = []
        self.exogenous_equations = {}
        
        # First pass: identify explicit exogenous process equations
        for eq in self.equations:
            # Try to find equations that directly define exogenous processes
            exo_match = None
            for exo_var in self.exogenous:
                if re.search(rf'\b{exo_var}\b\s*=', eq):
                    exo_match = exo_var
                    break
            
            if exo_match:
                self.exogenous_equations[exo_match] = eq
                print(f"Found exogenous equation: {eq}")
            else:
                # Check if any shock appears directly in this equation
                shock_in_eq = False
                for shock in self.shocks:
                    if re.search(rf'\b{shock}\b', eq):
                        shock_in_eq = True
                        shock_match = shock
                        break
                
                if shock_in_eq:
                    # Create an implicit exogenous process
                    # Try to identify the main endogenous variable
                    endo_match = re.search(r'([A-Za-z0-9_]+)\s*=', eq)
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
                        print(f"Created implicit exogenous equation: {impl_eq}")
                        
                        # Modify the original equation to use the implicit exogenous variable
                        mod_eq = eq.replace(shock_match, implicit_exo)
                        self.endogenous_equations.append(mod_eq)
                        print(f"Modified endogenous equation: {mod_eq}")
                    else:
                        # Couldn't identify main variable, keep as endogenous
                        self.endogenous_equations.append(eq)
                        print(f"Added endogenous equation with shock: {eq}")
                else:
                    # Regular endogenous equation
                    self.endogenous_equations.append(eq)
                    print(f"Added endogenous equation: {eq}")
        
        print(f"Classified {len(self.endogenous_equations)} endogenous and {len(self.exogenous_equations)} exogenous equations")
    
    def find_max_lead_lag(self):
        """
        Find maximum lead and lag in all endogenous equations
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
    
    def transform_endogenous_model(self):
        """
        Transform the endogenous part of the Dynare model into a system with only t and t+1 variables
        """
        print("Transforming endogenous model...")
        
        # Create the transformed variables list starting with original endogenous variables
        self.transformed_variables = self.endogenous.copy()
        
        # Track which variables have lags and leads and their maximum lag/lead
        var_max_lag = {}
        var_max_lead = {}
        
        # First pass - identify what needs transforming
        for eq in self.endogenous_equations:
            # Find all variables with leads
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq)
            for var, lead in lead_matches:
                if var in self.endogenous:  # Only transform endogenous variables
                    lead_val = int(lead)
                    if lead_val >= 1:
                        var_max_lead[var] = max(var_max_lead.get(var, 0), lead_val)
            
            # Find all variables with lags
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
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
        
        # Add lead variables to transformed variables list - CORRECTED VERSION
        # Only add lead variables for variables with leads > 1
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:  # Only if max lead is greater than 1
                # We need lead1, lead2, lead3, etc.
                for lead in range(1, max_lead + 1):
                    self.transformed_variables.append(f"{var}_lead{lead}")
        
        print(f"Transformed variable count: {len(self.transformed_variables)}")
        print(f"Variables with lags: {var_max_lag}")
        print(f"Variables with leads > 1: {[v for v, l in var_max_lead.items() if l > 1]}")
        
        # Transform endogenous equations
        self.transformed_equations = []
        for i, eq in enumerate(self.endogenous_equations):
            transformed_eq = eq
            
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
                transformed_eq = f"{rhs.strip()} - ({lhs.strip()})"  # Added parentheses for safety
            
            self.transformed_equations.append({f"eq{i+1}": transformed_eq.strip()})
            print(f"Transformed equation {i+1}: {transformed_eq.strip()}")
        
        # Add transition equations for lags
        eq_num = len(self.transformed_equations) + 1
        for var, max_lag in var_max_lag.items():
            # First lag: var_lag_p = var
            lag_eq = f"{var}_lag_p - {var}"
            self.transformed_equations.append({f"eq{eq_num}": lag_eq})
            print(f"Added lag transition equation {eq_num}: {lag_eq}")
            eq_num += 1
            
            # Additional lags: var_lagN_p = var_lag(N-1)
            for lag in range(2, max_lag + 1):
                prev_lag_suffix = str(lag-1) if lag-1 > 1 else ""
                lag_eq = f"{var}_lag{lag}_p - {var}_lag{prev_lag_suffix}"
                self.transformed_equations.append({f"eq{eq_num}": lag_eq})
                print(f"Added lag transition equation {eq_num}: {lag_eq}")
                eq_num += 1
        
        # Add transition equations for leads - CORRECTED VERSION
        # Only add lead transition equations for variables with leads > 1
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:  # Only create auxiliary variables and equations for leads > 1
                # First lead transition equation: var_p = var_lead1
                lead_eq = f"{var}_p - {var}_lead1"
                self.transformed_equations.append({f"eq{eq_num}": lead_eq})
                print(f"Added lead transition equation {eq_num}: {lead_eq}")
                eq_num += 1
                
                # Additional lead transition equations: var_leadN_p = var_lead(N+1)
                for lead in range(1, max_lead):
                    next_lead = lead + 1
                    lead_eq = f"{var}_lead{lead}_p - {var}_lead{next_lead}"
                    self.transformed_equations.append({f"eq{eq_num}": lead_eq})
                    print(f"Added lead transition equation {eq_num}: {lead_eq}")
                    eq_num += 1
        
        print(f"Transformed equation count: {len(self.transformed_equations)}")
    
    def prepare_exogenous_var(self):
        """
        Prepare exogenous processes in VAR form for JSON output and Phi matrix computation
        """
        print("Preparing exogenous variables...")
        # Skip if no exogenous variables
        if not self.exogenous:
            print("No exogenous variables to prepare.")
            return
        
        # Transform exogenous equations to use _lag notation
        transformed_equations = {}
        
        for exo, eq_str in self.exogenous_equations.items():
            # Replace lags with _lag notation
            transformed_eq = eq_str
            for lag in range(self.max_lag, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-' + str(lag) + r'\)'
                lag_suffix = str(lag) if lag > 1 else ""
                transformed_eq = re.sub(pattern, r'\1_lag' + lag_suffix, transformed_eq)
            
            transformed_equations[exo] = transformed_eq
            print(f"Transformed exogenous equation: {transformed_eq}")
        
        # Update exogenous equations
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
                    print(f"Created default exogenous equation: {exo} = {shock_name}")
                else:
                    # If no matching shock found, use a generic equation
                    self.exogenous_equations[exo] = f"{exo} = 0"
                    print(f"Created zero exogenous equation: {exo} = 0")
        
        print(f"Prepared {len(self.exogenous_equations)} exogenous equations")

    def save_model_to_json(self, output_file):
        """
        Save the parsed and transformed model to a JSON file
        
        Args:
            output_file (str): Path to the output JSON file
        
        Returns:
            dict: The model data dictionary
        """
        print(f"Saving model to JSON file: {output_file}")
        
        # Ensure exogenous equations are prepared
        self.prepare_exogenous_var()
        
        # Convert transformed equations from list of dicts to a single dict
        endogenous_eqs = {}
        for eq_dict in self.transformed_equations:
            endogenous_eqs.update(eq_dict)  # Merge dictionaries
        
        # Create the model data structure
        model_data = {
            "endogenous_variables": self.transformed_variables,
            "exogenous_variables": self.exogenous,
            "parameters": self.parameters,
            "parameter_values": self.param_values,
            "shocks": self.shocks,
            "original_equations": self.equations,
            "endogenous_equations": endogenous_eqs,  # Dictionary of equations
            "exogenous_equations": self.exogenous_equations,
            "max_lead": self.max_lead,
            "max_lag": self.max_lag
        }
        
        # Save to file if output_file is provided
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            print(f"Model saved to {output_file}")
        
        return model_data
    
    def create_var_representation(self):
        """
        Create a proper VAR representation for the exogenous processes,
        accounting for multiple lags
        
        Returns:
            dict: Information about the VAR representation
        """
        print("Creating VAR representation for exogenous processes...")
        
        # Create expanded exogenous list including lags
        expanded_exogenous = []
        exo_lag_map = {}  # Maps from exo variable to its lags
        
        # First, add original exogenous variables
        for exo in self.exogenous:
            expanded_exogenous.append(exo)
            exo_lag_map[exo] = [exo]
        
        # Then add all lags for each exogenous variable
        for exo in self.exogenous:
            for lag in range(1, self.max_lag + 1):
                lag_suffix = str(lag) if lag > 1 else ""
                lag_var = f"{exo}_lag{lag_suffix}"
                expanded_exogenous.append(lag_var)
                exo_lag_map[exo].append(lag_var)
        
        # Extract information about the structure of each exogenous process
        var_structure = {}
        
        for exo, eq_str in self.exogenous_equations.items():
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Initialize the structure
                var_structure[exo] = {
                    "equation": eq_str,
                    "ar_coefficients": [],
                    "shock": None,
                    "constant": 0
                }
                
                # Extract AR coefficients
                for lag in range(1, self.max_lag + 1):
                    lag_suffix = str(lag) if lag > 1 else ""
                    lag_pattern = rf'\b{exo}_lag{lag_suffix}\b'
                    
                    # Look for coefficient of the form: a*exo_lag
                    coef_pattern = rf'([0-9.]+)\s*\*\s*{exo}_lag{lag_suffix}\b'
                    coef_match = re.search(coef_pattern, rhs)
                    
                    if coef_match:
                        # We have a coefficient
                        coefficient = float(coef_match.group(1))
                        var_structure[exo]["ar_coefficients"].append(coefficient)
                    elif re.search(lag_pattern, rhs):
                        # The variable appears without explicit coefficient (implied 1)
                        var_structure[exo]["ar_coefficients"].append(1.0)
                    else:
                        # No coefficient for this lag
                        var_structure[exo]["ar_coefficients"].append(0.0)
                
                # Check for shock term
                for shock in self.shocks:
                    if re.search(rf'\b{shock}\b', rhs):
                        var_structure[exo]["shock"] = shock
                        break
                
                # Check for constant term
                const_pattern = r'([+-]?\s*[0-9.]+)(?!\s*\*)'
                const_match = re.search(const_pattern, rhs)
                if const_match:
                    try:
                        var_structure[exo]["constant"] = float(const_match.group(1))
                    except:
                        # If we can't parse it, leave at default
                        pass
        
        # Create the transition matrix representation
        phi_matrix = np.zeros((len(expanded_exogenous), len(expanded_exogenous)))
        shock_selection = np.zeros((len(expanded_exogenous), len(self.shocks)))
        
        # Map shocks to their indices
        shock_indices = {shock: i for i, shock in enumerate(self.shocks)}
        
        # Fill in the Phi matrix row by row
        row_idx = 0
        for exo in self.exogenous:
            # Fill row for main exogenous variable
            if exo in var_structure:
                # Add AR coefficients
                for lag, coef in enumerate(var_structure[exo]["ar_coefficients"]):
                    lag_suffix = str(lag+1) if lag+1 > 1 else ""
                    lag_var = f"{exo}_lag{lag_suffix}"
                    lag_col = expanded_exogenous.index(lag_var)
                    phi_matrix[row_idx, lag_col] = coef
                
                # Add shock selection
                if var_structure[exo]["shock"] is not None:
                    shock_col = shock_indices[var_structure[exo]["shock"]]
                    shock_selection[row_idx, shock_col] = 1.0
            
            row_idx += 1
            
            # Fill rows for lag variables
            for lag in range(1, self.max_lag):
                # Current lag
                lag_suffix = str(lag) if lag > 1 else ""
                lag_var = f"{exo}_lag{lag_suffix}"
                
                # Next lag 
                next_lag_suffix = str(lag-1) if lag-1 > 1 else ""
                next_lag_var = exo if lag-1 == 0 else f"{exo}_lag{next_lag_suffix}"
                
                # Add transition (identity): lag_var = next_lag_var
                next_lag_col = expanded_exogenous.index(next_lag_var)
                phi_matrix[row_idx, next_lag_col] = 1.0
                
                row_idx += 1
        
        self.expanded_exogenous = expanded_exogenous
        self.phi_matrix = phi_matrix
        self.shock_selection_matrix = shock_selection
        self.exogenous_structure = var_structure
        
        var_info = {
            "expanded_exogenous": expanded_exogenous,
            "phi_matrix": phi_matrix.tolist(),
            "shock_selection_matrix": shock_selection.tolist(),
            "exogenous_structure": var_structure
        }
        
        return var_info

    def generate_jacobian_evaluator(self, output_file=None):
        """
        Generate a Python function that evaluates the Jacobian matrices for the model
        with corrected VAR representation for exogenous variables with multiple lags
        
        Args:
            output_file (str, optional): Path to save the generated Python code
                
        Returns:
            str: The generated Python code for the Jacobian evaluator
        """
        print("Generating Jacobian evaluator with corrected VAR representation...")
        
        # Basic model components
        variables = self.transformed_variables
        exogenous = self.exogenous
        parameters = self.parameters
        
        # First, identify the lags used for each exogenous variable
        exo_lags = {}
        for exo in exogenous:
            exo_lags[exo] = []
            # Check each exogenous equation for lags
            if exo in self.exogenous_equations:
                eq_str = self.exogenous_equations[exo]
                for lag in range(1, self.max_lag + 1):
                    lag_suffix = str(lag) if lag > 1 else ""
                    lag_var = f"{exo}_lag{lag_suffix}"
                    if lag_var in eq_str:
                        exo_lags[exo].append(lag_var)
        
        # Create expanded exogenous list including lags
        expanded_exogenous = exogenous.copy()
        for exo in exogenous:
            expanded_exogenous.extend(exo_lags[exo])
        
        print(f"Exogenous variables with lags: {expanded_exogenous}")
        
        # Create variables with "_p" suffix for t+1 variables
        variables_p = [var + "_p" for var in variables]
        
        # Create symbolic variables
        var_symbols = {var: sy.symbols(var) for var in variables}
        var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
        exo_symbols = {exo: sy.symbols(exo) for exo in expanded_exogenous}
        param_symbols = {param: sy.symbols(param) for param in parameters}
        
        # Combine all symbols
        all_symbols = {**var_symbols, **var_p_symbols, **exo_symbols, **param_symbols}
        
        # Get endogenous equations from the JSON representation
        model_data = self.save_model_to_json(None)
        endogenous_eqs = model_data["endogenous_equations"]
        
        # Parse endogenous equations into sympy expressions
        equations = []
        success_count = 0
        error_count = 0
        
        for eq_name, eq_str in endogenous_eqs.items():
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
        
        # Compute Jacobians for endogenous system
        X_symbols = [var_symbols[var] for var in variables]
        X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
        Z_symbols = [exo_symbols[exo] for exo in exogenous]  # Only original exogenous
        
        # A = ∂F/∂X_p (Jacobian with respect to future variables)
        print("Computing A matrix...")
        A_symbolic = F.jacobian(X_p_symbols)
        
        # B = -∂F/∂X (negative Jacobian with respect to current variables)
        print("Computing B matrix...")
        B_symbolic = -F.jacobian(X_symbols)
        
        # C = -∂F/∂Z (negative Jacobian with respect to exogenous processes)
        print("Computing C matrix...")
        C_symbolic = -F.jacobian(Z_symbols)
        
        # Create exogenous VAR system - with proper handling of lags
        print("Computing exogenous VAR system with proper lag structure...")
        
        # We need the full expanded Z vector that includes all lags
        Z_expanded_symbols = [exo_symbols[exo] for exo in expanded_exogenous]
        
        # Create equations for the exogenous VAR system
        exo_equations = []
        
        # Process original exogenous equations
        for exo in exogenous:
            if exo in self.exogenous_equations:
                eq_str = self.exogenous_equations[exo]
                if "=" in eq_str:
                    lhs, rhs = eq_str.split("=", 1)
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    
                    # Convert to symbolic form
                    eq_sym = rhs
                    
                    # Replace variables and parameters with symbols
                    for name, symbol in all_symbols.items():
                        pattern = r'\b' + re.escape(name) + r'\b'
                        eq_sym = re.sub(pattern, str(symbol), eq_sym)
                    
                    try:
                        # Parse the expression
                        expr = sy.sympify(eq_sym) - exo_symbols[exo]
                        exo_equations.append(expr)
                    except Exception as e:
                        print(f"Error parsing exogenous equation for {exo}: {eq_str}")
                        print(f"Error: {str(e)}")
                        # Use placeholder
                        exo_equations.append(sy.sympify("0"))
                else:
                    # Add default equation if no equals sign
                    exo_equations.append(exo_symbols[exo])
            else:
                # Add default equation if no equation defined
                exo_equations.append(exo_symbols[exo])
        
        # Add transition equations for all lags
        for exo in exogenous:
            for i, lag_var in enumerate(exo_lags[exo]):
                if i == 0:
                    # First lag: exo_lag = exo
                    eq = exo_symbols[lag_var] - exo_symbols[exo]
                else:
                    # Additional lags: exo_lagN = exo_lag(N-1)
                    prev_lag = exo_lags[exo][i-1]
                    eq = exo_symbols[lag_var] - exo_symbols[prev_lag]
                
                exo_equations.append(eq)
        
        # Create exogenous system as sympy Matrix
        G = sy.Matrix(exo_equations)
        
        # Compute Phi matrix for VAR system
        Phi_symbolic = -G.jacobian(Z_expanded_symbols)
        
        print("Generating output code...")
        
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
            "        phi: Matrix for VAR representation of exogenous processes (includes lags)",
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
            f"    phi = np.zeros(({len(exo_equations)}, {len(expanded_exogenous)}))"
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
        
        # Add Phi matrix elements
        function_code.append("")
        function_code.append("    # Phi matrix elements (VAR representation with lags)")
        function_code.append(f"    # Order of variables: {expanded_exogenous}")
        for i in range(Phi_symbolic.rows):
            for j in range(Phi_symbolic.cols):
                if Phi_symbolic[i, j] != 0:
                    expr = str(Phi_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    phi[{i}, {j}] = {expr}")
        
        # Return all matrices
        function_code.append("")
        function_code.append("    return a, b, c, phi")
        
        # Join all lines to form the complete function code
        complete_code = "\n".join(function_code)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(complete_code)
            print(f"Jacobian evaluator saved to {output_file}")
        
        return complete_code


import os
import sys


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
    print(f"Processing DSGE model from {dynare_file}...")
    
    # Create parser instance
    parser = DynareParser()
    
    # Parse the file content
    parser.parse_file(dynare_file)
    
    # Transform the model
    parser.transform_endogenous_model()
    
    # Prepare exogenous processes in VAR form
    parser.prepare_exogenous_var()
    
    # Save the model to JSON
    output_prefix = "qpm_model_parsed"
    json_file = f"{output_prefix}.json"
    model_data = parser.save_model_to_json(json_file)
    print(f"Parsed model saved to {json_file}")
    
    # Generate Jacobian evaluator
    jacobian_file = f"{output_prefix}_jacobian.py"
    parser.generate_jacobian_evaluator(jacobian_file)
    print(f"Jacobian evaluator saved to {jacobian_file}")
    
    # Print summary of model components
    print("\nModel Summary:")
    print(f"- Endogenous variables: {len(model_data['endogenous_variables'])}")
    print(f"- Exogenous variables: {len(model_data['exogenous_variables'])}")
    print(f"- Parameters: {len(model_data['parameters'])}")
    print(f"- Equations: {len(model_data['endogenous_equations'])}")
    print(f"- Maximum lead: {model_data['max_lead']}")
    print(f"- Maximum lag: {model_data['max_lag']}")