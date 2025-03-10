#%%
import re
import json
import os
import numpy as np
import scipy.linalg as la
import sys
import matplotlib.pyplot as plt

class DynareParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.var_list = []
        self.varexo_list = []
        self.parameters = {}
        self.equations = []
        self.model_text = ""
        self.state_variables = []
        self.control_variables = []
        self.all_variables = []
        self.auxiliary_variables = []
        
    # def read_dynare_file(self):
    #     """Read the Dynare .mod file content"""
    #     with open(self.file_path, 'r') as file:
    #         self.content = file.read()

    def read_dynare_file(self):
        """Read and preprocess the Dynare .mod file content"""
        with open(self.file_path, 'r') as file:
            self.content = file.read()
        self.preprocess_content()  # Clean content immediately after reading


    def preprocess_content(self):
        """Remove comments and clean up content before parsing"""
        # Remove single-line comments
        self.content = re.sub(r'//.*', '', self.content)
        # Remove extra whitespace
        self.content = re.sub(r'\s+', ' ', self.content)

    def parse_variables(self):
        """Extract variable declarations from the Dynare file"""
        var_section = re.search(r'var\s+(.*?);', self.content, re.DOTALL)
        if var_section:
            var_text = var_section.group(1)
            # No need for comment removal here since we preprocessed
            var_list = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', var_text)
            self.var_list = [v for v in var_list if v]

        
    def parse_exogenous(self):
        """Extract exogenous variable declarations from the Dynare file"""
        varexo_section = re.search(r'varexo\s+(.*?);', self.content, re.DOTALL)
        if varexo_section:
            varexo_text = varexo_section.group(1)
            # Remove comments and split by whitespace
            varexo_text = re.sub(r'//.*', '', varexo_text)
            varexo_list = [v.strip() for v in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', varexo_text)]
            self.varexo_list = [v for v in varexo_list if v]  # Remove empty strings
            
    def parse_parameters(self):
        """Extract parameter declarations and values from the Dynare file"""
        # Get parameter names
        params_section = re.search(r'parameters\s+(.*?);', self.content, re.DOTALL)
        if params_section:
            params_text = params_section.group(1)
            params_text = re.sub(r'//.*', '', params_text)  # Remove comments
            param_list = [p.strip() for p in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', params_text)]
            param_list = [p for p in param_list if p]  # Remove empty strings
            
            # Initialize parameters dictionary
            for param in param_list:
                self.parameters[param] = None
                
            # Get parameter values
            for param in param_list:
                param_value = re.search(rf'{param}\s*=\s*([0-9.-]+)', self.content)
                if param_value:
                    self.parameters[param] = float(param_value.group(1))
    
    def parse_model(self):
        """Extract the model equations from the Dynare file"""
        model_section = re.search(r'model;(.*?)end;', self.content, re.DOTALL)
        if model_section:
            self.model_text = model_section.group(1).strip()
            
            # Split by semicolons to get individual equations
            equations = re.split(r';', self.model_text)
            equations = [eq.strip() for eq in equations if eq.strip()]
            
            self.equations = equations


    # def identify_state_control_variables(self):
    #     """Identify state/control variables from transformed var_list"""
    #     # This method is modified to use the transformed variable list
    #     if not hasattr(self, 'transformed_var_list'):
    #         # If transformed_var_list doesn't exist yet, generate it
    #         self.generate_auxiliary_equations()
        
    #     # State variables: All variables with _lag suffix
    #     self.state_variables = sorted([var for var in self.transformed_var_list if "_lag" in var])
        
    #     # Control variables: Remaining non-exogenous variables that aren't future variables
    #     self.control_variables = sorted([
    #         var for var in self.transformed_var_list 
    #         if var not in self.state_variables
    #         and var not in self.varexo_list
    #         and not var.endswith("_p")
    #     ])

    # def identify_state_control_variables(self):
    #     """Identify state/control variables and ensure exogenous states appear last"""
    #     if not hasattr(self, 'transformed_var_list'):
    #         self.generate_auxiliary_equations()
        
    #     # First identify all state variables
    #     all_state_variables = [var for var in self.transformed_var_list if "_lag" in var]
        
    #     # Separate endogenous and exogenous state variables
    #     # Typically exogenous state variables begin with "RES_" or "SHK_" in your model
    #     exogenous_states = [var for var in all_state_variables if var.startswith(("RES_"))]
    #     endogenous_states = [var for var in all_state_variables if not var.startswith(("RES_", "SHK_"))]
        
    #     # Combine with exogenous last
    #     self.state_variables = sorted(endogenous_states) + sorted(exogenous_states)
        
    #     # Control variables remain the same
    #     self.control_variables = sorted([
    #         var for var in self.transformed_var_list 
    #         if var not in self.state_variables
    #         and var not in self.varexo_list
    #         and not var.endswith("_p")
    #     ])


    def analyze_model_variables(self):
        """
        First pass: Analyze all variables and their time shifts across the entire model.
        
        Returns:
            variable_shifts: Dictionary mapping each variable to its set of time shifts
            all_variables: Set of all base variable names found in the model
        """
        variable_shifts = {}  # Maps variables to their time shifts
        all_variables = set()  # All base variable names
        
        # Process each equation to find variables and their time shifts
        for equation in self.equations:
            # Remove comments and clean up
            equation = re.sub(r'//.*', '', equation).strip()
            
            # Find all base variables (excluding parameters)
            base_vars = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', equation))
            base_vars = base_vars - set(self.parameters.keys())
            
            # Add to all_variables set
            all_variables.update(base_vars)
            
            # For each variable, find all its lead/lag patterns
            for var_name in base_vars:
                # Initialize if not already in dictionary
                if var_name not in variable_shifts:
                    variable_shifts[var_name] = set()
                    variable_shifts[var_name].add(0)  # Always include current period
                
                # Find all lead/lag patterns for this variable
                lead_lag_pattern = rf'{var_name}\(\s*([+-]?\d+)\s*\)'
                lead_lag_matches = re.findall(lead_lag_pattern, equation)
                
                # Add all time shifts found for this variable
                for time_shift_str in lead_lag_matches:
                    variable_shifts[var_name].add(int(time_shift_str))
        
        return variable_shifts, all_variables

    def create_transformation_plan(self, variable_shifts):
        """
        Create a comprehensive transformation plan for all variables in the model.
        
        Args:
            variable_shifts: Dictionary mapping each variable to its set of time shifts
        
        Returns:
            transformation_map: Dictionary mapping original variable expressions to transformed names
            aux_equations: List of auxiliary equations needed
            model_variables: Dictionary of all variable types in the transformed model
        """
        transformation_map = {}  # Maps original expressions to transformed variable names
        aux_equations = []       # List of auxiliary equations
        processed_aux_eqs = set() # Track which auxiliary equations have been added
        
        # Track different types of variables in the transformed model
        model_variables = {
            'state_variables': set(),     # Variables with _lag suffix
            'control_variables': set(),   # Non-state, non-exogenous, non-future variables
            'aux_variables': set(),       # Auxiliary variables
            'all_variables': set(),       # All variables in transformed model
            'lead_variables': set()       # Variables with _lead suffix
        }
        
        # For each variable, create transformation rules and auxiliary equations
        for var_name, shifts in variable_shifts.items():
            # Skip exogenous variables (shocks) in the main variable lists
            if var_name in self.varexo_list:
                continue
                
            # Process all time shifts for this variable
            for shift in sorted(shifts):  # Sort shifts to ensure consistent processing
                # Current period (shift = 0)
                if shift == 0:
                    # No transformation needed
                    transformation_map[var_name] = var_name
                    model_variables['all_variables'].add(var_name)
                    
                    # Add to control variables (unless it's an exogenous shock)
                    if var_name not in self.varexo_list:
                        model_variables['control_variables'].add(var_name)
                
                # One-period lead (shift = 1)
                elif shift == 1:
                    orig_expr = f"{var_name}(+1)"
                    transformed_var = f"{var_name}_p"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    # Note: _p variables are not considered control variables
                
                # One-period lag (shift = -1)
                elif shift == -1:
                    orig_expr = f"{var_name}(-1)"
                    transformed_var = f"{var_name}_lag"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    model_variables['state_variables'].add(transformed_var)
                    
                    # Add auxiliary equation for the lag if not already added
                    aux_eq = f"{transformed_var}_p = {var_name}"
                    if aux_eq not in processed_aux_eqs:
                        aux_equations.append(aux_eq)
                        processed_aux_eqs.add(aux_eq)
                    model_variables['aux_variables'].add(transformed_var)
                
                # Multi-period leads (shift > 1)
                elif shift > 1:
                    # Define direct transformation for the equation
                    orig_expr = f"{var_name}(+{shift})"
                    transformed_var = f"{var_name}_lead{shift}"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    model_variables['lead_variables'].add(transformed_var)
                    model_variables['control_variables'].add(transformed_var)
                    
                    # Generate auxiliary variables and equations for all leads
                    # For lead variables, we need: var_lead = var_p, var_lead2 = var_lead_p, etc.
                    for i in range(1, shift + 1):
                        # Add to all variables
                        lead_var = f"{var_name}_lead{i}"
                        model_variables['all_variables'].add(lead_var)
                        model_variables['lead_variables'].add(lead_var)
                        model_variables['control_variables'].add(lead_var)
                        
                        # Create auxiliary equations according to the pattern:
                        # var_lead = var_p
                        # var_lead2 = var_lead_p
                        # var_lead3 = var_lead2_p
                        if i == 1:
                            # First lead relates to the base variable
                            aux_eq = f"{lead_var} = {var_name}_p"
                        else:
                            # Higher leads relate to the previous lead
                            prev_lead = f"{var_name}_lead{i-1}"
                            aux_eq = f"{lead_var} = {prev_lead}_p"
                        
                        # Only add if not already processed
                        if aux_eq not in processed_aux_eqs:
                            aux_equations.append(aux_eq)
                            processed_aux_eqs.add(aux_eq)
                            model_variables['aux_variables'].add(lead_var)
                
                # Multi-period lags (shift < -1)
                elif shift < -1:
                    abs_shift = abs(shift)
                    orig_expr = f"{var_name}({shift})"
                    transformed_var = f"{var_name}_lag{abs_shift}"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    model_variables['state_variables'].add(transformed_var)
                    
                    # Generate auxiliary variables and equations for all lags:
                    # For lag variables, we need:
                    # var_lag_p = var
                    # var_lag2_p = var_lag
                    # var_lag3_p = var_lag2
                    
                    # First add the standard one-period lag if needed
                    std_lag_var = f"{var_name}_lag"
                    model_variables['all_variables'].add(std_lag_var)
                    model_variables['state_variables'].add(std_lag_var)
                    
                    std_lag_eq = f"{std_lag_var}_p = {var_name}"
                    if std_lag_eq not in processed_aux_eqs:
                        aux_equations.append(std_lag_eq)
                        processed_aux_eqs.add(std_lag_eq)
                        model_variables['aux_variables'].add(std_lag_var)
                    
                    # Generate additional lags as needed
                    for i in range(2, abs_shift + 1):
                        # The current lag variable
                        lag_var = f"{var_name}_lag{i}"
                        model_variables['all_variables'].add(lag_var)
                        model_variables['state_variables'].add(lag_var)
                        
                        # The previous lag variable
                        prev_lag = f"{var_name}_lag{i-1}"
                        
                        # Auxiliary equation: current_lag_p = previous_lag
                        aux_eq = f"{lag_var}_p = {prev_lag}"
                        
                        if aux_eq not in processed_aux_eqs:
                            aux_equations.append(aux_eq)
                            processed_aux_eqs.add(aux_eq)
                            model_variables['aux_variables'].add(lag_var)
        
        return transformation_map, aux_equations, model_variables


    def apply_transformation(self):
        """
        Two-pass transformation of the model:
        1. Analyze all variables and their time shifts
        2. Create a comprehensive transformation plan
        3. Apply transformations consistently across all equations
        4. Update model variables and add auxiliary equations
        
        Returns:
            Dictionary with transformed model information
        """
        # First pass: Analyze variables and their shifts
        variable_shifts, all_variables = self.analyze_model_variables()
        
        # Create transformation plan based on the analysis
        transformation_map, aux_equations, model_variables = self.create_transformation_plan(variable_shifts)
        
        # Apply transformations to all equations
        transformed_equations = []
        for i, equation in enumerate(self.equations):
            # Remove comments and clean up
            clean_eq = re.sub(r'//.*', '', equation).strip()
            transformed_eq = clean_eq
            
            # Process each variable in the equation
            for var_name in all_variables:
                # Replace var(+1) with var_p
                transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*\+1\s*\)', f'{var_name}_p', transformed_eq)
                
                # Replace var(-1) with var_lag
                transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*-1\s*\)', f'{var_name}_lag', transformed_eq)
                
                # Replace var(+n) with var_leadn for n > 1
                for j in range(2, 10):  # Assume no leads greater than +9
                    transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*\+{j}\s*\)', f'{var_name}_lead{j}', transformed_eq)
                
                # Replace var(-n) with var_lagn for n > 1
                for j in range(2, 10):  # Assume no lags greater than -9
                    transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*-{j}\s*\)', f'{var_name}_lag{j}', transformed_eq)
            
            transformed_equations.append(transformed_eq)
        
        # Update class properties
        self.transformed_equations = transformed_equations
        self.auxiliary_equations = aux_equations
        
        # Remove _p variables and shock variables from main model variables
        endogenous_vars = set([v for v in model_variables['all_variables'] 
                            if not v.endswith('_p') and v not in self.varexo_list])
        
        # # Properly classify state/control variables
        self.state_variables = list(set(model_variables['state_variables']))
        
        # First, separate the state variables into endogenous and exogenous
        self.exogenous_states = [var for var in self.state_variables if var.startswith("RES_") and var.endswith("_lag")]
        self.endogenous_states = [var for var in self.state_variables if var not in self.exogenous_states]

        # Reorder state variables with exogenous states last
        self.state_variables = sorted(self.endogenous_states) + sorted(self.exogenous_states)

        # Control variables are endogenous variables that are not state variables
        self.control_variables = list(endogenous_vars - set(self.state_variables))
        
        # Remove shock variables from all_variables
        self.all_variables = list(endogenous_vars)
        
        # Keep auxiliary variables as defined
        self.auxiliary_variables = list(set(model_variables['aux_variables']))
        
        # Format equations for output
        formatted_equations = self.format_transformed_equations(transformed_equations, aux_equations)
        
        return {
            'equations': formatted_equations,
            'state_variables': sorted(self.state_variables),
            'control_variables': sorted(self.control_variables),
            'auxiliary_variables': sorted(self.auxiliary_variables),
            'all_variables': sorted(self.all_variables)
        }

    # def format_transformed_equations(self, main_equations, aux_equations):
    #     """Format equations for output"""
    #     formatted_equations = []
        
    #     # Process main equations
    #     for i, equation in enumerate(main_equations):
    #         # Convert equation to standard form (right side - left side = 0)
    #         if "=" in equation:
    #             left_side, right_side = equation.split("=", 1)
    #             formatted_eq = f"{right_side.strip()} - ({left_side.strip()})"
    #         else:
    #             formatted_eq = equation
            
    #         eq_dict = {f"eq{i+1}": formatted_eq}
    #         formatted_equations.append(eq_dict)
        
    #     # Process auxiliary equations
    #     for i, aux_equation in enumerate(aux_equations):
    #         left_side, right_side = aux_equation.split("=", 1)
    #         formatted_eq = f"{right_side.strip()} - ({left_side.strip()})"
            
    #         eq_dict = {f"eq{len(main_equations) + i + 1}": formatted_eq}
    #         formatted_equations.append(eq_dict)
        
    #     return formatted_equations


    def format_transformed_equations(self, main_equations, aux_equations):
        """Format transformed equations for output"""
        formatted_equations = []
        
        # Process main equations
        for i, equation in enumerate(main_equations):
            # Convert equation to standard form (right side - left side = 0)
            if "=" in equation:
                left_side, right_side = equation.split("=", 1)
                formatted_eq = f"{right_side.strip()} - ({left_side.strip()})"
            else:
                formatted_eq = equation
            
            eq_dict = {f"eq{i+1}": formatted_eq}
            formatted_equations.append(eq_dict)
        
        # Process auxiliary equations
        for i, aux_equation in enumerate(aux_equations):
            left_side, right_side = aux_equation.split("=", 1)
            formatted_eq = f"{right_side.strip()} - ({left_side.strip()})"
            
            eq_dict = {f"eq{len(main_equations) + i + 1}": formatted_eq}
            formatted_equations.append(eq_dict)
        
        return formatted_equations

    
    def prepare_json_output(self):
        """Prepare the final JSON output"""
        # Combine all variables
        self.all_variables = self.state_variables + self.control_variables
        
        # Format equations
        formatted_equations = self.format_transformed_equations()
        
        # Create the JSON structure
        output = {
            "equations": formatted_equations,
            "state_variables": self.state_variables,
            "control_variables": self.control_variables,
            "parameters": list(self.parameters.keys()),
            "param_values": self.parameters,
            "shocks": self.varexo_list,
            "output_text": self.generate_output_text(formatted_equations)
        }
        
        return output
    
    def generate_output_text(self, formatted_equations):
        """Generate output text in the required format"""
        output_text = "equations = {\n"
        
        for i, eq_dict in enumerate(formatted_equations):
            for eq_name, eq_value in eq_dict.items():
                output_text += f'\t{{"{eq_name}": "{eq_value}"}}'
                if i < len(formatted_equations) - 1:
                    output_text += ",\n"
                else:
                    output_text += "\n"
        
        output_text += "};\n\n"
        
        output_text += "variables = ["
        output_text += ", ".join([f'"{var}"' for var in self.all_variables])
        output_text += "];\n\n"
        
        output_text += "parameters = ["
        output_text += ", ".join([f'"{param}"' for param in self.parameters.keys()])
        output_text += "];\n\n"
        
        for param, value in self.parameters.items():
            output_text += f"{param} = {value};\n"
        
        output_text += "\n"
        
        output_text += "shocks = ["
        output_text += ", ".join([f'"{shock}"' for shock in self.varexo_list])
        output_text += "];\n"
        
        return output_text
    
    def parse(self):
        """Main parsing function, updated with two-pass transformation"""
        self.read_dynare_file()
        self.parse_variables()
        self.parse_exogenous()
        self.parse_parameters()
        self.parse_model()
        
        # Use the two-pass transformation approach
        output = self.apply_transformation()
        
        # Add other necessary output fields
        output['parameters'] = list(self.parameters.keys())
        output['param_values'] = self.parameters
        output['shocks'] = self.varexo_list
        output['output_text'] = self.generate_output_text(output['equations'])
        
        return output
    
    def save_json(self, output_file):
        """Save the parsed model to a JSON file"""
        output = self.parse()
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Model parsed and saved to {output_file}")
        return output

    # def generate_jacobian_evaluator(self, output_file=None):
    #     """
    #     Generate a Python function that evaluates the Jacobian matrices for the model.
        
    #     Args:
    #         output_file (str, optional): Path to save the generated Python code
            
    #     Returns:
    #         str: The generated Python code for the Jacobian evaluator
    #     """
    #     import sympy as sy
    #     import re
        
    #     print("Generating Jacobian evaluator...")
        
    #     # Basic model components
    #     variables = self.state_variables + self.control_variables
    #     exogenous = self.varexo_list
    #     parameters = list(self.parameters.keys())
        
    #     # Create variables with "_p" suffix for t+1 variables
    #     variables_p = [var + "_p" for var in variables]
        
    #     # Create symbolic variables for all model components
    #     var_symbols = {var: sy.symbols(var) for var in variables}
    #     var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
    #     exo_symbols = {exo: sy.symbols(exo) for exo in exogenous}
    #     param_symbols = {param: sy.symbols(param) for param in parameters}
        
    #     # Combine all symbols
    #     all_symbols = {**var_symbols, **var_p_symbols, **exo_symbols, **param_symbols}
        
    #     # Get endogenous equations from the formatted equations
    #     formatted_equations = self.format_transformed_equations()
    #     endogenous_eqs = {}
    #     for eq_dict in formatted_equations:
    #         endogenous_eqs.update(eq_dict)
        
    #     # Parse endogenous equations into sympy expressions
    #     equations = []
    #     success_count = 0
    #     error_count = 0
        
    #     for eq_name, eq_str in endogenous_eqs.items():
    #         # Convert string to sympy expression
    #         eq_expr = eq_str
    #         for name, symbol in all_symbols.items():
    #             # Use regex to match whole words only
    #             pattern = r'\b' + re.escape(name) + r'\b'
    #             eq_expr = re.sub(pattern, str(symbol), eq_expr)
            
    #         # Try to parse the expression
    #         try:
    #             expr = sy.sympify(eq_expr)
    #             equations.append(expr)
    #             success_count += 1
    #         except Exception as e:
    #             print(f"Failed to parse equation {eq_name}: {eq_str}")
    #             print(f"Error: {str(e)}")
    #             # Try to recover by using a placeholder
    #             equations.append(sy.sympify("0"))
    #             error_count += 1
        
    #     print(f"Parsed {success_count} equations successfully, {error_count} with errors")
        
    #     # Create system as sympy Matrix
    #     F = sy.Matrix(equations)
        
    #     # Compute Jacobians for endogenous system
    #     X_symbols = [var_symbols[var] for var in variables]
    #     X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
    #     Z_symbols = [exo_symbols[exo] for exo in exogenous]  
        
    #     # A = ∂F/∂X_p (Jacobian with respect to future variables)
    #     print("Computing A matrix...")
    #     A_symbolic = -F.jacobian(X_p_symbols)
        
    #     # B = -∂F/∂X (negative Jacobian with respect to current variables)
    #     print("Computing B matrix...")
    #     B_symbolic = F.jacobian(X_symbols)
        
    #     # C = -∂F/∂Z (negative Jacobian with respect to exogenous processes)
    #     print("Computing C matrix...")
    #     C_symbolic = F.jacobian(Z_symbols)
        
    #     print("Generating output code...")
        
    #     # Generate code for the Jacobian evaluation function
    #     function_code = [
    #         "import numpy as np",
    #         "",
    #         "def evaluate_jacobians(theta):",
    #         "    \"\"\"",
    #         "    Evaluates Jacobian matrices for the Klein method and VAR representation",
    #         "    ",
    #         "    Args:",
    #         "        theta: List or array of parameter values in the order of:",
    #         f"            {parameters}",
    #         "        ",
    #         "    Returns:",
    #         "        a: Matrix ∂F/∂X_p (Jacobian with respect to future variables)",
    #         "        b: Matrix -∂F/∂X (negative Jacobian with respect to current variables)",
    #         "        c: Matrix -∂F/∂Z (negative Jacobian with respect to exogenous processes)",
    #         "    \"\"\"",
    #         "    # Unpack parameters from theta"
    #     ]
        
    #     # Add parameter unpacking
    #     for i, param in enumerate(parameters):
    #         function_code.append(f"    {param} = theta[{i}]")
        
    #     # Initialize matrices
    #     function_code.extend([
    #         "",
    #         f"    a = np.zeros(({len(equations)}, {len(variables)}))",
    #         f"    b = np.zeros(({len(equations)}, {len(variables)}))",
    #         f"    c = np.zeros(({len(equations)}, {len(exogenous)}))"   
    #     ])
        
    #     # Add A matrix elements
    #     function_code.append("")
    #     function_code.append("    # A matrix elements")
    #     for i in range(A_symbolic.rows):
    #         for j in range(A_symbolic.cols):
    #             if A_symbolic[i, j] != 0:
    #                 expr = str(A_symbolic[i, j])
    #                 # Clean up the expression
    #                 for param in parameters:
    #                     # Replace symbol with parameter name
    #                     pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
    #                     expr = re.sub(pattern, param, expr)
    #                 function_code.append(f"    a[{i}, {j}] = {expr}")
        
    #     # Add B matrix elements
    #     function_code.append("")
    #     function_code.append("    # B matrix elements")
    #     for i in range(B_symbolic.rows):
    #         for j in range(B_symbolic.cols):
    #             if B_symbolic[i, j] != 0:
    #                 expr = str(B_symbolic[i, j])
    #                 # Clean up the expression
    #                 for param in parameters:
    #                     pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
    #                     expr = re.sub(pattern, param, expr)
    #                 function_code.append(f"    b[{i}, {j}] = {expr}")
        
    #     # Add C matrix elements
    #     function_code.append("")
    #     function_code.append("    # C matrix elements")
    #     for i in range(C_symbolic.rows):
    #         for j in range(C_symbolic.cols):
    #             if C_symbolic[i, j] != 0:
    #                 expr = str(C_symbolic[i, j])
    #                 # Clean up the expression
    #                 for param in parameters:
    #                     pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
    #                     expr = re.sub(pattern, param, expr)
    #                 function_code.append(f"    c[{i}, {j}] = {expr}")
        
    #     # Return all matrices
    #     function_code.append("")
    #     function_code.append("    return a, b, c")
        
    #     # Join all lines to form the complete function code
    #     complete_code = "\n".join(function_code)
        
    #     # Save to file if specified
    #     if output_file:
    #         with open(output_file, 'w') as f:
    #             f.write(complete_code)
    #         print(f"Jacobian evaluator saved to {output_file}")
        
    #     return complete_code


    def generate_jacobian_evaluator(self, output_file=None):
        """
        Generate a Python function that evaluates the Jacobian matrices for the model.
        
        Args:
            output_file (str, optional): Path to save the generated Python code
                
        Returns:
            str: The generated Python code for the Jacobian evaluator
        """
        import sympy as sy
        import re
        
        print("Generating Jacobian evaluator...")
        
        # First, apply the model transformation if it hasn't been done yet
        if not hasattr(self, 'transformed_equations') or not self.transformed_equations:
            print("Applying model transformation first...")
            self.apply_transformation()
        
        # Get the relevant model components after transformation
        variables = self.state_variables + self.control_variables
        exogenous = self.varexo_list
        parameters = list(self.parameters.keys())
        
        # Create variables with "_p" suffix for t+1 variables
        variables_p = [var + "_p" for var in variables]
        
        # Create symbolic variables for all model components
        var_symbols = {var: sy.symbols(var) for var in variables}
        var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
        exo_symbols = {exo: sy.symbols(exo) for exo in exogenous}
        param_symbols = {param: sy.symbols(param) for param in parameters}
        
        # Combine all symbols
        all_symbols = {**var_symbols, **var_p_symbols, **exo_symbols, **param_symbols}
        
        # Get endogenous equations from the formatted equations
        formatted_equations = self.format_transformed_equations(self.transformed_equations, self.auxiliary_equations)
        endogenous_eqs = {}
        for eq_dict in formatted_equations:
            endogenous_eqs.update(eq_dict)
        
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
        Z_symbols = [exo_symbols[exo] for exo in exogenous]  
        
        # A = ∂F/∂X_p (Jacobian with respect to future variables)
        print("Computing A matrix...")
        A_symbolic = -F.jacobian(X_p_symbols)
        
        # B = -∂F/∂X (negative Jacobian with respect to current variables)
        print("Computing B matrix...")
        B_symbolic = F.jacobian(X_symbols)
        
        # C = -∂F/∂Z (negative Jacobian with respect to exogenous processes)
        print("Computing C matrix...")
        C_symbolic = F.jacobian(Z_symbols)
        
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
            f"            {parameters}",
            "        ",
            "    Returns:",
            "        a: Matrix ∂F/∂X_p (Jacobian with respect to future variables)",
            "        b: Matrix -∂F/∂X (negative Jacobian with respect to current variables)",
            "        c: Matrix -∂F/∂Z (negative Jacobian with respect to exogenous processes)",
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
            f"    c = np.zeros(({len(equations)}, {len(exogenous)}))"   
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
        
        # Return all matrices
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


def klein(a=None,b=None,n_states=None,eigenvalue_warnings=True):

    '''Solves linear dynamic models with the form of:
    
                a*Et[x(t+1)] = b*x(t)       
                
        [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
        a vector of nonpredetermined costate variables. z(t) is a vector of exogenous forcing variables with 
        autocorrelation matrix phi. The solution to the model is a set of matrices f, n, p, l such that:

                u(t)   = f*s(t)
                s(t+1) = p*s(t).

        The solution algorithm is based on Klein (2000) and his solab.m Matlab program.

    Args:
        a:                      (Numpy ndarray) Coefficient matrix on future-dated variables
        b:                      (Numpy ndarray) Coefficient matrix on current-dated variables
        c:                      (Numpy ndarray) Coefficient matrix on exogenous forcing variables
        n_states:               (int) Number of state variables
        eigenvalue_warnings:    (bool) Whether to print warnings that there are too many or few eigenvalues. Default: True

    Returns:
        f:          (Numpy ndarray) Solution matrix coeffients on s(t) for u(t)
        p:          (Numpy ndarray) Solution matrix coeffients on s(t) for s(t+1)
        stab:       (int) Indicates solution stability and uniqueness
                        stab == 1: too many stable eigenvalues
                        stab == -1: too few stable eigenvalues
                        stab == 0: just enoughstable eigenvalues
        eig:        The generalized eigenvalues from the Schur decomposition


    '''

    s,t,alpha,beta,q,z = la.ordqz(A=a,B=b,sort='ouc',output='complex')


        

    # Components of the z matrix
    z11 = z[0:n_states,0:n_states]
    z12 = z[0:n_states,n_states:]
    z21 = z[n_states:,0:n_states]
    z22 = z[n_states:,n_states:]

    # number of nonpredetermined variables
    n_costates = np.shape(a)[0] - n_states
    
    if n_states>0:
        if np.linalg.matrix_rank(z11)<n_states:
            sys.exit("Invertibility condition violated. Check model equations or parameter values.")

    s11 = s[0:n_states,0:n_states];
    if n_states>0:
        z11i = la.inv(z11)
        s11i = la.inv(s11)
    else:
        z11i = z11
        s11i = s11

    # Components of the s,t,and q matrices
    s12 = s[0:n_states,n_states:]
    s22 = s[n_states:,n_states:]
    t11 = t[0:n_states,0:n_states]
    t12 = t[0:n_states,n_states:]
    t22 = t[n_states:,n_states:]
    q1  = q[0:n_states,:]
    q2  = q[n_states:,:]

    # Verify that there are exactly n_states stable (inside the unit circle) eigenvalues:
    stab = 0

    if n_states>0:
        if np.abs(t[n_states-1,n_states-1])>np.abs(s[n_states-1,n_states-1]):
            if eigenvalue_warnings:
                print('Warning: Too few stable eigenvalues. Check model equations or parameter values.')
            stab = -1

    if n_states<n_states+n_costates:
        if np.abs(t[n_states,n_states])<np.abs(s[n_states,n_states]):
            if eigenvalue_warnings:
                print('Warning: Too many stable eigenvalues. Check model equations or parameter values.')
            stab = 1

    # Compute the generalized eigenvalues
    tii = np.diag(t)
    sii = np.diag(s)
    eig = np.zeros(np.shape(tii),dtype=np.complex128)
    # eig = np.zeros(np.shape(tii))

    for k in range(len(tii)):
        if np.abs(sii[k])>0:
            # eig[k] = np.abs(tii[k])/np.abs(sii[k])
            eig[k] = tii[k]/sii[k]    
        else:
            eig[k] = np.inf

    print('Eigenvalues:', np.abs(eig))
    # Solution matrix coefficients on the endogenous state
    if n_states>0:
            dyn = np.linalg.solve(s11,t11)
    else:
        dyn = np.array([])


    f = z21.dot(z11i)
    p = z11.dot(dyn).dot(z11i)

    f = np.real(f)
    p = np.real(p)

    return f, p,stab,eig



def build_fortran_state_space(F, P, C, n_states, n_controls, n_exogenous):
    """
    Builds the state space representation matching the Fortran code structure.
    
    This function creates the state-space matrices that match the structure used in
    the Fortran implementation (found in dsoltokalman.f90 and related files).
    
    Variable ordering in the state-space:
    1. Control variables (non-predetermined)
    2. Endogenous state variables (predetermined)
    3. Exogenous state variables (like shocks with AR processes)
    
    Matrix structures:
    - T matrix (transition):
      [0               F1              F2*P_exo] (controls)
      [0               P_endo          P_endo_exo*P_exo] (endogenous states)
      [0               0               P_exo] (exogenous states)
    
    - R matrix (shock impact):
      [F2*P_exo] (controls affected through exogenous states)
      [P_endo_exo] (endogenous states affected by exogenous states)
      [I] (identity matrix for direct shock impacts)
    
    Where:
    - F1: Impact of endogenous states on controls
    - F2: Impact of exogenous states on controls
    - P_endo: Transition of endogenous states
    - P_exo: Autoregressive process of exogenous states
    - P_endo_exo: Impact of exogenous states on endogenous states
    
    Args:
        F:          (numpy.ndarray) Solution matrix from Klein (control-to-state mapping)
        P:          (numpy.ndarray) Solution matrix from Klein (state transition)
        C:          (numpy.ndarray) Shock impact matrix from the linearized system
        n_states:   (int) Number of state variables (both endogenous and exogenous)
        n_controls: (int) Number of control variables
        n_exogenous: (int) Number of exogenous shock processes
        
    Returns:
        T:          (numpy.ndarray) Transition matrix for state space matching Fortran
        R:          (numpy.ndarray) Shock impact matrix for state space matching Fortran
    """
    # Total size of the state vector
    n_total = n_controls + n_states 
    
    # In Fortran representation, we have to separate endogenous from exogenous states
    n_endo_states = n_states - n_exogenous
    
    # Initialize the state transition matrix T
    T = np.zeros((n_total, n_total))
    
    # 1. Initialize T matrix according to blocks in Fortran code
    
    # Upper middle block: controls depend on endogenous states through F
    T[:n_controls, n_controls:n_controls+n_endo_states] = F[:, :n_endo_states]
    
    # Upper right block: controls depend on exogenous states 
    T[:n_controls, n_controls+n_endo_states:] = np.dot(F[:, n_endo_states:], P[n_endo_states:, n_endo_states:])
    
    # Middle middle block: endogenous state transitions
    T[n_controls:n_controls+n_endo_states, n_controls:n_controls+n_endo_states] = P[:n_endo_states, :n_endo_states]
    
    # Middle right block: endogenous states affected by exogenous states
    T[n_controls:n_controls+n_endo_states, n_controls+n_endo_states:] = np.dot(
        P[:n_endo_states, n_endo_states:], 
        P[n_endo_states:, n_endo_states:]
    )
    
    # Lower right block: exogenous state processes (AR processes)
    T[n_controls+n_endo_states:, n_controls+n_endo_states:] = P[n_endo_states:, n_endo_states:]
    
    # 2. Now create the R matrix (shock impact)
    R = np.zeros((n_total, n_exogenous))
    
    # Upper part: controls are affected by shocks through F
    R[:n_controls, :] = np.dot(F[:, n_endo_states:], P[n_endo_states:, n_endo_states:])
    
    # Middle part: endogenous states affected by shocks 
    R[n_controls:n_controls+n_endo_states, :] = P[:n_endo_states, n_endo_states:]
    
    # Lower part: unit shock impacts on exogenous processes - optimized
    exo_indices = np.arange(n_exogenous)
    R[n_controls+n_endo_states+exo_indices, exo_indices] = 1.0
        
    return T, R

def impulse_response_fortran(T, R, shock_idx, periods, scale=1.0, parser=None):
    """
    Computes impulse response functions for a specified shock using the Fortran-aligned state space.
    
    Args:
        T:          (numpy.ndarray) Transition matrix (Fortran aligned)
        R:          (numpy.ndarray) Shock impact matrix (Fortran aligned)
        shock_idx:  (int) Index of shock to analyze
        periods:    (int) Number of periods for IRF
        scale:      (float) Scale factor for the shock
        parser:     (DynareParser) Parser instance to get variable ordering
        
    Returns:
        irf:        (pandas.DataFrame) Impulse responses with variable names
    """
    import pandas as pd
    
    n_variables = T.shape[0]
    n_shocks = R.shape[1]
    
    # Check shock index
    if shock_idx >= n_shocks:
        raise ValueError(f"Shock index {shock_idx} out of range (0-{n_shocks-1})")
    
    # Create shock vector (one-time shock)
    shock = np.zeros(n_shocks)
    shock[shock_idx] = scale
    
    # Create shock matrix (all zeros except for first period)
    shocks = np.zeros((periods, n_shocks))
    shocks[0, :] = shock
    
    # Initialize with zeros
    x0 = np.zeros(n_variables)
    
    # Compute IRF using the simulation function
    irf_values = simulate_state_space(T, R, x0, shocks, periods)
    
    # Get variable ordering from parser
    if parser:
        # Order should be: controls first, then endogenous states, then exogenous states
        # This depends on how your parser identifies these variables
        
        # Separate exogenous states (often starting with RES_ or similar prefix)
        exogenous_states = [var for var in parser.state_variables if var.startswith(("RES_"))]
        endogenous_states = [var for var in parser.state_variables if not var.startswith(("RES_"))]
        
        # Combine in the order the state space expects
        ordered_vars = parser.control_variables + endogenous_states + exogenous_states
        
        # Create DataFrame with variable names
        irf = pd.DataFrame(
            data=irf_values[1:, :],  # Skip the initial zeros
            columns=ordered_vars,
            index=pd.RangeIndex(stop=periods, name='Period')
        )
    else:
        # If no parser provided, just use numerical indices
        irf = pd.DataFrame(
            data=irf_values[1:, :],
            columns=[f"var_{i}" for i in range(n_variables)],
            index=pd.RangeIndex(stop=periods, name='Period')
        )
    
    return irf







def plot_variables(df, variables, figsize=(12, 8)):
    """
    Plots selected variables from a DataFrame against its index (Period).
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    variables (list): List of column names to plot
    figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot each selected variable
    for var in variables:
        if var in df.columns:
            plt.plot(df.index, df[var], label=var)
        else:
            print(f"Warning: Variable {var} not found in DataFrame columns")
    
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.title('Economic Variables Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:




def print_model_details(parser):
    """Print detailed model information from parser components we've seen in the code context"""
    # Header
    print("="*60)
    print("MODEL DETAILS".center(60))
    print("="*60)
    
    # Variables (from Context 3)
    print("\nVARIABLES:")
    print(f"State variables ({len(parser.state_variables)}): {', '.join(parser.state_variables)}")
    print(f"Control variables ({len(parser.control_variables)}): {', '.join(parser.control_variables)}")
    print(f"Exogenous shocks ({len(parser.varexo_list)}): {', '.join(parser.varexo_list)}")  # From Context 2
    
    # Equations (from Context 2)
    print(f"\nEQUATIONS: {len(parser.equations)} total")  # From equations list in Context 2
    
    # Solution matrices (from Context 3)
    print("\nKLEIN SOLUTION MATRICES:")
    print(f"f matrix ({parser.f.shape}): Policy functions mapping states to controls")
    print("First 3 rows/columns:")
    print(parser.f[:3, :3])  # Show top-left corner
    
    print(f"\np matrix ({parser.p.shape}): State transition matrix")
    print("First 3 rows/columns:")
    print(parser.p[:3, :3])
    
    # Matrix-var relationships (from Context 3)
    print("\nSTRUCTURE:")
    print(f"Total equations: {len(parser.equations)}")
    print(f"State variables: {len(parser.state_variables)}")
    print(f"Control variables: {len(parser.control_variables)}")
    print(f"Exogenous shocks: {len(parser.varexo_list)}")
    print(f"Solution stability: {'Stable' if parser.stab else 'Unstable'}")  # From Context 3
    
    print("="*60)


def simulate_state_space(A, B, x0, shocks, periods):
    """
    Simulates the state space model for a given number of periods.
    
    Args:
        A:          (numpy.ndarray) Transition matrix
        B:          (numpy.ndarray) Shock impact matrix
        x0:         (numpy.ndarray) Initial state vector
        shocks:     (numpy.ndarray) Matrix of shocks (periods x n_shocks)
        periods:    (int) Number of periods to simulate
        
    Returns:
        x:          (numpy.ndarray) Simulated state variables (periods x n_variables)
    """
    n_variables = A.shape[0]
    n_shocks = B.shape[1]
    
    # Initialize simulation array
    x = np.zeros((periods+1, n_variables))
    x[0, :] = x0
    
    # Check shocks dimensions
    if shocks.shape[0] < periods:
        raise ValueError(f"Shock matrix must have at least {periods} rows")
    if shocks.shape[1] != n_shocks:
        raise ValueError(f"Shock matrix must have {n_shocks} columns")
    
    # Simulate forward
    for t in range(periods):
        x[t+1, :] = A @ x[t, :] + B @ shocks[t, :]
    
    return x


# Example usage
if __name__ == "__main__":
    import os 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
    parser = DynareParser(dynare_file)
    parser.save_json(os.path.join(script_dir,"transformed_model_claude.json"))
    parser.generate_jacobian_evaluator("_jacobian_evaluator.py")

    #Evaluate Jacobians
    # Load Jacobian evaluator function
    import importlib.util
    spec = importlib.util.spec_from_file_location("jacobian_evaluator", "_jacobian_evaluator.py")
    jacobian_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jacobian_module)
    evaluate_jacobians = jacobian_module.evaluate_jacobians
    
    # Get parameter values
    param_values = list(parser.parameters.values())
    
    #Number of state variables#print_model_output(parser)

    
    n_states = len(parser.state_variables)
    n_exogenous = len(parser.varexo_list)
    n_equations = len(parser.all_variables)
    n_controls = n_equations - n_states
    
    # Evaluate Jacobians
    a, b, c = evaluate_jacobians(param_values)

    f, p, stab, eig = klein(a=a, b=b, n_states=n_states, eigenvalue_warnings=True)
    parser.f = f
    parser.p = p
    parser.stab = stab
    parser.eig= eig

    print_model_details(parser)
    print("Done!")

    # T, R = build_state_space(f, p, c, n_states, n_equations)    
    # irfs = impulse_response(T, R,2, periods=40, scale=1.0)

    # Build state space using Fortran alignment
    T, R = build_fortran_state_space(f, p, c, n_states, n_controls, n_exogenous)
    
    # Compute impulse responses
    irfs = impulse_response_fortran(T, R, shock_idx=2, periods=40, scale=1.0, parser=parser)

    # Plot IRFs
    variables_to_plot = [
        "RR_GAP",
        "RS",    
        "DLA_CPI",
        "L_GDP_GAP",   
        "RES_RS",
        "RES_LGDP_GAP",
        "RES_DLA_CPI"
    ]
    plot_variables(irfs, variables_to_plot)

    