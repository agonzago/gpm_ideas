#%%

"""
Simple Dynare Parser for Klein's Method
"""

import re
import os

class DynareParser:
    def __init__(self):
        self.variables = []
        self.exogenous = []
        self.parameters = []
        self.param_values = {}
        self.equations = []
        self.max_lead = 0
        self.max_lag = 0
        self.transformed_variables = []
        self.transformed_equations = []
        
    def parse_file(self, file_content):
        """Parse a Dynare file content into structured data"""
        # Extract var section
        var_match = re.search(r'var\s+(.*?);', file_content, re.DOTALL)
        if var_match:
            var_section = var_match.group(1)
            # Remove comments and extract variable names
            var_section = re.sub(r'//.*?$', '', var_section, flags=re.MULTILINE)
            self.variables = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', var_section)]
        
        # Extract varexo section
        varexo_match = re.search(r'varexo\s+(.*?);', file_content, re.DOTALL)
        if varexo_match:
            varexo_section = varexo_match.group(1)
            varexo_section = re.sub(r'//.*?$', '', varexo_section, flags=re.MULTILINE)
            self.exogenous = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', varexo_section)]
        
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
            # Clean up comments and split into equations
            cleaned_lines = []
            for line in model_section.split(';'):
                line = re.sub(r'//.*?$', '', line, flags=re.MULTILINE).strip()
                if line:
                    cleaned_lines.append(line)
            self.equations = cleaned_lines
        
        # Find max lead and lag
        for eq in self.equations:
            # Search for leads like varname(+n)
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq)
            for var, lead in lead_matches:
                self.max_lead = max(self.max_lead, int(lead))
            
            # Search for lags like varname(-n)
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
            for var, lag in lag_matches:
                self.max_lag = max(self.max_lag, int(lag))
    
    def transform_model(self):
        """Transform the Dynare model into a system with only t and t+1 variables"""
        # Create the transformed variables list
        self.transformed_variables = self.variables.copy()
        
        # Track which variables have lags and leads and their maximum lag/lead
        var_max_lag = {}
        var_max_lead = {}
        
        # First pass - identify what needs transforming
        for eq in self.equations:
            # Clean the equation of any comments before processing
            eq_clean = re.sub(r'//.*$', '', eq).strip()
            
            # Find all variables with leads
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq_clean)
            for var, lead in lead_matches:
                lead_val = int(lead)
                if lead_val >= 1:
                    var_max_lead[var] = max(var_max_lead.get(var, 0), lead_val)
            
            # Find all variables with lags
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq_clean)
            for var, lag in lag_matches:
                lag_val = int(lag)
                if lag_val >= 1:
                    var_max_lag[var] = max(var_max_lag.get(var, 0), lag_val)
        
        # Add lag variables to transformed variables list only for variables that have lags
        for var, max_lag in var_max_lag.items():
            for lag in range(1, max_lag + 1):
                lag_suffix = str(lag) if lag > 1 else ""
                self.transformed_variables.append(f"{var}_lag{lag_suffix}")
        
        # Add lead variables beyond +1 to transformed variables list only for variables that have leads beyond +1
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:
                for lead in range(1, max_lead + 1):
                    self.transformed_variables.append(f"{var}_lead{lead}")
        
        # Transform equations
        for i, eq in enumerate(self.equations):
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
            
            # Split the equation on the equals sign
            if "=" in transformed_eq:
                lhs, rhs = transformed_eq.split("=", 1)
                transformed_eq = f"{rhs.strip()} - ({lhs.strip()})"
                
            self.transformed_equations.append({f"eq{i+1}": transformed_eq.strip()})
        
        # Add transition equations for lags, only for variables that have lags
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
        
        # Add transition equations for leads, only for variables that have leads beyond +1
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:
                # First lead equation: var_p = var_lead1
                self.transformed_equations.append({f"eq{eq_num}": f"{var}_p - {var}_lead1"})
                eq_num += 1
                
                # Additional lead equations: var_leadN_p = var_lead(N+1)
                for lead in range(1, max_lead):
                    self.transformed_equations.append({f"eq{eq_num}": f"{var}_lead{lead}_p - {var}_lead{lead+1}"})
                    eq_num += 1
    
    def generate_output(self):
        """Generate the output script with the transformed model"""
        output = ""
        
        # Output equations
        output += "equations = {\n"
        for i, eq_dict in enumerate(self.transformed_equations):
            for key, value in eq_dict.items():
                output += f'    {{"{key}": "{value}"}}'
                if i < len(self.transformed_equations) - 1:
                    output += ","
                output += "\n"
        output += "};\n\n"
        
        # Output variables
        variables_str = ", ".join([f'"{var}"' for var in self.transformed_variables])
        output += f"variables = [{variables_str}];\n\n"
        
        # Output parameters
        parameters_str = ", ".join([f'"{param}"' for param in self.parameters])
        output += f"parameters = [{parameters_str}];\n\n"
        
        # Output parameter values
        param_values_str = "\n".join([f"{param} = {value};" for param, value in self.param_values.items()])
        output += f"{param_values_str}\n\n"
        
        # Output shocks
        shocks_str = ", ".join([f'"{shock}"' for shock in self.exogenous])
        output += f"shocks = [{shocks_str}];\n"
        
        return output


def parse_dynare_file(filename):
    """
    Parse a Dynare model file and transform it for Klein's method
    
    Args:
        filename (str): Path to the Dynare file
    
    Returns:
        dict: Dictionary with transformed model components
    """
    # Read the Dynare file
    with open(filename, 'r') as f:
        dynare_content = f.read()
    
    # Parse and transform
    parser = DynareParser()
    parser.parse_file(dynare_content)
    parser.transform_model()
    
    # Return a dictionary with all components
    return {
        'equations': parser.transformed_equations,
        'variables': parser.transformed_variables,
        'parameters': parser.parameters,
        'param_values': parser.param_values,
        'shocks': parser.exogenous,
        'output_text': parser.generate_output()
    }


def save_transformed_model(parsed_model, output_file):
    """
    Save the transformed model to a file
    
    Args:
        parsed_model (dict): Parsed model from parse_dynare_file()
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write(parsed_model['output_text'])


# Example usage
if __name__ == "__main__":
    # This is just a simple example - you won't need to run this directly
    model = parse_dynare_file("/Volumes/TOSHIBA EXT/main_work/Work/Projects/iris_replacement/Simplest/qpm_simpl1.dyn")
    save_transformed_model(model, "qpm_simpl1.txt")