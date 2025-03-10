#%%
import os
import re
import json

import os
import re
import json

class DynareParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_content = self._read_file()
        self.variables = []
        self.shocks = []
        self.parameters = {}
        self.equations = []
        self.aux_vars = set()
        self.state_vars = set()
        self.control_vars = set()
        
        self._parse_variables()
        self._parse_shocks()
        self._parse_parameters()
        self._parse_model()
        self._classify_variables()

    def _read_file(self):
        with open(self.file_path, 'r') as f:
            return f.read()

    def _parse_variables(self):
        var_match = re.search(r'var\s*((?!varexo).*?);', self.raw_content, re.DOTALL|re.IGNORECASE)
        if var_match:
            var_block = var_match.group(1)
            self.variables = [v.strip() for v in re.findall(r'(\w+)(?=\s*//|;|,|\n)', var_block)]

    def _parse_shocks(self):
        shocks_match = re.search(r'varexo\s*(.*?);', self.raw_content, re.DOTALL|re.IGNORECASE)
        if shocks_match:
            self.shocks = [s.strip() for s in re.findall(r'(\w+)(?=\s*//|;|,|\n)', shocks_match.group(1))]

    def _parse_parameters(self):
        param_pattern = r'(\w+)\s*=\s*([\d.]+)(?=\s*;)'
        param_matches = re.findall(param_pattern, self.raw_content)
        self.parameters = {name: float(value) for name, value in param_matches}

    def _parse_model(self):
        model_block = re.search(r'model;(.*?)end;', self.raw_content, re.DOTALL|re.IGNORECASE)
        if not model_block:
            raise ValueError("No model block found")
            
        equations = [eq.split('//')[0].strip() 
                    for eq in model_block.group(1).split(';') if eq.strip()]
        
        for eq in equations:
            if eq:
                processed_eq = self._process_equation(eq)
                self.equations.append(processed_eq)

    def _process_equation(self, eq):
        # Process all lead/lag patterns with proper variable boundary detection
        return re.sub(r'\b(\w+)\s*\(([+-]\d+)\)\b', 
                    lambda m: self._handle_timing(m), eq)

    def _handle_timing(self, match):
        var_name, timing = match.groups()
        timing = int(timing)
        
        if abs(timing) == 1:
            return f"{var_name}_p" if timing > 0 else f"{var_name}_lag"
        else:
            return self._create_aux_vars(var_name, timing)

    def _create_aux_vars(self, base_var, timing):
        prefix = "lead" if timing > 0 else "lag"
        steps = abs(timing)
        aux_vars = [f"{base_var}_{prefix}{i}" for i in range(1, steps+1)]
        
        # Create chain of auxiliary equations
        for i in range(steps):
            if timing > 0:
                lhs = f"{base_var}_p" if i == 0 else f"{aux_vars[i-1]}_p"
                rhs = aux_vars[i]
            else:
                lhs = f"{aux_vars[i]}_p"
                rhs = base_var if i == 0 else aux_vars[i-1]
            
            eq = f"{lhs} = {rhs}"
            if eq not in self.equations:
                self.equations.insert(0, eq)  # Add auxiliary equations first
                
        self.aux_vars.update(aux_vars)
        return aux_vars[-1]

    def _classify_variables(self):
        all_vars = set(self.variables) | self.aux_vars
        
        for var in all_vars:
            if any(part.startswith('lag') for part in var.split('_')):
                self.state_vars.add(var)
            else:
                self.control_vars.add(var)
                
        # Ensure original variables without lags are controls
        for var in self.variables:
            if var not in self.state_vars:
                self.control_vars.add(var)

    def generate_json(self):
        return {
            "equations": [{"eq"+str(i+1): eq} for i, eq in enumerate(self.equations)],
            "state_variables": sorted(self.state_vars),
            "control_variables": sorted(self.control_vars),
            "parameters": self.parameters,
            "shocks": self.shocks,
            "variable_order": sorted(self.state_vars) + sorted(self.control_vars)
        }

    def save_json(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.generate_json(), f, indent=2, ensure_ascii=False)

# if __name__ == "__main__":
#     import os
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
#     parser = DynareParser(dynare_file)
#     parser.save_json(os.path.join(script_dir, "transformed_model_claude.json"))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = DynareParser(os.path.join(script_dir, "qpm_simpl1.dyn"))
    parser.save_json(os.path.join(script_dir, "transformed_model_deep.json"))


# if __name__ == "__main__":
#     import os 
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
#     parser = DynareParser(dynare_file)
#     parser.save_json(os.path.join(script_dir,"transformed_model_deep.json"))
