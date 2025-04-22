#!/usr/bin/env python3
# dynare_parser.py
# Parser for Dynare model files, transforming them for use with Klein's solution method

import os
import re
import json
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)



class DynareParser:
    """
    Parser for Dynare model files, converting them to a format suitable 
    for Klein's solution method with proper handling of leads/lags
    and support for trend blocks.
    """

    def __init__(self, input_file):
        self.input_file = input_file
        self.content = ""
        self.clean_content = ""

        # Standard lists
        self.var_list = []
        self.varexo_list = []
        self.parameters = []
        self.param_values = {}
        self.original_equations = []

        # After timing adjustment
        self.equations_with_timing = []
        self.var_lead_lag_info = {}

        # Auxiliary for leads/lags
        self.auxiliary_vars = []
        self.auxiliary_eqs = []
        self.final_equations = []

        # Classification
        self.state_variables = []
        self.control_variables = []
        self.shock_to_process_var_map = {}

        # trend‐related attributes ---
        self.varexo_trends = []            # names from varexo_trends block
        self.trends_vars = []              # names from trends_vars block
        self.trend_shocks = []             # names from trend_shocks block
        self.trend_equations = []          # lines in trend_model block
        self.measurement_equations = []    # lines in measument_equations


    # Getters here 
    def get_core_equations(self):
        "Return only the main `model; ... end;` equations."
        return self.original_equations

    def get_trend_equations(self):
        "Return the equations from the `trend_model` block."
        return self.trend_equations

    def get_measurement_equations(self):
        "Return the equations from the `measument_equations` block."
        return self.measurement_equations

    def get_trend_vars(self):
        "Return the list of variables in the `trends_vars` block."
        return self.trends_vars

    def get_trend_shocks(self):
        "Return all trend shocks (from `varexo_trends` + `trend_shocks`)."
        return self.varexo_trends + self.trend_shocks
    
    def _find_trend_model_block(self):
        """
        Extract lines between the 'trend_model' header (no semicolon) and the lone ';' line.
        Returns raw lines (with trailing semicolons) of actual equations.
        """
        lines = []
        in_block = False
        for raw in self._lines:
            l = raw.strip()
            if not in_block:
                # match a line exactly 'trend_model'
                if re.match(r'^(?i)trend_model\s*$', l):
                    in_block = True
            else:
                # stop at a line that is just ';'
                if l == ';':
                    break
                lines.append(raw)
        return lines
    

    # ----------------------------------------------------------------
    # 1) File I/O and cleaning
    # ----------------------------------------------------------------

    def read_file(self):
        """Read the Dynare file into self.content and lines."""
        try:
            with open(self.input_file, 'r') as f:
                self.content = f.read()
            self._lines = self.content.splitlines()
            return True
        except Exception as e:
            print("Error reading file:", e)
            return False

    
    def clean_file(self, out_folder):
        """Strip comments and normalize whitespace."""
        # block comments
        tmp = re.sub(r'/\*.*?\*/', '', self.content, flags=re.DOTALL)
        # line comments
        tmp = re.sub(r'//.*|%.*', '', tmp)
        # collapse whitespace
        tmp = re.sub(r'\s+', ' ', tmp).strip()
        self.clean_content = tmp

        # save
        os.makedirs(out_folder, exist_ok=True)
        with open(os.path.join(out_folder, "clean_file.txt"), 'w') as f:
            f.write(self.clean_content)
        return True

    # ----------------------------------------------------------------
    # 2) Generic block parsing
    # ----------------------------------------------------------------

    def _find_block(self, blockname, semicolon_header=True):
        """
        Return raw lines between 'blockname[;]' and next 'end;'.
        If semicolon_header=False, matches a header line exactly 'blockname'.
        """
        stoks = {blockname.lower()}
        if semicolon_header:
            stoks.add(blockname.lower() + ';')
        collected = []
        in_block = False
        for raw in self._lines:
            l = raw.strip().lower()
            if not in_block:
                if l in stoks:
                    in_block = True
            else:
                if l == 'end;':
                    break
                collected.append(raw)
        return collected

    def _find_subblock_until(self, header, terminator=';'):
        """
        Extract lines between a header line (e.g. "trends_vars" or "trends_vars;")
        and the first line that is exactly the terminator (e.g. ";").
        Handles optional spaces and an optional semicolon after the header.
        """
        import re
        start_re = re.compile(rf'^\s*{re.escape(header)}\s*;?\s*$', re.IGNORECASE)
        term_re  = re.compile(rf'^\s*{re.escape(terminator)}\s*$', re.IGNORECASE)

        out = []
        in_block = False
        for raw in self._lines:
            line = raw.strip()
            if not in_block:
                if start_re.match(line):
                    in_block = True
            else:
                if term_re.match(line):
                    break
                out.append(raw)
        return out
    
    def _clean_block_lines(self, raw_lines):
        """Strip blank/comment lines and trailing semicolons."""
        out = []
        for L in raw_lines:
            l = L.strip()
            if not l or l.startswith('%'):
                continue
            if '//' in l:
                l = l.split('//',1)[0].rstrip()
            if l.endswith(';'):
                l = l[:-1].rstrip()
            if l:
                out.append(l)
        return out
        
    # ----------------------------------------------------------------
    # 3) Declarations
    # ----------------------------------------------------------------

    def extract_declarations(self):
        """
        Extract all the declarations from the Dynare file:
        - var
        - varexo
        - varobs
        - parameters & parameter values
        - varexo_trends      (trend shocks)
        - trends_vars        (trend states)
        - trend_shocks block (var…; stderr…;)
        - shocks block       (var…; stderr…;)
        """

        cc = self.clean_content

        # 1) var (core endogenous)
        m = re.search(r'var\s+(.*?);', cc, re.IGNORECASE|re.DOTALL)
        self.var_list = re.findall(r'\b([A-Za-z]\w*)\b', m.group(1)) if m else []

        # 2) varexo (core exogenous)
        m = re.search(r'varexo\s+(.*?);', cc, re.IGNORECASE|re.DOTALL)
        self.varexo_list = re.findall(r'\b([A-Za-z]\w*)\b', m.group(1)) if m else []

        # 3) varobs
        m = re.search(r'varobs\s+(.*?);', cc, re.IGNORECASE|re.DOTALL)
        self.varobs_list = re.findall(r'\b([A-Za-z]\w*)\b', m.group(1)) if m else []

        # 4) parameters
        m = re.search(r'parameters\s+(.*?);', cc, re.IGNORECASE|re.DOTALL)
        self.parameters = re.findall(r'\b([A-Za-z]\w*)\b', m.group(1)) if m else []

        # 4b) parameter values
        for p in self.parameters:
            pm = re.search(rf'{p}\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*;', cc)
            self.param_values[p] = float(pm.group(1)) if pm else np.nan

        # 5) varexo_trends (list only)
        #raw = self._find_block('varexo_trends', semicolon_header=True)
        raw = self._find_subblock_until('varexo_trends', terminator =';')
        
        self.varexo_trends = []
        for line in raw:
            # strip comments, split commas
            line = re.sub(r'//.*', '', line).strip().rstrip(',')
            for v in line.split(','):
                v = v.strip()
                if v:
                    self.varexo_trends.append(v)

        # 6) trends_vars (list only)
        #raw = self._find_block('trends_vars', semicolon_header=True)
        raw = self._find_subblock_until('trends_vars', terminator=';')
        
        self.trends_vars = []
        for line in raw:
            line = re.sub(r'//.*', '', line).strip().rstrip(',')
            for v in line.split(','):
                v = v.strip()
                if v:
                    self.trends_vars.append(v)

        
        # 7) trend_shocks block: pick only var NAME; stderr VALUE;
        raw = self._find_block('trend_shocks', semicolon_header=True)
        #raw = self._find_subblock_until('trend_shocks', terminator=';')
        self.trend_shocks = []
        last_var = None

        for line in raw:
            # strip comments
            line = re.sub(r'//.*', '', line)
            # split into statements on ';'
            stmts = [s.strip() for s in line.split(';') if s.strip()]
            for s in stmts:
                parts = s.split()
                if len(parts) >= 2 and parts[0].lower() == 'var':
                    # var SHK_X
                    vname = parts[1]
                    self.trend_shocks.append(vname)
                    last_var = vname
                elif len(parts) >= 2 and parts[0].lower() == 'stderr' and last_var is not None:
                    # stderr 1
                    val = float(parts[1])
                    pname = f"stderr_{last_var}"
                    if pname not in self.parameters:
                        self.parameters.append(pname)
                    self.param_values[pname] = val
        

        # 8) core shocks block
        #raw = self._find_block('shocks', semicolon_header=True)
        raw = self._find_subblock_until('shocks', terminator =';')
        items = []
        for L in raw:
            for itm in L.split(';'):
                s = re.sub(r'//.*', '', itm).strip()
                if s:
                    items.append(s)
        last = None
        for s in items:
            parts = s.split()
            if parts[0].lower() == 'var' and len(parts) > 1:
                last = parts[1]
                # ensure in core exo list
                if last not in self.varexo_list:
                    self.varexo_list.append(last)
            elif parts[0].lower() == 'stderr' and last:
                val = float(parts[1])
                pname = f"stderr_{last}"
                if pname not in self.parameters:
                    self.parameters.append(pname)
                self.param_values[pname] = val

        return True

    # ----------------------------------------------------------------
    # 4) Model + trend equations
    # ----------------------------------------------------------------

    def extract_model(self, out_folder):
        mc = self.clean_content

        # main model
        m = re.search(r'model\s*;(.*?)end\s*;', mc, re.IGNORECASE|re.DOTALL)
        if m:
            eqs = [e.strip() for e in m.group(1).split(';') if e.strip()]
            self.original_equations = eqs


        # NEW: trend_model subblock (header 'trend_model;')
        raw_tm = self._find_block('trend_model', semicolon_header=True)
        # clean (strip comments, trailing semicolons, blank lines)
        self.trend_equations = self._clean_block_lines(raw_tm)

        # Then as before:
        raw_me = self._find_block('measument_equations', semicolon_header=True)
        self.measurement_equations = self._clean_block_lines(raw_me)

        # save for inspection
        with open(os.path.join(out_folder, "clean_equations.txt"), 'w') as f:
            f.write("Variables: " + ", ".join(self.var_list) + "\n")
            f.write("Shocks:    " + ", ".join(self.varexo_list) + "\n\n")
            f.write("Equations:\n")
            for eq in self.original_equations:
                f.write("  " + eq + ";\n")
        return True

    
    def identify_shock_equations(self, out_folder):
        """Identify which equations contain shocks and their associated variables."""
        for eq in self.original_equations:
            # Skip equations without =
            if "=" not in eq:
                continue
                
            left, right = [s.strip() for s in eq.split("=", 1)]
            
            # Find all shocks present in this equation
            present_shocks = [shock for shock in self.varexo_list if re.search(rf'\b{re.escape(shock)}\b', right)]
            
            if len(present_shocks) == 1:
                # This is an exogenous process equation
                shock = present_shocks[0]
                variable = left.split("(")[0].strip() if "(" in left else left
                self.shock_to_process_var_map[shock] = variable
                
                # Check if this is the exogenous process equation and lead it forward
                # Use a better regex pattern to find lag terms of the form variable(-n)
                lag_pattern = rf'{re.escape(variable)}\s*\(\s*-\d+\s*\)'
                if variable in self.var_list and re.search(lag_pattern, right):
                    # This is an exogenous process, lead it forward
                    print(f"Leading forward exogenous process equation: {eq}")
                    new_eq = self._lead_forward_equation(eq)
                    self.equations_with_timing.append(new_eq)
                else:
                    self.equations_with_timing.append(eq)
            else:
                # Regular equation, keep as is
                self.equations_with_timing.append(eq)
                
        # Save equations with correct timing
        full_path = os.path.join(out_folder, "clean_file_with_correct_timing.txt")
        with open(full_path, 'w') as f:
            for eq in self.equations_with_timing:
                f.write(eq + ";\n")
                
        return True
    
    def _lead_forward_equation(self, equation):
        """Lead forward an exogenous process equation by one period."""
        if "=" not in equation:
            return equation
            
        left, right = [s.strip() for s in equation.split("=", 1)]
        variable = left.split("(")[0].strip() if "(" in left else left
        
        # Modify left side to add (+1)
        new_left = f"{variable}(+1)"
        
        # Modify right side: replace var(-n) with var(-(n-1))
        new_right = right
        
        # Find all instances of var(-n) and replace them
        # More robust pattern that ensures we only match the specific variable name
        lag_pattern = re.compile(rf'\b({re.escape(variable)})\(\s*-(\d+)\s*\)')
        
        matches = list(lag_pattern.finditer(right))
        # Process in reverse order to avoid changing positions as we replace
        for match in reversed(matches):
            full_match = match.group(0)
            var_name = match.group(1)
            lag = int(match.group(2))
            
            if lag == 1:
                # var(-1) -> var
                replacement = var_name
            else:
                # var(-n) -> var(-(n-1))
                replacement = f"{var_name}(-{lag-1})"
                
            # Replace just this instance at the specific position
            start, end = match.span()
            new_right = new_right[:start] + replacement + new_right[end:]
            
        # Debug output
        print(f"  Original: {left} = {right}")
        print(f"  Led forward: {new_left} = {new_right}")
            
        return f"{new_left} = {new_right}"
    
    def analyze_variable_leads_lags(self):
        """Analyze all equations to identify the maximum lead and lag for each variable."""
        for var in self.var_list:
            self.var_lead_lag_info[var] = {'max_lead': 0, 'max_lag': 0}
            
        for eq in self.equations_with_timing:
            for var in self.var_list:
                # Use word boundary \b to ensure we match complete variable names
                # This prevents partial matches of variable names that are substrings of others
                lead_pattern = re.compile(rf'\b{re.escape(var)}\(\s*\+(\d+)\s*\)')
                lag_pattern = re.compile(rf'\b{re.escape(var)}\(\s*-(\d+)\s*\)')
                
                # Check for leads
                for match in lead_pattern.finditer(eq):
                    lead = int(match.group(1))
                    if lead > self.var_lead_lag_info[var]['max_lead']:
                        self.var_lead_lag_info[var]['max_lead'] = lead
                        print(f"Found lead {lead} for variable {var} in equation: {eq}")
                        
                # Check for lags
                for match in lag_pattern.finditer(eq):
                    lag = int(match.group(1))
                    if lag > self.var_lead_lag_info[var]['max_lag']:
                        self.var_lead_lag_info[var]['max_lag'] = lag
                        print(f"Found lag {lag} for variable {var} in equation: {eq}")
        
        # Print summary of leads and lags           
        print("\nVariable lead/lag summary:")
        for var, info in self.var_lead_lag_info.items():
            if info['max_lead'] > 0 or info['max_lag'] > 0:
                print(f"  {var}: max_lead={info['max_lead']}, max_lag={info['max_lag']}")
                        
        return True
    
    def generate_auxiliary_variables(self, out_folder):
        """Generate auxiliary variables and equations for leads and lags."""
        for var, info in self.var_lead_lag_info.items():
            # Generate auxiliary variables for lags
            if info['max_lag'] > 0:
                # Create state variable for the first lag
                lag_var = f"{var}_lag"
                if lag_var not in self.auxiliary_vars:
                    self.auxiliary_vars.append(lag_var)
                    self.state_variables.append(lag_var)
                    # Add auxiliary equation: lag_var_p = var
                    self.auxiliary_eqs.append(f"{lag_var}_p = {var}")
                
                # Create additional lag variables if needed
                prev_lag_var = lag_var
                for i in range(2, info['max_lag'] + 1):
                    curr_lag_var = f"{var}_lag{i}"
                    if curr_lag_var not in self.auxiliary_vars:
                        self.auxiliary_vars.append(curr_lag_var)
                        self.state_variables.append(curr_lag_var)
                        # Add auxiliary equation: lag_var{i}_p = lag_var{i-1}
                        self.auxiliary_eqs.append(f"{curr_lag_var}_p = {prev_lag_var}")
                    prev_lag_var = curr_lag_var
            
            # Generate auxiliary variables for leads
            if info['max_lead'] > 1:  # Only need auxiliary for lead > 1
                for i in range(1, info['max_lead']):
                    lead_var = f"{var}_lead{i}"
                    if lead_var not in self.auxiliary_vars:
                        self.auxiliary_vars.append(lead_var)
                        self.control_variables.append(lead_var)
                        
                        if i == 1:
                            # Add auxiliary equation: lead_var = var_p
                            self.auxiliary_eqs.append(f"{lead_var} = {var}_p")
                        else:
                            # Add auxiliary equation: lead_var = lead_var{i-1}_p
                            self.auxiliary_eqs.append(f"{lead_var} = {var}_lead{i-1}_p")
                            
        # Save file with auxiliary variables and equations
        full_path = os.path.join(out_folder, "clean_file_with_correct_timing_and_auxiliary_variables.txt")
        with open(full_path, 'w') as f:
            f.write("Original Equations:\n")
            for eq in self.equations_with_timing:
                f.write(eq + ";\n")
            f.write("\nAuxiliary Variables:\n")
            f.write(", ".join(self.auxiliary_vars) + "\n")
            f.write("\nAuxiliary Equations:\n")
            for eq in self.auxiliary_eqs:
                f.write(eq + ";\n")
                
        return True
    
    def substitute_leads_lags(self, out_folder):
        """Replace all lead and lag notations with the appropriate auxiliary variables."""
        # Process original equations
        for eq in self.equations_with_timing:
            new_eq = eq
            
            # Process variables in order of length (longest first) to avoid substring issues
            sorted_vars = sorted(self.var_list, key=len, reverse=True)
            
            for var in sorted_vars:
                # We'll use a two-stage approach:
                # 1. First identify all matches with their positions
                # 2. Then replace them in reverse order to avoid shifting positions
                
                # --- Replace leads ---
                # First replace var(+1) with var_p
                lead1_pattern = re.compile(rf'\b{re.escape(var)}\(\s*\+1\s*\)')
                matches = list(lead1_pattern.finditer(new_eq))
                for match in reversed(matches):
                    start, end = match.span()
                    new_eq = new_eq[:start] + f"{var}_p" + new_eq[end:]
                
                # Then replace var(+n) with var_lead{n-1}_p for n > 1
                lead_pattern = re.compile(rf'\b{re.escape(var)}\(\s*\+(\d+)\s*\)')
                matches = list(lead_pattern.finditer(new_eq))
                for match in reversed(matches):
                    lead = int(match.group(1))
                    if lead > 1:
                        start, end = match.span()
                        replacement = f"{var}_lead{lead-1}_p"
                        new_eq = new_eq[:start] + replacement + new_eq[end:]
                
                # --- Replace lags ---
                # First replace var(-1) with var_lag
                lag1_pattern = re.compile(rf'\b{re.escape(var)}\(\s*-1\s*\)')
                matches = list(lag1_pattern.finditer(new_eq))
                for match in reversed(matches):
                    start, end = match.span()
                    new_eq = new_eq[:start] + f"{var}_lag" + new_eq[end:]
                
                # Then replace var(-n) with var_lag{n} for n > 1
                lag_pattern = re.compile(rf'\b{re.escape(var)}\(\s*-(\d+)\s*\)')
                matches = list(lag_pattern.finditer(new_eq))
                for match in reversed(matches):
                    lag = int(match.group(1))
                    if lag > 1:
                        start, end = match.span()
                        replacement = f"{var}_lag{lag}"
                        new_eq = new_eq[:start] + replacement + new_eq[end:]
            
            print(f"Original: {eq}")
            print(f"Substituted: {new_eq}")
            self.final_equations.append(new_eq)
            
        # Add auxiliary equations
        self.final_equations.extend(self.auxiliary_eqs)

        # Add all auxiliary variables
        updated_var_names = self.var_list
        for aux_var in self.auxiliary_vars:
            if aux_var not in updated_var_names:
                updated_var_names.append(aux_var)
        
        #Add the updated variable names to the list
        self.var_names = updated_var_names

        # Save the file with substitutions
        full_path = os.path.join(out_folder, "clean_file_with_auxiliary_variables_substituted.txt")
        with open(full_path, 'w') as f:
            for eq in self.final_equations:
                f.write(eq + ";\n")
                
        return True
    

    def classify_variables(self):
        """
        Classifies variables into states and controls based on naming conventions
        and exogenous process identification after auxiliary variables are created.
        - State variables:
            - Variables directly driven by exogenous shocks.
            - Auxiliary variables representing lags (e.g., 'var_lag', 'var_lag2').
        - Control variables: All other variables (including original non-state variables
            and auxiliary lead variables like 'var_p', 'var_lead1_p').
        """
        # Use sets for efficient handling and uniqueness
        # self.var_names should contain original + auxiliary vars at this point
        all_vars_set = set(self.var_names)
        add_state_variables_set = set()

        # 1. Add variables directly driven by shocks (exogenous states)
        # Ensure these variables actually exist in the final list
        exogenous_states = set(self.shock_to_process_var_map.values())
        

        # 2. Add auxiliary variables representing lags based on naming convention
        #    Regex to match '_lag' or '_lag' followed by digits at the end of the string
        lag_var_pattern = re.compile(r'_lag\d*$') # Matches _lag or _lagN at the end
        for var in all_vars_set:
            # Check if the variable name ends with _lag or _lagN
            if lag_var_pattern.search(var):
                # Check if the base variable name exists (e.g., for 'X_lag', check if 'X' exists)
                base_var = lag_var_pattern.sub('', var)
                if base_var in self.var_list: # or base_var in self.auxiliary_vars:
                    add_state_variables_set.add(var)
        
        # 3. Control variables are all variables not classified as states
    
        state_variables_set =  exogenous_states | add_state_variables_set
        control_variables_set = all_vars_set - state_variables_set

        # Convert sets back to sorted lists for consistent ordering
        self.state_variables = list(exogenous_states) + list(add_state_variables_set)
        self.control_variables = list(control_variables_set)

        # Sanity check
        if len(self.state_variables) + len(self.control_variables) != len(all_vars_set):
            print("Warning: Variable classification mismatch. Some variables might be unclassified or double-counted.")
            print(f"Total unique vars: {len(all_vars_set)}")
            print(f"States found ({len(self.state_variables)}): {self.state_variables}")
            print(f"Controls found ({len(self.control_variables)}): {self.control_variables}")


        # Print classification results
        print("Variable Classification:")
        print(f"  States: {len(self.state_variables)}")
        print(f"  Controls: {len(self.control_variables)}")
        print(f"  Total Variables (incl. auxiliary): {len(self.var_names)} (Unique: {len(all_vars_set)})")
        print(f"  State Variables: {self.state_variables}")
        # print(f"  Control Variables: {self.control_variables}") # Uncomment to print controls
        return True
        
    def format_equations_for_json(self):
        """Format equations for JSON output in the required form."""
        formatted_equations = []
        
        for i, eq in enumerate(self.final_equations, 1):
            if "=" in eq:
                left, right = [s.strip() for s in eq.split("=", 1)]
                # Format: right - (left)
                formatted_eq = f"{right} - ({left})"
                formatted_equations.append({f"eq{i}": formatted_eq})
            else:
                # If there's no =, just use the equation as is
                formatted_equations.append({f"eq{i}": eq.strip()})
                
        return formatted_equations
    
    def generate_json_output(self, out_folder):
        """Generate JSON output with the model information."""
        # Format equations
        formatted_equations = self.format_equations_for_json()
        
        # Create the JSON structure
        model_json = {
            "parameters": self.parameters,
            "param_values": self.param_values,
            "states": self.state_variables,
            "controls": self.control_variables,
            "all_variables": self.state_variables + self.control_variables,
            "shocks": self.varexo_list,
            "equations": formatted_equations,
            "shock_to_process_var_map": self.shock_to_process_var_map
        }
        
        # Write to file
        full_path = os.path.join(out_folder, "model_json.json")
        with open(full_path, 'w') as f:
            json.dump(model_json, f, indent=2)
            
        return model_json
    
    def generate_jacobian_matrices(self, out_folder):
        """Generate Jacobian matrices A, B, C using symbolic differentiation for the CORE model."""
        # 1) Build symbols
        param_symbols = {p: sp.Symbol(p) for p in self.parameters}
        # Core variables at t and t+1
        all_vars    = self.state_variables + self.control_variables
        var_symbols = {v: sp.Symbol(v) for v in all_vars}
        var_p_symbols = {f"{v}_p": sp.Symbol(f"{v}_p") for v in all_vars}

        # CORE shocks: use only the shocks that map to a process variable
        core_shocks = list(self.shock_to_process_var_map.keys())
        shock_symbols = {s: sp.Symbol(s) for s in core_shocks}

        # Combine all symbols
        all_symbols = {**param_symbols, **var_symbols, **var_p_symbols, **shock_symbols}

        # 2) Parse core equations only
        symbolic_eqs = []
        transformers = standard_transformations + (implicit_multiplication_application,)
        for eq_dict in self.format_equations_for_json():
            eqstr = next(iter(eq_dict.values()))
            # parse into symbolic
            sym = parse_expr(eqstr, local_dict=all_symbols,
                              transformations=transformers)
            symbolic_eqs.append(sym)
        F = sp.Matrix(symbolic_eqs)

        # 3) Build parameter, variable, shock lists in the correct order
        theta_syms = [param_symbols[p] for p in self.parameters]
        x_syms     = [var_symbols[v] for v in all_vars]
        x_p_syms   = [var_p_symbols[f"{v}_p"] for v in all_vars]
        eps_syms   = [shock_symbols[s] for s in core_shocks]

        # 4) Compute Jacobians
        A = F.jacobian(x_p_syms)          # ∂F/∂x_p
        B = -F.jacobian(x_syms)           # -∂F/∂x
        C = -F.jacobian(eps_syms)         # -∂F/∂ε (only core shocks)

        # 5) Lambdify into Python code
        import numpy as _np
        code_lines = [
            "import numpy as np",
            "def evaluate_jacobians(theta):",
            "    # Unpack parameters",
        ]
        for i, p in enumerate(self.parameters):
            code_lines.append(f"    {p} = theta[{i}]")
        code_lines += [
            f"    A = np.zeros(({A.rows}, {A.cols}))",
            f"    B = np.zeros(({B.rows}, {B.cols}))",
            f"    C = np.zeros(({C.rows}, {C.cols}))",
            "",
            "    # Fill A",
        ]
        # A entries
        for i in range(A.rows):
            for j in range(A.cols):
                if A[i,j] != 0:
                    expr = str(A[i,j]).replace("exp","np.exp")
                    code_lines.append(f"    A[{i},{j}] = {expr}")
        # B entries
        code_lines += ["", "    # Fill B"]
        for i in range(B.rows):
            for j in range(B.cols):
                if B[i,j] != 0:
                    expr = str(B[i,j]).replace("exp","np.exp")
                    code_lines.append(f"    B[{i},{j}] = {expr}")
        # C entries
        code_lines += ["", "    # Fill C (core shocks)"]
        for i in range(C.rows):
            for j in range(C.cols):
                if C[i,j] != 0:
                    expr = str(C[i,j]).replace("exp","np.exp")
                    code_lines.append(f"    C[{i},{j}] = {expr}")

        code_lines.append("")
        code_lines.append("    return A, B, C")

        # Write to file
        with open(os.path.join(out_folder, "jacobian_matrices.py"), 'w') as f:
            f.write("\n".join(code_lines))

        return True
    
    def generate_model_structure(self, out_folder):
        """Generate model structure with indices and selection matrices."""
        all_vars = self.state_variables + self.control_variables
        
        # Calculate indices
        n_states = len(self.state_variables)
        n_controls = len(self.control_variables)
        n_vars = len(all_vars)
        n_shocks = len(self.varexo_list)
        
        # Create R structure (shock->state direct impact)
        R_struct = np.zeros((n_states, n_shocks))
        
        # Create C selection matrix (selects states)
        C_selection = np.zeros((n_vars, n_states))
        for i, state in enumerate(self.state_variables):
            state_idx = all_vars.index(state)
            C_selection[state_idx, i] = 1.0
            
        # Create D structure (shock->var direct impact)
        D_struct = np.zeros((n_vars, n_shocks))
        for shock, var in self.shock_to_process_var_map.items():
            if shock in self.varexo_list and var in all_vars:
                shock_idx = self.varexo_list.index(shock)
                var_idx = all_vars.index(var)
                D_struct[var_idx, shock_idx] = 1.0
                
        # Create indices dictionary
        indices = {
            'n_states': n_states,
            'n_controls': n_controls,
            'n_vars': n_vars,
            'n_shocks': n_shocks,
            'n_endogenous_states': 0,  # This would need to be calculated based on further analysis
            'n_exo_states_ws': 0,      # Same here
            'n_exo_states_wos': 0,     # Same here
            'zero_persistence_processes': []  # This would need to be calculated
        }
        
        # Create labels dictionary
        labels = {
            'state_labels': self.state_variables,
            'control_labels': self.control_variables,
            'variable_labels': all_vars,
            'shock_labels': self.varexo_list
        }
        
        # Create model structure file
        structure_code = [
            "import numpy as np",
            "",
            f"indices = {indices}",
            "",
            f"R_struct = np.array({R_struct.tolist()})",
            "",
            f"C_selection = np.array({C_selection.tolist()})",
            "",
            f"D_struct = np.array({D_struct.tolist()})",
            "",
            "# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls",
            "",
            f"labels = {labels}"
            "",
            f"shock_to_process_var_map = {self.shock_to_process_var_map}" # Added shock map
        ]
        
        # Write to file
        full_path = os.path.join(out_folder, "model_structure.py")
        with open(full_path, 'w') as f:
            f.write('\n'.join(structure_code))
            
        return True
    
    def clean_log_files(self, folder_path):
        """
        Checks if the specified folder exists. If it does, deletes the entire folder
        and all its contents recursively.

        Args:
            folder_path (str): The path to the directory to be cleaned (deleted).
        """
        import shutil
        print(f"--- Running clean_log_files for: '{folder_path}' ---")

        # 1. Check if the path exists and is a directory
        if os.path.isdir(folder_path):
            print(f"Directory '{folder_path}' exists. Proceeding with deletion...")
            try:
                # 2. Delete the directory tree
                shutil.rmtree(folder_path)
                print(f"Successfully deleted directory: '{folder_path}'")
            except OSError as e:
                # Handle potential errors during deletion (e.g., permissions, file in use)
                print(f"Error deleting directory '{folder_path}': {e}")
            except Exception as e:
                # Catch any other unexpected errors
                print(f"An unexpected error occurred while deleting '{folder_path}': {e}")
        # Optional: Check if the path exists but is not a directory
        elif os.path.exists(folder_path):
            print(f"Path '{folder_path}' exists but is NOT a directory. No action taken.")
        else:
            # 3. If the directory doesn't exist, just report it
            print(f"Directory '{folder_path}' does not exist. No action needed.")

        print(f"--- Finished clean_log_files for: '{folder_path}' ---")

    def parse(self, out_folder="model_files"):
        """Execute the full parsing process."""
        print("Starting Dynare model parsing...")
        #Step 0: Delete the output folder if it exists
        self.clean_log_files(out_folder)

        # Create output folder if it doesn't exist
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(f"Created output folder: {out_folder}")

        # Step 1: Read and clean the file
        if not self.read_file():
            return False
        if not self.clean_file(out_folder):
            return False
            
        # Step 2: Extract declarations
        if not self.extract_declarations():
            return False
            
        # Step 3: Extract model equations
        if not self.extract_model(out_folder):
            return False
            
        # Step 4: Identify and lead forward exogenous equations
        if not self.identify_shock_equations(out_folder):
            return False
            
        # Step 5: Analyze variables for leads and lags
        if not self.analyze_variable_leads_lags():
            return False
            
        # Step 6: Generate auxiliary variables and equations
        if not self.generate_auxiliary_variables(out_folder):
            return False
            
        # Step 7: Substitute leads and lags with auxiliary variables
        if not self.substitute_leads_lags(out_folder):
            return False
            
        # Step 8: Classify variables as state or control
        if not self.classify_variables():
            return False
            
        # Step 9: Generate JSON output
        model_json = self.generate_json_output(out_folder)
            
        # Step 10: Generate Jacobian matrices
        if not self.generate_jacobian_matrices(out_folder):
            return False
            
        # Step 11: Generate model structure
        if not self.generate_model_structure(out_folder):
            return False
            
        print("Dynare model parsing completed successfully!")
        return True

    
    def generate_trend_state_space(self, out_folder):
        """
        1) Rewrite Dynare 'trend_model' eqns into canonical form X_p = f(X, eps)
        2) Symbolically compute A_tr = ∂f/∂X, B_tr = ∂f/∂eps
        3) Emit trend_jacobians.py
        4) Build measurement blocks D_tr (cycle→obs) and C_tr (trend→obs)
        """
        import os, re
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application
        )
        import numpy as np

        # 1) Preprocess: drop all VAR(-1)->VAR, rename LHS var to VAR_p
        processed = []
        for eq in self.trend_equations:
            lhs, rhs = [s.strip() for s in eq.split('=',1)]
            v = re.match(r'([A-Za-z]\w*)', lhs).group(1)
            new_lhs = f"{v}_p"
            rhs_mod = rhs
            for vv in self.trends_vars:
                rhs_mod = re.sub(rf'\b{vv}\(\s*-\s*1\s*\)', vv, rhs_mod)
            processed.append((new_lhs, rhs_mod))

        # 2) Build symbols
        x_syms = {v: sp.Symbol(v) for v in self.trends_vars}
        sh_syms = {s: sp.Symbol(s) for s in self.trend_shocks}
        p_syms = {p: sp.Symbol(p) for p in self.parameters}

        local = {**x_syms, **sh_syms, **p_syms}

        # 3) Build numeric Jacobian expressions for f
        #    f_i(x,eps) is the RHS expression of processed[i]
        f_exprs = []
        for lhs, rhs in processed:
            expr = parse_expr(rhs, local_dict=local,
                              transformations=standard_transformations
                                + (implicit_multiplication_application,))
            f_exprs.append(expr)
        Fm = sp.Matrix(f_exprs)

        # A_tr_syms[i,j] = d f_i / d x_j
        A_tr_syms = Fm.jacobian([x_syms[v] for v in self.trends_vars])
        # B_tr_syms[i,k] = d f_i / d eps_k
        B_tr_syms = Fm.jacobian([sh_syms[s] for s in self.trend_shocks])

        # 4) Write trend_jacobians.py
        n_tr, _    = A_tr_syms.shape
        _, n_sh_tr = B_tr_syms.shape
        jacfile = os.path.join(out_folder, "trend_jacobians.py")
        code = [
            "import numpy as np",
            "",
            "def evaluate_trend_jacobians(theta):",
            "    \"\"\"Return A_tr, B_tr\"\"\""
        ]
        # Unpack theta
        for i,p in enumerate(self.parameters):
            code.append(f"    {p} = theta[{i}]")
        code.append(f"    A_tr = np.zeros(({n_tr},{n_tr}))")
        code.append(f"    B_tr = np.zeros(({n_tr},{n_sh_tr}))")
        code.append("    # fill A_tr")
        for i in range(n_tr):
            for j in range(n_tr):
                v = A_tr_syms[i,j]
                if v != 0:
                    s = str(v).replace("exp","np.exp")
                    code.append(f"    A_tr[{i},{j}] = {s}")
        code.append("    # fill B_tr")
        for i in range(n_tr):
            for j in range(n_sh_tr):
                v = B_tr_syms[i,j]
                if v != 0:
                    s = str(v).replace("exp","np.exp")
                    code.append(f"    B_tr[{i},{j}] = {s}")
        code.append("    return A_tr, B_tr")

        with open(jacfile, "w") as f:
            f.write("\n".join(code))
        print("Wrote trend_jacobians.py to", jacfile)

        # 5) Build measurement blocks by token‐matching
        obs_raw = self._find_subblock_until('varobs', terminator=';')
        obs = []
        for line in obs_raw:
            clean = re.sub(r'//.*', '', line).strip().rstrip(',')
            if clean:
                obs += [v.strip() for v in clean.split(',') if v.strip()]
        self.varobs_list = obs

        core_cycle = [v for v in self.var_list if v not in self.trends_vars]
        D_tr = np.zeros((len(obs), len(core_cycle)))
        C_tr = np.zeros((len(obs), len(self.trends_vars)))
        for i, me in enumerate(self.measurement_equations):
            _, rhs = [s.strip() for s in me.split('=',1)]
            toks = re.findall(r'\b[A-Za-z]\w*\b', rhs)
            for tok in toks:
                if tok in core_cycle:
                    D_tr[i, core_cycle.index(tok)] = 1.0
                elif tok in self.trends_vars:
                    C_tr[i, self.trends_vars.index(tok)] = 1.0

        self.D_tr = D_tr
        self.C_tr = C_tr
        return
    

        
# def main():
#     """Main function to run the parser on a file."""
#     # import argparse
    
#     # parser = argparse.ArgumentParser(description='Parse a Dynare model file for use with Klein solution method.')
#     # parser.add_argument('file', help='The Dynare model file to parse')
#     # args = parser.parse_args()
#     import os
#     script_dir = os.path.dirname(__file__)
#     os.chdir(script_dir)
#     dynare_file = "qpm_simpl1_with_trends.dyn"
#     dynare_parser = DynareParser(dynare_file)
#     success = dynare_parser.parse()
    
#     if success:
#         print("Files generated:")
#         print("  - clean_file.txt")
#         print("  - clean_file_with_correct_timing.txt")
#         print("  - clean_file_with_correct_timing_and_auxiliary_variables.txt")
#         print("  - clean_file_with_auxiliary_variables_substituted.txt")
#         print("  - model_json.json")
#         print("  - jacobian_matrices.py")
#         print("  - model_structure.py")
#     else:
#         print("Parsing failed.")
        
# if __name__ == "__main__":
#     main()