#!/usr/bin/env python3
"""
dynare_parser_main.py
A simplified Dynare-style parser with example usage.
"""

import os
import re
import sympy
import numpy as np
import pickle
import collections
import importlib.util
import datetime 

def time_shift_expression(expr, shift, parser_symbols, var_names_set):
    """
    Shift the time index of variables (x_p1, x_m2, etc.) in 'expr' by 'shift' steps.
    """
    if shift == 0:
        return expr

    subs_dict = {}
    for atom in expr.free_symbols:
        nm = atom.name
        base = None
        k = 0
        is_var = False

        m_aux_lead = re.match(r'(aux_\w+_lead)(\d+)$', nm)
        m_aux_lag  = re.match(r'(aux_\w+_lag)(\d+)$', nm)
        m_lead     = re.match(r'(\w+)_p(\d+)$', nm)
        m_lag      = re.match(r'(\w+)_m(\d+)$', nm)

        if m_aux_lead:
            base = m_aux_lead.group(1)
            k = int(m_aux_lead.group(2))
            is_var = True
        elif m_aux_lag:
            base = m_aux_lag.group(1)
            k = -int(m_aux_lag.group(2))
            is_var = True
        elif m_lead:
            vb = m_lead.group(1)
            if vb in var_names_set and not vb.lower().startswith('aux_'):
                base = vb
                k = int(m_lead.group(2))
                is_var = True
        elif m_lag:
            vb = m_lag.group(1)
            if vb in var_names_set and not vb.lower().startswith('aux_'):
                base = vb
                k = -int(m_lag.group(2))
                is_var = True
        elif nm in var_names_set:
            base = nm
            k = 0
            is_var = True

        if is_var:
            new_k = k + shift
            if new_k == 0:
                mo = re.match(r'aux_(\w+)_(?:lead|lag)', base)
                cbase = mo.group(1) if mo else base
                new_name = cbase if cbase in var_names_set else base
            elif new_k > 0:
                mo = re.match(r'aux_(\w+)_(?:lead|lag)', base)
                cbase = mo.group(1) if mo else base
                prf = 'aux_' if base.lower().startswith('aux_') else ''
                new_name = f'{prf}{cbase}_p{new_k}'
            else:
                mo = re.match(r'aux_(\w+)_(?:lead|lag)', base)
                cbase = mo.group(1) if mo else base
                prf = 'aux_' if base.lower().startswith('aux_') else ''
                new_name = f'{prf}{cbase}_m{abs(new_k)}'

            if new_name not in parser_symbols:
                parser_symbols[new_name] = sympy.Symbol(new_name)
            subs_dict[atom] = parser_symbols[new_name]

    return expr.xreplace(subs_dict)

class DynareParser:

    def __init__(self, mod_file_path):
        self.mod_file_path = mod_file_path
        self.param_names = []
        self.var_names = []
        self.var_names_set = set()
        self.shock_names = []
        self.equations_str = []
        self.symbols = {}
        self.sympy_equations_original = []
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        self.static_subs = {}
        self.equations_after_static_elim = []
        self.equations_after_static_sub = []
        self.aux_lead_vars = {}
        self.aux_lag_vars = {}
        self.aux_var_definitions = []
        self.equations_after_aux_sub = []
        self.final_dynamic_var_names = []
        self.state_vars_ordered = []
        self.state_var_map = {}
        self.final_equations_for_jacobian = []
        self._parse_mod_file()
        self.var_names_set = set(self.var_names)
        self._create_initial_sympy_symbols()
        self._parse_equations_to_sympy()

    def _parse_mod_file(self):
        if not os.path.isfile(self.mod_file_path):
            raise FileNotFoundError(f"Mod file not found: {self.mod_file_path}")
        with open(self.mod_file_path, 'r') as f:
            content = f.read()
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        vpat = re.compile(r'var\s+(.*?);', re.I | re.DOTALL)
        vxpat = re.compile(r'varexo\s+(.*?);', re.I | re.DOTALL)
        ppat = re.compile(r'parameters\s+(.*?);', re.I | re.DOTALL)
        mpat = re.compile(r'model\s*;\s*(.*?)\s*end\s*;', re.I | re.DOTALL)
        vm = vpat.search(content)
        if vm:
            self.var_names = [x.strip() for x in vm.group(1).split() if x.strip()]
        vxm = vxpat.search(content)
        if vxm:
            self.shock_names = [x.strip() for x in vxm.group(1).split() if x.strip()]
        pm = ppat.search(content)
        if pm:
            self.param_names = [x.strip().rstrip(',') for x in pm.group(1).split() if x.strip()]
        mm = mpat.search(content)
        if not mm:
            raise ValueError("model block not found.")
        self.equations_str = [e.strip() for e in mm.group(1).split(';') if e.strip()]

    def _create_initial_sympy_symbols(self):
        alln = self.var_names + self.param_names + self.shock_names
        for nm in alln:
            self.symbols[nm] = sympy.Symbol(nm)

    def _parse_equations_to_sympy(self):
        for eqstr in self.equations_str:
            if not eqstr:
                continue
            eqproc = self._replace_dynare_timing(eqstr)
            if '=' in eqproc:
                lhs_s, rhs_s = eqproc.split('=', 1)
                lhs = sympy.parse_expr(lhs_s, local_dict=self.symbols, evaluate=False)
                rhs = sympy.parse_expr(rhs_s, local_dict=self.symbols, evaluate=False)
                self.sympy_equations_original.append(sympy.Eq(lhs - rhs, 0))
            else:
                expr = sympy.parse_expr(eqproc, local_dict=self.symbols, evaluate=False)
                self.sympy_equations_original.append(sympy.Eq(expr, 0))

    def _replace_dynare_timing(self, eqstr):
        pat = re.compile(r'\b([A-Za-z_]\w*)\s*\(([+\-]?)(\d+)\)')
        out = eqstr
        replacements = []
        needed = set()
        for m in pat.finditer(out):
            s, e = m.span()
            vn, sign, num = m.groups()
            if vn in self.var_names_set:
                k = int(num)
                rep = f"{vn}_p{k}" if sign == '+' else f"{vn}_m{k}" if sign == '-' else vn
                replacements.append((s, e, rep))
                needed.add(rep)
        for (s, e, rp) in sorted(replacements, key=lambda x: x[0], reverse=True):
            out = out[:s] + rp + out[e:]
        for nn in needed:
            if nn not in self.symbols:
                self.symbols[nn] = sympy.Symbol(nn)
        return out

    def _analyze_variable_timing(self, max_k=10):
        vs = {self.symbols[v] for v in self.var_names}
        for eq in self.sympy_equations_original:
            free_at = eq.lhs.free_symbols
            for var in vs:
                if var in free_at:
                    self.var_timing_info[var]['appears_current'] = True
                for k in range(1, max_k + 1):
                    pk = f"{var.name}_p{k}"
                    mk = f"{var.name}_m{k}"
                    if pk in self.symbols and self.symbols[pk] in free_at:
                        self.var_timing_info[var]['max_lead'] = max(self.var_timing_info[var]['max_lead'], k)
                    if mk in self.symbols and self.symbols[mk] in free_at:
                        self.var_timing_info[var]['min_lag'] = min(self.var_timing_info[var]['min_lag'], -k)

    def _identify_and_eliminate_static_vars(self):
        all_varsym = {self.symbols[v] for v in self.var_names}
        dynamic_syms = set()
        for eq in self.sympy_equations_original:
            at = eq.lhs.free_symbols
            for vsym in all_varsym:
                if vsym in at:
                    has_leadlag = False
                    for k in range(1, 15):
                        pk = f"{vsym.name}_p{k}"
                        mk = f"{vsym.name}_m{k}"
                        if ((pk in self.symbols and self.symbols[pk] in at) or
                            (mk in self.symbols and self.symbols[mk] in at)):
                            has_leadlag = True
                            break
                    if has_leadlag:
                        dynamic_syms.add(vsym)
        candidate_static = all_varsym - dynamic_syms
        self.equations_after_static_elim = list(self.sympy_equations_original)
        self.static_subs = {}
        solved = set()
        changed = True
        while changed:
            changed = False
            next_eqs = []
            used_eqs = set()
            for eq in self.equations_after_static_elim:
                freea = eq.lhs.free_symbols
                candhere = [c for c in candidate_static if c in freea]
                exclusively_current = []
                for c in candhere:
                    has_ll = False
                    for k in range(1, 15):
                        if (f"{c.name}_p{k}" in self.symbols and self.symbols[f"{c.name}_p{k}"] in freea) or \
                           (f"{c.name}_m{k}" in self.symbols and self.symbols[f"{c.name}_m{k}"] in freea):
                            has_ll = True
                            break
                    if not has_ll:
                        exclusively_current.append(c)
                if exclusively_current:
                    unsolved = [u for u in exclusively_current if u not in solved]
                    if len(unsolved) == 1:
                        target = unsolved[0]
                        try:
                            sol = sympy.solve(eq.lhs, target)
                            if len(sol) == 1:
                                new_sol = sol[0].subs(self.static_subs)
                                if target not in new_sol.free_symbols:
                                    self.static_subs[target] = new_sol
                                    solved.add(target)
                                    used_eqs.add(eq)
                                    changed = True
                                else:
                                    next_eqs.append(eq)
                            else:
                                next_eqs.append(eq)
                        except Exception:
                            next_eqs.append(eq)
                    else:
                        next_eqs.append(eq)
                else:
                    next_eqs.append(eq)
            self.equations_after_static_elim = [e for e in next_eqs if e not in used_eqs]

    def _substitute_static_vars(self):
        if not self.static_subs:
            self.equations_after_static_sub = list(self.equations_after_static_elim)
            return
        subsdict = {}
        max_l = 10
        for sv, mapping_expr in self.static_subs.items():
            subsdict[sv] = mapping_expr
            for k in range(1, max_l + 1):
                leadk = f"{sv.name}_p{k}"
                if leadk in self.symbols:
                    subsdict[self.symbols[leadk]] = time_shift_expression(mapping_expr, k, self.symbols, self.var_names_set)
                lagk = f"{sv.name}_m{k}"
                if lagk in self.symbols:
                    subsdict[self.symbols[lagk]] = time_shift_expression(mapping_expr, -k, self.symbols, self.var_names_set)
        self.equations_after_static_sub = []
        for eq in self.equations_after_static_elim:
            lhs_sub = eq.lhs.xreplace(subsdict)
            lhs_simpl = sympy.simplify(lhs_sub)
            self.equations_after_static_sub.append(sympy.Eq(lhs_simpl, 0))

    def _handle_aux_vars(self, file_path=None):
        """
        Handles leads > +1 and lags < -1 by creating auxiliary variables
        and applying the correct substitutions. Adheres to Python style.
        """
        print("\n--- Stage 4: Handling Aux Vars (Readable & Corrected Sub Target) ---")
        self.aux_lead_vars = {} # name -> symbol
        self.aux_lag_vars = {}  # name -> symbol
        self.aux_var_definitions = [] # Equations defining aux_leads AND aux_lags
        current_equations = list(self.equations_after_static_sub) # Use result from Stage 3
        subs_long_leads_lags = {} # Combined substitutions

        # Get base dynamic variable names (excluding static ones)
        dynamic_base_var_names = [v for v in self.var_names if v not in [s.name for s in self.static_subs.keys()]]
        output_lines = ["Handling long leads/lags using 'aux_VAR_leadK'/'aux_VAR_lagK':"]

        # --- Identify and Define Aux Lead Vars (for leads > +1) ---
        leads_to_replace = collections.defaultdict(int)
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                match = re.match(r"(\w+)_p(\d+)", atom.name)
                if match:
                    base, k_str = match.groups()
                    k = int(k_str)
                    if k > 1 and base in dynamic_base_var_names:
                        leads_to_replace[base] = max(leads_to_replace[base], k)

        if leads_to_replace:
            output_lines.append("\nCreating auxiliary LEAD variables:")
            # Sort for consistent processing order
            for var_name in sorted(leads_to_replace.keys()):
                max_lead = leads_to_replace[var_name]
                # Need aux vars up to aux_lead(max_lead - 1) to represent E[var(+2)]...E[var(+max_lead)]
                for k in range(1, max_lead):
                    aux_name = f"aux_{var_name}_lead{k}"

                    # Ensure aux symbol exists and track it
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    self.aux_lead_vars[aux_name] = self.symbols[aux_name]

                    # Define the aux equation: aux_k(t) = E[PrevAux(t+1)] or E[Var(t+1)]
                    if k == 1:
                        # aux_lead1 = Var_p1
                        var_p1_name = f"{var_name}_p1"
                        if var_p1_name not in self.symbols:
                            self.symbols[var_p1_name] = sympy.Symbol(var_p1_name)
                        rhs_sym = self.symbols[var_p1_name]
                    else:
                        # aux_leadK = aux_lead(K-1)_p1
                        prev_aux_name = f"aux_{var_name}_lead{k-1}"
                        prev_aux_p1_name = f"{prev_aux_name}_p1"
                        if prev_aux_p1_name not in self.symbols:
                            self.symbols[prev_aux_p1_name] = sympy.Symbol(prev_aux_p1_name)
                        rhs_sym = self.symbols[prev_aux_p1_name]

                    new_eq = sympy.Eq(self.symbols[aux_name] - rhs_sym, 0)

                    # Add definition only if unique
                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)
                        line = f"- Added aux var '{aux_name}'. Definition: {new_eq.lhs} = 0"
                        print(line)
                        output_lines.append(line)

                    # --- CORRECTED Substitution Rule Generation ---
                    # Rule: Replace Var_p(k+1) symbol with Aux_Var_leadK_p1 (lead of aux var k)
                    orig_lead_key = f"{var_name}_p{k+1}"
                    # The substitution target represents E[Var(t+k+1)] which is E[Aux_Var_leadK(t+1)]
                    aux_lead_k_p1_name = f"{aux_name}_p1" # Aux var k at time t+1

                    if orig_lead_key in self.symbols:
                        # Ensure the target symbol (aux_..._p1) exists
                        if aux_lead_k_p1_name not in self.symbols:
                            self.symbols[aux_lead_k_p1_name] = sympy.Symbol(aux_lead_k_p1_name)
                        # Add the rule: Original Lead -> Lead of Aux
                        subs_long_leads_lags[self.symbols[orig_lead_key]] = self.symbols[aux_lead_k_p1_name]
                        output_lines.append(f"  - Sub rule created: {orig_lead_key} -> {aux_lead_k_p1_name}")
                    # --- End of Correction ---

        # --- Identify and Define Aux Lag Vars (for lags < -1) ---
        # (Readable formatting, logic was okay)
        lags_to_replace = collections.defaultdict(int)
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                match = re.match(r"(\w+)_m(\d+)", atom.name)
                if match:
                    base, k_str = match.groups()
                    k = int(k_str)
                    if k > 1 and base in dynamic_base_var_names:
                        lags_to_replace[base] = max(lags_to_replace[base], k)

        if lags_to_replace:
            output_lines.append("\nCreating auxiliary LAG variables:")
            for var_name in sorted(lags_to_replace.keys()):
                max_lag = lags_to_replace[var_name]
                # Need aux vars up to aux_lag(max_lag - 1)
                for k in range(1, max_lag):
                    aux_name = f"aux_{var_name}_lag{k}"

                    # Ensure aux symbol exists and track it
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    self.aux_lag_vars[aux_name] = self.symbols[aux_name]

                    # Define aux equation: aux_k(t) = PrevAux(t-1) or Var(t-1)
                    if k == 1:
                        # aux_lag1 = Var_m1
                        var_m1_name = f"{var_name}_m1"
                        if var_m1_name not in self.symbols:
                            self.symbols[var_m1_name] = sympy.Symbol(var_m1_name)
                        rhs_sym = self.symbols[var_m1_name]
                    else:
                        # aux_lagK = aux_lag(K-1)_m1
                        prev_aux_name = f"aux_{var_name}_lag{k-1}"
                        prev_aux_m1_name = f"{prev_aux_name}_m1"
                        if prev_aux_m1_name not in self.symbols:
                            self.symbols[prev_aux_m1_name] = sympy.Symbol(prev_aux_m1_name)
                        rhs_sym = self.symbols[prev_aux_m1_name]

                    new_eq = sympy.Eq(self.symbols[aux_name] - rhs_sym, 0)

                    # Add definition only if unique
                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)
                        line = f"- Added aux var '{aux_name}'. Definition: {new_eq.lhs}=0"
                        print(line)
                        output_lines.append(line)

                    # Substitution rule: Var_m(k+1) -> aux_lagK_m1
                    orig_lag_key = f"{var_name}_m{k+1}"
                    aux_lag_m1_key = f"{aux_name}_m1" # Need lag of aux state k

                    if orig_lag_key in self.symbols:
                        # Ensure the target symbol exists
                        if aux_lag_m1_key not in self.symbols:
                            self.symbols[aux_lag_m1_key] = sympy.Symbol(aux_lag_m1_key)
                        # Add the rule: Original Lag -> Lag of Aux
                        subs_long_leads_lags[self.symbols[orig_lag_key]] = self.symbols[aux_lag_m1_key]
                        output_lines.append(f"  - Sub rule created: {orig_lag_key} -> {aux_lag_m1_key}")

        # Apply substitutions
        self.equations_after_aux_sub = []
        output_lines.append("\nApplying long lead/lag substitutions:")
        for i, eq in enumerate(current_equations):
            subbed_lhs = eq.lhs.xreplace(subs_long_leads_lags)
            try:
                simplified_lhs = sympy.simplify(subbed_lhs)
            except Exception:
                simplified_lhs = subbed_lhs # Fallback if simplify fails
            subbed_eq = sympy.Eq(simplified_lhs, 0)
            self.equations_after_aux_sub.append(subbed_eq)
            line2 = f"  Eq {i+1} substituted: {sympy.sstr(subbed_eq.lhs, full_prec=False)} = 0"
            output_lines.append(line2)

        # Update final dynamic var list (base + ALL aux vars)
        self.final_dynamic_var_names = dynamic_base_var_names + list(self.aux_lead_vars.keys()) + list(self.aux_lag_vars.keys())

        line = f"\nSubst complete. Final dynamic vars (base+aux_lead+aux_lag): {len(self.final_dynamic_var_names)}"
        print(line)
        output_lines.append(line)
        print(f"  {self.final_dynamic_var_names}") # Print the list
        line = f"Eqs after aux sub (excluding defs): {len(self.equations_after_aux_sub)}"
        print(line)
        output_lines.append(line)
        line = f"Aux var defs (lead+lag): {len(self.aux_var_definitions)}"
        print(line)
        output_lines.append(line)

        if file_path:
            self._save_intermediate_file(file_path, output_lines, self.equations_after_aux_sub + self.aux_var_definitions, "Equations After Aux Handling")
        return self.equations_after_aux_sub, self.aux_var_definitions

    def _define_state_vector(self):
        eqall = self.equations_after_aux_sub + self.aux_var_definitions
        csyms = [self.symbols[n] for n in self.final_dynamic_var_names]
        pred_vars = []
        mixed_vars = []
        for s in csyms:
            sp1 = f'{s.name}_p1'
            has_lead = any(sp1 in eq.lhs.free_symbols for eq in eqall)
            if 'aux_' in s.name and 'lead' in s.name:
                mixed_vars.append(s)
            elif has_lead:
                mixed_vars.append(s)
            else:
                pred_vars.append(s)
        pred_orig = [v for v in pred_vars if not (v.name.startswith('aux_') and '_lag' in v.name)]
        pred_lags = [v for v in pred_vars if (v.name.startswith('aux_') and '_lag' in v.name)]
        mixed_orig = [v for v in mixed_vars if not (v.name.startswith('aux_') and '_lead' in v.name)]
        mixed_leads = [v for v in mixed_vars if v.name.startswith('aux_') and '_lead' in v.name]
        self.state_vars_ordered = pred_orig + pred_lags + mixed_orig + mixed_leads
        self.state_var_map = {v: i for i, v in enumerate(self.state_vars_ordered)}

    def _build_final_equations(self):
        self.final_equations_for_jacobian = list(self.equations_after_aux_sub) + list(self.aux_var_definitions)
        ns = len(self.state_vars_ordered)
        ne = len(self.final_equations_for_jacobian)
        # Typically ns==ne for a well-formed system

    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        """
        Calculates numerical A, B, C, D from final equations and states.
        Uses standard Python formatting. D = -d(eq)/d(eps).

        Args:
            param_dict_values (dict): Dictionary mapping parameter names to values.
            file_path (str, optional): Path to save numerical matrices (.pkl/.txt).

        Returns:
            tuple: (A, B, C, D, state_names, shock_names) or raises error.
        """
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing failed: Final equations/states not generated.")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)
        self.last_param_values = param_dict_values # Store for function generation example
        # Create substitution dictionary for parameters
        param_subs = {self.symbols[p]: v for p, v in param_dict_values.items() if p in self.symbols}

        # --- Create symbolic vectors (Readable Version) ---
        state_vec_t = sympy.Matrix(self.state_vars_ordered)
        state_vec_tp1_list = []
        state_vec_tm1_list = []

        # Build lists for t+1 and t-1 states, ensuring symbols exist
        for state_sym in self.state_vars_ordered:
            lead_name = f"{state_sym.name}_p1"
            lag_name = f"{state_sym.name}_m1"

            # Handle lead symbol (t+1)
            if lead_name not in self.symbols:
                self.symbols[lead_name] = sympy.Symbol(lead_name)
            state_vec_tp1_list.append(self.symbols[lead_name])

            # Handle lag symbol (t-1)
            if lag_name not in self.symbols:
                self.symbols[lag_name] = sympy.Symbol(lag_name)
            state_vec_tm1_list.append(self.symbols[lag_name])

        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)

        # Create shock vector
        shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
        if shock_syms_list:
             shock_vec = sympy.Matrix(shock_syms_list)
        else:
             shock_vec = None

        # Create equation vector (LHS of final equations, assuming Expression=0 form)
        eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])
        # --- End Create symbolic vectors ---

        print("Calculating Jacobians...")
        try:
            # --- Calculate Jacobians (Readable Version) ---
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)

            # Calculate D matrix Jacobian (D = -d(eq)/d(eps))
            if shock_vec and n_shocks > 0:
                D_sym = -eq_vec.jacobian(shock_vec)
            else:
                n_eqs = len(eq_vec) # Should equal n_state if build_final_eqs passed
                D_sym = sympy.zeros(n_eqs, n_shocks)
            # --- End Calculate Jacobians ---
        except Exception as e:
            print(f"Error during symbolic Jacobian calculation: {e}")
            # Optional: Add more debug info here if necessary
            raise

        print("Substituting parameter values...")
        try:
            # --- Numerical Substitution ---
            # Use evalf(subs=...) for numerical evaluation
            A_num = np.array(A_sym.evalf(subs=param_subs).tolist(), dtype=float)
            B_num = np.array(B_sym.evalf(subs=param_subs).tolist(), dtype=float)
            C_num = np.array(C_sym.evalf(subs=param_subs).tolist(), dtype=float)
            if n_shocks > 0:
                D_num = np.array(D_sym.evalf(subs=param_subs).tolist(), dtype=float)
            else:
                # Ensure D has correct shape even if empty
                D_num = np.zeros((n_state, 0), dtype=float)
            # --- End Numerical Substitution ---
        except Exception as e:
            print(f"Error during numerical substitution (evalf): {e}")
            # Optional: Print symbolic matrices that caused the error
            # print("A_sym:", A_sym) ... etc.
            raise

        # --- Final checks ---
        expected_d_shape = (n_state, n_shocks)
        if A_num.shape != (n_state, n_state) or B_num.shape != (n_state, n_state) or \
           C_num.shape != (n_state, n_state) or D_num.shape != expected_d_shape:
            print(f"ERROR: Final matrix dimension mismatch!")
            print(f" A:{A_num.shape}, B:{B_num.shape}, C:{C_num.shape}, D:{D_num.shape}")
            print(f" Expected A,B,C:({n_state},{n_state}), Expected D:{expected_d_shape}")
            raise ValueError("Matrix dimension mismatch.")
        print("Numerical matrices A, B, C, D calculated.")

        # --- Save (Optional) ---
        if file_path:
            # Make sure helper is defined or moved outside the class
            self._save_final_matrices(file_path, A_num, B_num, C_num, D_num)

        return A_num, B_num, C_num, D_num, [s.name for s in self.state_vars_ordered], self.shock_names

    # --- Helper Function (Defined within the class or move outside) ---
    def _generate_matrix_assignments_code_helper(self, matrix_sym, matrix_name):
        """
        Generates Python code lines for element-wise matrix assignments.
        Internal helper function. Uses standard Python formatting.
        """
        try:
            rows, cols = matrix_sym.shape
        except Exception as e:
            print(f"Error getting shape for matrix {matrix_name}. Matrix: {matrix_sym}")
            raise ValueError(f"Could not get shape for symbolic matrix {matrix_name}") from e

        indent = "    " # Standard 4 spaces
        # Initialize the matrix creation line correctly indented
        code_lines = [f"{indent}{matrix_name} = np.zeros(({rows}, {cols}), dtype=float)"]
        assignments = []

        # Iterate using actual dimensions from the symbolic matrix
        for r in range(rows):
            for c in range(cols):
                try:
                    element = matrix_sym[r, c]
                except IndexError:
                    # Defensive check
                    print(f"Internal Error: Index ({r},{c}) out of bounds for {matrix_name} shape {matrix_sym.shape}")
                    continue

                # Check for structural zero before converting to string
                if element != 0 and element is not sympy.S.Zero:
                    try:
                        # Get standard Python string representation
                        expr_str = sympy.sstr(element, full_prec=False)
                        # Add assignment line with standard indentation
                        assignments.append(f"{indent}{matrix_name}[{r}, {c}] = {expr_str}")
                    except Exception as str_e:
                        print(f"Warning: String conversion failed for {matrix_name}[{r},{c}]: {str_e}")
                        assignments.append(f"{indent}# Error generating code for {matrix_name}[{r},{c}] = {element}")

        # Add the assignment block only if there are non-zero elements found
        if assignments:
            code_lines.append(f"{indent}# Fill {matrix_name} non-zero elements")
            code_lines.extend(assignments)

        # Return the complete code block as a single string
        return "\n".join(code_lines)    

    def generate_matrix_function_file(self, filename="jacobian_matrices.py"):
        """
        Generates jacobian_matrices(theta) -> A, B, C, D with clean code.
        The generated function itself has no docstring or example usage block.
        Uses standard Python formatting.
        """
        function_name = "jacobian_matrices"
        print(f"\n--- Generating Python Function File (Readable): {filename} ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing must be run successfully first.")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)

        # --- 1. Recalculate Symbolic Jacobians (Readable) ---
        try:
            # (Same readable Jacobian calculation as in get_numerical_ABCD)
            state_vec_t = sympy.Matrix(self.state_vars_ordered)
            state_vec_tp1_list = []
            state_vec_tm1_list = []
            for state_sym in self.state_vars_ordered:
                lead_name = f"{state_sym.name}_p1"
                lag_name = f"{state_sym.name}_m1"
                if lead_name not in self.symbols: self.symbols[lead_name] = sympy.Symbol(lead_name)
                state_vec_tp1_list.append(self.symbols[lead_name])
                if lag_name not in self.symbols: self.symbols[lag_name] = sympy.Symbol(lag_name)
                state_vec_tm1_list.append(self.symbols[lag_name])
            state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
            state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)
            shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
            shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None
            eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])

            print("Calculating symbolic Jacobians...")
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            # Calculate D = -d(eq)/d(eps)
            if shock_vec and n_shocks > 0 :
                D_sym = -eq_vec.jacobian(shock_vec)
            else:
                n_eqs = len(eq_vec)
                D_sym = sympy.zeros(n_eqs, n_shocks)
            print("Jacobians calculated.")
        except Exception as e:
            print(f"Error calculating Jacobians: {e}")
            raise
        # --- End Recalculate Symbolic Jacobians ---

        # --- 2. Prepare Parameter Info (Readable) ---
        ordered_params_from_mod = self.param_names
        param_symbols_in_matrices = set().union(*(mat.free_symbols for mat in [A_sym, B_sym, C_sym, D_sym]))
        used_params_ordered = [p for p in ordered_params_from_mod if p in [s.name for s in param_symbols_in_matrices]]
        param_indices = {p: i for i, p in enumerate(ordered_params_from_mod)}
        # --- End Prepare Parameter Info ---

        # --- 3. Generate Python code strings using Internal Helper ---
        print("Generating Python code strings for matrices...")
        # Call the INTERNAL helper (defined above)
        code_A = self._generate_matrix_assignments_code_helper(A_sym, 'A')
        code_B = self._generate_matrix_assignments_code_helper(B_sym, 'B')
        code_C = self._generate_matrix_assignments_code_helper(C_sym, 'C')
        code_D = self._generate_matrix_assignments_code_helper(D_sym, 'D')
        print("Code generation complete.")
        # --- End Generate Python code strings ---

        # --- 4. Assemble File Content (Readable - No Semicolons/Docstring/Example) ---
        file_lines = []
        file_lines.append(f"# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'")
        file_lines.append(f"# Generated: {datetime.datetime.now().isoformat()}")
        file_lines.append("import numpy as np")
        file_lines.append("from math import * # Provides standard math functions")
        file_lines.append("")
        file_lines.append(f"def {function_name}(theta):")
        # --- NO DOCSTRING ---
        file_lines.append("    # Unpack parameters (using original order)")
        file_lines.append(f"    expected_len = {len(ordered_params_from_mod)}")
        # Corrected f-string formatting for ValueError
        file_lines.append(f"    if len(theta) != expected_len:")
        file_lines.append(f"        raise ValueError(f'Expected {{expected_len}} parameters, got {{len(theta)}}')")
        file_lines.append("    try:")
        # Generate unpacking lines correctly indented
        for p_name in used_params_ordered:
            idx = param_indices[p_name]
            file_lines.append(f"        {p_name} = theta[{idx}]")
        file_lines.append("    except IndexError:")
        # Corrected f-string formatting for IndexError
        file_lines.append(f"        raise IndexError('Parameter vector theta has incorrect length.')")
        file_lines.append("")
        file_lines.append("    # Initialize and fill matrices")
        # Add generated matrix code blocks with correct indentation
        file_lines.append(code_A)
        file_lines.append("")
        file_lines.append(code_B)
        file_lines.append("")
        file_lines.append(code_C)
        file_lines.append("")
        file_lines.append(code_D)
        file_lines.append("")
        file_lines.append("    # --- Return results ---")
        file_lines.append(f"    state_names = {repr([s.name for s in self.state_vars_ordered])}")
        file_lines.append(f"    shock_names = {repr(self.shock_names)}")
        file_lines.append("")
        file_lines.append("    return A, B, C, D, state_names, shock_names")
        # --- NO if __name__ == '__main__': block ---

        # Join all lines
        final_file_content = "\n".join(file_lines)
        # --- End Assemble File Content ---

        # --- 5. Write File ---
        try:
            dir_name = os.path.dirname(filename)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(final_file_content)
            print(f"Successfully generated function file: {filename}")
        except Exception as e:
            print(f"Error writing function file {filename}: {e}")

    # ===========================================
    # process_model (Corrected Arguments and Internal Calls)
    # ===========================================
    def process_model(self, param_dict_values_or_list, output_dir_intermediate=None,
                    output_dir_final=None, generate_function=True):
        """
        Runs the full parsing and matrix generation pipeline. Correctly returns results.
        Uses standard Python formatting. Internal calls don't handle file paths.

        Args:
            param_dict_values_or_list: Dict or ordered list/array of parameter values.
            output_dir_intermediate: Directory to save intermediate text files.
            output_dir_final: Directory to save final .pkl and .py files.
            generate_function: Boolean flag to generate the python function file.

        Returns:
            tuple: (A, B, C, D, state_names, shock_names) on success, None on failure.
        """
        # --- Parameter Input Handling (Readable) ---
        param_dict_values = {}
        if isinstance(param_dict_values_or_list, (list, tuple, np.ndarray)):
            if len(param_dict_values_or_list) != len(self.param_names):
                raise ValueError(f"Input parameter list/array length ({len(param_dict_values_or_list)}) "
                                f"!= declared parameters ({len(self.param_names)})")
            param_dict_values = {name: val for name, val in zip(self.param_names, param_dict_values_or_list)}
            # print("Received parameter list/array, converted to dict.") # Optional print
        elif isinstance(param_dict_values_or_list, dict):
            param_dict_values = param_dict_values_or_list
            missing_keys = set(self.param_names) - set(param_dict_values.keys())
            if missing_keys:
                print(f"Warning: Input parameter dict missing keys: {missing_keys}")
        else:
            raise TypeError("param_dict_values_or_list must be a dict, list, tuple, or numpy array.")
        # --- End Parameter Input Handling ---

        # --- Define file paths (Readable) ---
        base_name = os.path.splitext(os.path.basename(self.mod_file_path))[0]
        fpaths_inter = {i: None for i in range(6)}
        final_matrices_pkl = None
        function_py = None

        if output_dir_intermediate:
            os.makedirs(output_dir_intermediate, exist_ok=True)
            inter_names = ["timing", "static_elim", "static_sub", "aux_handling", "state_def", "final_eqs"]
            # Create the dictionary for file paths *before* calling the stages
            fpaths_inter = {i: os.path.join(output_dir_intermediate, f"{i+1}_{base_name}_{name}.txt")
                            for i, name in enumerate(inter_names)}

        if output_dir_final:
            os.makedirs(output_dir_final, exist_ok=True)
            final_matrices_pkl = os.path.join(output_dir_final, f"{base_name}_matrices.pkl")
            if generate_function:
                function_py = os.path.join(output_dir_final, f"{base_name}_jacobian_matrices.py")
        # --- End Define file paths ---

        # --- Run pipeline steps ---
        try:
            print("\n--- Starting Model Processing Pipeline ---")
            # Call internal methods WITHOUT file_path argument
            # The file saving will happen *after* the stage completes
            self._analyze_variable_timing()
            self._save_intermediate_file(fpaths_inter[0], ["Stage 1 Output..."], self.sympy_equations_original, "Original Equations After Timing Analysis") # Example save

            self._identify_and_eliminate_static_vars()
            self._save_intermediate_file(fpaths_inter[1], ["Stage 2 Output..."], self.equations_after_static_elim, "Equations After Static Elimination") # Example save

            self._substitute_static_vars()
            self._save_intermediate_file(fpaths_inter[2], ["Stage 3 Output..."], self.equations_after_static_sub, "Equations After Static Substitution") # Example save

            self._handle_aux_vars()
            self._save_intermediate_file(fpaths_inter[3], ["Stage 4 Output..."], self.equations_after_aux_sub + self.aux_var_definitions, "Equations After Aux Handling") # Example save

            self._define_state_vector()
            # No equations to save specifically after state definition, maybe save state list
            state_lines = ["Stage 5 Output:", f"State Vector: {[s.name for s in self.state_vars_ordered]}"]
            self._save_intermediate_file(fpaths_inter[4], state_lines)

            self._build_final_equations()
            self._save_intermediate_file(fpaths_inter[5], ["Stage 6 Output..."], self.final_equations_for_jacobian, "Final Equation System")

            #Order final equations and classify final variables as forward-looking, mixed, or backward-looking (exogenous states), 
            #Order equations with exogenous states last (to get block diagonal structure)

            # Save final equations before calculating Jacobians for debugging
            if output_dir_intermediate:
                eq_out_path = os.path.join(output_dir_intermediate, f"DEBUG_{base_name}_final_equations_used.txt")
                self.save_final_equations_to_txt(filename=eq_out_path)

            # Get numerical matrices using the processed data
            A, B, C, D, state_names, shock_names = self.get_numerical_ABCD(
                param_dict_values, # Pass the DICT here
                file_path=final_matrices_pkl # Save numerical matrices
            )

            # Generate the function file if requested
            if generate_function and function_py:
                # This method uses self.last_param_values set by get_numerical_ABCD
                self.generate_matrix_function_file(filename=function_py)

            print("\n--- Model Processing Successful ---")
            # Return the calculated numerical matrices and names
            return A, B, C, D, state_names, shock_names

        except Exception as e:
            print(f"\n--- ERROR during model processing: {type(e).__name__}: {e} ---")
            import traceback
            traceback.print_exc()
            return None # Indicate failure
        # --- End Run pipeline steps ---

    # --- Helper methods (_save_intermediate_file, _save_final_matrices, save_final_equations_to_txt) ---
    # (Ensure these are defined within the class with standard Python formatting)
    def _save_intermediate_file(self, file_path, lines, equations=None, equations_title="Equations"):
        if not file_path: return
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(f"--- {os.path.basename(file_path)} ---\n\n")
                f.write("\n".join(lines))
                if equations:
                    f.write(f"\n\n{equations_title} ({len(equations)}):\n")
                    for i, eq in enumerate(equations):
                        f.write(f"  {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
            print(f"Intermediate results saved to {file_path}")
        except Exception as e:
            print(f"Warning: Could not save intermediate file {file_path}. Error: {e}")

    def _save_final_matrices(self, file_path, A, B, C, D):
        matrix_data = {'A': A, 'B': B, 'C': C, 'D': D,
                    'state_names': [s.name for s in self.state_vars_ordered],
                    'shock_names': self.shock_names,
                    'param_names': self.param_names}
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                pickle.dump(matrix_data, f)
            print(f"Matrices saved to {file_path}")
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write("State Names:\n" + str(matrix_data['state_names']) + "\n\n")
                f.write("Shock Names:\n" + str(matrix_data['shock_names']) + "\n\n")
                np.set_printoptions(linewidth=200, precision=4, suppress=True)
                f.write("A Matrix:\n" + np.array2string(A) + "\n\n")
                f.write("B Matrix:\n" + np.array2string(B) + "\n\n")
                f.write("C Matrix:\n" + np.array2string(C) + "\n\n")
                f.write("D Matrix:\n" + np.array2string(D) + "\n\n")
            print(f"Human-readable matrices saved to {txt_path}")
        except Exception as e:
            print(f"Warning: Could not save matrices file {file_path}. Error: {e}")

    def save_final_equations_to_txt(self, filename="final_equations.txt"):
        """Saves the final list of equations to a text file for inspection."""
        print(f"\n--- Saving Final Equations to: {filename} ---")
        if not hasattr(self, 'final_equations_for_jacobian') or not self.final_equations_for_jacobian:
            print("Warning: Final equations have not been generated yet. File not saved.")
            return
        try:
            dir_name = os.path.dirname(filename)
            if dir_name: 
                os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"# Final System Equations ({len(self.final_equations_for_jacobian)} equations)\n")
                if hasattr(self, 'state_vars_ordered') and self.state_vars_ordered:
                    f.write(f"# State Variables Order: {[s.name for s in self.state_vars_ordered]}\n\n")
                else: f.write("# State variable order not determined yet.\n\n")
                for i, eq in enumerate(self.final_equations_for_jacobian):
                    f.write(f"Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
            print(f"Successfully saved final equations to {filename}")
        except Exception as e: print(f"Error writing final equations file {filename}: {e}")




# ===========================================
# Example Usage Script (Corrected process_model call)
# ===========================================
if __name__ == "__main__":
    
    import sys
    # --- Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    os.chdir(script_dir)
    mod_file = "qpm_model.dyn"
    output_dir_inter = "model_files_intermediate_final"
    output_dir_final = "model_files_numerical_final"
    os.makedirs(output_dir_inter, exist_ok=True)
    os.makedirs(output_dir_final, exist_ok=True)


    # --- Define parameters DICT ---
    parameter_values_dict = { 'b1': 0.7, 
                            'b4': 0.7, 
                            'a1': 0.5, 
                            'a2': 0.1, 
                            'g1': 0.7, 
                            'g2': 0.3, 
                            'g3': 0.25, 
                            'rho_DLA_CPI': 0.75, 
                            'rho_L_GDP_GAP': 0.75, 
                            'rho_rs': 0.75, 
                            'rho_rs2': 0.01 }

    # --- Instantiate parser ---
    try: 
        parser = DynareParser(mod_file)
    except Exception as e: 
        print(f"Error initializing parser: {e}"); 
        sys.exit(1)

    # --- Create theta list IN ORDER ---
    try: 
        parameter_theta = [parameter_values_dict[pname] for pname in parser.param_names]; 
        print(f"\nTheta created (order: {parser.param_names})")
    except KeyError as e: 
        print(f"\nERROR: Param '{e}' missing from dict."); 
        sys.exit(1)
    except Exception as e: 
        print(f"\nERROR creating theta list: {e}"); 
        sys.exit(1)

    # --- Process the model ---
    # Call process_model with the ordered list as the FIRST argument (positional)
    result = parser.process_model(parameter_theta, # <<< Use positional argument
                                output_dir_intermediate=output_dir_inter,
                                output_dir_final=output_dir_final,
                                generate_function=True)

    # --- Check Results & Test Generated Function ---
    if result:
        A_direct, B_direct, C_direct, D_direct, state_names_direct, shock_names_direct = result
        print("\n--- Results from Direct Calculation ---"); 
        print("States:", state_names_direct); 
        print(f"A:{A_direct.shape} B:{B_direct.shape} C:{C_direct.shape} D:{D_direct.shape}")
        function_file = os.path.join(output_dir_final, f"{os.path.splitext(mod_file)[0]}_jacobian_matrices.py")
        if os.path.exists(function_file):
            print("\n--- Testing Generated Function ---"); # Test code...
            # ... (Keep importlib testing code as before - it was correct) ...
            abs_function_file = os.path.abspath(function_file); 
            module_name = os.path.splitext(os.path.basename(function_file))[0]
            try:
                spec = importlib.util.spec_from_file_location(module_name, abs_function_file);
                if spec is None: print(f"Error: Could not load spec for {module_name}")
                else:
                    mod_matrices = importlib.util.module_from_spec(spec); 
                    sys.modules[module_name] = mod_matrices; 
                    spec.loader.exec_module(mod_matrices)
                    A_f, B_f, C_f, D_f, states_f, shocks_f = mod_matrices.jacobian_matrices(parameter_theta) # Call correct function name
                    print("Function call successful. Comparing matrices...")
                    try: # Assertions
                        assert np.allclose(A_direct, A_f, atol=1e-8, equal_nan=True), "A mismatch"; assert np.allclose(B_direct, B_f, atol=1e-8, equal_nan=True), "B mismatch"
                        assert np.allclose(C_direct, C_f, atol=1e-8, equal_nan=True), "C mismatch"; assert np.allclose(D_direct, D_f, atol=1e-8, equal_nan=True), "D mismatch"
                        assert state_names_direct == states_f, "State mismatch"; 
                        assert shock_names_direct == shocks_f, "Shock mismatch"
                        print("Generated function tested successfully.")
                    except AssertionError as ae: print(f"!!! Assertion Error: {ae} !!!")
            except Exception as test_e: 
                print(f"Error testing generated func: {test_e}"); 
                import traceback; traceback.print_exc()
        else: 
            print(f"\nGenerated file not found: {function_file}. Cannot test.")
    else: 
        print("\nModel processing failed.")