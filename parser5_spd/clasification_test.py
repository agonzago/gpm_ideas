import sympy

class YourModelSolverClass: # Replace with your actual class name
    def __init__(self):
        # --- Dummy attributes for demonstration ---
        # You would replace these with your actual model setup
        self.final_dynamic_var_names = [
            'L_GDP_GAP', 'DLA_CPI', 'RS', 'RR_GAP',
            'RES_L_GDP_GAP', 'RES_DLA_CPI', 'RES_RS',
            'aux_DLA_CPI_lead1', 'aux_DLA_CPI_lead2', 'aux_RES_RS_lag1'
        ]
        self.symbols = sympy.symbols(
            'L_GDP_GAP L_GDP_GAP_m1 L_GDP_GAP_p1 '
            'DLA_CPI DLA_CPI_m1 DLA_CPI_p1 '
            'RS RS_m1 ' # RS_p1 doesn't appear in the example equations
            'RR_GAP RR_GAP_p1 ' # RR_GAP_m1 doesn't appear
            'RES_L_GDP_GAP RES_L_GDP_GAP_m1 '
            'RES_DLA_CPI RES_DLA_CPI_m1 '
            'RES_RS RES_RS_m1 RES_RS_p1 aux_RES_RS_lag1_m1 ' # Added missing symbol for Eq 7
            'aux_DLA_CPI_lead1 aux_DLA_CPI_lead1_p1 '
            'aux_DLA_CPI_lead2 aux_DLA_CPI_lead2_p1 ' # Added missing symbol for Eq 3
            'aux_RES_RS_lag1 ' # aux_RES_RS_lag1_m1 already defined
            'SHK_L_GDP_GAP SHK_DLA_CPI SHK_RS '
            'b1 b4 a1 a2 g1 g2 g3 rho_L_GDP_GAP rho_DLA_CPI rho_rs rho_rs2' # Parameters
        )
        # Ensure self.symbols is a dictionary mapping names to symbols
        if not isinstance(self.symbols, dict):
             self.symbols = {s.name: s for s in self.symbols}

        # --- Define equations (ensure they are Eq objects or expressions = 0) ---
        # Using the names from self.symbols
        s = self.symbols
        eq1 = sympy.Eq(s['L_GDP_GAP'] - s['L_GDP_GAP_m1']*s['b1'] + s['L_GDP_GAP_p1']*(s['b1'] - 1) - s['RES_L_GDP_GAP'] + s['RR_GAP_p1']*s['b4'], 0)
        eq2 = sympy.Eq(s['DLA_CPI'] - s['DLA_CPI_m1']*s['a1'] + s['DLA_CPI_p1']*(s['a1'] - 1) - s['L_GDP_GAP']*s['a2'] - s['RES_DLA_CPI'], 0)
        eq3 = sympy.Eq(-s['RES_RS'] + s['RS'] - s['RS_m1']*s['g1'] + (s['g1'] - 1)*(s['DLA_CPI_p1'] + s['L_GDP_GAP']*s['g3'] + s['aux_DLA_CPI_lead2_p1']*s['g2']), 0)
        eq4 = sympy.Eq(s['DLA_CPI_p1'] + s['RR_GAP'] - s['RS'], 0) # Defines RR_GAP(t)
        eq5 = sympy.Eq(s['RES_L_GDP_GAP'] - s['RES_L_GDP_GAP_m1']*s['rho_L_GDP_GAP'] - s['SHK_L_GDP_GAP'], 0)
        eq6 = sympy.Eq(s['RES_DLA_CPI'] - s['RES_DLA_CPI_m1']*s['rho_DLA_CPI'] - s['SHK_DLA_CPI'], 0)
        eq7 = sympy.Eq(s['RES_RS'] - s['RES_RS_m1']*s['rho_rs'] - s['SHK_RS'] - s['aux_RES_RS_lag1_m1']*s['rho_rs2'], 0)
        eq8 = sympy.Eq(-s['DLA_CPI_p1'] + s['aux_DLA_CPI_lead1'], 0)
        eq9 = sympy.Eq(-s['aux_DLA_CPI_lead1_p1'] + s['aux_DLA_CPI_lead2'], 0)
        eq10 = sympy.Eq(-s['RES_RS_m1'] + s['aux_RES_RS_lag1'], 0)

        self.final_equations_for_jacobian = [
            eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10
        ]

        # --- Define aux/shock variables (if known) ---
        self.aux_lead_vars = {'aux_DLA_CPI_lead1', 'aux_DLA_CPI_lead2'}
        self.aux_lag_vars = {'aux_RES_RS_lag1'}
        self.shock_vars = {'SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS'} # Good practice to identify shocks
        # Parameters are implicitly handled as they don't end in _p1 or _m1
        # --- End of Dummy Attributes ---

        self.state_vars_ordered = []
        self.state_var_map = {}


    def _get_equation_symbols(self, eq):
        """Helper to get free symbols from both sides of an Equality."""
        if isinstance(eq, sympy.Equality):
            return eq.lhs.free_symbols.union(eq.rhs.free_symbols)
        elif isinstance(eq, sympy.Expr):
            return eq.free_symbols
        else:
            print(f"Warning: Equation is not a SymPy Equality or Expr: {type(eq)}. Cannot extract symbols.")
            return set()

    def _classify_and_order_variables_corrected(self):
        """
        Classifies dynamic variables based on time indices of symbols
        within their defining equation. Orders as [Backward, Mixed, Forward].
        Assumes k-th equation defines k-th variable in final_dynamic_var_names.
        """
        print("\n--- Stage 5: Classifying and Ordering (Corrected Logic) ---")
        if not self.final_dynamic_var_names:
            raise ValueError("Final dynamic variable list is empty.")
        if not self.final_equations_for_jacobian:
            raise ValueError("Final equations not defined.")
        if len(self.final_dynamic_var_names) != len(self.final_equations_for_jacobian):
            raise ValueError(f"Mismatch between number of variables ({len(self.final_dynamic_var_names)}) "
                             f"and equations ({len(self.final_equations_for_jacobian)}).")

        backward_vars, mixed_vars, forward_vars = [], [], []
        classified_vars = set() # Keep track of vars already classified (e.g., aux)

        # --- Step 1: Pre-classify Aux and Shocks (Optional but recommended) ---
        print("Pre-classifying known Aux/Shock variables...")
        final_var_symbols_map = {name: self.symbols[name] for name in self.final_dynamic_var_names if name in self.symbols}

        for var_name, var_sym in final_var_symbols_map.items():
             # Check aux classifications first
            if hasattr(self, 'aux_lag_vars') and var_name in self.aux_lag_vars:
                print(f"  {var_name} -> BACKWARD (Aux Lag)")
                backward_vars.append(var_sym)
                classified_vars.add(var_name)
            elif hasattr(self, 'aux_lead_vars') and var_name in self.aux_lead_vars:
                print(f"  {var_name} -> FORWARD (Aux Lead)")
                forward_vars.append(var_sym)
                classified_vars.add(var_name)
            # Can add shock classification here if needed (shocks usually aren't states)
            # elif hasattr(self, 'shock_vars') and var_name in self.shock_vars:
            #     print(f"  {var_name} -> EXOGENOUS (Shock)")
            #     classified_vars.add(var_name) # Exclude shocks from state vector

        # --- Step 2: Classify remaining variables based on defining equation ---
        print("Classifying remaining variables based on their defining equation...")
        for i, var_name in enumerate(self.final_dynamic_var_names):
            if var_name in classified_vars:
                continue # Already handled (e.g., was an aux var)

            if var_name not in self.symbols:
                 print(f"Warning: Variable '{var_name}' not found in symbols dictionary. Skipping.")
                 continue

            var_sym = self.symbols[var_name]
            equation = self.final_equations_for_jacobian[i]
            equation_symbols = self._get_equation_symbols(equation)

            has_lag = False
            has_lead = False

            for sym in equation_symbols:
                if not isinstance(sym, sympy.Symbol): continue # Skip non-symbols if any

                # Check name for lag/lead indicators - Adjust if your convention differs
                if sym.name.endswith('_m1'):
                    has_lag = True
                elif sym.name.endswith('_p1'):
                    has_lead = True

                # Optimization: if both found, no need to check further for this equation
                if has_lag and has_lead:
                    break

            # --- Determine classification ---
            classification = "Unknown"
            if has_lag and has_lead:
                classification = "MIXED"
                mixed_vars.append(var_sym)
            elif has_lag: # Only lag found in defining equation
                classification = "BACKWARD"
                backward_vars.append(var_sym)
            elif has_lead: # Only lead found in defining equation
                classification = "FORWARD"
                forward_vars.append(var_sym)
            else: # Neither lag nor lead found in defining equation (Static or depends only on t)
                  # Grouping with backward is common for ordering state vectors
                classification = "BACKWARD (Static)"
                backward_vars.append(var_sym)

            print(f"  {var_name} (Eq {i+1}) -> {classification} (Eq has lag: {has_lag}, Eq has lead: {has_lead})")
            classified_vars.add(var_name) # Mark as classified

        # --- Step 3 & 4: Ordering, Checks, Map ---
        backward_vars.sort(key=lambda s: s.name)
        mixed_vars.sort(key=lambda s: s.name)
        forward_vars.sort(key=lambda s: s.name)

        self.state_vars_ordered = backward_vars + mixed_vars + forward_vars
        self.state_var_map = {s: i for i, s in enumerate(self.state_vars_ordered)}

        # --- Final Checks ---
        # Ensure all originally intended variables were classified
        expected_vars_set = set(self.final_dynamic_var_names)
        # If shocks were explicitly excluded, remove them from the expected set for the check
        if hasattr(self, 'shock_vars'):
             expected_vars_set = expected_vars_set - self.shock_vars # Adjust if shocks *are* states

        final_classified_names = {s.name for s in self.state_vars_ordered}

        if final_classified_names != expected_vars_set:
            print("\nError: Mismatch between expected dynamic variables and classified state variables!")
            print(f"  Expected ({len(expected_vars_set)}): {sorted(list(expected_vars_set))}")
            print(f"  Classified ({len(final_classified_names)}): {sorted(list(final_classified_names))}")
            missing = expected_vars_set - final_classified_names
            extra = final_classified_names - expected_vars_set
            if missing: print(f"  Missing: {missing}")
            if extra: print(f"  Extra: {extra}")
            # Decide whether to raise an error or just warn
            # raise RuntimeError("State variable classification mismatch.")
            print("Warning: State variable classification mismatch detected.")


        print("\nFinal variable classification and ordering complete.")
        print(f"  Order: Backward ({len(backward_vars)}), Mixed ({len(mixed_vars)}), Forward ({len(forward_vars)})")
        print(f"  Total State Variables: {len(self.state_vars_ordered)}")
        print(f"  Final Ordered State Vector Names: {[s.name for s in self.state_vars_ordered]}")


# --- Example Usage ---
solver_instance = YourModelSolverClass()
solver_instance._classify_and_order_variables_corrected()

# You can then access the results:
# print("\nOrdered Symbols:", solver_instance.state_vars_ordered)
# print("Variable Map:", solver_instance.state_var_map)