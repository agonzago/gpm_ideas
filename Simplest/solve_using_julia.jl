using Dynare
using DataFrames
using CSV

# Solve the Dynare model
solved_model = dynare("qpm_simpl1.mod", dynare_version="4.6") # Or your dynare version

# Extract the IRFs
irfs = solved_model.oo_.irfs

# Convert the IRFs to a DataFrame
irf_df = DataFrame()
for (var, irf_array) in irfs
    irf_df[string(var)] = irf_array[:]  # Flatten the array
end

# Get the shock names - to make sure you are getting the correct IRFs
shocks = solved_model.M_.exo_names
println(shocks)

# Add a "Period" column
irf_df[!, "Period"] = 0:(size(irf_df, 1) - 1)

# Reorder columns with "Period" as first column
col_order = ["Period", names(irf_df)...]
irf_df = irf_df[:, col_order]

# Save the DataFrame to a CSV file
CSV.write("irfs_dynare.csv", irf_df)

println("IRFs saved to irfs_dynare.csv")