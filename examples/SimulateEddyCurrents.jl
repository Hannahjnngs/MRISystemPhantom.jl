using KomaMRI
using MRISystemPhantom, Random

seq = read_seq(raw"/Users/hannah.jennings/Documents/Sequences/DWI_Summer_2026.seq");

cfg     = PhantomConfig(field = :T15, voxel_size_mm = 1.0)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

#rng = Random.Xoshiro(123);

outdir = "/Users/hannah.jennings/Documents/Code/TestFolder/Data/eddy_sweep"
mkpath(outdir)

# per-harmonic settings: edit these
tau_per_row   = fill(100e-3, 9)                 # decay constant for each harmonic
coeff_per_row = [[0.02, 0.02, 0.02] for _ in 1:9]  # its 3 coefficients [col1, col2, col3]

function run_case(obj, seq, scanner, r, tau_per_row, coeff_per_row, outdir)
    fname = joinpath(outdir, r == 0 ? "baseline.bson" : "harmonic_$(r).bson")
    isfile(fname) && (println("skipping $fname"); return)

    tau    = zeros(9)
    coeffs = zeros(9, 3)
    if r > 0
        tau[r]       = tau_per_row[r]
        coeffs[r, :] = coeff_per_row[r]
    end

    params = Dict{String, Any}(
        "return_type" => "raw",
        "sim_method"  => BlochEddy(),
        "tau"         => tau,
        "coeffs"      => coeffs,
    )

    t = @elapsed raw = simulate(obj, seq, scanner; sim_params=params)

    tmp = fname * ".tmp"
    bson(tmp, Dict(:raw => raw, :tau => tau, :coeffs => coeffs))
    mv(tmp, fname; force=true)
    println("done harmonic $r in $(round(t; digits=1)) s")
end

for r in 0:9                      # 0 = baseline with everything zero
    try
        run_case(obj, seq, scanner, r, tau_per_row, coeff_per_row, outdir)
    catch e
        @error "case failed" r exception=(e, catch_backtrace())
    end
    GC.gc()
end