using Optim
using KomaMRI, MRIReco, MAT
using MRISystemPhantom, Random
using Plots
import Statistics

seq = read_seq(raw"/Users/hannah.jennings/Documents/Sequences/DWI_Summer_2026.seq");
real_img = matread("/Users/hannah.jennings/Documents/Code/TestFolder/Slice 2 all contrasts/OneSlice_contrast6_slice2.mat")["OneSlice"]

#initialisation
# Select one contrast/slice. Contrast 1 is the three acquisitions discarded for the initial EPI correction; remaining acquisitions are ordered by slice
contrast_to_reconstruct = 5
slice_to_reconstruct = 2
num_slices = 3
num_contrasts = 9
etl = 55

1 < contrast_to_reconstruct <= num_contrasts ||
    error("contrast_to_reconstruct must be between 2 and $num_contrasts")
1 <= slice_to_reconstruct <= num_slices ||
    error("slice_to_reconstruct must be between 1 and $num_slices")

acquisition_index = (contrast_to_reconstruct - 1) * num_slices + slice_to_reconstruct
sequence_block_stride = 56
sequence_start = 1 + (acquisition_index - 1) * sequence_block_stride
sequence_end = sequence_start + sequence_block_stride
sequence_end <= length(seq) ||
    error("Selected acquisition extends beyond the sequence")
seq_oneslice = seq[sequence_start:sequence_end]
plot_seq(seq_oneslice)

cfg     = PhantomConfig(field = :T15, voxel_size_mm = 1.0)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

#initialise tau and coeffs placeholders
tau = zeros(9)
coeffs= zeros(9, 3)


# ---------------- 2. Helpers ----------------
function normalize_img(I)
    A = Float64.(abs.(I))
    return clamp.(A ./ quantile(vec(A), 0.99), 0.0, 1.0)
end

real_norm = normalize_img(real_img)
mask      = real_norm .> 0.05                 # where the real image has signal

function ssim_cost(sim, real_norm, mask)
    s = normalize_img(sim) .* mask
    r = real_norm          .* mask
    return 1 - assess_ssim(s, r)              # 0 = identical, lower is better
end

#---------------- 3. Simulation with Eddy Currents ----------------
function simulate_with_eddy(new_coeffs, taus)
    length(new_coeffs) == 3 || error("Expected 3 x/y/z coefficients")
    length(taus) == 3 || error("Expected 3 time constants")

    tau[1:3] .= taus
    coeffs[1, :] .= new_coeffs

params_eddy = Dict{String, Any}(
    "return_type" => "raw", 
    "sim_method" => BlochEddy(),
    "tau" => tau, 
    "coeffs" => coeffs
)

raw = simulate(obj, seq_oneslice, scanner; sim_params=params_eddy);

#----Reconstruction of the simulated data-----------

# The single-slice sequence contains one unencoded correction profile followed by the 55 phase-encoded imaging profiles.
length(raw.profiles) == etl + 1 ||
    error("Expected one correction profile and $etl imaging profiles; got $(length(raw.profiles)) profiles")
image_profiles = raw.profiles[2:end]

npnts = length(image_profiles[1].data)
all(length(profile.data) == npnts for profile in image_profiles) ||
    error("All imaging profiles must contain the same number of samples")

raw_data = Array{ComplexF64}(undef, npnts, etl)
kro = Array{Float64}(undef, npnts, etl)
kph = Array{Float64}(undef, npnts, etl)

for (line, profile) in enumerate(image_profiles)
    raw_data[:, line] .= profile.data
    kro[:, line] .= profile.traj[1, :]
    kph[:, line] .= profile.traj[2, :]
end

# The acquired imaging lines alternate readout polarity.
@assert all(isapprox.(reshape(kro[:, 1], :, 1), kro[:, 1:2:end], atol=1e-6))
@assert all(isapprox.(-reshape(kro[:, 1], :, 1), kro[:, 2:2:end], atol=1e-6))

@info "Approximate FOV in phase encoding direction = $(1 / Statistics.median(Statistics.median(kph[:, 1:end-1] - kph[:, 2:end]))) mm"

@info "Approximate FOV in readout direction = $(1 / maximum(kro[2:end, 1] - kro[1:end-1, 1]))"

# Prepare the data for MRIReco
sequenceInfo = Dict{Symbol, Any}()
encodingSize = (122, etl)
fov = (488.0, 220.0, 4.0) # it says this should be in mm ...

TrajFloat = Float64                               # type to be used
adc = seq_oneslice[3].ADC[1]
deltaTE = only(seq_oneslice[4].DUR)               # echo spacing (in s)
dwelltime = adc.T / adc.N                         # dwell time (in s)

name = "EPI"                                      # name of the trajectory
numProfiles = etl                                 # number of profiles
numSamplingPerProfile = npnts                     # number of sampling points per profile
numSlices = 1                                     # number of slices (for 3d trajectories)
AQ = TrajFloat(adc.T)                             # readout duration in s (per profile)
TE = TrajFloat(0.086)                             # echo time in s
cartesian = false                                 # true if sampling points lie on a cartesian grid (they don't - ramp sampling)
circular = false                                 # true if kspace is covered in a circular domain
times = (TE .+ (-npnts÷2:(npnts-1)÷2) .* dwelltime .+ (-etl÷2:(etl-1)÷2)' .* deltaTE)[:] # sampling times in s

nodes = TrajFloat.(vcat(mod.(vec(kro)' * fov[1] / encodingSize[1] .+ 0.5, 1) .- 0.5,
                        mod.(vec(kph)' * fov[2] / encodingSize[2] .+ 0.5, 1) .- 0.5))
traj = [Trajectory{TrajFloat}(name, nodes, times, TE, AQ, numProfiles,
                              numSamplingPerProfile, numSlices, cartesian, circular)]

# One acquisition, one slice, one echo; data are ordered sample-by-sample by profile.
kdata = Array{Matrix{ComplexF64}, 3}(undef, 1, 1, 1)
kdata[1, 1, 1] = reshape(raw_data, npnts * etl, 1)
subsampleIndices = [collect(1:npnts * etl)]

# parameters for image reconstruction:
params = Dict{Symbol, Any}(:reconSize => encodingSize)

acqdata = AcquisitionData{Float64, 2}(sequenceInfo, traj, kdata, subsampleIndices, encodingSize, fov)

img = reconstruction(acqdata, params)

img_mag = abs.(img)
img2d = Array(img_mag[:, :, 1, 1, 1, 1])
heatmap(img2d, title="Magnitude MRI image", aspect_ratio=:equal, yflip=true)

    return img_mag
end

###---------

function cost(p)
    coeffs = p                                     # signed: eddy currents can be +/-
    sim = simulate_with_eddy(new_coeffs, taus)
    return ssim_cost(sim, real_norm, mask)       # 1 - SSIM, from earlier
end

res = optimize(cost, zeros(3), NelderMead(),
               Optim.Options(iterations = 150, show_trace = true))
println("Coefficients (x, y, z): ", Optim.minimizer(res))
println("Final cost: ", Optim.minimum(res))