using KomaMRI, MRISystemPhantom, Random

seq = read_seq(raw"/Users/hannah.jennings/Documents/Sequences/DWI_Summer_2026.seq");

# Select one contrast/slice. Contrast 1 is the three acquisitions discarded
# for the initial EPI correction; remaining acquisitions are ordered by slice.
contrast_to_reconstruct = 4
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

#rng = Random.Xoshiro(123);

# Define hardware parameters
tau = zeros(9)
coeffs = zeros(9, 3)
#tau[8] = 50e-3 # 50 ms decay constant for XY harmonic 
#tau[4] = 70e-3 # 70 ms decay constant for Z harmonic 

#tau = rand(rng, 9) # this results in a bit long time constants, just to make sure ...
#coeffs = rand(rng, 9, 3)

#coeffs[8, 1] = 0.050 # X-gradient driving the XY cross-term
#coeffs[4, 1] = 0.030 # X-gradient driving the Z cross-term

#tau[1] = 100e-3
#tau[2] = 50e-3
#tau[3] = 50e-3
#tau[4] = 50e-3

#coeffs[1, 1] = 0.01
#coeffs[1, 2] = 0.01
#coeffs[1, 3] = 0.01
#coeffs[2, 1] = 0.02
#coeffs[3, 2] = 0.02
#coeffs[4, 3] = 0.02



# Simulation parameters
params = Dict{String, Any}(
    "return_type" => "raw", 
    "sim_method" => Bloch(),
    "tau" => tau, 
    "coeffs" => coeffs
)

raw = simulate(obj, seq_oneslice, scanner; sim_params=params);

# Let's save the raw data, so we don't need to simulate it again.
using BSON: @save
raw_filename = "OneSlice_contrast$(contrast_to_reconstruct)_slice$(slice_to_reconstruct).bson"
@save raw_filename raw

# -------------------------------------
# Reconstruction
# Remove comment out if I want to open from BSON file - i this case It was still saved in variable 
#using BSON: @load
using KomaMRI, MRIReco, MAT
# Loading it again with:
#@load "raw_no_eddy.bson" raw

# we might have to re-load the sequence, if Julia crashed on us:
#seq = read_seq(raw"/Users/hannah.jennings/Documents/Sequences/DWI_Summer_2026.seq");

# The single-slice sequence contains one unencoded correction profile followed
# by the 55 phase-encoded imaging profiles.
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

# we could export the trajectory to MATLAB:
#import MAT
#trajectory_filename = "OneSlice_contrast$(contrast_to_reconstruct)_slice$(slice_to_reconstruct)_trajectory.mat"
#file = MAT.matopen(trajectory_filename, "w")
#write(file, "kro", kro)
#write(file, "kph", kph)
#close(file)

# The acquired imaging lines alternate readout polarity.
@assert all(isapprox.(reshape(kro[:, 1], :, 1), kro[:, 1:2:end], atol=1e-6))
@assert all(isapprox.(-reshape(kro[:, 1], :, 1), kro[:, 2:2:end], atol=1e-6))

import Statistics
@info "Approximate FOV in phase encoding direction = $(1 / Statistics.median(Statistics.median(kph[:, 1:end-1] - kph[:, 2:end]))) mm"

@info "Approximate FOV in readout direction = $(1 / maximum(kro[2:end, 1] - kro[1:end-1, 1]))"

# We need to prepare the data for MRIReco
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

images = reconstruction(acqdata, params)

# export images for MATLAB:
image_filename = "OneSlice_contrast$(contrast_to_reconstruct)_slice$(slice_to_reconstruct).mat"
file = MAT.matopen(image_filename, "w")
write(file, "OneSlice", images)
close(file)
