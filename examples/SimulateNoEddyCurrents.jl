using KomaMRI, MRISystemPhantom, Random

seq = read_seq(raw"/Users/hannah.jennings/Documents/Sequences/DWI_Summer_2026.seq");

cfg     = PhantomConfig(field = :T15, voxel_size_mm = 1.0)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

rng = Random.Xoshiro(123);

# Define hardware parameters
#tau = zeros(9)
#coeffs = zeros(9, 3)
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
    #"tau" => tau, 
    #"coeffs" => coeffs
)

raw = simulate(obj, seq, scanner; sim_params=params);

# Let's save the raw data, so we don't need to simulate it again.
using BSON: @save
@save "raw_no_eddy.bson" raw 

# -------------------------------------
# Reconstruction
# Remove comment out if I want to open from BSON file - i this case It was still saved in variable 
using BSON: @load
using KomaMRI, MRIReco
# Loading it again with:
@load "raw_no_eddy.bson" raw

# we might have to re-load the sequence, if Julia crashed on us:
seq = read_seq(raw"/Users/hannah.jennings/Documents/Sequences/DWI_Summer_2026.seq");

# We bring the raw data into a more convenient format.
# In this case all readouts have the same number of points:
npnts = length(raw.profiles[1].data)
# We also know the echo-train length and number of slices:
etl = 55
slc = 3
# Here we have 9 acquisitions for each slice
contrasts = length(raw.profiles) ÷ etl ÷ slc

# Arrays for storing the raw data and the (simulated) k-space trajectory
raw_data = Array{ComplexF64}(undef, npnts, etl, slc, contrasts);
kro = Array{Float64}(undef, npnts, etl, slc, contrasts);
kph = Array{Float64}(undef, npnts, etl, slc, contrasts);

# This uses knowledge about the loop structure of the sequence:
for i in eachindex(raw.profiles)
    line = (i-1) % etl + 1
    slice = div(i-1, etl) % slc + 1
    contrast = div(i-1, etl * slc) + 1
    raw_data[:, line, slice, contrast] .= raw.profiles[i].data
    kro[:, line, slice, contrast] .= raw.profiles[i].traj[1, :]
    kph[:, line, slice, contrast] .= raw.profiles[i].traj[2, :]
end

# we could export the trajectory to MATLAB:
#import MAT
file = MAT.matopen("Simulation_261026", "w")
write(file, "kro", kro)
write(file, "kph", kph)
close(file)

# check the trajectory (the phase encoding should be the same for all slices and contrasts).
# we skip the first contrast (indices 1..slc and start with slc + 1), since the first contrast has
# no phase encoding (it's used for some sort of EPI correction)
@assert(all(all(isapprox.(reshape(kph, npnts * etl, :)[:, slc + 1], reshape(kph, npnts * etl, :)[:, slc+1:contrasts*slc], atol=1e-6))))

import Statistics
@info "Approximate FOV in phase encoding direction = $(1 / Statistics.median(Statistics.median((kph[:, 1:end-1, 1, 2] - kph[:, 2:end, 1, 2])))) mm"

# check k-space encoding in readout directions (odd and even lines have opposite polarity)
@assert(all(all(isapprox.( kro[:, 1, 1, 1], reshape(kro[:, 1:2:end, :, [1:2; slc+1:end]], npnts, :), atol=1e-6))))
@assert(all(all(isapprox.(-kro[:, 1, 1, 1], reshape(kro[:, 2:2:end, :, [1:2; slc+1:end]], npnts, :), atol=1e-6))))

# here, the third contrast is special - it is acquired with opposite polarity for corrections ...
@assert(all(all(isapprox.(-kro[:, 1, 1, 1], reshape(kro[:, 1:2:end, :, 3], npnts, :), atol=1e-6))))
@assert(all(all(isapprox.( kro[:, 1, 1, 1], reshape(kro[:, 2:2:end, :, 3], npnts, :), atol=1e-6))))

@info "Approximate FOV in readout direction = $(1 / maximum(kro[2:end, 1, 1, 1] - kro[1:end-1, 1, 1, 1]))"

# We need to prepare the data for MRIReco
sequenceInfo = Dict{Symbol, Any}()
subsampleIndices = [collect(1:npnts * etl) for _ in 2:contrasts]
encodingSize = (122, etl)
fov = (488.0, 220.0, 4.0) # it says this should be in mm ...

TrajFloat = Float64                               # type to be used
deltaTE = seq[4].DUR                              # echo spacing (in s)
dwelltime = seq[3].ADC[1].T / seq[3].ADC[1].N     # dwell time (in s)

name = "EPI"                                      # name of the trajectory
numProfiles = etl                                 # number of profiles
numSamplingPerProfile = npnts                     # number of sampling points per profile
numSlices = 1                                     # number of slices (for 3d trajectories)
AQ = TrajFloat(seq[3].ADC[1].T)                   # readout duration in s (per profile)
TE = TrajFloat(0.086)                             # echo time in s
cartesian = false                                 # true if sampling points lie on a cartesian grid (they don't - ramp sampling)
circular = false                                  # true if kspace is covered in a circular domain
times = (TE .+ (-npnts÷2:(npnts-1)÷2) .* dwelltime .+ (-etl÷2:(etl-1)÷2)' .* deltaTE)[:] # sampling times in s

traj = Vector{Trajectory{TrajFloat}}(undef, contrasts-1)
for i in 2:contrasts # skip first "contrast" (no phase encoding, correction for EPI readout)
    nodes = TrajFloat.(vcat(mod.(kro[:, :, 1, i][:]' * fov[1] / encodingSize[1] .+ 0.5, 1) .- 0.5, mod.(kph[:, :, 1, i][:]' * fov[2] / encodingSize[2] .+ 0.5, 1) .- 0.5)) # sampling locations in k-space (same for each slice here) ??? UNITS ???
    traj[i-1] = Trajectory{TrajFloat}(name, nodes, times, TE, AQ, numProfiles, numSamplingPerProfile, numSlices, cartesian, circular)  
end

# outer dims: 8 x 3 x 1, inner dims: 9240 x 1
kdata = Array{Matrix{ComplexF64}, 3}(undef, contrasts-1, slc, 1)
for i in 2:contrasts # don't reconstruct 1st contrast
    for j in 1:slc
        kdata[i-1, j, 1] = reshape(raw_data[:, :, j, i], 9240, 1)
    end
end

# parameters for image reconstruction:
params = Dict{Symbol, Any}(:reconSize => encodingSize)

acqdata = AcquisitionData{Float64, 2}(sequenceInfo, traj, kdata, subsampleIndices, encodingSize, fov)

images = reconstruction(acqdata, params)

# export images for MATLAB:
file = MAT.matopen(raw"/Users/hannah.jennings/Documents/Code/TestFolder/Simulation_261026.mat", "w")
write(file, "images_2", images)
close(file)
