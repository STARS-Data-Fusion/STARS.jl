module STARS

export STARS_fusion
export coarse_fine_data_fusion
export coarse_fine_data_fusion_SS
export coarse_fine_data_fusion_SS_seq

export MLE_estimation
export fast_var_est
export compute_n_eff

export DataFusionState

# Write your package code here.

using Suppressor
using Dates
using DelimitedFiles
using DataFrames
using Rasters
using Plots
using Sobol
#using Kalman
using LinearAlgebra
# using GaussianDistributions: ⊕ # independent sum of Gaussian r.v.
using GaussianDistributions
using Distributions
using Statistics
using StatsBase
using SparseArrays
using BlockDiagonals
using DynamicIterators
using Trajectories
using ProgressMeter
using Distances
using Optim
import GaussianRandomFields.CovarianceFunction
import GaussianRandomFields.Matern
import GaussianRandomFields.apply

include("BBoxes.jl")

using .BBoxes

include("sentinel_tiles.jl")

include("HLS.jl")

using .HLS

include("VNP43.jl")

using .VNP43

include("FilterSmoother.jl")
using .FilterSmoother

#BLAS.set_num_threads(1)

"struct representing state of data fusion operation as fine mean, fine standard deviation, coarse bias mean, and coarse bias standard deviation"
struct DataFusionState
    mean::Raster # fine resolution mean (n,n,nsteps)
    SD::Raster # fine resolution standard deviation (n,n,nsteps)
    mean_bias::Raster # coarse resolution bias mean (n_c,n_c,nsteps)
    SD_bias::Raster # coarse resolution bias standard deviation (n_c,n_c,nsteps)
    cond_sims::Union{Raster, Nothing} # M conditional mean simulations (n,n,nsteps,M)
end

function unif_weighted_obs_operator(sensor, target, sensor_res, target_res)
    d1 = pairwise(Euclidean(), sensor[:, 1]' .+ sensor_res[1] / 2, target[:, 1]' .+ target_res[1] / 2, dims=2) .<= sensor_res[1] / 2
    d2 = pairwise(Euclidean(), sensor[:, 2]' .+ sensor_res[2] / 2, target[:, 2]' .+ target_res[2] / 2, dims=2) .<= sensor_res[2] / 2

    H = d1 .* d2
    return sparse(broadcast(/, H, sum(H, dims=2)))
end

## need to update for corner coordinates
function gauss_weighted_obs_operator(sensor, target, res; p=1.0, scale=1.0)
    H = exp.(-0.5 * pairwise(SqEuclidean(), sensor ./ transpose(res ./ scale), target ./ transpose(res ./ scale), dims=1))

    H[H.<exp(-0.5 * (p * scale)^2)] .= 0

    return sparse(broadcast(/, H, sum(H, dims=2)))
end

function kernel_matrix(X::AbstractArray{T}; reg=1e-10, σ=1.0) where {T<:Real}
    Diagonal(reg * ones(size(X)[1])) + exp.(-0.5 * pairwise(SqEuclidean(), X, dims=1) ./ σ^2)
end

function matern_cor(X::AbstractArray{T}; reg=1e-10, ν=0.5, σ=1.0) where {T<:Real}
    cc = CovarianceFunction(2, Matern(σ, ν))
    Diagonal(reg * ones(size(X)[2])) + apply(cc, X, X)
end

## neighborhood functions on a grid
function col_major(row, col, nrow)
    return row .+ nrow * (col .- 1)
end

function row_major(row, col, ncol)
    return col .+ ncol * (row .- 1)
end

function filter_val(x, lb, ub)
    return x[(x.>=lb).*(x.<=ub)]
end


function replace_val(x, lb, ub, val=-999999)
    x[(x.<lb).|(x.>ub)] .= val
    return x
end

## for now we will only do queen, extend to regional neighborhoods, or clustered
function neighborhood(rows, cols, nrow, ncol; column_major::Bool=false, include::Bool=true)
    n = nrow * ncol
    lr = minimum(rows)
    ur = maximum(rows)
    lc = minimum(cols)
    uc = maximum(cols)

    r0 = rows .- lr .+ 1
    c0 = cols .- lc .+ 1
    r1 = replace_val(r0 .+ 1, lr, ur)
    r2 = replace_val(r0 .- 1, lr, ur)
    c1 = replace_val(c0 .+ 1, lc, uc)
    c2 = replace_val(c0 .- 1, lc, uc)

    if column_major

        x0 = col_major(r0, c0, nrow)
        x1 = col_major(r1, c0, nrow)
        x2 = col_major(r2, c0, nrow)
        x3 = col_major(r0, c1, nrow)
        x4 = col_major(r0, c2, nrow)
        x5 = col_major(r1, c1, nrow)
        x6 = col_major(r2, c2, nrow)
        x7 = col_major(r1, c2, nrow)
        x8 = col_major(r2, c1, nrow)

    else

        x0 = row_major(r0, c0, ncol)
        x1 = row_major(r1, c0, ncol)
        x2 = row_major(r2, c0, ncol)
        x3 = row_major(r0, c1, ncol)
        x4 = row_major(r0, c2, ncol)
        x5 = row_major(r1, c1, ncol)
        x6 = row_major(r2, c2, ncol)
        x7 = row_major(r1, c2, ncol)
        x8 = row_major(r2, c1, ncol)

    end

    if include
        nbs = [x0 x1 x2 x3 x4 x5 x6 x7 x8]
    else
        nbs = [x1 x2 x3 x4 x5 x6 x7 x8]
    end

    df = DataFrame(ind=x0)
    df.neighbs = [filter_val(row, 1, nrow * ncol) for row in eachrow(nbs)]

    return df
end


function get_index(neighbs, index)
    return DataFrame(ind=neighbs.ind, indices=[index[x] for x in neighbs.neighbs])
end

function get_inds(index1, index2)
    findall(in(index1).(index2))
end

function get_fine_ind(coarse_neighbs_indices, fine_index)
    df = DataFrame(ind=coarse_neighbs_indices.ind, fine_inds=[get_inds(x, fine_index) for x in coarse_neighbs_indices.indices])
    df.keep_ind = [findall(in(x[1]).(fine_index[y])) for (x, y) in zip(coarse_neighbs_indices.indices, df.fine_inds)]
    df.replace_ind = [x[y] for (x, y) in zip(df.fine_inds, df.keep_ind)]
    return df
end

function get_fine_neighbs2(coords, column_major::Bool=false)
    ## assumes box
    nr = length(unique(coords[:, 2]))
    nc = length(unique(coords[:, 1]))

    cols = repeat(1:nc, inner=[1], outer=nr)
    rows = repeat(1:nr, inner=nc, outer=[1])

    nbs = neighborhood(rows, cols, nr, nc, column_major=column_major, include=false)

    num_neighbs = [length(x) for x in nbs.neighbs]

    return flatten(nbs, :neighbs), num_neighbs
end

function get_fine_neighbs(coords, column_major::Bool=false)

    nr = length(unique(coords.row))
    nc = length(unique(coords.col))
    nbs = neighborhood(coords.row, coords.col, nr, nc, column_major=column_major, include=false)

    num_neighbs = [length(x) for x in nbs.neighbs]

    return flatten(nbs, :neighbs), num_neighbs
end

function build_CAR_prec(nbs, nnbs, sigma, phi)
    return spdiagm(0 => nnbs ./ sigma) - phi / sigma .* sparse(nbs.ind, nbs.neighbs, 1.0)
end

function build_CAR_var(nbs, nnbs, sigma, phi)
    A = build_CAR_prec(nbs, nnbs, sigma, phi)
    return inv(Matrix(A))
end

function build_GP_var(locs, sigma, phi, nugget=1e-10)
    A = sigma .* kernel_matrix(locs, reg=nugget, σ=phi)
    return A
end

function get_missing_indices(raster::Raster)
    if isnan(raster.missingval)
        return isnan.(raster)
    elseif ismissing(raster.missingval)
        return ismissing.(raster)
    else
        return raster .== raster.missingval
    end
end

function cell_size(raster::Raster)
    # Julia is column-major so raster arrays are transposed from the way they are in Numpy
    # in Julia, rows translate to X, and cols translate to Y
    rows = size(raster)[1]
    cols = size(raster)[2]

    if length(size(raster)) == 3
        bbox = bounds(raster[:, :, 1])
    else
        bbox = bounds(raster)
    end

    xmin, xmax = bbox[1]
    ymin, ymax = bbox[2]
    width = xmax - xmin
    height = ymax - ymin
    cell_width = width / rows
    cell_height = -height / cols
    cell_width, cell_height
end

function get_x_matrix(image::Raster)
    reshape(repeat(vec(image.dims[1].val[:]), inner=1, outer=size(image)[2]), size(image)[1], size(image)[2])
end

function get_y_matrix(image::Raster)
    transpose(reshape(repeat(vec(image.dims[2].val[:]), inner=1, outer=size(image)[1]), size(image)[2], size(image)[1]))
end

"STARS fusion"
function STARS_fusion(
        measurements, 
        measurement_error_vars,                                 
        measurement_coords, 
        resolutions,                   
        target_coords,
        target_resolution,
        prior_mean,
        prior_sd;
        ensembles::Union{AbstractArray{>:Missing},Nothing} = nothing,
        target_times = [1], 
        smooth = true,           
        spatial_mod = "Matern",                   
        cov_pars = [0.002, 150, 1e-10, 0.5], 
        wt = [1,0.5],   
        offset_ar = [0.95, 0.0], 
        offset_var = [0.0001,0.0001],                        
        obs_operator = "uniform", 
        historic_tseries = ones((1,1))) 
                              
    # measurements: List of n_i x T matrices of instrument measurements, columns are flattened vector of pixels within window at each time point
    # measurement_error_vars: List of n_i x T matrices of measurement error variances, columns are flattened vector of variances within window at each time point
    # measurement_coords: List of n_i x 2 matrices of x,y centroids of instrument grids, assumes (col,row) format
    # resolutions: List of 2-dim vector of instrument grid resolution 
    # target_coords: nf x 2 matrix of x,y centroids of target grid within window
    # prior_mean: vector of prior mean at time t=0
    # prior_sd: vector of prior sd at time t=0
    # cov_pars: vector of spatial covariance parameters [spatial var, length scale or MRF dependence, nugget]
    # offset_ar: vector AR(1) correlation parameter for each sensor, value of 0 means no bias
    # offset_var: innovation variance parameter for bias terms
    # obs_operator: "uniform" or "gaussian" specifiying weights of observation operator

    ni = size(measurements)[1] # number of instruments
    nnobs = [size(measurements[i])[1] for i in 1:ni] # number of grid cells per instrument
    nf = size(target_coords)[1] # number of target resolution grid cells
    nsteps = size(measurements[end])[2] # number of time steps

    ## build observation operator, stack observations and variances 
    H = Array{Float64}(undef, 0, nf)
    ys = Array{Float64}(undef, 0, nsteps)
    err_vars = Array{Float64}(undef, 0, nsteps)

    for i in 1:ni
        if obs_operator == "uniform"
            H1 = unif_weighted_obs_operator(measurement_coords[i], target_coords, resolutions[i], target_resolution)
        elseif obs_operator == "gaussian"
            H1 = gauss_weighted_obs_operator(measurement_coords[i], target_coords, resolutions[i], p=wt[i], scale=1)
        else
            error("operator must be either uniform or gaussian")
        end

        H = vcat(H, H1)
        ys = vcat(ys, measurements[i])
        err_vars = vcat(err_vars, measurement_error_vars[i])
    end

    ## build spatial models
    if spatial_mod == "MRF"
        nbs, nn = get_fine_neighbs2(target_coords)
        Q = build_CAR_var(nbs, nn, cov_pars[1], cov_pars[2]) # spatial innovation covariance
    elseif spatial_mod == "Ind"
        Q = cov_pars[1] .* Diagonal(ones(size(target_coords)[1]))
    elseif spatial_mod == "SqExp"
        Q = build_GP_var(target_coords, cov_pars[1], cov_pars[2], nugget=cov_pars[3])
    elseif spatial_mod == "Matern"
        Q = cov_pars[1] .* matern_cor(transpose(target_coords), reg=cov_pars[3], ν=cov_pars[4], σ=cov_pars[2])
    elseif spatial_mod == "TS"
        Q = cov_pars[1] .* kernel_matrix(historic_tseries, reg=cov_pars[3], σ=cov_pars[2])
    elseif spatial_mod == "Mixed"
        Q = cov_pars[1] .* kernel_matrix(historic_tseries, reg=cov_pars[3], σ=cov_pars[2]) .* build_GP_var(target_coords, 1., cov_pars[5], 0.0)
    # elseif spatial_mod == "Mixed"
    #     Q = cov_pars[1] .* (0.8 .* kernel_matrix(historic_tseries, reg=cov_pars[3], σ=cov_pars[2]) + 0.2 .* matern_cor(transpose(target_coords), reg=cov_pars[3], ν=1.5, σ=150))
    else
        error("spatial_mod must be MRF, SqExp, Matern, TS, or Mixed")
    end

    ## add additive biases and set-up remaining model matrices
    nfs = 0
    add_bias = offset_ar .> 0 # if offset_ar is greater than 0 add bias for that instrument

    if any(add_bias)
        nfs = sum(nnobs[add_bias])
        ar_vals = vcat(fill.(offset_ar[add_bias], nnobs[add_bias])...)
        ar_vars = vcat(fill.(offset_var[add_bias], nnobs[add_bias])...)
        Hb = zeros((sum(nnobs), sum(nnobs[add_bias])))
        Hb[vcat(fill.(add_bias, nnobs)...), :] = Matrix(I, nfs, nfs)
        H = hcat(Hb, H)

        # dynamics
        F = Φ = Diagonal(vcat(ar_vals, ones(nf))) # identity transition matrix (random walk)
        Q = Matrix(BlockDiagonal([diagm(ar_vars), Q])) # spatial innovation covariance
        if spatial_mod == "Ind"
            Q = Diagonal(Q)
        end
    else

        F = Φ = UniformScaling(1)

    end

    x0 = convert(Vector{Float64}, prior_mean[:]) # don't need this but here to help with synergizing code later
    P0 = convert(Matrix{Float64}, Diagonal(prior_sd[:] .^ 2)) # just assuming diagonal C0

    if size(P0) != size(Q)
        error("Dimension of prior_mean/prior_sd doesn't match model dimension. Check if prior includes bias elements.")
    end

    #target_time_indices = target_times
    Tt = length(target_times)
    t0 = minimum(target_times)
    tn = maximum(target_times)

    if !smooth & (tn < nsteps)
        #println("Column dimension of measurement matrices suggest measurements included after last target time but smooth = false. Are you sure you don't want to smooth?")
        nsteps = tn
    end

    # if target is only the last day there is no need to smooth, filtering is equal to smoothing distribution
    if (t0 == tn) & (tn == nsteps)
        smooth = false
    end
    
    if isnothing(ensembles)
        cond_sims = nothing
    end

    #H = Matrix(H)
    M = KSModel(H, Q, F)

    predicted_means, predicted_covs, filtering_means, filtering_covs =
        FilterSmoother.filter_series(M, x0, P0, ys[:, 1:nsteps], err_vars[:, 1:nsteps])

    if !smooth
        if ~isnothing(ensembles)
            cond_sims = FilterSmoother.conditional_sim_series(filtering_means[(target_times .+ 1)], filtering_covs[(target_times .+ 1)], ensembles, nfs=nfs+1) 
        end

        fused_images = reduce(hcat,filtering_means[(target_times .+ 1)])
        fused_sd_images = reduce(hcat,[sqrt.(diag(filtering_covs[x])) for x in (target_times .+ 1)])
    else
        smoothed_means, smoothed_covs =
            FilterSmoother.smooth_series(M, predicted_means[t0:end], predicted_covs[t0:end],
                                       filtering_means[t0:end], filtering_covs[t0:end])

        if ~isnothing(ensembles)
            cond_sims = FilterSmoother.conditional_sim_series(smoothed_means[(target_times .- t0 .+ 1)], smoothed_covs[(target_times .- t0 .+ 1)], ensembles, nfs=nfs+1) 
        end

        fused_images = reduce(hcat,smoothed_means[(target_times .- t0 .+ 1)]) # probably do a reduce here
        fused_sd_images = reduce(hcat,[sqrt.(diag(smoothed_covs[x])) for x in (target_times .- t0 .+ 1)])

    end

    return fused_images, fused_sd_images, cond_sims
end

"function to perform data fusion on pair of coarse and fine raster timeseries using a moving window"
function coarse_fine_data_fusion(
        coarse_images::Raster,
        fine_images::Raster,
        cov_pars::Union{Raster,AbstractVector{Float64}};
        coarse_err_var::Union{Raster, Float64} = 1e-6,
        fine_err_var::Union{Raster, Float64} = 1e-6,
        prior::Union{DataFusionState, Nothing} = nothing,
        n_ensembles = 0,
        target_times = nothing, 
        smooth = true,              
        spatial_mod = "Matern",                   
        wt = [1, 0.5],   
        offset_ar = [0.98, 0.0], 
        offset_var = [1e-5, 1e-5],                        
        obs_operator = "uniform", 
        default_mean::Float64 = 0.3,
        default_SD::Float64 = 0.01,
        default_bias_mean::Float64 = 0.0,
        default_bias_SD::Float64 = 1e-5,
        buffer_distance::Union{Float64, Nothing} = nothing,
        historic_tseries = nothing,
        show_progress_bar::Bool = true)::DataFusionState

    coarse_dims = coarse_images.dims[1:2]
    fine_dims = fine_images.dims[1:2]
    coarse_rows, coarse_cols, coarse_timesteps = size(coarse_images)
    coarse_shape = (coarse_rows, coarse_cols)
    fine_rows, fine_cols, fine_timesteps = size(fine_images)
    fine_shape = (fine_rows, fine_cols)

    if prior === nothing
        prior_fused_mean = Raster(default_mean .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_fused_sd = Raster(default_SD .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_bias_mean = Raster(default_bias_mean .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
        prior_bias_sd = Raster(default_bias_SD .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
    else
        prior_fused_mean = ifelse(length(prior.mean.dims) == 3, prior.mean[:, :, end], prior.mean)
        prior_fused_sd = ifelse(length(prior.SD.dims) == 3, prior.SD[:, :, end], prior.SD)
        prior_bias_mean = ifelse(length(prior.mean_bias.dims) == 3, prior.mean_bias[:, :, end], prior.mean_bias)
        prior_bias_sd = ifelse(length(prior.SD_bias.dims) == 3, prior.SD_bias[:, :, end], prior.SD_bias)
    end

    if target_times === nothing
        target_times = 1:coarse_timesteps
    end

    if isa(target_times, Vector{String})
        target_times = map(d -> Date(d), target_times)
    end

    target_timesteps = length(target_times)

    if isa(target_times, Vector{Date})
        time_slice = At(target_times)
        bias_dims = (coarse_dims[1:2]..., Ti(target_times))
        target_dims = (fine_dims[1:2]..., Ti(target_times))
    else
        time_slice = target_times
        #bias_dims = (coarse_dims[1:2]..., coarse_images.dims[3][target_times])
        #target_dims = (fine_dims[1:2]..., fine_images.dims[3][target_times])
        bias_dims = (coarse_dims[1:2]..., Band(1:coarse_timesteps)[target_times])
        target_dims = (fine_dims[1:2]..., Band(1:fine_timesteps)[target_times])
    end

    target_time_indices = target_times

    if isa(target_times, Vector{Date})
        target_time_indices = findall(t -> t in target_times, Array(fine_images.dims[3].val))

        if length(target_time_indices) == 0
            throw(error("no target times $(target_times) found in fine images $(Array(fine_images.dims[3].val))"))
        end
    end

    start_col = 1
    end_col = coarse_cols
    start_row = 1
    end_row = coarse_rows

    fine_cell_width, fine_cell_height = cell_size(fine_images)
    coarse_cell_width, coarse_cell_height = cell_size(coarse_images)

    if buffer_distance === nothing
        buffer_distance = coarse_cell_width
    end

    if spatial_mod == "Ind"
        buffer_distance = 0.0
    end

    resolutions = [[coarse_cell_width, -coarse_cell_height], [fine_cell_width, -fine_cell_height]]
    target_resolution = [fine_cell_width, -fine_cell_height]

    fused_target_images = Raster(fill(NaN, fine_rows, fine_cols, target_timesteps), dims=target_dims, missingval=fine_images.missingval)
    fused_UQ_target_images = Raster(fill(NaN, fine_rows, fine_cols, target_timesteps), dims=target_dims, missingval=fine_images.missingval)

    fused_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, target_timesteps), dims=bias_dims, missingval=coarse_images.missingval)
    fused_UQ_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, target_timesteps), dims=bias_dims, missingval=coarse_images.missingval)

    m1 = hcat(repeat(start_col:end_col, inner=[1], outer=[end_row]),
        repeat(start_row:end_row, inner=[end_col], outer=[1]))

    n = size(m1)[1] 
    
    fused_vectors_list = Vector{Matrix{Float64}}(undef, end_col*end_row)
    fused_UQ_vectors_list = Vector{Matrix{Float64}}(undef, end_col*end_row)
    fused_condsim_vectors_list = Vector{Union{Array{Float64}, Nothing}}(undef, end_col*end_row)

    #if n_ensembles > 0
        #fused_condsim_target_images = Raster(fill(NaN, fine_rows, fine_cols, n_ensembles,target_timesteps), dims=(target_dims[1:2]...,Z(1:n_ensembles),target_dims[3]), missingval=fine_images.missingval)
    fused_condsim_target_images = Raster(fill(NaN, fine_rows, fine_cols, n_ensembles*target_timesteps), dims=(target_dims[1:2]...,Z(1:(n_ensembles*target_timesteps))), missingval=fine_images.missingval)
        #ugly 3d workaround for 4d resampling later
    ensembles = Raster(reshape(rand(Normal(0,1),fine_rows*fine_cols*target_timesteps*n_ensembles),size(fused_condsim_target_images)), fused_condsim_target_images.dims)
        #else
        #ensemble_window_values = nothing
    #end

    if show_progress_bar
        p = Progress(n)
        update!(p, 0)
        jj = Threads.Atomic{Int}(0)
        l = Threads.SpinLock()
    end

    # time_subset = zeros(n)
    # time_fused = zeros(n)
    # time_putback = zeros(n)

    #@showprogress 
    Threads.@threads for k in 1:n
        if show_progress_bar
            Threads.atomic_add!(jj, 1)
            Threads.lock(l)
            update!(p, jj[])
            Threads.unlock(l)
        end

        #t1 = time()
        coarse_col, coarse_row = m1[k, :]
        # select the target bounding box along the edges of the center coarse pixel
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        # select the window bounding box by expanding the target bounding box by a given distance
        window_bbox = buffer(target_bbox, buffer_distance)
        # select coarse pixels using the window bounding box
        coarse_window = view(coarse_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select prior mean bias pixels using the window bounding box
        bias_mean_window = view(prior_bias_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select prior standard deviation bias using the window bounding box
        bias_sd_window = view(prior_bias_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # dimensions of the moving window on the coarse grid
        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
        # extract coarse pixel values
        coarse_window_values = Matrix(reshape(read(coarse_window), coarse_window_rows * coarse_window_cols, coarse_window_timesteps))
        # extract coarse mean bias values
        bias_mean_window_values = Matrix(reshape(read(bias_mean_window), coarse_window_rows * coarse_window_cols, 1))
        # extract coarse standard deviation values
        bias_sd_window_values = Matrix(reshape(read(bias_sd_window), coarse_window_rows * coarse_window_cols, 1))
        # select fine pixels using the window bounding box
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        #if n_ensembles > 0
            # select ensemble pixels using the window bounding box
        ensemble_window = view(ensembles, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        #end

        # select fine prior mean values using the window bounding box
        prior_mean_window = view(prior_fused_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select fine prior standard deviation values using the window bounding box
        prior_sd_window = view(prior_fused_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # dimensions of the moving window on the fine grid
        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)
        # extract fine pixel values
        fine_window_values = Matrix(reshape(read(fine_window), fine_window_rows * fine_window_cols, fine_window_timesteps))
        #if n_ensembles > 0
            # select ensemble pixels using the window bounding box
        ensemble_window_values = Array(reshape(replace(read(ensemble_window), fine_window.missingval => missing), fine_window_rows * fine_window_cols, n_ensembles,target_timesteps))
        #end

        if n_ensembles == 0
            ensemble_window_values = nothing
        end

        # extract fine prior mean values
        prior_mean_window_values = Matrix(reshape(read(prior_mean_window), fine_window_rows * fine_window_cols, 1))
        # extract fine prior standard deviation values
        prior_sd_window_values = Matrix(reshape(read(prior_sd_window), fine_window_rows * fine_window_cols, 1))
        # get coarse x coordinates
        coarse_window_x_matrix = get_x_matrix(coarse_window)
        # get coarse y coordinates
        coarse_window_y_matrix = get_y_matrix(coarse_window)
        # combine coarse x and y coordinates
        coarse_window_coords = cat(vec(coarse_window_x_matrix), vec(coarse_window_y_matrix), dims=2)
        # get fine x coordinates
        fine_window_x_matrix = get_x_matrix(fine_window)
        # get fine y coordinates
        fine_window_y_matrix = get_y_matrix(fine_window)
        # combine fine x and y coordinates
        fine_window_coords = cat(vec(fine_window_x_matrix), vec(fine_window_y_matrix), dims=2)
        # set the target coordinates to the fine coordinates
        target_coords = fine_window_coords

        if isa(cov_pars, Raster)
            cov_pars_window = view(cov_pars, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
            cov_values = read(cov_pars_window)[:]
        else
            cov_values = cov_pars
        end

        if historic_tseries === nothing
            tseries_window_values = ones((1, 1))
        else
            tseries_window = view(historic_tseries, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
            tseries_window_values = Matrix{Float64}(reshape(read(tseries_window), fine_window_rows * fine_window_cols, size(historic_tseries)[3]))
        end

        # separate handling for time dimension if target times are a set of dates or indices
        if isa(target_times, Vector{Date})
            # assign time index dimension
            bias_window_dims = (coarse_window.dims[1], coarse_window.dims[2], Ti(target_times))
            target_window_dims = (fine_window.dims[1], fine_window.dims[2], Ti(target_times))
        else
            # assign enumerated index for time dimension
            bias_window_dims = (coarse_window.dims[1],coarse_window.dims[2], Band(1:coarse_timesteps)[target_times])
            target_window_dims = (fine_window.dims[1],fine_window.dims[2], Band(1:fine_timesteps)[target_times])
        end

        # assign measurement errors
        if isa(coarse_err_var, Raster)
            coarse_err_vars = read(view(coarse_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            coarse_err_vars = coarse_err_var .* ones(size(coarse_window_values))
        end

        if isa(fine_err_var, Raster)
            fine_err_vars = read(view(fine_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            fine_err_vars = fine_err_var .* ones(size(fine_window_values))
        end

        measurement_error_vars = [coarse_err_vars, fine_err_vars ]

         # combine fine and coarse input values
        measurements = [coarse_window_values, fine_window_values]
        # combine fine and coarse coordinates
        measurement_coords = [coarse_window_coords, fine_window_coords]

        prior_mean = vcat(bias_mean_window_values, prior_mean_window_values)
        prior_sd = vcat(bias_sd_window_values, prior_sd_window_values)

        #time_subset[k] = time() - t1
        #t2 = time()
        ## run offset+MRF model filtering/smoothing over full time period
        fused_vectors_list[k], fused_UQ_vectors_list[k], fused_condsim_vectors_list[k] = STARS_fusion(
            measurements,                                                                                
            measurement_error_vars,
            measurement_coords,
            resolutions,
            target_coords,
            target_resolution,
            prior_mean,
            prior_sd,
            ensembles = ensemble_window_values,
            target_times = target_time_indices, 
            smooth = smooth,  
            spatial_mod = spatial_mod,
            cov_pars = cov_values, 
            wt = wt,
            offset_ar = offset_ar,
            offset_var = offset_var,
            obs_operator = obs_operator,
            historic_tseries = tseries_window_values
        )

        #time_fused[k] = time() - t2
    end

    ### put back into rasters
    for k in 1:n

        #t3 = time()
        coarse_col, coarse_row = m1[k, :]

        # select the target bounding box along the edges of the center coarse pixel
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        # select the window bounding box by expanding the target bounding box by a given distance
        window_bbox = buffer(target_bbox, buffer_distance)

        # select coarse pixels using the window bounding box
        coarse_window = view(coarse_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select fine pixels using the window bounding box
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
    
        # dimensions of the moving window on the coarse grid
        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
        # dimensions of the moving window on the fine grid
        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)

        # separate handling for time dimension if target times are a set of dates or indices
        if isa(target_times, Vector{Date})
            # assign time index dimension
            bias_window_dims = (coarse_window.dims[1], coarse_window.dims[2], Ti(target_times))
            target_window_dims = (fine_window.dims[1], fine_window.dims[2], Ti(target_times))
        else
            # assign enumerated index for time dimension
            bias_window_dims = (coarse_window.dims[1],coarse_window.dims[2], Band(1:coarse_timesteps)[target_times])
            target_window_dims = (fine_window.dims[1],fine_window.dims[2], Band(1:fine_timesteps)[target_times])
        end

        fused_target_images_subset = view(fused_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_fine_window = reshape(replace(fused_vectors_list[k], missing => fine_window.missingval)[(end-fine_window_rows*fine_window_cols+1):end, :], fine_window_rows, fine_window_cols, target_timesteps)
        resampled_subset = resample(Raster(fused_fine_window, dims=(fine_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_target_images_subset)
        inds = get_missing_indices(fused_target_images_subset)
        fused_target_images_subset[inds] = resampled_subset[inds]
    
        if n_ensembles > 0
            fused_condsim_target_images_subset = view(fused_condsim_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
            #fused_ensemble_window = reshape(replace(fused_condsim_vectors_list[k], missing => fine_window.missingval), fine_window_rows, fine_window_cols, n_ensembles, target_timesteps)
            #resampled_subset = resample(Raster(fused_ensemble_window, dims = (fine_window.dims[1:2]..., Z(1:n_ensembles), Band(1:target_timesteps))), to = fused_condsim_target_images_subset)
            # for whatever reason Julia is swapping the last two indices in the resample, an ugly workaround here
            fused_ensemble_window = reshape(replace(fused_condsim_vectors_list[k], missing => fine_window.missingval), fine_window_rows, fine_window_cols, n_ensembles*target_timesteps)
            resampled_subset = resample(Raster(fused_ensemble_window, dims = (fine_window.dims[1:2]..., Z(1:(n_ensembles*target_timesteps)))), to = fused_condsim_target_images_subset)
            inds = get_missing_indices(fused_condsim_target_images_subset)
            fused_condsim_target_images_subset[inds] = resampled_subset[inds]
        end

        fused_UQ_target_images_subset = view(fused_UQ_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_UQ_fine_window = reshape(replace(fused_UQ_vectors_list[k], missing => fine_window.missingval)[(end-fine_window_rows*fine_window_cols+1):end, :], fine_window_rows, fine_window_cols, target_timesteps)
        resampled_subset = resample(Raster(fused_UQ_fine_window, dims=(fine_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_UQ_target_images_subset)
        inds = get_missing_indices(fused_UQ_target_images_subset)
        fused_UQ_target_images_subset[inds] = resampled_subset[inds]

        fused_bias_images_subset = view(fused_coarse_bias_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_bias_window = reshape(replace(fused_vectors_list[k], missing => coarse_window.missingval)[1:(end-fine_window_rows*fine_window_cols), :], coarse_window_rows, coarse_window_cols, target_timesteps)

        if size(fused_bias_window)[1:2] == (1, 1) && size(fused_bias_images_subset)[1:2] == (1, 1)
            fused_bias_images_subset[:, :, :] = fused_bias_window[:, :, :]
        else
            resampled_subset = resample(Raster(fused_bias_window, dims=(coarse_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_bias_images_subset)
            inds = get_missing_indices(fused_bias_images_subset)
            fused_bias_images_subset[inds] = resampled_subset[inds]
        end

        fused_UQ_bias_images_subset = view(fused_UQ_coarse_bias_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_UQ_bias_window = reshape(replace(fused_UQ_vectors_list[k], missing => coarse_window.missingval)[1:(end-fine_window_rows*fine_window_cols), :], coarse_window_rows, coarse_window_cols, target_timesteps)

        if size(fused_UQ_bias_window)[1:2] == (1, 1) && size(fused_UQ_bias_images_subset)[1:2] == (1, 1)
            fused_UQ_bias_images_subset[:, :, :] = fused_UQ_bias_window[:, :, :]
        else
            resampled_subset = resample(Raster(fused_UQ_bias_window, dims=(coarse_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_UQ_bias_images_subset)
            inds = get_missing_indices(fused_UQ_bias_images_subset)
            fused_UQ_bias_images_subset[inds] = resampled_subset[inds]
        end
        #time_putback[k] = time() - t3
    end

    if n_ensembles > 0
        # ugly workaround
        fused_condsim_target_images2 = Raster(reshape(fused_condsim_target_images, fine_rows, fine_cols, n_ensembles, target_timesteps), dims = (fine_images.dims[1:2]..., Z(1:n_ensembles),Band(1:target_timesteps)))
        result = DataFusionState(
            fused_target_images,
            fused_UQ_target_images,
            fused_coarse_bias_images,
            fused_UQ_coarse_bias_images,
            fused_condsim_target_images2
        )    
    else
        result = DataFusionState(
            fused_target_images,
            fused_UQ_target_images,
            fused_coarse_bias_images,
            fused_UQ_coarse_bias_images,
            nothing
        )
    end



    return result #, time_subset, time_fused, time_putback

end

"function to perform data fusion on pair of coarse and fine raster timeseries using a moving window and subsampling buffer"
function coarse_fine_data_fusion_SS(
        coarse_images::Raster,
        fine_images::Raster,
        cov_pars::Union{Raster,AbstractVector{Float64}};
        coarse_err_var::Union{Raster, Float64} = 1e-6,
        fine_err_var::Union{Raster, Float64} = 1e-6,
        prior::Union{DataFusionState, Nothing} = nothing,
        n_ensembles = 0,
        target_times = nothing, 
        smooth = true,              
        spatial_mod = "Matern",                   
        wt = [1, 0.5],   
        offset_ar = [0.98, 0.0], 
        offset_var = [1e-5, 1e-5],                        
        obs_operator = "uniform", 
        default_mean::Float64 = 0.3,
        default_SD::Float64 = 0.01,
        default_bias_mean::Float64 = 0.0,
        default_bias_SD::Float64 = 1e-5,
        buffer_distance::Union{Float64, Nothing} = 0.,
        prop_subsample::Float64 = 0.2,
        historic_tseries = nothing)::DataFusionState

    coarse_dims = coarse_images.dims[1:2]
    fine_dims = fine_images.dims[1:2]
    coarse_rows, coarse_cols, coarse_timesteps = size(coarse_images)
    coarse_shape = (coarse_rows, coarse_cols)
    fine_rows, fine_cols, fine_timesteps = size(fine_images)
    fine_shape = (fine_rows, fine_cols)

    if prior === nothing
        prior_fused_mean = Raster(default_mean .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_fused_sd = Raster(default_SD .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_bias_mean = Raster(default_bias_mean .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
        prior_bias_sd = Raster(default_bias_SD .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
    else
        prior_fused_mean = ifelse(length(prior.mean.dims) == 3, prior.mean[:, :, end], prior.mean)
        prior_fused_sd = ifelse(length(prior.SD.dims) == 3, prior.SD[:, :, end], prior.SD)
        prior_bias_mean = ifelse(length(prior.mean_bias.dims) == 3, prior.mean_bias[:, :, end], prior.mean_bias)
        prior_bias_sd = ifelse(length(prior.SD_bias.dims) == 3, prior.SD_bias[:, :, end], prior.SD_bias)
    end

    if target_times === nothing
        target_times = 1:coarse_timesteps
    end

    if isa(target_times, Vector{String})
        target_times = map(d -> Date(d), target_times)
    end

    target_timesteps = length(target_times)

    if isa(target_times, Vector{Date})
        time_slice = At(target_times)
        bias_dims = (coarse_dims[1:2]..., Ti(target_times))
        target_dims = (fine_dims[1:2]..., Ti(target_times))
    else
        time_slice = target_times
        #bias_dims = (coarse_dims[1:2]..., coarse_images.dims[3][target_times])
        #target_dims = (fine_dims[1:2]..., fine_images.dims[3][target_times])
        bias_dims = (coarse_dims[1:2]..., Band(1:coarse_timesteps)[target_times])
        target_dims = (fine_dims[1:2]..., Band(1:fine_timesteps)[target_times])
    end

    target_time_indices = target_times

    if isa(target_times, Vector{Date})
        target_time_indices = findall(t -> t in target_times, Array(fine_images.dims[3].val))

        if length(target_time_indices) == 0
            throw(error("no target times $(target_times) found in fine images $(Array(fine_images.dims[3].val))"))
        end
    end

    start_col = 1
    end_col = coarse_cols
    start_row = 1
    end_row = coarse_rows

    fine_cell_width, fine_cell_height = cell_size(fine_images)
    coarse_cell_width, coarse_cell_height = cell_size(coarse_images)

    if buffer_distance === nothing
        buffer_distance = coarse_cell_width
    end

    if spatial_mod == "Ind"
        buffer_distance = 0.0
    end

    resolutions = [[coarse_cell_width, -coarse_cell_height], [fine_cell_width, -fine_cell_height]]
    target_resolution = [fine_cell_width, -fine_cell_height]

    fused_target_images = Raster(fill(NaN, fine_rows, fine_cols, target_timesteps), dims=target_dims, missingval=fine_images.missingval)
    fused_UQ_target_images = Raster(fill(NaN, fine_rows, fine_cols, target_timesteps), dims=target_dims, missingval=fine_images.missingval)

    fused_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, target_timesteps), dims=bias_dims, missingval=coarse_images.missingval)
    fused_UQ_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, target_timesteps), dims=bias_dims, missingval=coarse_images.missingval)

    m1 = hcat(repeat(start_col:end_col, inner=[1], outer=[end_row]),
        repeat(start_row:end_row, inner=[end_col], outer=[1]))

    n = size(m1)[1] 
    
    fused_vectors_list = Vector{Matrix{Float64}}(undef, end_col*end_row)
    fused_UQ_vectors_list = Vector{Matrix{Float64}}(undef, end_col*end_row)
    fused_condsim_vectors_list = Vector{Union{Array{Float64}, Nothing}}(undef, end_col*end_row)

    # p = Progress(n)
    # update!(p, 0)
    # jj = Threads.Atomic{Int}(0)
    # l = Threads.SpinLock()
    #time_subset = zeros(n)
    #time_fused = zeros(n)
    #time_putback = zeros(n)

    #@showprogress 
    Threads.@threads for k in 1:n
    #for k in 1:n
        # Threads.atomic_add!(jj, 1)
        # Threads.lock(l)
        # update!(p, jj[])
        # Threads.unlock(l)

        #t1 = time()
        coarse_col, coarse_row = m1[k, :]
        # select the target bounding box along the edges of the center coarse pixel
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        # select the window bounding box by expanding the target bounding box by a given distance
        mask_bbox = buffer(target_bbox, buffer_distance)

        coarse_bbox = buffer(target_bbox, max(coarse_cell_height, coarse_cell_width))

        # select coarse pixels using the window bounding box
        coarse_window = view(coarse_images, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))

        window_bbox = buffer(BBox(coarse_window), fine_cell_width)

        # select prior mean bias pixels using the window bounding box
        bias_mean_window = view(prior_bias_mean, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))
                # select prior standard deviation bias using the window bounding box
        bias_sd_window = view(prior_bias_sd, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))
                # dimensions of the moving window on the coarse grid
        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
                # extract coarse pixel values
        coarse_window_values = Matrix(reshape(read(coarse_window), coarse_window_rows * coarse_window_cols, coarse_window_timesteps))
                # extract coarse mean bias values
        bias_mean_window_values = Matrix(reshape(read(bias_mean_window), coarse_window_rows * coarse_window_cols, 1))
                # extract coarse standard deviation values
        bias_sd_window_values = Matrix(reshape(read(bias_sd_window), coarse_window_rows * coarse_window_cols, 1))
                # select fine pixels using the window bounding box
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))

        maskk = Raster(fill(false, size(fine_window)[1:2]), dims = fine_window.dims[1:2])
        maskk[Rasters.Between(mask_bbox.xmin, mask_bbox.xmax),Rasters.Between(mask_bbox.ymin, mask_bbox.ymax)] .= true

        ## sobol sequence
        nr,nc = size(maskk)
        
        ns = Int64(round(prop_subsample*nr*nc))
        s = SobolSeq(2)
        p = reduce(hcat, Sobol.next!(s) for i = 1:ns)'
        p[:,1] = abs.(p[:,1] .- 1.)
        p[:,1] = Int64.(ceil.(p[:,1].*nc))
        p[:,2] = Int64.(ceil.(p[:,2].*nr))
        p2 = Int64.(p)

        for i in 1:size(p2)[1]
            maskk[p2[i,2], p2[i,1]] = true
        end

        ## srr
        #ss = sort(rand(findall(.~maskk[:]),(Int.(round(0.3*sum(.~maskk))),)))
        #maskk[ss] .= true

        # select fine prior mean values using the window bounding box
        prior_mean_window = view(prior_fused_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select fine prior standard deviation values using the window bounding box
        prior_sd_window = view(prior_fused_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # dimensions of the moving window on the fine grid
        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)
        # extract fine pixel values
        fine_window_values = Matrix(reshape(read(fine_window), fine_window_rows * fine_window_cols, fine_window_timesteps))

        mask_values = read(maskk)[:]

        # extract fine prior mean values
        prior_mean_window_values = Matrix(reshape(read(prior_mean_window), fine_window_rows * fine_window_cols, 1))
        # extract fine prior standard deviation values
        prior_sd_window_values = Matrix(reshape(read(prior_sd_window), fine_window_rows * fine_window_cols, 1))
        # get coarse x coordinates
        coarse_window_x_matrix = get_x_matrix(coarse_window)
        # get coarse y coordinates
        coarse_window_y_matrix = get_y_matrix(coarse_window)
        # combine coarse x and y coordinates
        coarse_window_coords = cat(vec(coarse_window_x_matrix), vec(coarse_window_y_matrix), dims=2)
        # get fine x coordinates
        fine_window_x_matrix = get_x_matrix(fine_window)
        # get fine y coordinates
        fine_window_y_matrix = get_y_matrix(fine_window)
        # combine fine x and y coordinates
        fine_window_coords = cat(vec(fine_window_x_matrix), vec(fine_window_y_matrix), dims=2)
        # set the target coordinates to the fine coordinates
        target_coords = fine_window_coords[mask_values,:]

        if isa(cov_pars, Raster)
            cov_pars_window = view(cov_pars, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
            cov_values = read(cov_pars_window)[:]
        else
            cov_values = cov_pars
        end

        if historic_tseries === nothing
            tseries_window_values = ones((1, 1))
        else
            tseries_window = view(historic_tseries, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
            tseries_window_values = Matrix{Float64}(reshape(read(tseries_window), fine_window_rows * fine_window_cols, size(historic_tseries)[3]))[mask_values,:]
        end

        # separate handling for time dimension if target times are a set of dates or indices
        if isa(target_times, Vector{Date})
            # assign time index dimension
            bias_window_dims = (coarse_window.dims[1], coarse_window.dims[2], Ti(target_times))
            target_window_dims = (fine_window.dims[1], fine_window.dims[2], Ti(target_times))
        else
            # assign enumerated index for time dimension
            bias_window_dims = (coarse_window.dims[1],coarse_window.dims[2], Band(1:coarse_timesteps)[target_times])
            target_window_dims = (fine_window.dims[1],fine_window.dims[2], Band(1:fine_timesteps)[target_times])
        end

        # assign measurement errors
        if isa(coarse_err_var, Raster)
            coarse_err_vars = read(view(coarse_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            coarse_err_vars = coarse_err_var .* ones(size(coarse_window_values))
        end

        if isa(fine_err_var, Raster)
            fine_err_vars = read(view(fine_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            fine_err_vars = fine_err_var .* ones(size(fine_window_values))
        end

        measurement_error_vars = [coarse_err_vars, fine_err_vars[mask_values,:] ]

        # combine fine and coarse input values
        measurements = [coarse_window_values, fine_window_values[mask_values,:]]
        # combine fine and coarse coordinates
        measurement_coords = [coarse_window_coords, fine_window_coords[mask_values,:]]
        
        prior_mean = vcat(bias_mean_window_values, prior_mean_window_values[mask_values,:])
        prior_sd = vcat(bias_sd_window_values, prior_sd_window_values[mask_values,:])
    
        #time_subset[k] = time() - t1

        #t2 = time()
        ## run offset+MRF model filtering/smoothing over full time period
        fused_vectors_list[k], fused_UQ_vectors_list[k], fused_condsim_vectors_list[k] = STARS_fusion(
            measurements,                                                                                
            measurement_error_vars,
            measurement_coords,
            resolutions,
            target_coords,
            target_resolution,
            prior_mean,
            prior_sd,
            ensembles = nothing,
            target_times = target_time_indices, 
            smooth = smooth,  
            spatial_mod = spatial_mod,
            cov_pars = cov_values, 
            wt = wt,
            offset_ar = offset_ar,
            offset_var = offset_var,
            obs_operator = obs_operator,
            historic_tseries = tseries_window_values
        )
        #time_fused[k] = time() - t2
    end

    println("putting back into raster")
    
    ### put back into rasters
    for k in 1:n
        #t3 = time()
        coarse_col, coarse_row = m1[k, :]

        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        # select the window bounding box by expanding the target bounding box by a given distance
        mask_bbox = buffer(target_bbox, buffer_distance)

        coarse_bbox = buffer(target_bbox, max(coarse_cell_height, coarse_cell_width))

        # select coarse pixels using the window bounding box
        coarse_window = view(coarse_images, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))

        window_bbox = buffer(BBox(coarse_window), fine_cell_width)

        # select fine pixels using the window bounding box
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))

        maskk = Raster(fill(false, size(fine_window)[1:2]), dims = fine_window.dims[1:2])
        maskk[Rasters.Between(mask_bbox.xmin, mask_bbox.xmax),Rasters.Between(mask_bbox.ymin, mask_bbox.ymax)] .= true

        ## sobol sequence
        nr,nc = size(maskk)
        ns = Int64(round(prop_subsample*nr*nc))
        s = SobolSeq(2)
        p = reduce(hcat, Sobol.next!(s) for i = 1:ns)'
        p[:,1] = abs.(p[:,1] .- 1.)
        p[:,1] = Int64.(ceil.(p[:,1].*nc))
        p[:,2] = Int64.(ceil.(p[:,2].*nr))
        p2 = Int64.(p)

        for i in 1:size(p2)[1]
            maskk[p2[i,2], p2[i,1]] = true
        end

        mask_values = read(maskk)[:]
        
        # dimensions of the moving window on the coarse grid
        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
        # dimensions of the moving window on the fine grid
        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)

        # separate handling for time dimension if target times are a set of dates or indices
        if isa(target_times, Vector{Date})
            # assign time index dimension
            bias_window_dims = (coarse_window.dims[1], coarse_window.dims[2], Ti(target_times))
            target_window_dims = (fine_window.dims[1], fine_window.dims[2], Ti(target_times))
        else
            # assign enumerated index for time dimension
            bias_window_dims = (coarse_window.dims[1],coarse_window.dims[2], Band(1:coarse_timesteps)[target_times])
            target_window_dims = (fine_window.dims[1],fine_window.dims[2], Band(1:fine_timesteps)[target_times])
        end

        fused_fine_window = zeros(fine_window_rows*fine_window_cols,target_timesteps)
        fused_UQ_fine_window = zeros(fine_window_rows*fine_window_cols,target_timesteps)
        fused_ensemble_window = zeros(fine_window_rows*fine_window_cols,n_ensembles*target_timesteps)

        fused_target_images_subset = view(fused_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        subset_fine_window = replace(fused_vectors_list[k], missing => fine_window.missingval)[(end-sum(mask_values)[1]+1):end, :]
        fused_fine_window[mask_values,:] = subset_fine_window
        resampled_subset = resample(Raster(reshape(fused_fine_window,fine_window_rows, fine_window_cols, target_timesteps), dims=(fine_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_target_images_subset)
        inds = get_missing_indices(fused_target_images_subset)
        fused_target_images_subset[inds] = resampled_subset[inds]

        fused_UQ_target_images_subset = view(fused_UQ_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))

        subset_UQ_fine_window = replace(fused_UQ_vectors_list[k], missing => fine_window.missingval)[(end-sum(mask_values)[1]+1):end, :]
        fused_UQ_fine_window[mask_values,:] = subset_UQ_fine_window

        resampled_subset = resample(Raster(reshape(fused_UQ_fine_window, fine_window_rows, fine_window_cols, target_timesteps), dims=(fine_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_UQ_target_images_subset)
        inds = get_missing_indices(fused_UQ_target_images_subset)
        fused_UQ_target_images_subset[inds] = resampled_subset[inds]

        fused_bias_images_subset = view(fused_coarse_bias_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_bias_window = reshape(replace(fused_vectors_list[k], missing => coarse_window.missingval)[1:(end-sum(mask_values)[1]), :], coarse_window_rows, coarse_window_cols, target_timesteps)

        if size(fused_bias_window)[1:2] == (1, 1) && size(fused_bias_images_subset)[1:2] == (1, 1)
            fused_bias_images_subset[:, :, :] = fused_bias_window[:, :, :]
        else
            resampled_subset = resample(Raster(fused_bias_window, dims=(coarse_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_bias_images_subset)
            inds = get_missing_indices(fused_bias_images_subset)
            fused_bias_images_subset[inds] = resampled_subset[inds]
        end

        fused_UQ_bias_images_subset = view(fused_UQ_coarse_bias_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_UQ_bias_window = reshape(replace(fused_UQ_vectors_list[k], missing => coarse_window.missingval)[1:(end-sum(mask_values)[1]), :], coarse_window_rows, coarse_window_cols, target_timesteps)

        if size(fused_UQ_bias_window)[1:2] == (1, 1) && size(fused_UQ_bias_images_subset)[1:2] == (1, 1)
            fused_UQ_bias_images_subset[:, :, :] = fused_UQ_bias_window[:, :, :]
        else
            resampled_subset = resample(Raster(fused_UQ_bias_window, dims=(coarse_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_UQ_bias_images_subset)
            inds = get_missing_indices(fused_UQ_bias_images_subset)
            fused_UQ_bias_images_subset[inds] = resampled_subset[inds]
        end
        #time_putback[k] = time() - t3
    end

    
    result = DataFusionState(
            fused_target_images,
            fused_UQ_target_images,
            fused_coarse_bias_images,
            fused_UQ_coarse_bias_images,
            nothing)
    

    return result #, time_subset, time_fused, time_putback

end

"function to perform data fusion on pair of coarse and fine raster timeseries using a moving window and subsampling buffer, without parallelization"
function coarse_fine_data_fusion_SS_seq(
        coarse_images::Raster,
        fine_images::Raster,
        cov_pars::Union{Raster,AbstractVector{Float64}};
        coarse_err_var::Union{Raster, Float64} = 1e-6,
        fine_err_var::Union{Raster, Float64} = 1e-6,
        prior::Union{DataFusionState, Nothing} = nothing,
        n_ensembles = 0,
        target_times = nothing, 
        smooth = true,              
        spatial_mod = "Matern",                   
        wt = [1, 0.5],   
        offset_ar = [0.98, 0.0], 
        offset_var = [1e-5, 1e-5],                        
        obs_operator = "uniform", 
        default_mean::Float64 = 0.3,
        default_SD::Float64 = 0.01,
        default_bias_mean::Float64 = 0.0,
        default_bias_SD::Float64 = 1e-5,
        buffer_distance::Union{Float64, Nothing} = 0.,
        prop_subsample::Float64 = 0.2,
        historic_tseries = nothing)::DataFusionState

    coarse_dims = coarse_images.dims[1:2]
    fine_dims = fine_images.dims[1:2]
    coarse_rows, coarse_cols, coarse_timesteps = size(coarse_images)
    coarse_shape = (coarse_rows, coarse_cols)
    fine_rows, fine_cols, fine_timesteps = size(fine_images)
    fine_shape = (fine_rows, fine_cols)

    if prior === nothing
        prior_fused_mean = Raster(default_mean .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_fused_sd = Raster(default_SD .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_bias_mean = Raster(default_bias_mean .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
        prior_bias_sd = Raster(default_bias_SD .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
    else
        prior_fused_mean = ifelse(length(prior.mean.dims) == 3, prior.mean[:, :, end], prior.mean)
        prior_fused_sd = ifelse(length(prior.SD.dims) == 3, prior.SD[:, :, end], prior.SD)
        prior_bias_mean = ifelse(length(prior.mean_bias.dims) == 3, prior.mean_bias[:, :, end], prior.mean_bias)
        prior_bias_sd = ifelse(length(prior.SD_bias.dims) == 3, prior.SD_bias[:, :, end], prior.SD_bias)
    end

    if target_times === nothing
        target_times = 1:coarse_timesteps
    end

    if isa(target_times, Vector{String})
        target_times = map(d -> Date(d), target_times)
    end

    target_timesteps = length(target_times)

    if isa(target_times, Vector{Date})
        time_slice = At(target_times)
        bias_dims = (coarse_dims[1:2]..., Ti(target_times))
        target_dims = (fine_dims[1:2]..., Ti(target_times))
    else
        time_slice = target_times
        #bias_dims = (coarse_dims[1:2]..., coarse_images.dims[3][target_times])
        #target_dims = (fine_dims[1:2]..., fine_images.dims[3][target_times])
        bias_dims = (coarse_dims[1:2]..., Band(1:coarse_timesteps)[target_times])
        target_dims = (fine_dims[1:2]..., Band(1:fine_timesteps)[target_times])
    end

    target_time_indices = target_times

    if isa(target_times, Vector{Date})
        target_time_indices = findall(t -> t in target_times, Array(fine_images.dims[3].val))

        if length(target_time_indices) == 0
            throw(error("no target times $(target_times) found in fine images $(Array(fine_images.dims[3].val))"))
        end
    end

    start_col = 1
    end_col = coarse_cols
    start_row = 1
    end_row = coarse_rows

    fine_cell_width, fine_cell_height = cell_size(fine_images)
    coarse_cell_width, coarse_cell_height = cell_size(coarse_images)

    if buffer_distance === nothing
        buffer_distance = coarse_cell_width
    end

    if spatial_mod == "Ind"
        buffer_distance = 0.0
    end

    resolutions = [[coarse_cell_width, -coarse_cell_height], [fine_cell_width, -fine_cell_height]]
    target_resolution = [fine_cell_width, -fine_cell_height]

    fused_target_images = Raster(fill(NaN, fine_rows, fine_cols, target_timesteps), dims=target_dims, missingval=fine_images.missingval)
    fused_UQ_target_images = Raster(fill(NaN, fine_rows, fine_cols, target_timesteps), dims=target_dims, missingval=fine_images.missingval)

    fused_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, target_timesteps), dims=bias_dims, missingval=coarse_images.missingval)
    fused_UQ_coarse_bias_images = Raster(fill(NaN, coarse_rows, coarse_cols, target_timesteps), dims=bias_dims, missingval=coarse_images.missingval)

    m1 = hcat(repeat(start_col:end_col, inner=[1], outer=[end_row]),
        repeat(start_row:end_row, inner=[end_col], outer=[1]))

    n = size(m1)[1] 
    
    #fused_vectors_list = Vector{Matrix{Float64}}(undef, end_col*end_row)
    #fused_UQ_vectors_list = Vector{Matrix{Float64}}(undef, end_col*end_row)
    #fused_condsim_vectors_list = Vector{Union{Array{Float64}, Nothing}}(undef, end_col*end_row)

    # p = Progress(n)
    # update!(p, 0)
    # jj = Threads.Atomic{Int}(0)
    # l = Threads.SpinLock()
    
    # time_subset = zeros(n)
    # time_fused = zeros(n)
    # time_putback = zeros(n)
    # num_fine = zeros(n)
    # num_coarse = zeros(n)

    #@showprogress 
    for k in 1:n
    #for k in 1:n
        # Threads.atomic_add!(jj, 1)
        # Threads.lock(l)
        # update!(p, jj[])
        # Threads.unlock(l)
        #t1 = time()
        coarse_col, coarse_row = m1[k, :]
        # select the target bounding box along the edges of the center coarse pixel
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        # select the window bounding box by expanding the target bounding box by a given distance
        mask_bbox = buffer(target_bbox, buffer_distance)

        coarse_bbox = buffer(target_bbox, max(coarse_cell_height, coarse_cell_width))

        # select coarse pixels using the window bounding box
        coarse_window = view(coarse_images, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))

        window_bbox = buffer(BBox(coarse_window), fine_cell_width)

        # select prior mean bias pixels using the window bounding box
        bias_mean_window = view(prior_bias_mean, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))
                # select prior standard deviation bias using the window bounding box
        bias_sd_window = view(prior_bias_sd, X(Rasters.Between(coarse_bbox.xmin,coarse_bbox.xmax)), Y(Rasters.Between(coarse_bbox.ymin, coarse_bbox.ymax)))
                # dimensions of the moving window on the coarse grid
        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
                # extract coarse pixel values
        coarse_window_values = Matrix(reshape(read(coarse_window), coarse_window_rows * coarse_window_cols, coarse_window_timesteps))
                # extract coarse mean bias values
        bias_mean_window_values = Matrix(reshape(read(bias_mean_window), coarse_window_rows * coarse_window_cols, 1))
                # extract coarse standard deviation values
        bias_sd_window_values = Matrix(reshape(read(bias_sd_window), coarse_window_rows * coarse_window_cols, 1))
                # select fine pixels using the window bounding box
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))

        maskk = Raster(fill(false, size(fine_window)[1:2]), dims = fine_window.dims[1:2])
        maskk[Rasters.Between(mask_bbox.xmin, mask_bbox.xmax),Rasters.Between(mask_bbox.ymin, mask_bbox.ymax)] .= true

        ## sobol sequence
        nr,nc = size(maskk)
        
        ns = Int64(round(prop_subsample*nr*nc))
        s = SobolSeq(2)
        p = reduce(hcat, Sobol.next!(s) for i = 1:ns)'
        p[:,1] = abs.(p[:,1] .- 1.)
        p[:,1] = Int64.(ceil.(p[:,1].*nc))
        p[:,2] = Int64.(ceil.(p[:,2].*nr))
        p2 = Int64.(p)

        for i in 1:size(p2)[1]
            maskk[p2[i,2], p2[i,1]] = true
        end

        ## srr
        #ss = sort(rand(findall(.~maskk[:]),(Int.(round(0.3*sum(.~maskk))),)))
        #maskk[ss] .= true

        # select fine prior mean values using the window bounding box
        prior_mean_window = view(prior_fused_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select fine prior standard deviation values using the window bounding box
        prior_sd_window = view(prior_fused_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # dimensions of the moving window on the fine grid
        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)
        # extract fine pixel values
        fine_window_values = Matrix(reshape(read(fine_window), fine_window_rows * fine_window_cols, fine_window_timesteps))

        mask_values = read(maskk)[:]

        # extract fine prior mean values
        prior_mean_window_values = Matrix(reshape(read(prior_mean_window), fine_window_rows * fine_window_cols, 1))
        # extract fine prior standard deviation values
        prior_sd_window_values = Matrix(reshape(read(prior_sd_window), fine_window_rows * fine_window_cols, 1))
        # get coarse x coordinates
        coarse_window_x_matrix = get_x_matrix(coarse_window)
        # get coarse y coordinates
        coarse_window_y_matrix = get_y_matrix(coarse_window)
        # combine coarse x and y coordinates
        coarse_window_coords = cat(vec(coarse_window_x_matrix), vec(coarse_window_y_matrix), dims=2)
        # get fine x coordinates
        fine_window_x_matrix = get_x_matrix(fine_window)
        # get fine y coordinates
        fine_window_y_matrix = get_y_matrix(fine_window)
        # combine fine x and y coordinates
        fine_window_coords = cat(vec(fine_window_x_matrix), vec(fine_window_y_matrix), dims=2)
        # set the target coordinates to the fine coordinates
        target_coords = fine_window_coords[mask_values,:]

        if isa(cov_pars, Raster)
            cov_pars_window = view(cov_pars, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
            cov_values = read(cov_pars_window)[:]
        else
            cov_values = cov_pars
        end

        if historic_tseries === nothing
            tseries_window_values = ones((1, 1))
        else
            tseries_window = view(historic_tseries, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
            tseries_window_values = Matrix{Float64}(reshape(read(tseries_window), fine_window_rows * fine_window_cols, size(historic_tseries)[3]))[mask_values,:]
        end

        # separate handling for time dimension if target times are a set of dates or indices
        if isa(target_times, Vector{Date})
            # assign time index dimension
            bias_window_dims = (coarse_window.dims[1], coarse_window.dims[2], Ti(target_times))
            target_window_dims = (fine_window.dims[1], fine_window.dims[2], Ti(target_times))
        else
            # assign enumerated index for time dimension
            bias_window_dims = (coarse_window.dims[1],coarse_window.dims[2], Band(1:coarse_timesteps)[target_times])
            target_window_dims = (fine_window.dims[1],fine_window.dims[2], Band(1:fine_timesteps)[target_times])
        end

        # assign measurement errors
        if isa(coarse_err_var, Raster)
            coarse_err_vars = read(view(coarse_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            coarse_err_vars = coarse_err_var .* ones(size(coarse_window_values))
        end

        if isa(fine_err_var, Raster)
            fine_err_vars = read(view(fine_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            fine_err_vars = fine_err_var .* ones(size(fine_window_values))
        end

        measurement_error_vars = [coarse_err_vars, fine_err_vars[mask_values,:] ]

        # combine fine and coarse input values
        measurements = [coarse_window_values, fine_window_values[mask_values,:]]
        # combine fine and coarse coordinates
        measurement_coords = [coarse_window_coords, fine_window_coords[mask_values,:]]
        
        prior_mean = vcat(bias_mean_window_values, prior_mean_window_values[mask_values,:])
        prior_sd = vcat(bias_sd_window_values, prior_sd_window_values[mask_values,:])

        #time_subset[k] = time() - t1

        #num_fine[k] = size(measurements[2])[1]
        #num_coarse[k]  = size(measurements[1])[1]

        #t2 = time()
        ## run offset+MRF model filtering/smoothing over full time period
        fused_vectors_list, fused_UQ_vectors_list, fused_condsim_vectors_list = STARS_fusion(
            measurements,                                                                                
            measurement_error_vars,
            measurement_coords,
            resolutions,
            target_coords,
            target_resolution,
            prior_mean,
            prior_sd,
            ensembles = nothing,
            target_times = target_time_indices, 
            smooth = smooth,  
            spatial_mod = spatial_mod,
            cov_pars = cov_values, 
            wt = wt,
            offset_ar = offset_ar,
            offset_var = offset_var,
            obs_operator = obs_operator,
            historic_tseries = tseries_window_values
        )

        #time_fused[k] = time() - t2

        #t3 = time()

         # separate handling for time dimension if target times are a set of dates or indices
         if isa(target_times, Vector{Date})
            # assign time index dimension
            bias_window_dims = (coarse_window.dims[1], coarse_window.dims[2], Ti(target_times))
            target_window_dims = (fine_window.dims[1], fine_window.dims[2], Ti(target_times))
        else
            # assign enumerated index for time dimension
            bias_window_dims = (coarse_window.dims[1],coarse_window.dims[2], Band(1:coarse_timesteps)[target_times])
            target_window_dims = (fine_window.dims[1],fine_window.dims[2], Band(1:fine_timesteps)[target_times])
        end

        fused_fine_window = zeros(fine_window_rows*fine_window_cols,target_timesteps)
        fused_UQ_fine_window = zeros(fine_window_rows*fine_window_cols,target_timesteps)
        fused_ensemble_window = zeros(fine_window_rows*fine_window_cols,n_ensembles*target_timesteps)

        fused_target_images_subset = view(fused_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        subset_fine_window = replace(fused_vectors_list, missing => fine_window.missingval)[(end-sum(mask_values)[1]+1):end, :]
        fused_fine_window[mask_values,:] = subset_fine_window
        resampled_subset = resample(Raster(reshape(fused_fine_window,fine_window_rows, fine_window_cols, target_timesteps), dims=(fine_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_target_images_subset)
        inds = get_missing_indices(fused_target_images_subset)
        fused_target_images_subset[inds] = resampled_subset[inds]

        fused_UQ_target_images_subset = view(fused_UQ_target_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))

        subset_UQ_fine_window = replace(fused_UQ_vectors_list, missing => fine_window.missingval)[(end-sum(mask_values)[1]+1):end, :]
        fused_UQ_fine_window[mask_values,:] = subset_UQ_fine_window

        resampled_subset = resample(Raster(reshape(fused_UQ_fine_window, fine_window_rows, fine_window_cols, target_timesteps), dims=(fine_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_UQ_target_images_subset)
        inds = get_missing_indices(fused_UQ_target_images_subset)
        fused_UQ_target_images_subset[inds] = resampled_subset[inds]

        fused_bias_images_subset = view(fused_coarse_bias_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_bias_window = reshape(replace(fused_vectors_list, missing => coarse_window.missingval)[1:(end-sum(mask_values)[1]), :], coarse_window_rows, coarse_window_cols, target_timesteps)

        if size(fused_bias_window)[1:2] == (1, 1) && size(fused_bias_images_subset)[1:2] == (1, 1)
            fused_bias_images_subset[:, :, :] = fused_bias_window[:, :, :]
        else
            resampled_subset = resample(Raster(fused_bias_window, dims=(coarse_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_bias_images_subset)
            inds = get_missing_indices(fused_bias_images_subset)
            fused_bias_images_subset[inds] = resampled_subset[inds]
        end

        fused_UQ_bias_images_subset = view(fused_UQ_coarse_bias_images, X(Rasters.Between(target_bbox.xmin, target_bbox.xmax)), Y(Rasters.Between(target_bbox.ymin, target_bbox.ymax)))
        fused_UQ_bias_window = reshape(replace(fused_UQ_vectors_list, missing => coarse_window.missingval)[1:(end-sum(mask_values)[1]), :], coarse_window_rows, coarse_window_cols, target_timesteps)

        if size(fused_UQ_bias_window)[1:2] == (1, 1) && size(fused_UQ_bias_images_subset)[1:2] == (1, 1)
            fused_UQ_bias_images_subset[:, :, :] = fused_UQ_bias_window[:, :, :]
        else
            resampled_subset = resample(Raster(fused_UQ_bias_window, dims=(coarse_window.dims[1:2]..., Band(1:target_timesteps))), to=fused_UQ_bias_images_subset)
            inds = get_missing_indices(fused_UQ_bias_images_subset)
            fused_UQ_bias_images_subset[inds] = resampled_subset[inds]
        end

        #time_putback[k] = time() - t3
    end

    result = DataFusionState(
            fused_target_images,
            fused_UQ_target_images,
            fused_coarse_bias_images,
            fused_UQ_coarse_bias_images,
            nothing)
    

    return result #, time_subset, time_fused, time_putback, num_fine, num_coarse

end

"evaluate the neg loglikelihood for Matern-based covariance models"
function nll_evaluate(ys, err_vars, 
        H, F, Q, x0, P0,
        target_coords,
        target_times;      
        spatial_mod = "Matern",                   
        cov_pars = [0.002, 150, 1e-10, 0.5],
        historic_tseries::Union{AbstractArray{Float64}, Nothing} = nothing)

    p = size(ys)[2]
    Qb = copy(Q)
    nfs = size(target_coords)[1]
    nq = size(x0)[1]

    st = nq - nfs + 1

    if spatial_mod == "MRF"
       # nbs, nn = get_fine_neighbs2(target_coords)
       # Qb[st:end,st:end] .= build_CAR_var(nbs, nn, cov_pars[1], cov_pars[2]) # spatial innovation covariance
        error("estimation is not yet implemented for spatial_mod = MRF")
    elseif spatial_mod == "SqExp"
        Qb[st:end,st:end] .= build_GP_var(target_coords, cov_pars[1], cov_pars[2], nugget=cov_pars[3])
    elseif spatial_mod == "Matern"
        Qb[st:end,st:end] .= cov_pars[1].*matern_cor(transpose(target_coords), reg = cov_pars[3], ν=cov_pars[4], σ = cov_pars[2])
    elseif spatial_mod == "TS"
        Qb[st:end,st:end] .= cov_pars[1].*kernel_matrix(historic_tseries, reg = cov_pars[3], σ = cov_pars[2])
    else
        error("spatial_mod must be SqExp, Matern, or TS")
    end


    nll = 0.
    filtering_means = [x0] # these will be T + 1 long
    filtering_covs = [P0]

    for t ∈ 1:p
        x_pred = F * filtering_means[t] # filtering_means[1], covs[1] is prior mean
        P_pred = F * filtering_covs[t] * F' + Qb

        #ids = fill(false, sum(nnobs))

        ym = .!ismissing.(ys[:,t])

        if sum(ym) > 0

            # subset out missing values
            R = @views Diagonal(err_vars[ym,t])
            y = @views ys[ym,t] 
            Ht = H[ym,:] 

            res_pred = y - Ht * copy(x_pred) # innovation

            HPpT = P_pred * Ht'

            S = Ht * HPpT + R # innovation covariance

            if t ∈ target_times
                nll += (0.5*logdet(Symmetric(S)) + (0.5 * res_pred' * (cholesky(Symmetric(S)) \ res_pred)))
            end

            # Kalman gain; K = P_pred * H' * inv(S)
            begin
                LAPACK.potrf!('U', S)
                LAPACK.potri!('U', S)
                K = BLAS.symm('R', 'U', S, HPpT)
            end

            # With K
            x_new = x_pred + K * res_pred # filtering distribution mean
            P_new = P_pred - K * (Ht * P_pred) # filtering distribution covariance

            push!(filtering_means, x_new)
            push!(filtering_covs, P_new)

            #nll += (0.5*logdet(Symmetric(S)) + 0.5 * res_pred' * (cholesky(Symmetric(S)) \ res_pred))
        else
            push!(filtering_means, x_pred)
            push!(filtering_covs, P_pred)
        end
        
    end

    return nll
end

"minimize the neg loglikelihood for Matern-based covariance models"
function nll_minimize(
        measurements, 
        measurement_error_vars,                                 
        measurement_coords, 
        target_times,
        resolutions,                   
        target_coords,
        target_resolution,
        prior_mean,
        prior_sd;      
        spatial_mod = "Matern",                   
        cov_pars = [0.002, 150, 1e-10, 0.5], 
        wt = [1,0.5],   
        offset_ar = [0.95, 0.0], 
        offset_var = [0.0001,0.0001],                        
        obs_operator = "uniform", 
        historic_tseries = ones(1,1))
        
                            
    # measurements: List of n_i x T matrices of instrument measurements, columns are flattened vector of pixels within window at each time point
    # measurement_error_vars: List of n_i x T matrices of measurement error variances, columns are flattened vector of variances within window at each time point
    # measurement_coords: List of n_i x 2 matrices of x,y centroids of instrument grids, assumes (col,row) format
    # resolutions: List of 2-dim vector of instrument grid resolution 
    # target_coords: nf x 2 matrix of x,y centroids of target grid within window
    # prior_mean: vector of prior mean at time t=0
    # prior_sd: vector of prior sd at time t=0
    # cov_pars: vector of spatial covariance parameters [spatial var, length scale or MRF dependence, nugget]
    # offset_ar: vector AR(1) correlation parameter for each sensor, value of 0 means no bias
    # offset_var: innovation variance parameter for bias terms
    # obs_operator: "uniform" or "gaussian" specifiying weights of observation operator

    ni = size(measurements)[1] # number of instruments
    nnobs = [size(measurements[i])[1] for i in 1:ni] # number of grid cells per instrument
    nf = size(target_coords)[1] # number of target resolution grid cells
    T = size(measurements[end])[2] # number of time steps

    ## build observation operator, stack observations and variances 
    H = Array{Float64}(undef, 0, nf)
    ys = Array{Float64}(undef, 0, T)
    err_vars = Array{Float64}(undef, 0, T)

    for i in 1:ni
        if obs_operator == "uniform"
            H1 = unif_weighted_obs_operator(measurement_coords[i], target_coords, resolutions[i], target_resolution)
        elseif obs_operator == "gaussian"
            H1 = gauss_weighted_obs_operator(measurement_coords[i], target_coords, resolutions[i], p=wt[i], scale=1)
        else
            error("operator must be either uniform or gaussian")
        end

        H = vcat(H,H1)
        ys = vcat(ys, measurements[i])
        err_vars = vcat(err_vars, measurement_error_vars[i])
    end

    ## build spatial models
    if spatial_mod == "MRF"
        #nbs, nn = get_fine_neighbs2(target_coords)
        #Q = build_CAR_var(nbs, nn, cov_pars[1], cov_pars[2]) # spatial innovation covariance
        error("estimation is not yet implemented for spatial_mod = MRF")
    elseif spatial_mod == "SqExp"
        Q = build_GP_var(target_coords, cov_pars[1], cov_pars[2], nugget=cov_pars[3])
    elseif spatial_mod == "Matern"
        Q = cov_pars[1].*matern_cor(transpose(target_coords), reg = cov_pars[3], ν=cov_pars[4], σ = cov_pars[2])
    elseif spatial_mod == "TS"
        Q = cov_pars[1].*kernel_matrix(historic_tseries, reg = cov_pars[3], σ = cov_pars[2])
    else
        error("spatial_mod must be MRF, SqExp, Matern, or TS")
    end

    ## add additive biases and set-up remaining model matrices
    nfs = 0
    add_bias = offset_ar .> 0 # if offset_ar is greater than 0 add bias for that instrument

    if any(add_bias)
        nfs = sum(nnobs[add_bias])
        ar_vals = vcat(fill.(offset_ar[add_bias], nnobs[add_bias])...)
        ar_vars = vcat(fill.(offset_var[add_bias], nnobs[add_bias])...)
        Hb = zeros((sum(nnobs),sum(nnobs[add_bias])))
        Hb[vcat(fill.(add_bias, nnobs)...),:] = Matrix(I, nfs, nfs)
        H = hcat(Hb,H)

        # dynamics
        F = Φ = Diagonal(vcat(ar_vals, ones(nf))) # identity transition matrix (random walk)
        Q = Matrix(BlockDiagonal([ diagm(ar_vars), Q])) # spatial innovation covariance
    else

        F = Φ = UniformScaling(1)
        
    end

    x0 = prior_mean[:] # don't need this but here to help with synergizing code later
    P0 = Matrix(Diagonal(prior_sd[:].^2)) # just assuming diagonal C0

    if size(P0) != size(Q)
        error("Dimension of prior_mean/prior_sd doesn't match model dimension. Check if prior includes bias elements.")
    end

    H = Matrix(H)    

    pars1 = log.(cov_pars[1:2])
    
    g(pars::Vector) = nll_evaluate(ys, err_vars,
                                    H, F, Q, x0, P0,
                                    target_coords,
                                    target_times, 
                                    spatial_mod = spatial_mod,
                                    cov_pars = vcat(exp.(pars), cov_pars[3:end]),
                                    historic_tseries = historic_tseries)

    test1 = optimize(g, pars1, NelderMead(), Optim.Options(iterations=50, g_tol=3))

    result = vcat(exp.(Optim.minimizer(test1)),Optim.converged(test1))

    return result
end

"estimate parameters for a subsample of coarse pixels"
function MLE_estimation(
        coarse_images::Raster,
        fine_images::Raster;
        coarse_err_var::Union{Raster, Float64} = 1e-6,
        fine_err_var::Union{Raster, Float64} = 1e-6,
        subsample_inds::Union{AbstractArray{Int64}, Nothing} = nothing,
        prior::Union{DataFusionState, Nothing} = nothing,
        target_times = nothing,             
        spatial_mod = "Matern",  
        cov_pars = [0.001, 120, 1e-10, 1.5],                 
        wt = [1, 0.5],   
        offset_ar = [0.98, 0.0], 
        offset_var = [1e-5, 1e-5],                        
        obs_operator = "uniform", 
        default_mean::Float64 = 0.3,
        default_SD::Float64 = 0.01,
        default_bias_mean::Float64 = 0.0,
        default_bias_SD::Float64 = 1e-5,
        buffer_distance::Union{Float64, Nothing} = nothing,
        historic_tseries::Union{AbstractArray{Float64}, Nothing} = nothing)
        
    coarse_dims = coarse_images.dims[1:2]
    fine_dims = fine_images.dims[1:2]
    coarse_rows, coarse_cols, coarse_timesteps = size(coarse_images)
    coarse_shape = (coarse_rows, coarse_cols)
    fine_rows, fine_cols, fine_timesteps = size(fine_images)
    fine_shape = (fine_rows, fine_cols)

    if prior === nothing
        prior_fused_mean = Raster(default_mean .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_fused_sd = Raster(default_SD .* ones(fine_shape), dims=fine_dims, missingval=fine_images.missingval)
        prior_bias_mean = Raster(default_bias_mean .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
        prior_bias_sd = Raster(default_bias_SD .* ones(coarse_shape), dims=coarse_dims, missingval=coarse_images.missingval)
    else
        prior_fused_mean = ifelse(length(prior.mean.dims) == 3, prior.mean[:, :, end], prior.mean)
        prior_fused_sd = ifelse(length(prior.SD.dims) == 3, prior.SD[:, :, end], prior.SD)
        prior_bias_mean = ifelse(length(prior.mean_bias.dims) == 3, prior.mean_bias[:, :, end], prior.mean_bias)
        prior_bias_sd = ifelse(length(prior.SD_bias.dims) == 3, prior.SD_bias[:, :, end], prior.SD_bias)
    end

    if target_times === nothing
        target_times = 1:coarse_timesteps
    end

    fine_cell_width, fine_cell_height = cell_size(fine_images)
    coarse_cell_width, coarse_cell_height = cell_size(coarse_images)

    if buffer_distance === nothing
        buffer_distance = coarse_cell_width
    end

    resolutions = [[coarse_cell_width, -coarse_cell_height], [fine_cell_width, -fine_cell_height]]
    target_resolution = [fine_cell_width, -fine_cell_height]
    
    #p = Threads.nthreads()
    #println("Parallelizing over "*string(p)*" threads...")

    if subsample_inds === nothing
        subsample_inds = hcat(repeat(1:coarse_cols,inner=[1],outer=[coarse_rows]),
                repeat(1:coarse_rows,inner=[coarse_cols],outer=[1]))
    end
        
    n = size(subsample_inds)[1]

    result = Matrix{Float64}(0.0I, n, 3) 

    ## using rasters
    #result = Raster(fill(NaN, coarse_rows, coarse_cols, 1:3), dims=(coarse_dims..., Band(1:3)), missingval=fine_images.missingval)
    
    #@showprogress 
    p = Progress(n)
    Threads.@threads for k in 1:n
        
        coarse_col, coarse_row = subsample_inds[k,:]
        
        # select the target bounding box along the edges of the center coarse pixel
        target_bbox = buffer(BBox(coarse_images[coarse_row:coarse_row, coarse_col:coarse_col, :]), fine_cell_width)
        # select the window bounding box by expanding the target bounding box by a given distance
        window_bbox = buffer(target_bbox, buffer_distance)
        # select coarse pixels using the window bounding box
        coarse_window = view(coarse_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select prior mean bias pixels using the window bounding box
        bias_mean_window = view(prior_bias_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select prior standard deviation bias using the window bounding box
        bias_sd_window = view(prior_bias_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # dimensions of the moving window on the coarse grid
        coarse_window_rows, coarse_window_cols, coarse_window_timesteps = size(coarse_window)
        # extract coarse pixel values
        coarse_window_values = Matrix(reshape(replace(read(coarse_window), coarse_window.missingval => missing), coarse_window_rows * coarse_window_cols, coarse_window_timesteps))
        # extract coarse mean bias values
        bias_mean_window_values = Matrix(reshape(replace(read(bias_mean_window), coarse_window.missingval => missing), coarse_window_rows * coarse_window_cols, 1))
        # extract coarse standard deviation values
        bias_sd_window_values = Matrix(reshape(replace(read(bias_sd_window), coarse_window.missingval => missing), coarse_window_rows * coarse_window_cols, 1))
        # select fine pixels using the window bounding box
        fine_window = view(fine_images, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select fine prior mean values using the window bounding box
        prior_mean_window = view(prior_fused_mean, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # select fine prior standard deviation values using the window bounding box
        prior_sd_window = view(prior_fused_sd, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
        # dimensions of the moving window on the fine grid
        fine_window_rows, fine_window_cols, fine_window_timesteps = size(fine_window)
        # extract fine pixel values
        fine_window_values = Matrix(reshape(replace(read(fine_window), fine_window.missingval => missing), fine_window_rows * fine_window_cols, fine_window_timesteps))
        # extract fine prior mean values
        prior_mean_window_values = Matrix(reshape(replace(read(prior_mean_window), fine_window.missingval => missing), fine_window_rows * fine_window_cols, 1))
        # extract fine prior standard deviation values
        prior_sd_window_values = Matrix(reshape(replace(read(prior_sd_window), fine_window.missingval => missing), fine_window_rows * fine_window_cols, 1))
        # get coarse x coordinates
        coarse_window_x_matrix = get_x_matrix(coarse_window)
        # get coarse y coordinates
        coarse_window_y_matrix = get_y_matrix(coarse_window)
        # combine coarse x and y coordinates
        coarse_window_coords = cat(vec(coarse_window_x_matrix), vec(coarse_window_y_matrix), dims=2)
        # get fine x coordinates
        fine_window_x_matrix = get_x_matrix(fine_window)
        # get fine y coordinates
        fine_window_y_matrix = get_y_matrix(fine_window)
        # combine fine x and y coordinates
        fine_window_coords = cat(vec(fine_window_x_matrix), vec(fine_window_y_matrix), dims=2)
        # set the target coordinates to the fine coordinates
        target_coords = fine_window_coords

        # assign measurement errors
        if isa(coarse_err_var, Raster)
            coarse_err_vars = read(view(coarse_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            coarse_err_vars = coarse_err_var .* ones(size(coarse_window_values))
        end

        if isa(fine_err_var, Raster)
            fine_err_vars = read(view(fine_err_var, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax))))
        else
            fine_err_vars = fine_err_var .* ones(size(fine_window_values))
        end

        measurement_error_vars = [coarse_err_vars, fine_err_vars ]
        # combine fine and coarse input values
        measurements = [coarse_window_values, fine_window_values]
        # combine fine and coarse coordinates
        measurement_coords = [coarse_window_coords, fine_window_coords]

        prior_mean = vcat(bias_mean_window_values, prior_mean_window_values)
        prior_sd = vcat(bias_sd_window_values, prior_sd_window_values)  

        if historic_tseries == nothing
            tseries_window_values = ones((1,1))
        else
            tseries_window = view(historic_tseries, X(Rasters.Between(window_bbox.xmin, window_bbox.xmax)), Y(Rasters.Between(window_bbox.ymin, window_bbox.ymax)))
            tseries_window_values = Matrix{Float64}(reshape(replace(read(tseries_window), fine_window.missingval => missing), fine_window_rows * fine_window_cols, size(historic_tseries)[3]))
        end

        #result[coarse_row,coarse_col,:] =
        result[k,:] = nll_minimize(
                            measurements, 
                            measurement_error_vars,                                 
                            measurement_coords, 
                            target_times,
                            resolutions,                   
                            target_coords,
                            target_resolution,
                            prior_mean,
                            prior_sd;      
                            spatial_mod = spatial_mod,                   
                            cov_pars = cov_pars, 
                            wt = wt,   
                            offset_ar = offset_ar, 
                            offset_var = offset_var,                        
                            obs_operator = obs_operator, 
                            historic_tseries = tseries_window_values)
        
        ProgressMeter.next!(p)
    end

    return result
end

"fast, adhoc estimate of innovation var"
function fast_var_est(
        coarse_images::Raster;
        n_eff_agg = 50,
        min_num_obs = 8,
        default_var = 1e-4
    )

    if ismissing(coarse_images.missingval)
        num_obs = sum(.!(ismissing.(coarse_images)),dims=3)
    elseif isnan(coarse_images.missingval)
        num_obs = sum(.!(isnan.(coarse_images)),dims=3)
    end

    result = mapslices(x -> var(x[.!(isnan.(x))]), diff(coarse_images,dims=3), dims=3).*n_eff_agg
    result[num_obs .< min_num_obs] .= default_var

    return result
end

function compute_n_eff(
        agg_scale,
        range;
        smoothness=0.5
    )
    subsample_inds = hcat(repeat(1:agg_scale,inner=[1],outer=[agg_scale]),
                    repeat(1:agg_scale,inner=[agg_scale],outer=[1]))'

    n = size(subsample_inds)[2]
    n_eff = n^2/sum(matern_cor(subsample_inds, reg = 1e-16,ν=smoothness, σ=range))

    return n_eff
end

end