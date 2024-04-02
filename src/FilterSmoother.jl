module FilterSmoother


export KSModel #, filter, smoother, filter_series, smooth_series, filter_and_smooth!


using LinearAlgebra

T = Float64

TM = Vector{AbstractMatrix{T}}
Tv = Vector{AbstractVector{T}}

struct KSModel{T}
    H::AbstractMatrix{T}
    Q::AbstractMatrix{T}
    F::Union{AbstractMatrix{T}, UniformScaling{T}}
    #R::Union{UniformScaling{T},Diagonal{T}}
end


"""
    Kalman Filter

    Arguments:
    y::observation vector
    M::KSModel{T}
"""
function filter(M::KSModel{T}, y::AbstractVector{T}, err_vars::AbstractVector{T}, x_pred::Vector{T}, P_pred::AbstractMatrix{T}; nfs::Int = 0) where T <: Real

    ym = .!isnan.(y)
    sum(ym) == 0 && return (x_pred, P_pred)

    # subset out missing values
    # R = @views Diagonal(err_vars[ym,t])
    R = @views Diagonal(err_vars[ym])
    y = @views y[ym]
    # y = @views ys[ym,t]
    #Ht = H[ym,:]

    # Shortcuts to use old syntax for now:
    H = M.H
    # Q = M.Q

    #ids[ym] .= true
    # ck = findall(ym[1:nfs]) ## nfs is only coincidentally correct, it is the number of bias components with corresponds to coarse pixels here
    # cf = findall(ym[nfs + 1:end]) .+ nfs
    # nnk = size(ck)[1]
    # nnf = size(cf)[1]

    # #### Need to generalize
    # # Construct helper matrices that make computation faster.
    # HH = H[ck, :] # non-diagonal part of H
    # # HD is the diagonal part of H. Infer its type to be either
    # # UniformScaling or Diagonal
    # #HD = sum(diag(H) .== H[1]) == length(diag(H)) ? H[1] * I : 1*Diagonal(H)
    # HD = 1*Diagonal(H)
    # #HD = HD[ym,:]
    # # HHmD is HH with diagonal set to zero
    # HHmD = H[1:nfs, :] # memory copy
    # HHmD[diagind(HHmD)] .= 0.0
    # HHmD = HHmD[ck,:]

    # begin # much faster than HPp = vcat(HH * P_pred, P_pred[nfs + 1:end, :])
    #     HPp = (HD * P_pred)[ym,:] # H * P_pred
    #     HPp[1:nnk, :] .+= HHmD * P_pred
    #     HPpT = (P_pred * HD)[:,ym] # Transpose of HPp; P_pred * H'
    #     HPpT[:, 1:nnk] .+= P_pred * HHmD'
    # end

    # #### need to generalize
    # res_pred = y - vcat(HH * x_pred, x_pred[cf]) # innovation

    # begin # faster way here
    #     S = (HPp * HD)[:,ym]
    #     S[:, 1:nnk] .+= HPp * HHmD'
    #     S .+= R
    # end

    # # S = Symmetric(Ht * P_pred * Ht' + R) # innovation covariance
    # # Kalman gain; K = P_pred * H' * inv(S)
    # begin
    #     LAPACK.potrf!('U', S)
    #     LAPACK.potri!('U', S)
    #     K = BLAS.symm('R', 'U', S, HPpT)
    # end

    # # With K
    # x_new = x_pred + K * res_pred # filtering distribution mean
    # P_new = P_pred - K * HPp # filtering distribution covariance

    Ht = H[ym,:] 

    res_pred = y - Ht * x_pred # innovation

    HPpT = P_pred * Ht'

    S = Ht * HPpT + R # innovation covariance

    # Kalman gain; K = P_pred * H' * inv(S)
    begin
        LAPACK.potrf!('U', S)
        LAPACK.potri!('U', S)
        K = BLAS.symm('R', 'U', S, HPpT)
    end

    # With K
    x_new = x_pred + K * res_pred # filtering distribution mean
    P_new = P_pred - K * (Ht * P_pred) # filtering distribution covariance

    return (x_new, P_new)
end


function filter_series(M::KSModel{T}, x0::AbstractVector{T}, P0::AbstractMatrix{T}, Y::Matrix{T}, Y_err_vars::Matrix{T}) where T <: Real

    # Shorthands
    F = M.F
    Q = M.Q
    nsteps = size(Y)[2]

    filtering_means = Vector{AbstractVector{Float64}}(undef, 0)
    predicted_means = Vector{AbstractVector{Float64}}(undef, 0)
    filtering_covs = Vector{AbstractMatrix{Float64}}(undef, 0)
    predicted_covs = Vector{AbstractMatrix{Float64}}(undef, 0)
    push!(filtering_means, x0)
    push!(filtering_covs, P0)

    for t ∈ 1:nsteps
        # Predictive mean and covariance here
        x_pred = F * filtering_means[t] # filtering_means[1], covs[1] is prior mean
        P_pred = F * filtering_covs[t] * F' + Q
        push!(predicted_means, x_pred)
        push!(predicted_covs, P_pred)

        # Filtering is done here
        x_new, P_new = filter(M, Y[:,t], Y_err_vars[:,t], x_pred, P_pred)
        push!(filtering_means, x_new)
        push!(filtering_covs, P_new)
    end

    return predicted_means, predicted_covs, filtering_means, filtering_covs
end


# function smooth_series(M::KSModel{T}, predicted_means::Tv, predicted_covs::TM, filtering_means::Tv, filtering_covs::TM) where T <: Real
function smooth_series(M::KSModel{T}, predicted_means, predicted_covs, filtering_means, filtering_covs) where T <: Real

    # These arrays start at the final smoothed (= filtered) state
    smoothed_means = [filtering_means[end]]
    smoothed_covs = [filtering_covs[end]]

    # First step that we are interested here in is i = nsteps - 1
    nsteps = length(predicted_means) # This was T previously
    t0 = 1 #FIXME ADDED BLINDLY

    for i ∈ nsteps:-1:(t0+1)
        # NB. filtering_covs[i] is P_{i-1|i-1}, predicted_covs[i] is P_{i|i-1}
        begin # C = filtering_covs[i] * F * inv(predicted_covs[i])
            # This block assumes F is identity
            CC = predicted_covs[i][:,:]
            LAPACK.potrf!('U', CC)
            LAPACK.potri!('U', CC)
            C = zeros(size(predicted_covs[i]))
            BLAS.symm!('R', 'U', 1., CC, filtering_covs[i], 0., C)
        end
        x_smooth = filtering_means[i] .+ C * (smoothed_means[nsteps - i + 1] .- predicted_means[i])

        # P_smooth = filtering_covs[i] + C * (smoothed_covs[nsteps - i + 1] - predicted_covs[i]) * C'
        # Compute P_smooth = filtering_covs[i] + C *
        # (smoothed_covs[nsteps - i + 1] - predicted_covs[i]) * C'
        begin
            CC .= smoothed_covs[nsteps - i + 1] .- predicted_covs[i]
            D = BLAS.symm('R', 'U', CC, C) # D = C * CC
            CC .= filtering_covs[i]
            P_smooth = BLAS.gemm!('N', 'T', 1., D, C, 1., CC)
        end

        push!(smoothed_means, x_smooth)
        push!(smoothed_covs, P_smooth)
    end

    return reverse(smoothed_means), reverse(smoothed_covs)
end

function conditional_sim(post_mean::AbstractVector{T}, post_cov::AbstractMatrix{T}, ensembles; nfs = 1) where T <: Real
    Ω = sqrt(post_cov[nfs:end,nfs:end])
    return post_mean[nfs:end] .+ Ω * ensembles
end

function conditional_sim_series(post_means, post_covs, ensembles; nfs=1) 
    # post_means: vector of nx1
    # post_vars: vector of nxn
    # ensembles: nxMxT abstract array

    post_ensembles = []
    for (i,m) in collect(enumerate(post_means))
        pp = conditional_sim(m,post_covs[i], ensembles[:,:,i], nfs=nfs)
        push!(post_ensembles,pp)
    end

    return cat(post_ensembles...,dims=3)
end

# function filter_and_smooth!(H::Matrix{T}, F::UniformScaling, Q::Matrix{T}, ym::Matrix{T},
#                             fused_images::TM, fused_sd_images::TM,
#                             filtering_means::Tv, filtering_covs::TM;
#                             predicted_means::Tv = [], predicted_covs::TM = [],
#                             smooth = false)

#     filtering_means, filtering_covs, predicted_means, predicted_covs =
#         filter_series()

#     smoothed_means, smoothed_covs


#     # for t ∈ 1:T
#     #     x_pred = F * filtering_means[t] # filtering_means[1], covs[1] is prior mean
#     #     P_pred = filtering_covs[t] + Q



#     #     ym = .!ismissing.(ys[:,t])

#     #     if sum(ym)==0
#     #         push!(filtering_means, x_pred)
#     #         push!(filtering_covs, P_pred)
#     #         push!(predicted_means, x_pred)
#     #         push!(predicted_covs, P_pred)

#     #         if !smooth
#     #             if t0 <= t <= tn
#     #                 fused_images[:,t-t0+1] .= x_pred
#     #                 fused_sd_images[:,t-t0+1] .= sqrt.(diag(P_pred))
#     #             end
#     #         end

#     #     else

#     #         push!(filtering_means, x_new)
#     #         push!(filtering_covs, P_new)
#     #         push!(predicted_means, x_pred)
#     #         push!(predicted_covs, P_pred)

#     #         if !smooth
#     #             if t0 <= t <= tn
#     #                 fused_images[:,t-t0+1] .= x_new
#     #                 fused_sd_images[:,t-t0+1] .= sqrt.(diag(P_new))
#     #             end
#     #         end

#     #     end

#     # end

#     # if smooth



#     if T == tn
#         fused_images[:,tn-t0+1] .= filtering_means[end]
#         fused_sd_images[:,tn-t0+1] .= sqrt.(diag(filtering_covs[end]))
#     end


#     if i <= (tn+1)
#         fused_images[:,i-t0] .= x_smooth
#         fused_sd_images[:,i-t0] .= sqrt.(diag(P_smooth))
#     end


#     # end

#     return fused_images, fused_sd_images
# end




end
