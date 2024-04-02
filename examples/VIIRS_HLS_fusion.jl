using Plots
using Dates
using Rasters
using STARS
using STARS.BBoxes
using STARS.sentinel_tiles
using STARS.HLS
using STARS.VNP43
using STARS.STARS

tile = "10SGD"
start_date = "2022-04-01"
end_date = "2022-04-15"
variable = "NIR"
username = "" #add earthdata username
password = "" #add earthdata password
HLS_download_directory = "~/data/HLS_julia_demo"
VIIRS_download_directory = "~/data/VNP43_julia_demo"
HLS_cell_size = 60
VIIRS_cell_size = 500

## pull and resample HLS data, tile 10SGD, April 2022
HLS_series = generate_HLS_timeseries(
    tile,
    start_date,
    end_date,
    variable,
    HLS_download_directory,
    username,
    password,
    HLS_cell_size
);

## pull and resample VNP43 data, tile 10SGD, April 2022
VIIRS_series = generate_VNP43_timeseries_UTM(
    tile,
    start_date,
    end_date,
    variable,
    VIIRS_download_directory,
    username,
    password,
    VIIRS_cell_size
);

## check missing value
VIIRS_series.missingval
HLS_series.missingval

## fast estimation of innovation variance
n_eff = compute_n_eff(7,1.5,smoothness=0.5) ## 70 - 500 is roughly 7X7 HLS pixels within VIIRS, length scale of 1.5 times HLS resolution, exponential spatial covariance
sp_var = fast_var_est(VIIRS_series, n_eff_agg = n_eff);

## build raster of spatial covariance parameters, range, nugget, smoothness constant across pixels
cov_pars_raster = Raster(fill(NaN, size(VIIRS_series)[1], size(VIIRS_series)[2], 4), dims=(VIIRS_series.dims[1:2]...,Band(1:4)), missingval=VIIRS_series.missingval)
cov_pars_raster[:,:,1] = sp_var
cov_pars_raster[:,:,2] .= 150
cov_pars_raster[:,:,3] .= 1e-8
cov_pars_raster[:,:,4] .= 0.5;

plot(log.(cov_pars_raster[:,:,1]))

plot(VIIRS_series[:,:,1])
plot(HLS_series[:,:,6])

result_raster = coarse_fine_data_fusion(
    VIIRS_series[:,:,1:1], 
    HLS_series[:,:,1:1], 
    cov_pars_raster,
    n_ensembles = 0,
    target_times = [1],
    buffer_distance = 100.,
    smooth = false
)

plot(result_raster.mean)

result_raster2 = coarse_fine_data_fusion(
    VIIRS_series[:,:,2:2], 
    HLS_series[:,:,2:2], 
    cov_pars_raster,
    prior = result_raster,
    n_ensembles = 0,
    target_times = [1],
    buffer_distance = 100.,
    smooth = false
)