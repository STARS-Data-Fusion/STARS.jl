module VNP43

using Rasters
using DimensionalData.Dimensions.LookupArrays
using HDF5
using Dates
using StatsModels
using DataFrames
using GLM
using Statistics
using NaNStatistics

include("sentinel_tiles.jl")

using .sentinel_tiles: sentinel_tile_dims, sentinel_tile_polygon_latlon

include("VIIRS.jl")

using .VIIRS: download_VIIRS_tile

include("MODLAND.jl")

using .MODLAND: MODLAND_tiles_in_polygon

export download_VIIRS_tile

SINUSOIDAL_CRS = WellKnownText("PROJCS[\"unknown\",GEOGCS[\"unknown\",DATUM[\"unknown\",SPHEROID[\"unknown\",6371007.181,0]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]]],PROJECTION[\"Sinusoidal\"],PARAMETER[\"longitude_of_center\",0],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]]")

export SINUSOIDAL_CRS

CELL_SIZE = 500

function sinusoidal_tile_dims(h::Int, v::Int, tile_width_cells::Int)::Tuple{X,Y}
    # boundaries of sinusodial projection
    GLOBE_UPPER_LEFT_X = -20015109.355798
    GLOBE_UPPER_LEFT_Y = 10007554.677899
    GLOBE_LOWER_RIGHT_X = 20015109.355798
    GLOBE_LOWER_RIGHT_Y = -10007554.677899

    # size across (width or height) of any equal-area sinusoidal tile
    TILE_SIZE = 1111950.5197665554

    # rows and columns of sinusoidal tile grid
    TOTAL_ROWS = 18
    TOTAL_COLUMNS = 36

    cell_size = TILE_SIZE / tile_width_cells
    tile_left_x = GLOBE_UPPER_LEFT_X + h * TILE_SIZE
    tile_right_x = GLOBE_UPPER_LEFT_X + (h + 1) * TILE_SIZE - cell_size
    tile_upper_y = GLOBE_LOWER_RIGHT_Y + (TOTAL_ROWS - v) * TILE_SIZE - cell_size
    tile_lower_y = GLOBE_LOWER_RIGHT_Y + (TOTAL_ROWS - 1 - v) * TILE_SIZE
    sampling = Intervals(Start())
    x_dim = X(Projected(LinRange(tile_left_x, tile_right_x, tile_width_cells), order=ForwardOrdered(), span=Regular(cell_size), sampling=sampling, crs=SINUSOIDAL_CRS))
    y_dim = Y(Projected(LinRange(tile_upper_y, tile_lower_y, tile_width_cells), order=ReverseOrdered(), span=Regular(-cell_size), sampling=sampling, crs=SINUSOIDAL_CRS))
    dims = (x_dim, y_dim)

    return dims
end

export sinusoidal_tile_dims

struct VNP43FilenameAttributes
    filename::String
    product::String
    timestamp::String
    year::Int
    doy::Int
    date::Date
    tile::String
    h::Int
    v::Int
    collection::Int
    processing_timestamp::String

    function VNP43FilenameAttributes(filename::String)
        filename = basename(filename)
        product = String(split(filename, ".")[1])
        timestamp = String(split(filename, ".")[2][2:end])
        year = parse(Int, timestamp[1:4])
        tile = String(split(filename, ".")[3])
        doy = parse(Int, timestamp[5:7])
        date = Date(year, 1, 1) + Day(doy - 1)
        h, v = parse(Int, tile[2:3]), parse(Int, tile[5:6])
        collection = parse(Int, split(filename, ".")[4])
        processing_timestamp = String(split(filename, ".")[5])

        return new(
            filename,
            product,
            timestamp,
            year,
            doy,
            date,
            tile,
            h,
            v,
            collection,
            processing_timestamp
        )
    end
end

export VNP43FilenameAttributes

abstract type VNP43Granule end

struct VNP43IA4Granule <: VNP43Granule
    filename::String
    attr::VNP43FilenameAttributes
    fill::Int
    scale::Float64

    function VNP43IA4Granule(filename::String)
        attr = VNP43FilenameAttributes(filename)

        if attr.product != "VNP43IA4"
            error("mis-matched product $(attr.product) in filename $(filename)")
        end

        return new(filename, attr, 32767, 0.0001)
    end
end

export VNP43IA4Granule

struct VNP43MA4Granule <: VNP43Granule
    filename::String
    attr::VNP43FilenameAttributes
    fill::Int
    scale::Float64

    function VNP43MA4Granule(filename::String)
        attr = VNP43FilenameAttributes(filename)

        if attr.product != "VNP43MA4"
            error("mis-matched product $(attr.product) in filename $(filename)")
        end

        return new(filename, attr, 32767, 0.0001)
    end
end

export VNP43MA4Granule

function read_DN(granule::VNP43Granule, dataset_name::String)::Raster
    h = granule.attr.h
    v = granule.attr.v

    data_array = h5open(granule.filename) do file
        read(file[dataset_name])
    end

    rows, cols = size(data_array)[1:2]
    dims = sinusoidal_tile_dims(h, v, rows)
    raster = Raster(data_array, dims=dims, missingval=NaN)

    return raster
end

export read_DN

function read_SR(granule::VNP43IA4Granule, band::Int)::Raster
    if band ∉ 1:3
        error("band $(band) not in valid range for VNP43IA4 from 1 to 3")
    end

    dataset_name = "HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Nadir_Reflectance_I$(band)"
    DN = read_DN(granule, dataset_name)
    raster = replace(DN, granule.fill => NaN) * granule.scale

    return raster
end

export read_SR

function read_SR(granule::VNP43MA4Granule, band::Int)::Raster
    if band ∉ 1:11
        error("band $(band) not in valid range for VNP43MA4 from 1 to 3")
    end

    dataset_name = "HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Nadir_Reflectance_M$(band)"
    DN = read_DN(granule, dataset_name)
    raster = replace(DN, granule.fill => NaN) * granule.scale

    return raster
end

export read_SR

function read_blue(granule::VNP43MA4Granule)::Raster
    read_SR(granule, 3)
end

export read_blue

function read_green(granule::VNP43MA4Granule)::Raster
    read_SR(granule, 4)
end

export read_green

function read_red(granule::VNP43IA4Granule)::Raster
    read_SR(granule, 1)
end

function read_red(granule::VNP43MA4Granule)::Raster
    read_SR(granule, 5)
end

export read_red

function read_NIR(granule::VNP43IA4Granule)::Raster
    read_SR(granule, 2)
end

function read_NIR(granule::VNP43MA4Granule)::Raster
    read_SR(granule, 7)
end

export read_NIR

function read_SWIR1(granule::VNP43IA4Granule)::Raster
    read_SR(granule, 3)
end

function read_SWIR1(granule::VNP43MA4Granule)::Raster
    read_SR(granule, 10)
end

export read_SWIR1

function read_SWIR2(granule::VNP43MA4Granule)::Raster
    read_SR(granule, 10)
end

export read_SWIR2

function collect_samples(granule::VNP43MA4Granule)::DataFrame
    blue = read_blue(granule)
    green = read_green(granule)
    red = read_red(granule)
    NIR = read_NIR(granule)
    SWIR1 = read_SWIR1(granule)
    SWIR2 = read_SWIR2(granule)

    rows, cols = size(red)[1:2]
    count = rows * cols

    samples = DataFrame(
        blue=Array{Float64}(reshape(blue, count)),
        green=Array{Float64}(reshape(green, count)),
        red=Array{Float64}(reshape(red, count)),
        NIR=Array{Float64}(reshape(NIR, count)),
        SWIR1=Array{Float64}(reshape(SWIR1, count)),
        SWIR2=Array{Float64}(reshape(SWIR2, count))
    )

    samples = filter!(row -> all(x -> !isnan(x), row), samples)

    return samples
end

export collect_samples

function calculate_I_coefficients(granule::VNP43MA4Granule)::Matrix
    samples = collect_samples(granule)

    coefficients = hcat(
        coef(lm(@formula(blue ~ red + NIR + SWIR1), samples)),
        coef(lm(@formula(green ~ red + NIR + SWIR1), samples)),
        coef(lm(@formula(SWIR2 ~ red + NIR + SWIR1), samples))
    )

    return coefficients
end

export calculate_I_coefficients

function read_blue(granule_M::VNP43MA4Granule, granule_I::VNP43IA4Granule)::Tuple{Raster,Raster}
    c = calculate_I_coefficients(granule_M)
    red_I = read_red(granule_I)
    NIR_I = read_NIR(granule_I)
    SWIR1_I = read_SWIR1(granule_I)
    blue_I = red_I .* c[2, 2] .+ NIR_I .* c[3, 2] .+ SWIR1_I .* c[4, 2] .+ c[1, 2]
    blue_M = read_blue(granule_M)
    blue_bias = resample(resample(blue_I, to=blue_M, method=:average) - blue_M, to=blue_I, method=:cubic)
    blue_I = blue_I - blue_bias
    blue_error = resample(resample(blue_I, to=blue_M, method=:average) - blue_M, to=blue_I, method=:cubic)

    return blue_I, blue_error
end

export read_blue

function read_green(granule_M::VNP43MA4Granule, granule_I::VNP43IA4Granule)::Tuple{Raster,Raster}
    c = calculate_I_coefficients(granule_M)
    red_I = read_red(granule_I)
    NIR_I = read_NIR(granule_I)
    SWIR1_I = read_SWIR1(granule_I)
    green_I = red_I .* c[2, 2] .+ NIR_I .* c[3, 2] .+ SWIR1_I .* c[4, 2] .+ c[1, 2]
    green_M = read_green(granule_M)
    green_bias = resample(resample(green_I, to=green_M, method=:average) - green_M, to=green_I, method=:cubic)
    green_I = green_I - green_bias
    green_error = resample(resample(green_I, to=green_M, method=:average) - green_M, to=green_I, method=:cubic)

    return green_I, green_error
end

export read_green

function read_SWIR2(granule_M::VNP43MA4Granule, granule_I::VNP43IA4Granule)::Tuple{Raster,Raster}
    c = calculate_I_coefficients(granule_M)
    red_I = read_red(granule_I)
    NIR_I = read_NIR(granule_I)
    SWIR1_I = read_SWIR1(granule_I)
    SWIR2_I = red_I .* c[2, 3] .+ NIR_I .* c[3, 3] .+ SWIR1_I .* c[4, 3] .+ c[1, 3]
    SWIR2_M = read_SWIR2(granule_M)
    SWIR2_bias = resample(resample(SWIR2_I, to=SWIR2_M, method=:average) - SWIR2_M, to=SWIR2_I, method=:cubic)
    SWIR2_I = SWIR2_I - SWIR2_bias
    SWIR2_error = resample(resample(SWIR2_I, to=SWIR2_M, method=:average) - SWIR2_M, to=SWIR2_I, method=:cubic)

    return SWIR2_I, SWIR2_error
end

export read_SWIR2

function get_VNP43MA4_granule(
    tile::String,
    date::Union{Date,String},
    download_directory::String,
    username::String,
    password::String)::VNP43MA4Granule
    filename = download_VIIRS_tile("VNP43MA4.001", tile, date, download_directory, username, password)
    granule = VNP43MA4Granule(filename)

    return granule
end

export get_VNP43MA4_granule

function get_VNP43IA4_granule(
    tile::String,
    date::Union{Date,String},
    download_directory::String,
    username::String,
    password::String)::VNP43IA4Granule
    filename = download_VIIRS_tile("VNP43IA4.001", tile, date, download_directory, username, password)
    granule = VNP43IA4Granule(filename)

    return granule
end

export get_VNP43IA4_granule

function generate_VNP43_I_tile_sinusoidal(
    tile::String,
    date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String)::Tuple{Raster,Raster}
    date = Date(date)
    @info "generating VNP43 I-band sinusoidal tile $(tile) variable $(variable) date $(date)"

    VNP43IA4_granule = get_VNP43IA4_granule(
        tile,
        date,
        download_directory,
        username,
        password
    )

    VNP43MA4_granule = get_VNP43IA4_granule(
        tile,
        date,
        download_directory,
        username,
        password
    )

    if variable == "blue"
        image, error = read_blue(VNP43MA4_granule, VNP43IA4_granule)
    elseif variable == "green"
        image, error = read_green(VNP43MA4_granule, VNP43IA4_granule)
    elseif variable == "red"
        image = read_red(VNP43IA4_granule)
        error = image * 0
    elseif variable == "NIR"
        image = read_NIR(VNP43IA4_granule)
        error = image * 0
    elseif variable == "SWIR1"
        image = read_SWIR1(VNP43IA4_granule)
        error = image * 0
    elseif variable == "SWIR2"
        image, error = read_SWIR2(VNP43MA4_granule, VNP43IA4_granule)
    end

    return image, error
end

export generate_VNP43_I_tile_sinusoidal

function generate_VNP43_M_tile_sinusoidal(
    tile::String,
    date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String)::Raster
    date = Date(date)
    @info "generating VNP43 M-band sinusoidal tile $(tile) variable $(variable) date $(date)"

    VNP43MA4_granule = get_VNP43IA4_granule(
        tile,
        date,
        download_directory,
        username,
        password
    )

    if variable == "blue"
        image = read_blue(VNP43MA4_granule)
    elseif variable == "green"
        image = read_green(VNP43MA4_granule)
    elseif variable == "red"
        image = read_red(VNP43MA4_granule)
    elseif variable == "NIR"
        image = read_NIR(VNP43MA4_granule)
    elseif variable == "SWIR1"
        image = read_SWIR1(VNP43MA4_granule)
    elseif variable == "SWIR2"
        image = read_SWIR2(VNP43MA4_granule)
    end

    return image
end

export generate_VNP43_M_tile_sinusoidal

function generate_VNP43_I_tile_UTM(
    tile::String,
    date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String,
    cell_size::Union{Float64, Int64} = CELL_SIZE)::Tuple{Raster,Raster}
    date = Date(date)
    @info "generating VNP43 I-band UTM tile $(tile) variable $(variable) date $(date)"
    sinusoidal_tiles = MODLAND_tiles_in_polygon(sentinel_tile_polygon_latlon(tile))

    composite_image = nothing
    composite_error = nothing

    for sinusoidal_tile in sinusoidal_tiles
        image, error = generate_VNP43_I_tile_sinusoidal(
            sinusoidal_tile,
            date,
            variable,
            download_directory,
            username,
            password
        )

        dims = sentinel_tile_dims(tile, cell_size)
        resampled_image = resample(image, to=dims)
        resampled_error = resample(error, to=dims)

        if composite_image == nothing
            composite_image = resampled_image
        else
            composite_image = rebuild(composite_image, data=ifelse.(isnan.(composite_image.data), resampled_image, composite_image), missingval=NaN)
        end

        if composite_error == nothing
            composite_error = resampled_error
        else
            composite_error = rebuild(composite_error, data=ifelse.(isnan.(composite_error.data), resampled_error, composite_error), missingval=NaN)
        end
    end

    return composite_image, composite_error
end

export generate_VNP43_I_tile_UTM

function generate_VNP43_M_tile_UTM(
    tile::String,
    date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String,
    cell_size::Union{Float64, Int64} = CELL_SIZE)::Tuple{Raster,Raster}
    date = Date(date)
    @info "generating VNP43 M-band UTM tile $(tile) variable $(variable) date $(date)"
    sinusoidal_tiles = MODLAND_tiles_in_polygon(sentinel_tile_polygon_latlon(tile))

    composite_image = nothing

    for sinusoidal_tile in sinusoidal_tiles
        image = generate_VNP43_M_tile_sinusoidal(
            sinusoidal_tile,
            date,
            variable,
            download_directory,
            username,
            password
        )

        dims = sentinel_tile_dims(tile, cell_size)
        resampled_image = resample(image, to=dims)

        if composite_image == nothing
            composite_image = resampled_image
        else
            composite_image = rebuild(composite_image, data=ifelse.(isnan.(composite_image.data), resampled_image, composite_image), missingval=NaN)
        end
    end

    return composite_image
end

export generate_VNP43_M_tile_UTM

function generate_VNP43_I_timeseries_UTM(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String,
    cell_size::Union{Float64, Int64} = CELL_SIZE)::Raster
    start_date = Date(start_date)
    end_date = Date(end_date)

    @info "generating VNP43 I-band sources from $(start_date) to $(end_date)"

    dates = [start_date + Day(d - 1) for d in 1:((end_date - start_date).value + 1)]

    images = []
    x, y = sentinel_tile_dims(tile, cell_size)
    t = Ti(dates)
    dims = (x, y, t)

    for date in dates
        @info "generating VNP43 image for $(variable) on $(date)"
        image, bias = generate_VNP43_tile_UTM(
            tile,
            date,
            variable,
            download_directory,
            username,
            password,
            cell_size
        )
    
        push!(images, image)
    end

    shape = [size(images[1])..., 1]

    if size(images)[1] == 1
        stack = Raster(reshape(images[1], shape...), dims=(x, y, t), missingval=NaN)
    else
        stack = Raster(cat(reshape.(images, shape...)..., dims=3), dims=dims, missingval=NaN)
    end

    return stack
end

export generate_VNP43_I_timeseries_UTM

function generate_VNP43_M_timeseries_UTM(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String,
    cell_size::Union{Float64, Int64} = CELL_SIZE)::Raster
    start_date = Date(start_date)
    end_date = Date(end_date)

    @info "generating VNP43 M-band sources from $(start_date) to $(end_date)"

    dates = [start_date + Day(d - 1) for d in 1:((end_date - start_date).value + 1)]

    images = []
    x, y = sentinel_tile_dims(tile, cell_size)
    t = Ti(dates)
    dims = (x, y, t)

    for date in dates
        @info "generating VNP43 image for $(variable) on $(date)"
        image = generate_VNP43_M_tile_UTM(
            tile,
            date,
            variable,
            download_directory,
            username,
            password,
            cell_size
        )
    
        push!(images, image)
    end

    shape = [size(images[1])..., 1]

    if size(images)[1] == 1
        stack = Raster(reshape(images[1], shape...), dims=(x, y, t), missingval=NaN)
    else
        stack = Raster(cat(reshape.(images, shape...)..., dims=3), dims=dims, missingval=NaN)
    end

    return stack
end

export generate_VNP43_M_timeseries_UTM

end
