module HLS

using Crayons.Box
using Logging
using DataFrames
using Dates
using JSON
using HTTP
using Rasters
using DimensionalData.Dimensions.LookupArrays
using NaNStatistics
using GeoFormatTypes
import GeoDataFrames as GDF
import ArchGDAL as AG


include("sentinel_tiles.jl")

using .sentinel_tiles: sentinel_tile_dims

logger = ConsoleLogger()
global_logger(logger)

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search"
CMR_GRANULES_JSON_URL = "$(CMR_SEARCH_URL)/granules.json"
L30_CONCEPT = "C2021957657-LPCLOUD"
S30_CONCEPT = "C2021957295-LPCLOUD"
PAGE_SIZE = 2000
URL_FORMAT = merge(BLUE_FG, UNDERLINE)
VALUE_FORMAT = YELLOW_FG
TILE_FORMAT = YELLOW_FG
CONCEPT_ID_FORMAT = YELLOW_FG
DATE_FORMAT = BLUE_FG
CELL_SIZE = 30

export L30_CONCEPT
export S30_CONCEPT


"function to generate CMR date-range query string"
function generate_CMR_date_range(start_date::Union{Date,String}, end_date::Union{Date,String})::String
    "$(start_date)T00:00:00Z/$(end_date)T23:59:59Z"
end

export generate_CMR_date_range

"function to generate CMR API URL to search for HLS"
function generate_CMR_query_URL(
    concept_ID::String,
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    page_size::Int=PAGE_SIZE)::String
    datetime_range = generate_CMR_date_range(start_date, end_date)
    URL = "$(CMR_GRANULES_JSON_URL)?concept_id=$(concept_ID)&temporal=$(datetime_range)&page_size=$(page_size)&producer_granule_id=*.T$(tile).*&options[producer_granule_id][pattern]=true"

    return URL
end

export generate_CMR_query_URL

"function to search for HLS at tile within data range, given concept ID"
function CMR_query_JSON(
    concept_ID::String,
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    page_size::Int=PAGE_SIZE)::Dict
    # generate CMR URL to search for Sentinel tile in date range for given concept ID
    query_URL = generate_CMR_query_URL(concept_ID, tile, start_date, end_date)

    # send get request for the CMR query URL
    @info string("CMR API query for concept ID ", CONCEPT_ID_FORMAT(concept_ID), " at tile ", TILE_FORMAT(tile), " from ", DATE_FORMAT("$(start_date)"), " to ", DATE_FORMAT("$(end_date)"), " with URL: ", URL_FORMAT(query_URL))
    response = HTTP.get(query_URL)

    # check the status code of the response and make sure it's 200 before parsing JSON

    status = response.status

    if status != 200
        error("CMR API status $(status) for URL: $(query_URL)")
    end

    # parse JSON response from successful (200) CMR query
    response_dict = JSON.parse(String(response.body))

    return response_dict
end

export CMR_query_JSON

# need to include header with client ID and make "Client-Id: HLS.jl" the default header
"function to search for HLS at tile within data range, given concept ID"
function CMR_query(
    concept_ID::String,
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    page_size::Int=PAGE_SIZE)::DataFrame

    response_dict = CMR_query_JSON(
        concept_ID,
        tile,
        start_date,
        end_date,
        page_size
    )

    # build list of URLs returned by CMR API
    URLs = Vector{String}([])

    for entry in response_dict["feed"]["entry"]
        for link in entry["links"]
            push!(URLs, String(link["href"]))
        end
    end

    URL_count = length(URLs)
    @info string("CMR API query for concept ID ", CONCEPT_ID_FORMAT(concept_ID), " at tile ", TILE_FORMAT(tile), " from ", DATE_FORMAT("$(start_date)"), " to ", DATE_FORMAT("$(end_date)"), " included ", VALUE_FORMAT("$(URL_count)"), " URLs")

    https_df = DataFrame(
        granule_ID=[],
        sensor=[],
        tile=[],
        date=[],
        time=[],
        band=[],
        https=[]
    )

    s3_df = DataFrame(
        granule_ID=[],
        sensor=[],
        tile=[],
        date=[],
        time=[],
        band=[],
        s3=[]
    )

    for URL in URLs
        try
            # parse URL
            protocol = split(URL, ":")[1]
            filename_base = split(URL, "/")[end]
            granule_ID = join(split(filename_base, ".")[1:6], ".")
            sensor = split(filename_base, ".")[2]
            tile = split(filename_base, ".")[3][2:end]
            timestamp = replace(String(split(filename_base, ".")[4]), "T" => "")
            year = parse(Int, timestamp[1:4])
            doy = parse(Int, timestamp[5:7])
            hour = parse(Int, timestamp[8:9])
            minute = parse(Int, timestamp[10:11])
            second = parse(Int, timestamp[12:13])
            time = DateTime(year, 1, 1, hour, minute, second) + Day(doy - 1)
            date = Date(year, 1, 1) + Day(doy - 1)
            band = split(filename_base, ".")[end-1]

            # filter out invalid URLs
            if band in ["0_stac", "cmr", "0"]
                continue
            end

            if protocol == "https"
                append!(https_df, DataFrame(
                    granule_ID=[granule_ID],
                    sensor=[sensor],
                    tile=[tile],
                    date=[date],
                    time=[time],
                    band=[band],
                    https=[URL]
                ))
            elseif protocol == "s3"
                append!(s3_df, DataFrame(
                    granule_ID=[granule_ID],
                    sensor=[sensor],
                    tile=[tile],
                    date=[date],
                    time=[time],
                    band=[band],
                    s3=[URL]
                ))
            end
        catch
        end
    end

    df = outerjoin(https_df, s3_df, on=[(:granule_ID => :granule_ID), (:sensor => :sensor), (:tile => :tile), (:date => :date), (:time => :time), (:band => :band)])

    return df
end

export CMR_query

"function to search for HLS L30 Landsat product at tile in date range"
function L30_CMR_query(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    page_size::Int=PAGE_SIZE)::DataFrame
    @info string("CMR API query for ", VALUE_FORMAT("HLS Landsat L30"), " at tile ", TILE_FORMAT(tile), " from ", DATE_FORMAT("$(start_date)"), " to ", DATE_FORMAT("$(end_date)"))

    CMR_query(
        L30_CONCEPT,
        tile,
        start_date,
        end_date,
        page_size
    )
end

export L30_CMR_query

"function to search for HLS S30 Sentinel product at tile in date range"
function S30_CMR_query(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    page_size::Int=PAGE_SIZE)::DataFrame
    @info string("CMR API query for ", VALUE_FORMAT("HLS Sentinel S30"), " at tile ", TILE_FORMAT(tile), " from ", DATE_FORMAT("$(start_date)"), " to ", DATE_FORMAT("$(end_date)"))

    CMR_query(
        S30_CONCEPT,
        tile,
        start_date,
        end_date,
        page_size
    )
end

export S30_CMR_query

"function to search for HLS at tile in date range"
function HLS_CMR_query(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    page_size::Int=PAGE_SIZE)::DataFrame
    S30_listing = S30_CMR_query(
        tile,
        start_date,
        end_date,
        page_size
    )

    L30_listing = L30_CMR_query(
        tile,
        start_date,
        end_date,
        page_size
    )

    listing = vcat(S30_listing, L30_listing)
    listing = sort!(listing, :time)

    return listing
end

export HLS_CMR_query

"function to extract available Landsat dates from listing generated by HLS_CMR_query or L30_CMR_query"
function extract_L30_dates(listing::DataFrame)::Vector{Date}
    sort(unique(filter(row -> row.sensor == "L30", listing).date))
end

export extract_L30_dates

"function to extract available Sentinel dates from listing generated by HLS_CMR_query to S30_CMR_query"
function extract_S30_dates(listing::DataFrame)::Vector{Date}
    sort(unique(filter(row -> row.sensor == "S30", listing).date))
end

export extract_S30_dates

"function to determine which recent Landsat dates are missing from set of known Landsat dates available"
function find_missing_L30_dates(L30_dates::Vector{Date}, end_date::Union{Date,String})::Vector{Date}
    last_known_date = sort(L30_dates)[end]
    L30_mising_dates = []

    for L30_date in L30_dates
        prediction = L30_date + Day(16)

        if prediction > last_known_date && prediction <= end_date && prediction ∉ L30_dates
            push!(L30_dates_propagated, prediction)
        end

        prediction = L30_date + Day(32)

        if prediction > last_known_date && prediction <= end_date && prediction ∉ L30_dates
            push!(L30_dates_propagated, prediction)
        end
    end

    L30_mising_dates = sort(unique(L30_mising_dates))

    return L30_mising_dates
end

export find_missing_L30_dates

"function to determine which recent Sentinel dates are missing from set of known Landsat dates available"
function find_missing_S30_dates(S30_dates::Vector{Date}, end_date::Union{Date,String})::Vector{Date}
    last_known_date = sort(S30_dates)[end]
    S30_mising_dates = []

    for S30_date in S30_dates
        prediction = S30_date + Day(5)

        if prediction > last_known_date && prediction <= end_date && prediction ∉ S30_dates
            push!(S30_mising_dates, prediction)
        end

        prediction = S30_date + Day(10)

        if prediction > last_known_date && prediction <= end_date && prediction ∉ S30_dates
            push!(S30_mising_dates, prediction)
        end
    end

    S30_mising_dates = sort(unique(S30_mising_dates))
end

export find_missing_S30_dates

function generate_HLS_directory(URL::String, download_directory::String)::String
    filename_base = String(split(URL, "/")[end])
    granule_ID = join(split(filename_base, ".")[1:6], ".")
    timestamp = replace(String(split(filename_base, ".")[4]), "T" => "")
    year = parse(Int, timestamp[1:4])
    doy = parse(Int, timestamp[5:7])
    date = Date(year, 1, 1) + Day(doy - 1)

    directory = joinpath(
        expanduser(download_directory),
        Dates.format(date, "yyyy.mm.dd"),
        granule_ID
    )

    return directory
end

function generate_HLS_filename(URL::String, download_directory::String)::String
    filename_base = String(split(URL, "/")[end])
    directory = generate_HLS_directory(URL, download_directory)
    filename = joinpath(directory, filename_base)

    return filename
end

"function to download HLS at tile in date range"
function HLS_CMR_download(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    HLS_download_directory::String,
    username::String,
    password::String,
    sensor::String="both",
    page_size::Int=PAGE_SIZE)::DataFrame
    @info "using CMR to acquire HLS at tile $(tile) from $(start_date) to $(end_date)"
    HLS_download_directory = expanduser(HLS_download_directory)
    @info string("HLS download directory: $(HLS_download_directory)")
    mkpath(HLS_download_directory)

    if sensor == "S30"
        listing = S30_CMR_query(
            tile,
            start_date,
            end_date,
            page_size
        )
    elseif sensor == "L30"
        listing = L30_CMR_query(
            tile,
            start_date,
            end_date,
            page_size
        )
    else
        listing = HLS_CMR_query(
            tile,
            start_date,
            end_date,
            page_size
        )
    end

    filename_df = DataFrame(filename=[])

    for row in eachrow(listing)
        granule_ID = row.granule_ID
        @info "granule ID: $(granule_ID)"
        URL = row.https
        @info "URL: $(URL)"
        filename = generate_HLS_filename(URL, HLS_download_directory)
        @info "filename: $(filename)"
        file_found = isfile(filename)

        if file_found
            @info "file already downloaded: $(filename)"
        else
            @info "downloading file: $(filename)"
            directory, filename_base = splitdir(filename)
            mkpath(directory)
            run(`wget -q -c --user $username --password $password -O "$filename" "$URL"`)
        end

        append!(filename_df, DataFrame(filename=[filename]))
    end

    listing = hcat(listing, filename_df)

    return listing
end

export HLS_CMR_download

abstract type HLSGranule end

struct HLSS30Granule <: HLSGranule
    directory::String
    granule_ID::String

    function HLSS30Granule(directory::String)::HLSS30Granule
        granule_ID = basename(directory)
        sensor = split(granule_ID, ".")[2]

        if sensor != "S30"
            error("granule $(granule_ID) is not S30")
        end

        new(directory, granule_ID)
    end
end

struct HLSL30Granule <: HLSGranule
    directory::String
    granule_ID::String

    function HLSL30Granule(directory::String)::HLSL30Granule
        granule_ID = basename(directory)
        sensor = split(granule_ID, ".")[2]

        if sensor != "L30"
            error("granule $(granule_ID) is not L30")
        end

        new(directory, granule_ID)
    end
end

function get_band_filename(granule::HLSGranule, band::String)::String
    joinpath(granule.directory, "$(granule.granule_ID).$(band).tif")
end

function read_band(granule::HLSGranule, band::String)::Raster
    if band == "blue"
        return read_blue(granule)
    elseif band == "green"
        return read_green(granule)
    elseif band == "red"
        return read_red(granule)
    elseif band == "NIR"
        return read_NIR(granule)
    elseif band == "SWIR1"
        return read_SWIR1(granule)
    elseif band == "SWIR2"
        return read_SWIR2(granule)
    elseif band == "Fmask"
        return Raster(get_band_filename(granule, band))
    elseif band == "cloud"
        Fmask = read_band(granule, "Fmask")
        cloud = Fmask .& 15 .> 0

        return cloud
    else
        DN = Raster(get_band_filename(granule, band))
        DN = DN * 1.0
        DN[DN.==-9999] .= NaN
        raster = DN * 0.0001
        cloud = read_band(granule, "cloud")
        raster = ifelse.(cloud, NaN, raster)
        raster = rebuild(raster, missingval=NaN)

        return raster
    end
end

export read_band

function read_coastal_aerosol(granule::HLSS30Granule)::Raster
    read_band(granule, "B01")
end

function read_blue(granule::HLSS30Granule)::Raster
    read_band(granule, "B02")
end

function read_green(granule::HLSS30Granule)::Raster
    read_band(granule, "B03")
end

function read_red(granule::HLSS30Granule)::Raster
    read_band(granule, "B04")
end

function read_rededge1(granule::HLSS30Granule)::Raster
    read_band(granule, "B05")
end

function read_rededge2(granule::HLSS30Granule)::Raster
    read_band(granule, "B06")
end

function read_rededge3(granule::HLSS30Granule)::Raster
    read_band(granule, "B07")
end

function read_NIR_broad(granule::HLSS30Granule)::Raster
    read_band(granule, "B08")
end

function read_NIR(granule::HLSS30Granule)::Raster
    read_band(granule, "B8A")
end

function read_water_vapor(granule::HLSS30Granule)::Raster
    read_band(granule, "B09")
end

function read_cirrus(granule::HLSS30Granule)::Raster
    read_band(granule, "B10")
end

function read_SWIR1(granule::HLSS30Granule)::Raster
    read_band(granule, "B11")
end

function read_SWIR2(granule::HLSS30Granule)::Raster
    read_band(granule, "B12")
end

function read_coastal_aerosol(granule::HLSL30Granule)::Raster
    return read_band(granule, "B01")
end

function read_blue(granule::HLSL30Granule)::Raster
    return read_band(granule, "B02")
end

function read_green(granule::HLSL30Granule)::Raster
    return read_band(granule, "B03")
end

function read_red(granule::HLSL30Granule)::Raster
    return read_band(granule, "B04")
end

function read_NIR(granule::HLSL30Granule)::Raster
    return read_band(granule, "B05")
end

function read_SWIR1(granule::HLSL30Granule)::Raster
    return read_band(granule, "B06")
end

function read_SWIR2(granule::HLSL30Granule)::Raster
    return read_band(granule, "B07")
end

function read_cirrus(granule::HLSL30Granule)::Raster
    return read_band(granule, "B09")
end

function get_S30(
    tile::String,
    date::Union{Date,String},
    HLS_download_directory::String,
    username::String,
    password::String)::Union{HLSS30Granule,Nothing}

    listing = HLS_CMR_download(
        tile,
        date,
        date,
        HLS_download_directory,
        username,
        password,
        "S30"
    )

    if size(listing, 1) == 0
        return nothing
    end

    directory = dirname(listing.filename[1])
    granule = HLSS30Granule(directory)

    return granule
end

export get_S30

function get_L30(
    tile::String,
    date::Union{Date,String},
    HLS_download_directory::String,
    username::String,
    password::String)::Union{HLSL30Granule,Nothing}

    listing = HLS_CMR_download(
        tile,
        date,
        date,
        HLS_download_directory,
        username,
        password,
        "L30"
    )

    if size(listing, 1) == 0
        return nothing
    end

    directory = dirname(listing.filename[1])
    granule = HLSL30Granule(directory)

    return granule
end

export get_L30

function generate_HLS_tile(
    tile::String,
    date::Union{Date,String},
    band::String,
    download_directory::String,
    username::String,
    password::String,
    cell_size::Union{Float64,Int64}=CELL_SIZE)::Raster
    S30_granule = get_S30(tile, date, download_directory, username, password)

    if S30_granule == nothing
        x, y = sentinel_tile_dims(tile, 30)
        S30_image = Raster(fill(NaN, size(x)[1], size(y)[1], 1), dims=(x, y, Band(1:1)), missingval=NaN)
    else
        S30_image = read_band(S30_granule, band)
    end

    L30_granule = get_L30(tile, date, download_directory, username, password)

    if L30_granule == nothing
        x, y = sentinel_tile_dims(tile, 30)
        L30_image = Raster(fill(NaN, size(x)[1], size(y)[1], 1), dims=(x, y, Band(1:1)), missingval=NaN)
    else
        L30_image = read_band(L30_granule, band)
    end

    composite = (S30_image + L30_image) / 2
    composite = ifelse.(isnan.(composite), S30_image, composite)
    composite = ifelse.(isnan.(composite), L30_image, composite)

    dims = sentinel_tile_dims(tile, cell_size)
    resampled_image = resample(composite, to=dims)

    return resampled_image
end

export generate_HLS_tile

function generate_HLS_timeseries(
    tile::String,
    start_date::Union{Date,String},
    end_date::Union{Date,String},
    variable::String,
    download_directory::String,
    username::String,
    password::String,
    cell_size::Union{Float64,Int64}=CELL_SIZE)::Raster
    start_date = Date(start_date)
    end_date = Date(end_date)

    @info "generating HLS sources from $(start_date) to $(end_date)"

    dates = [start_date + Day(d - 1) for d in 1:((end_date-start_date).value+1)]

    images = []
    x, y = sentinel_tile_dims(tile, cell_size)
    t = Ti(dates)
    dims = (x, y, t)

    for date in dates
        @info "generating HLS image for $(variable) on $(date)"
        image = generate_HLS_tile(
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

    shape = [size(x)[1], size(y)[1], size(dates)[1]]

    if size(images)[1] == 1
        stack = Raster(images[1], dims=(x, y, t), missingval=NaN)
    else
        stack = Raster(cat(images..., dims=3), dims=dims, missingval=NaN)
    end

    return stack
end

export generate_HLS_timeseries

end
