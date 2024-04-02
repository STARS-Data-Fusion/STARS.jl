module VIIRS

using HTTP
using EzXML
using Logging
using Dates
using DataFrames

LPDAAC_DATE_DIRECTORY_REGEX = r"^\d{4}\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\/$"

function match_extensions(filename::String, extensions::Vector{String})::Bool
    any([endswith(filename, extension) for extension in extensions])
end

function HTTP_text(URL::String)::String
    String(HTTP.request("GET", URL).body)
end

function HTTP_links(URL::String)::Vector{String}
    with_logger(NullLogger()) do
        [link["href"] for link in findall("//a[@href and normalize-space(text()) != '']", parsehtml(HTTP_text(URL)))]
    end
end

function HTTP_regex(URL::String, regex::Regex)::Vector{String}
    [link for link in HTTP_links(URL) if occursin(regex, link)]
end

function HTTP_filenames(URL::String, extensions::Vector{String})::Vector{String}
    sort([link for link in HTTP_links(URL) if match_extensions(link, extensions)])
end

function VIIRS_dates(product::String)::Vector{Date}
    sort([Date(item[1:10], dateformat"yyyy.mm.dd") for item in HTTP_regex("https://e4ftl01.cr.usgs.gov/VIIRS/$(product)", LPDAAC_DATE_DIRECTORY_REGEX)])
end

function VIIRS_most_recent_date(product::String)::Date
    VIIRS_dates(product)[end]
end

function VIIRS_URLs(product::String, date::Union{Date,String}; tile::Union{String,Nothing}=nothing)::Vector{String}
    date = Date(date)
    date_string = Dates.format(date, "yyyy.mm.dd")
    URL = "https://e4ftl01.cr.usgs.gov/VIIRS/$(product)/$(date_string)"
    @info "searching files: $(URL)"
    filenames = HTTP_filenames(URL, [".h5", ".xml", ".jpg"])

    if tile !== nothing
        filenames = [filename for filename in filenames if occursin(tile, filename)]
    end

    URLs = ["$(URL)/$(filename)" for filename in filenames]

    return URLs
end

export VIIRS_URLs

function generate_VIIRS_filename(
    product::String,
    date::Union{Date,String},
    URL::String,
    download_directory::String)::String
    joinpath(
        expanduser(download_directory),
        product,
        Dates.format(date, "yyyy.mm.dd"),
        String(split(URL, "/")[end])
    )
end

export generate_VIIRS_filename

"function to download HLS at tile in date range"
function download_VIIRS_tile(
    product::String,
    tile::String,
    date::Union{Date,String},
    download_directory::String,
    username::String,
    password::String)::String
    date = Date(date)
    @info "downloading VIIRS $(product) at tile $(tile) on $(date)"
    download_directory = expanduser(download_directory)
    @info "VIIRS download directory: $(download_directory)"
    mkpath(download_directory)

    URLs = VIIRS_URLs(product, date, tile=tile)

    granule_filename = ""

    for URL in URLs
        @info "URL: $(URL)"
        filename = generate_VIIRS_filename(product, date, URL, download_directory)
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

        if endswith(filename, ".h5")
            granule_filename = filename
        end
    end

    return granule_filename
end

export download_VIIRS_tile

end
