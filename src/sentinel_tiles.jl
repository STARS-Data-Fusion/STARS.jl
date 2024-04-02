module sentinel_tiles

import GeoDataFrames as GDF
import ArchGDAL as AG
using Rasters
using DimensionalData.Dimensions.LookupArrays

include("Points.jl")

using .Points: Point

include("BBoxes.jl")

using .BBoxes: BBox

SENTINEL_FILENAME = joinpath(@__DIR__, "sentinel2.geojson")
SENTINEL_DF = GDF.read(SENTINEL_FILENAME)

"function to query internally stored information for Sentinel tile"
function sentinel_tile_record(tile::String)
    search_results = SENTINEL_DF[isequal.(SENTINEL_DF.tile, tile), :]

    if size(search_results)[1] == 0
        error("tile $(tile) not found")
    end

    tile_row = search_results[1, :]

    return tile_row
end

export sentinel_tile_record

"function to look up the polygon boundary of a Sentinel tile in latitude and longitude"
# function sentinel_tile_polygon(tile::String)::AG.IGeometry{AG.wkbPolygon25D}
#     tile_row = sentinel_tile_record(tile)
#     polygon = tile_row.geometry[1]

#     return polygon
# end

# export sentinel_tile_polygon

"function to look up bounding box of Sentinel tile"
function sentinel_tile_bbox(tile::String)::BBox
    tile_row = sentinel_tile_record(tile)
    bbox = BBox(tile_row.xmin, tile_row.ymin, tile_row.xmax, tile_row.ymax)

    return bbox
end

export sentinel_tile_bbox

"function to calculate the UTM centroid of a Sentinel tile"
function sentinel_tile_centroid(tile::String)::Point
    calculate_centroid(sentinel_tile_bbox(tile))
end

export sentinel_tile_centroid

"function to calculate the geographic centroid of a Sentinel tile"
function sentinel_tile_centroid_latlon(tile::String)::Point
    centroid_UTM = sentinel_tile_centroid(tile)
    tile_EPSG = sentinel_tile_EPSG(tile)
    centroid_latlon = from_AG_latlon(AG.reproject(to_AG(centroid_UTM), EPSG(tile_EPSG), EPSG(4326)))

    return centroid_latlon
end

export sentinel_tile_centroid_latlon

"function to look up the EPSG projection code for a Sentinel tile"
function sentinel_tile_EPSG(tile::String)::Int64
    tile_row = sentinel_tile_record(tile)

    if tile_row.EPSG == ""
        error("no projection found for tile $(tile)")
    end

    tile_EPSG = parse(Int64, tile_row.EPSG)

    return tile_EPSG
end

export sentinel_tile_EPSG

function sentinel_tile_CRS(tile::String)
    EPSG(sentinel_tile_EPSG(tile))
end

export sentinel_tile_CRS

"function to generate the raster dimensions of a Sentinel tile"
function sentinel_tile_dims(tile::String, cell_size::Union{Float64,Int64})::Tuple{X,Y}
    bbox = sentinel_tile_bbox(tile)
    xmin = bbox.xmin
    ymin = bbox.ymin
    xmax = bbox.xmax
    ymax = bbox.ymax
    tile_EPSG = sentinel_tile_EPSG(tile)
    cols = Int(trunc((ymax - ymin) / cell_size))
    rows = Int(trunc((xmax - xmin) / cell_size))
    x_dim = X(Projected(LinRange(xmin, xmin + (rows - 1) * cell_size, rows), order=ForwardOrdered(), span=Regular(cell_size), sampling=Intervals(Start()), crs=convert(WellKnownText, EPSG(tile_EPSG))))
    y_dim = Y(Projected(LinRange(ymax - cell_size, ymax - cols * cell_size, cols), order=ReverseOrdered(), span=Regular(-cell_size), sampling=Intervals(Start()), crs=convert(WellKnownText, EPSG(tile_EPSG))))
    dims = (x_dim, y_dim)

    return dims
end

export sentinel_tile_dims

function sentinel_tile_polygon(tile::String)::AG.IGeometry{AG.wkbPolygon}
    bbox = sentinel_tile_bbox(tile)
    polygon = AG.createpolygon([(bbox.xmin, bbox.ymax), (bbox.xmax, bbox.ymax), (bbox.xmax, bbox.ymin), (bbox.xmin, bbox.ymin)])

    return polygon
end

export sentinel_tile_polygon

function sentinel_tile_polygon_latlon(tile::String)::AG.IGeometry{AG.wkbPolygon}
    polygon = sentinel_tile_polygon(tile)
    crs = sentinel_tile_CRS(tile)
    polygon_latlon = AG.reproject(polygon, crs, EPSG(4326))
end

export sentinel_tile_polygon_latlon

"function to search for Sentinel tiles intersecting given geometry"
function find_sentinel_tiles(geometry::AG.IGeometry)::Vector{String}
    SENTINEL_DF[ArchGDAL.intersects.(SENTINEL_DF.geometry, [geometry]), :].tile
end

export find_sentinel_tiles

end