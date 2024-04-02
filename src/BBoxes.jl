module BBoxes

using Rasters

include("Points.jl")

using .Points: Point

"struct representing two-dimensional bounding box as min and max x and y values"
struct BBox
    xmin::Float64
    ymin::Float64
    xmax::Float64
    ymax::Float64
end

export BBox

"constructor for bounding box from raster"
function BBox(raster::Raster)::BBox
    if length(size(raster)) == 3
        ((xmin, xmax), (ymin, ymax)) = bounds(raster[:,:,1])[1:2]
    else
        ((xmin, xmax), (ymin, ymax)) = bounds(raster)[1:2]
    end
    
    bbox = BBox(xmin, ymin, xmax, ymax)
    
    return bbox
end



export BBox

"function to expand bounding box by the same distance in each direction"
function buffer(bbox::BBox, buffer::Float64)::BBox
    return BBox(bbox.xmin - buffer, bbox.ymin - buffer, bbox.xmax + buffer, bbox.ymax + buffer)
end

export buffer

"function to calculate the centroid of a bounding box"
function calculate_centroid(bbox::BBox)::Point
    Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2)
end

export calculate_centroid

end
