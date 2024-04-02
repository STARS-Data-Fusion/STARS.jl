module Points

import ArchGDAL as AG

"struct representing two-dimensional point as x and y values"
struct Point
    x::Float64 # projected x or geographic longitude
    y::Float64 # projected y or geographic latitude
end

export Point

"type definition for ArchGDAL point"
AGPoint = AG.IGeometry{AG.wkbPoint}

export AGPoint

"function to convert ArchGDAL projected point to Point"
function from_AG(point::AGPoint)::Point
    Point(AG.getx(point, 0), AG.gety(point, 0))
end

export from_AG

"function to convert ArchGDAL geographic point to Point"
function from_AG_latlon(point::AGPoint)::Point
    # apparently geographic points in ArchGDAL are (latitude, longitude), reversed from (x, y) used for projected points
    Point(AG.gety(point, 0), AG.getx(point, 0))
end

export from_AG_latlon

"function to convert projected Point to ArchGDAL point"
function to_AG(point::Point)::AGPoint
    AG.createpoint(point.x, point.y)
end

"function to convert geographic Point to ArchGDAL point"
function to_AG_latlon(point::Point)::AGPoint
    AG.createpoint(point.y, point.x)
end

export to_AG

end
