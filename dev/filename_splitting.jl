filename = "/Users/halverso/data/HLS_julia_testing/2022.10.02/HLS.S30.T11SPS.2022275T182201.v2.0/HLS.S30.T11SPS.2022275T182201.v2.0.B8A.tif"
directory, filename_base = splitdir(filename)
println(directory)
println(filename_base)
mkpath(directory)
