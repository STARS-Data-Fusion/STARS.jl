println("PATH:")
println(ENV["PATH"])
# println("wget:")
# command = "wget"
# wget_location = read(`$command`)
println("downloading URL")
URL = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T11SPS.2022275T182201.v2.0/HLS.S30.T11SPS.2022275T182201.v2.0.B8A.tif"
command = "wget $(URL)"
run(`wget $URL`)
