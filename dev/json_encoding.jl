using JSON
using Base64

filename = "test_file.txt"
credentials = Dict("username" => "someuser", "password" => "somepass")
println("credentials: $(credentials)")
credentials_json = json(credentials)
println("credentials JSON: $(credentials_json)")
encoded_credentials = base64encode(credentials_json)
println("encoded credentials: $(encoded_credentials)")

open(filename, "w") do file
    write(file, encoded_credentials)
end

read_credentials = open(filename, "r") do file
    String(read(file))
end

println("read credentials: $(read_credentials)")
decoded_credentials = String(base64decode(read_credentials))
println("decoded credentials: $(decoded_credentials)")
credentials_dict = JSON.parse(decoded_credentials)
println("credentials: $(credentials_dict)")
