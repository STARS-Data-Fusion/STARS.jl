using JSON
using Base64

DISPLAYED = ["username"]
HIDDEN = ["password"]

function enter_credentials(
    displayed::Vector{String}=DISPLAYED,
    hidden::Vector{String}=HIDDEN)::Dict{String,String}
    credentials = Dict()

    for key in displayed
        credentials[key] = Base.prompt(key)
    end

    for key in hidden
        hidden_buffer = Base.getpass(key)
        credentials[key] = read(hidden_buffer, String)
        Base.shred!(hidden_buffer)
    end

    return credentials
end

function encode_credentials(credentials::Dict{String,String})::String
    String(base64encode(JSON.json(credentials)))
end

function decode_credentials(credentials::String)::Dict{String,String}
    JSON.parse(String(base64decode(credentials)))
end

function write_credentials(credentials::Dict{String,String}, filename::String)
    open(filename, "w") do file
        write(file, encode_credentials(credentials))
    end
end

function read_credentials(filename::String)::Dict{String,String}
    open(filename, "r") do file
        decode_credentials(read(file, String))
    end
end

function get_credentials(
    filename::String,
    displayed::Vector{String}=DISPLAYED,
    hidden::Vector{String}=HIDDEN)

end

filename = "credentials.txt"
credentials = enter_credentials()
write_credentials(credentials, filename)
retrieved_credentials = read_credentials(filename)
println(retrieved_credentials)
