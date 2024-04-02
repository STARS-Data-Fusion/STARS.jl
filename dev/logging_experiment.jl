using Crayons.Box
using Logging

logger = ConsoleLogger()
global_logger(logger)

tile = "11SPS"

# @info string("tile: ", YELLOW_FG("$(tile)"))
@info string("tile: ", YELLOW_FG(tile))
