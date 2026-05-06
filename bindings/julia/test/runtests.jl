using Test

include("contract.jl")
include("wrappers.jl")
include("packaging.jl")
if get(ENV, "INNOVATE_JULIA_RUN_INSTALLED_PACKAGE_SMOKE", "false") == "true"
    include("installed_package_smoke.jl")
else
    include("runtime_paths.jl")
end
include("schema_compatibility.jl")
