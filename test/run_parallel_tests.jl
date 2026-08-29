# ==============================================================================
# Fast Parallel Test Runner for BayesianFootball.jl
# Dispatches individual test suites across concurrent worker processes
# ==============================================================================

using Test
using Dates
using Base.Threads

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const TEST_FILES = [
    "data_tests.jl",
    "splitting_tests.jl",
    "features_tests.jl",
    "pregame_tests.jl",
    "builder_tests.jl",
    "portfolio_tests.jl",
    "matchday_tests.jl",
    "caching_tests.jl",
    "recombination_tests.jl",
    "latents_tests.jl",
    "inference_tests.jl",
    "evaluation_tests.jl",
    "unified_portfolio_tests.jl",
]

println("================================================================================")
println("Running BayesianFootball.jl test suites concurrently across worker processes")
println("================================================================================")

start_time = time()
results = Dict{String, Tuple{Bool, Float64, String}}()
lock_res = ReentrantLock()

sem = Base.Semaphore(4)

@sync for file in TEST_FILES
    test_path = joinpath(@__DIR__, file)
    isfile(test_path) || continue

    Base.acquire(sem)
    @async begin
        t0 = time()
        code = "using Test, BayesianFootball, DataFrames, Dates, InlineStrings; include(\"$(escape_string(test_path))\")"
        cmd = `julia --project=$ROOT_DIR -t 2 -e $code`
        io = IOBuffer()
        success = false
        try
            p = run(pipeline(cmd, stdout=io, stderr=io))
            success = p.exitcode == 0
        catch e
            success = false
        finally
            Base.release(sem)
        end
        elapsed = round(time() - t0, digits=1)
        out_str = String(take!(io))
        lock(lock_res) do
            results[file] = (success, elapsed, out_str)
            status_str = success ? "✓ PASS" : "✗ FAIL"
            println(" [$(rpad(status_str, 6))] $(rpad(file, 28)) ($(elapsed)s)")
        end
    end
end

total_elapsed = round(time() - start_time, digits=1)
all_passed = all(r[1] for r in values(results))

println("================================================================================")
println("Summary: $(count(r[1] for r in values(results))) / $(length(results)) test suites passed in $(total_elapsed)s")
println("================================================================================")

if !all_passed
    println("\nFailed suites details:")
    for (f, (s, _, out)) in results
        if !s
            println("--- $f ---")
            println(out)
        end
    end
    exit(1)
end
