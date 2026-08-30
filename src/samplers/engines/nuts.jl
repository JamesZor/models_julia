# src/samplers/engines/nuts.jl
using Logging
export NUTSConfig, QueuedNUTSConfig

# --- 1. Configuration ---

struct NUTSConfig <: AbstractNUTSConfig 
    n_samples::Int
    n_chains::Int
    n_warmup::Int
    accept_rate::Real 
    max_depth::Int
    initialisation::AbstractInitStrategy 
    show_progress::Union{Symbol, Bool}
    silence_initial_stepsize::Bool
end

# Smart Constructor
function NUTSConfig(; 
    n_samples=1000, 
    n_chains=4, 
    n_warmup=500, 
    accept_rate=0.65,
    max_depth=10,
    show_progress=:perchain,
    # Helper to allow passing :map or :uniform symbol for convenience
    init_type=:uniform, 
    map_iters=50,
    jitter=0.001,
    initialisation=nothing,
    silence_initial_stepsize::Bool=true
)
    # Factory logic to create the correct strategy object
    strategy = if !isnothing(initialisation)
        initialisation
    elseif init_type == :map
        MapInit(max_iters=map_iters)
    else
        UniformInit()
    end

    return NUTSConfig(n_samples, n_chains, n_warmup, accept_rate, max_depth, strategy, show_progress, silence_initial_stepsize)
end

struct QueuedNUTSConfig <: AbstractNUTSConfig 
    n_samples::Int
    n_chains::Int
    n_warmup::Int
    accept_rate::Real 
    max_depth::Int
    initialisation::AbstractInitStrategy 
    show_progress::Union{Symbol, Bool}
    silence_initial_stepsize::Bool
end

function QueuedNUTSConfig(; 
    n_samples=1000, 
    n_chains=4, 
    n_warmup=500, 
    accept_rate=0.65,
    max_depth=10,
    show_progress=false,
    init_type=:uniform, 
    map_iters=50,
    jitter=0.001,
    initialisation=nothing,
    silence_initial_stepsize::Bool=true
)
    strategy = if !isnothing(initialisation)
        initialisation
    elseif init_type == :map
        MapInit(max_iters=map_iters)
    else
        UniformInit()
    end

    return QueuedNUTSConfig(n_samples, n_chains, n_warmup, accept_rate, max_depth, strategy, show_progress, silence_initial_stepsize)
end

# --- 2. Display ---

function Base.show(io::IO, ::MIME"text/plain", c::AbstractNUTSConfig)
    printstyled(io, "$(typeof(c))\n", color=:cyan, bold=true)
    println(io, "==========")
    println(io, "  Samples: $(c.n_samples)")
    println(io, "  Chains:  $(c.n_chains)")
    println(io, "  Accept Rate:  $(c.accept_rate)")
    println(io, "  Max Depth:  $(c.max_depth)")
    println(io, "  Init:    $(c.initialisation)") 
end

# --- 3. Execution ---

function run_sampler(turing_model, config::NUTSConfig)
    # Delegate initialisation logic to the strategy
    init_params = get_init_params(turing_model, config.initialisation, config.n_chains)
    
    logger = config.silence_initial_stepsize ? ConsoleLogger(stderr, Logging.Warn) : current_logger()
    
    chain = with_logger(logger) do
        sample(
            turing_model, 
            NUTS(config.n_warmup, config.accept_rate, max_depth=config.max_depth), 
            MCMCThreads(), 
            config.n_samples, 
            config.n_chains,
            progress = config.show_progress,
            adtype = AutoReverseDiff(compile=true),
            initial_params = init_params # Injection
        )
    end
    return chain
end

function run_sampler(turing_model, config::QueuedNUTSConfig, chain_id::Int)
    # Get params for this specific chain. 
    # get_init_params should handle getting the right slice or single vector if needed.
    # Actually, get_init_params returns a vector of vectors for N chains.
    # We will get all of them and just pick the one for `chain_id`.
    all_init_params = get_init_params(turing_model, config.initialisation, config.n_chains)
    init_params = all_init_params[chain_id]
    
    logger = config.silence_initial_stepsize ? ConsoleLogger(stderr, Logging.Warn) : current_logger()
    
    chain = with_logger(logger) do
        sample(
            turing_model, 
            NUTS(config.n_warmup, config.accept_rate, max_depth=config.max_depth), 
            config.n_samples, 
            progress = false, # explicitly disable to avoid conflict with ProgressMeter
            adtype = AutoReverseDiff(compile=true),
            initial_params = init_params # Injection
        )
    end
    return chain
end
