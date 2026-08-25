# src/data/splitting/display.jl

using Dates

# ==============================================================================
# PRETTY PRINTING (Configuration Objects)
# ==============================================================================

# --- CVConfig Display ---

function Base.show(io::IO, ::MIME"text/plain", config::CVConfig)
    # Title
    printstyled(io, "CVConfig", color=:cyan, bold=true)
    printstyled(io, " (Expanding Window Scheme)\n", color=:light_black)
    println(io, "========")

    # 1. Scope (What data are we touching?)
    printstyled(io, "[Scope]\n", color=:magenta)
    
    print(io, "  Target Seasons: ")
    printstyled(io, join(config.target_seasons, ", "), "\n", color=:white, bold=true)
    
    print(io, "  Tournaments:    ")
    if length(config.tournament_ids) > 5
        print(io, "[$(config.tournament_ids[1])...$(config.tournament_ids[end])] (N=$(length(config.tournament_ids)))")
    else
        print(io, "$(config.tournament_ids)")
    end
    println(io)
    println(io)

    # 2. Dynamics (How does time move?)
    printstyled(io, "[Dynamics]\n", color=:magenta)
    println(io, "  History Depth:  $(config.history_seasons) season(s)")
    println(io, "  Dynamics Col:   :$(config.dynamics_col)")
    
    # Details (Constraints)
    details = String[]
    push!(details, "warmup=$(config.warmup_period)")
    if !isnothing(config.end_dynamics)
        push!(details, "end=$(config.end_dynamics)")
    end
    if config.stop_early
        push!(details, "stop_early=true")
    end
    
    if !isempty(details)
        print(io, "  Constraints:    ")
        printstyled(io, join(details, ", "), "\n", color=:light_black)
    end
end

# Compact inline show (for arrays/logging)
function Base.show(io::IO, config::CVConfig)
    print(io, "CVConfig(Targets=$(config.target_seasons), Hist=$(config.history_seasons))")
end


# --- GroupedCVConfig Display ---

function Base.show(io::IO, ::MIME"text/plain", config::GroupedCVConfig)
    printstyled(io, "GroupedCVConfig", color=:cyan, bold=true)
    printstyled(io, " (Multi-Tournament Scheme)\n", color=:light_black)
    println(io, "===============")

    # 1. Scope (What data are we touching?)
    printstyled(io, "[Scope]\n", color=:magenta)
    
    print(io, "  Target Seasons: ")
    printstyled(io, isempty(config.target_seasons) ? "ALL" : join(config.target_seasons, ", "), "\n", color=:white, bold=true)
    
    print(io, "  Groups:         ")
    print(io, "$(length(config.tournament_groups)) group(s)")
    println(io)
    println(io)

    # 2. Dynamics (How does time move?)
    printstyled(io, "[Dynamics]\n", color=:magenta)
    println(io, "  History Depth:  $(config.history_seasons) season(s)")
    println(io, "  Dynamics Col:   :$(config.dynamics_col)")
    has_pooled_group = any(length(unique(group)) > 1 for group in config.tournament_groups)
    if has_pooled_group
        println(io, "  Pooled Clock:   shared calendar (blank/terminal empty steps skipped)")
    end

    # Details (Constraints)
    details = String[]
    push!(details, "warmup=$(config.warmup_period)")
    if !isnothing(config.end_dynamics)
        push!(details, "end=$(config.end_dynamics)")
    end
    if config.stop_early
        push!(details, has_pooled_group ? "stop_early=true (pooled no-op)" : "stop_early=true")
    end

    if !isempty(details)
        print(io, "  Constraints:    ")
        printstyled(io, join(details, ", "), "\n", color=:light_black)
    end
end

function Base.show(io::IO, config::GroupedCVConfig)
    print(io, "GroupedCVConfig(Targets=$(config.target_seasons), Hist=$(config.history_seasons))")
end

# --- ExpandingWindowCV Display ---

function Base.show(io::IO, ::MIME"text/plain", config::ExpandingWindowCV)
    printstyled(io, "ExpandingWindowCV\n", color=:cyan, bold=true)
    println(io, "=================")

    printstyled(io, "[Seasons]\n", color=:magenta)
    println(io, "  Train:  $(join(config.train_seasons, ", "))")
    println(io, "  Test:   $(join(config.test_seasons, ", "))")
    println(io)

    printstyled(io, "[Configuration]\n", color=:magenta)
    println(io, "  Window Col: :$(config.window_col)")
    println(io, "  Method:     $(config.method)")
end


# --- WindowCV Display ---

function Base.show(io::IO, ::MIME"text/plain", config::WindowCV)
    printstyled(io, "WindowCV", color=:cyan, bold=true)
    printstyled(io, " (Sliding Window)\n", color=:light_black)
    println(io, "========")
    
    printstyled(io, "[Seasons]\n", color=:magenta)
    println(io, "  Base:   $(join(config.base_seasons, ", "))")
    println(io, "  Target: $(join(config.target_seasons, ", "))")
    println(io)

    printstyled(io, "[Window]\n", color=:magenta)
    println(io, "  Size:     $(config.window_size) steps")
    println(io, "  Column:   :$(config.window_col)")
    println(io, "  Ordering: $(config.ordering)")
end
