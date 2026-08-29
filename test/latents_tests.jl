using Test
using BayesianFootball
using BayesianFootball.Models
using BayesianFootball.Predictions
using BayesianFootball.Data
using Distributions
using DataFrames

struct TypedLatentsMockModel <: BayesianFootball.AbstractFootballModel end

@testset "Typed posterior latents" begin
    ids = [101, 102]
    λh = [1.2 1.4 1.1; 0.8 1.0 1.3]
    λa = [0.9 1.1 0.7; 1.4 1.2 1.0]
    poisson = CountLatents(ids, λh, λa)

    @test poisson isa AbstractPosteriorLatents
    @test n_matches(poisson) == 2
    @test n_draws(poisson) == 3
    @test latent_match_ids(poisson) == ids
    @test match_index(poisson, 102) == 2
    @test match_index(poisson, -1) == 0
    @test observation_family(poisson) === :poisson
    @test latent_allocations(poisson) == 2
    @test_throws ErrorException CountLatents([1, 1], λh, λa)
    @test_throws ErrorException CountLatents(ids, copy(λh), fill(0.0, 2, 3))

    ws = GridWorkspace(12)
    S = alloc_score_grid(poisson, 12)
    @test compute_score_grid!(S, ws, poisson, 1) === S
    @test size(S) == (12, 12, 3)
    for k in 1:n_draws(poisson), a in 0:11, h in 0:11
        @test S[h + 1, a + 1, k] == pdf(Poisson(λh[1, k]), h) * pdf(Poisson(λa[1, k]), a)
    end

    one_x_two = Market1X2()
    book = alloc_market_book(one_x_two, n_draws(poisson))
    @test price_market!(book, S, one_x_two) === book
    @test all((book[1] .+ book[2] .+ book[3]) .≈ vec(sum(S, dims = (1, 2))))

    # Warm the kernels before checking their steady-state allocation contract.
    compute_score_grid!(S, ws, poisson, 1)
    price_market!(book, S, one_x_two)
    @test @allocated(compute_score_grid!(S, ws, poisson, 1)) == 0
    @test @allocated(price_market!(book, S, one_x_two)) == 0

    r_h = fill(4.0, 2, 3)
    r_a = fill(5.0, 2, 3)
    negbin = CountLatents(ids, λh, λa, (; r_h, r_a))
    @test observation_family(negbin) === :negbin
    @test latent_allocations(negbin) == 4
    Snb = alloc_score_grid(negbin)
    compute_score_grid!(Snb, ws, negbin, 2)
    @test all(isfinite, Snb)
    @test all(>(0.0), sum(Snb, dims = (1, 2)))

    recomb = RecombLatents(ids,
        0.8 .* λh, 0.8 .* λa,
        0.15 .* λh, 0.15 .* λa,
        0.05 .* λh, 0.05 .* λa,
        0.8 .* λh, 0.8 .* λa)
    @test observation_family(recomb) === :recombination
    Sr = alloc_score_grid(recomb)
    compute_score_grid!(Sr, ws, recomb, 1)
    @test all(vec(sum(Sr, dims = (1, 2))) .≈ 1.0)

    strikes = [0.5, 1.5, 2.5]
    φ = ones(2, 3, 3)
    φ[:, 3, :] .= 1.25
    smile = SmileLatents(ids, λh, λa, nothing, λh .+ λa, φ, strikes)
    @test observation_family(smile) === :smile_poisson
    @test n_strikes(smile) == 3
    smile_grid = compute_score_grid(smile, 1)
    @test smile_grid isa SmileScoreGrid
    ou = MarketOverUnder(2.5)
    smile_book = alloc_market_book(ou, n_draws(smile))
    price_market!(smile_book, smile_grid, ou)
    @test smile_book[1] .+ smile_book[2] == ones(n_draws(smile))

    legacy = to_legacy_dataframe(negbin)
    restored = latents_from_legacy_dataframe(NegBinCountFamily(), legacy)
    @test restored.λ_home == negbin.λ_home
    @test restored.observation_params.r_h == negbin.observation_params.r_h
    @test_throws ErrorException latents_from_legacy_dataframe(TypedLatentsMockModel(), legacy)
end

@testset "Typed model inference" begin
    ids = [201, 202]
    λh = fill(1.3, 2, 4)
    λa = fill(0.9, 2, 4)
    latents = CountLatents(ids, λh, λa)
    model = TypedLatentsMockModel()
    config = MarketConfig(Market1X2(), MarketBTTS(), MarketOverUnder(2.5))

    ppd = model_inference(latents, model; market_config = config)
    @test ppd.model === model
    @test sort(unique(ppd.df.match_id)) == ids
    @test nrow(ppd.df) == 14
    @test all(length(d) == 4 for d in ppd.df.distribution)
    @test model_inference(latents; model = model, market_config = config) === ppd
    @test_throws ErrorException model_inference(latents; market_config = config)
end
