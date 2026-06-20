using Random

function test_tau()
    rng = Random.default_rng()
    for _ in 1:10_000_000
        λ_h = exp(randn(rng) * 5 + 2) # can be huge
        λ_a = exp(randn(rng) * 5 + 2)
        ρ = randn(rng) * 2.0
        
        mx_rho = min(1.0 / (λ_h * λ_a) - 1e-4, 1.0 - 1e-4)
        mn_rho = max(-1.0 / λ_h + 1e-4, -1.0 / λ_a + 1e-4)
        
        r = clamp(ρ, mn_rho, mx_rho)
        
        t00 = 1.0 - (λ_h * λ_a * r)
        t10 = 1.0 + (λ_a * r)
        t01 = 1.0 + (λ_h * r)
        t11 = 1.0 - r
        
        if t00 <= 0 || t10 <= 0 || t01 <= 0 || t11 <= 0
            println("CRASH FOUND!")
            println("λ_h = $λ_h, λ_a = $λ_a, ρ = $ρ")
            println("mx_rho = $mx_rho, mn_rho = $mn_rho, r = $r")
            println("t00 = $t00, t10 = $t10, t01 = $t01, t11 = $t11")
            return
        end
    end
    println("No crash found in 10M tests.")
end

test_tau()
