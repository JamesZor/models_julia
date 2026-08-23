# 5. AD Performance, Masking & Stability

**Module:** `BayesianFootball.Models.PreGame`  
**Backend:** Turing.jl, ReverseDiff.jl, DynamicPPL.jl

---

## ⚡ 1. The ReverseDiff Static Tape Contract

ReverseDiff compiles the execution graph of a Turing `@model` into a static gradient tape (`ReverseDiff.GradientTape`). To achieve maximum sampling throughput and eliminate memory allocations:

1. **No Runtime Dynamic Conditionals (`if / else`):**
   Conditional execution based on data presence breaks tape compilation or causes tape thrashing.
2. **Strict Vectorization:**
   Avoid scalar loops over match rows inside the Turing `@model`. Vectorized broadcasting (`logpdf.(Distribution, array)`) allows the AD tape to optimize SIMD operations.
3. **Continuous Support:**
   Continuous observations (`pxG`) and parameters must remain within their valid support domains ($> 0.0$ for Gamma, etc.).

---

## 🎭 2. Zero-Allocation Binary Masking

When historical matches contain missing or unrecorded Proxy xG data (e.g., historical seasons without shot coordinates):

### ❌ Anti-Pattern: Runtime Filter
```julia
# BAD: Dynamic branch alters tape graph per iteration
for i in 1:N
    if has_pxg[i]
        Turing.@addlogprob! logpdf(Gamma(ν, scale[i]), pxg[i])
    end
end
```

### ✅ Best Practice: Impute & Vectorized Mask
```julia
# GOOD: Fully static graph with zero allocations
# 1. Feature extractor imputes dummy 1.0 for missing matches
# 2. Vectorized logpdf is multiplied by binary mask {0.0, 1.0}
ll_pxg_h = logpdf.(Gamma.(ν_xg, scale_pxg_h), pxg_open_h)
Turing.@addlogprob! sum(ll_pxg_h .* mask_pxg_h .* match_weights)
```

---

## 🛡️ 3. Numerical Clamping & Gradient Protection

To prevent explosive gradients ($\text{NaN} / \pm\infty$) during the NUTS warm-up phase:

```julia
# Clamp log-rates before exponentiation
log_μ_open_h = clamp.(inter_match .+ γ_h .+ α_h .- β_a .+ w_shift, -5.0, 4.0)
μ_open_h = exp.(log_μ_open_h)

# Clamp finishing factors
κ_team = exp.(clamp.(log_κ, -0.50, 0.50))
```

---

## 🧵 4. CPU Thread Pinning

When running multi-threaded MCMC sampling across CPU cores:
```julia
using ThreadPinning
pinthreads(:cores)
```
Locking Julia worker threads to physical CPU cores eliminates OS thread migration, preserving L1/L2 cache locality and yielding up to a **2.5x sampling speedup**.
