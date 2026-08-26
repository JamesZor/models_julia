using SpecialFunctions
function tp02_equation_data(m,fs)
 d=_dat(m,fs); (;home=d.h,away=d.a,yh=d.yh,ya=d.ya,weights=d.wt,wealth=d.xw,distance=zeros(Float64,length(d.h)),lfh=loggamma.(Float64.(d.yh).+1),lfa=loggamma.(Float64.(d.ya).+1))
end
function tp02_logjoint(m,p,d)
 α,β=slfp_team_effects(p); q=p.w_wealth.*d.wealth; eh=clamp.(p.μ.+p.γ.+α[d.home].+β[d.away].+q,-10.,10.); ea=clamp.(p.μ.+α[d.away].+β[d.home].-q,-10.,10.); lp=logpdf(m.interception_config.μ,p.μ)+logpdf(m.homeadvantage_config.γ_global,p.γ)+logpdf(m.dynamics_config.σ_att,p.σ_a)+logpdf(m.dynamics_config.σ_def,p.σ_d)+sum(logpdf.(Normal(),p.raw_a))+sum(logpdf.(Normal(),p.raw_d))+logpdf(m.w_wealth_prior,p.w_wealth); lp+sum(d.weights.*(d.yh.*eh.-exp.(eh).-d.lfh))+sum(d.weights.*(d.ya.*ea.-exp.(ea).-d.lfa))
end