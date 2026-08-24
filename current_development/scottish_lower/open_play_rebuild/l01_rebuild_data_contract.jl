module RebuildDataContract

using DataFrames

# History-only Stage-2 incident ledger. This module does not mutate `ds` or write files.
export audit_component_history, select_own_goal_hypothesis

# Keep schema access local: cache snapshots have evolved, but these are the provider names.
_col(df, candidates; required::Bool = true) = begin
    found = findfirst(c -> c in propertynames(df), candidates)
    found === nothing && required && throw(ArgumentError("missing required column; expected one of $(candidates), found $(propertynames(df))"))
    found === nothing ? nothing : candidates[found]
end
_value(r, col, default = missing) = col === nothing ? default : r[col]
_text(x) = ismissing(x) ? "" : String(x)
_id(x) = ismissing(x) ? nothing : try Int(x) catch; nothing end
_side(x) = ismissing(x) ? nothing : x isa Bool ? x : (x == 1 || lowercase(string(x)) in ("home", "true") ? true : (x == 0 || lowercase(string(x)) in ("away", "false") ? false : nothing))

"""Return `:beneficiary`, `:committing`, or `nothing` under an explicit policy.

`:validated_unique` selects only when exactly one hypothesis has passed the complete
incident-to-official reconciliation. It deliberately does not select the indistinguishable
no-own-goal case.
"""
function select_own_goal_hypothesis(row; policy::Symbol = :validated_unique)
    policy == :validated_unique || throw(ArgumentError("unsupported selection policy: $policy"))
    b = Bool(row.beneficiary_valid)
    c = Bool(row.committing_valid)
    return b == c ? nothing : (b ? :beneficiary : :committing)
end

"""
    audit_component_history(ds, history_match_ids; target_match_ids = Int[])

Build a non-mutating, history-only component ledger. Incident rows are restricted before any
classification; duplicate provider incident IDs are retained only once. `is_home = missing` is
never imputed: its component is excluded and its match is quarantined. Own-goal side remains two
parallel hypotheses (`beneficiary` means provider side is the scorer; `committing` flips it).
"""
function audit_component_history(ds, history_match_ids; target_match_ids = Int[])
    history = Set(Int.(collect(history_match_ids)))
    targets = Set(Int.(collect(target_match_ids)))
    isempty(history) && throw(ArgumentError("history_match_ids must not be empty"))
    overlap = intersect(history, targets)
    isempty(overlap) || throw(ArgumentError("history/target ID leakage: $(collect(overlap))"))

    matches, incidents = ds.matches, ds.incidents
    mid_m = _col(matches, (:match_id, :id)); mid_i = _col(incidents, (:match_id, :fixture_id))
    iid = _col(incidents, (:id, :incident_id)); typ = _col(incidents, (:incident_type, :type))
    cls = _col(incidents, (:incident_class, :incidentClass); required = false)
    side = _col(incidents, (:is_home, :isHome); required = false)
    resc = _col(incidents, (:rescinded, :is_rescinded); required = false)
    home_score = _col(matches, (:home_score, :homeScore)); away_score = _col(matches, (:away_score, :awayScore))
    tourn = _col(matches, (:tournament_id, :tournamentId); required = false)
    season = _col(matches, (:season, :season_id, :seasonId); required = false)

    match_rows = Dict{Int, Any}()
    duplicate_match_ids = Int[]
    for r in eachrow(matches)
        m = _id(r[mid_m]); m === nothing && continue
        if m in history
            haskey(match_rows, m) && push!(duplicate_match_ids, m)
            match_rows[m] = r
        end
    end
    missing_matches = sort!(collect(setdiff(history, Set(keys(match_rows)))))

    # Restrict first, then deduplicate. A repeated ID is diagnostic even when byte-identical.
    seen = Set{Int}(); duplicate_ids = Int[]; missing_incident_ids = Int[]
    event_rows = NamedTuple[]
    per_match = Dict{Int, Vector{NamedTuple}}()
    for r in eachrow(incidents)
        m = _id(r[mid_i]); (m === nothing || !(m in history)) && continue
        @assert !(m in targets) "target incident leaked into history audit"
        e = _id(r[iid])
        if e === nothing
            push!(missing_incident_ids, m)
            ev = (match_id=m, incident_id=missing, classification=:unusable_missing_incident_id,
                is_home=missing, rescinded=false, included=false, note="missing incident id")
            push!(event_rows, ev); push!(get!(per_match, m, NamedTuple[]), ev)
            continue
        elseif e in seen
            push!(duplicate_ids, e); continue
        end
        push!(seen, e)
        isresc = coalesce(_value(r, resc, false), false) === true
        t, c, s = _text(_value(r, typ)), _text(_value(r, cls)), _side(_value(r, side))
        kind = t == "goal" && c == "penalty" ? :converted_penalty :
               t == "inGamePenalty" ? :missed_award :
               t == "goal" && c == "ownGoal" ? :own_goal :
               t == "goal" ? :ordinary_goal : :other
        included = kind != :other && !isresc && s !== nothing
        note = isresc ? "rescinded component excluded" : s === nothing && kind != :other ? "missing is_home; not imputed" : ""
        ev = (match_id=m, incident_id=e, classification=kind, is_home=s === nothing ? missing : s,
              rescinded=isresc, included=included, note=note)
        push!(event_rows, ev); push!(get!(per_match, m, NamedTuple[]), ev)
    end

    ledger = DataFrame(match_id=Int[], tournament_id=Any[], season=Any[], official_G_h=Union{Missing,Int}[], official_G_a=Union{Missing,Int}[],
        penalty_C_h=Int[], penalty_C_a=Int[], penalty_A_h=Int[], penalty_A_a=Int[], missed_award_h=Int[], missed_award_a=Int[],
        raw_ordinary_h=Int[], raw_ordinary_a=Int[], own_goal_beneficiary_h=Int[], own_goal_beneficiary_a=Int[], own_goal_committing_h=Int[], own_goal_committing_a=Int[],
        np_nog_Y_beneficiary_h=Union{Missing,Int}[], np_nog_Y_beneficiary_a=Union{Missing,Int}[], np_nog_Y_committing_h=Union{Missing,Int}[], np_nog_Y_committing_a=Union{Missing,Int}[],
        residual_beneficiary_h=Union{Missing,Int}[], residual_beneficiary_a=Union{Missing,Int}[], residual_committing_h=Union{Missing,Int}[], residual_committing_a=Union{Missing,Int}[],
        nonnegative_ok=Bool[], beneficiary_nonnegative_ok=Bool[], committing_nonnegative_ok=Bool[], conversion_le_awards_ok=Bool[], beneficiary_accounting_ok=Bool[], committing_accounting_ok=Bool[], beneficiary_valid=Bool[], committing_valid=Bool[], quarantine_reasons=String[])

    for m in sort!(collect(history))
        r = get(match_rows, m, nothing)
        if r === nothing
            push!(ledger, (m, missing, missing, missing, missing, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, missing,missing,missing,missing, false,false,false,false,false,false,false,false,"missing_match")); continue
        end
        gh, ga = _value(r, home_score), _value(r, away_score)
        gh = ismissing(gh) ? missing : Int(gh); ga = ismissing(ga) ? missing : Int(ga)
        counts = Dict{Symbol,NTuple{2,Int}}(k => (0,0) for k in (:converted_penalty,:missed_award,:ordinary_goal,:own_goal_b,:own_goal_c))
        reasons = String[]
        for ev in get(per_match, m, NamedTuple[])
            ev.rescinded && ev.classification != :other && push!(reasons, "rescinded_component_incident")
            ev.classification == :unusable_missing_incident_id && push!(reasons, "missing_incident_id")
            ev.note == "missing is_home; not imputed" && push!(reasons, "missing_is_home")
            !ev.included && continue
            h = ev.is_home === true
            key = ev.classification == :own_goal ? :own_goal_b : ev.classification
            old = counts[key]; counts[key] = h ? (old[1]+1,old[2]) : (old[1],old[2]+1)
            if ev.classification == :own_goal # committing-side interpretation flips scoring side
                oldc = counts[:own_goal_c]; counts[:own_goal_c] = h ? (oldc[1],oldc[2]+1) : (oldc[1]+1,oldc[2])
            end
        end
        C, M, ordinary, Ob, Oc = counts[:converted_penalty], counts[:missed_award], counts[:ordinary_goal], counts[:own_goal_b], counts[:own_goal_c]
        A = (C[1]+M[1], C[2]+M[2])
        nonneg = all(x -> x >= 0, (C..., M..., ordinary..., Ob..., Oc...))
        clea = C[1] <= A[1] && C[2] <= A[2]

        # DESIGN canonical NP-NOG is score-derived under each own-goal convention;
        # ordinary incidents are an independent reconciliation count, never the target.
        Yb = (ismissing(gh) ? missing : gh-C[1]-Ob[1], ismissing(ga) ? missing : ga-C[2]-Ob[2])
        Yc = (ismissing(gh) ? missing : gh-C[1]-Oc[1], ismissing(ga) ? missing : ga-C[2]-Oc[2])
        bnonneg = !any(ismissing, Yb) && all(x -> x >= 0, Yb)
        cnonneg = !any(ismissing, Yc) && all(x -> x >= 0, Yc)
        rb = (ismissing(Yb[1]) ? missing : Yb[1]-ordinary[1], ismissing(Yb[2]) ? missing : Yb[2]-ordinary[2])
        rc = (ismissing(Yc[1]) ? missing : Yc[1]-ordinary[1], ismissing(Yc[2]) ? missing : Yc[2]-ordinary[2])
        bok = !any(ismissing, rb) && rb == (0,0)
        cok = !any(ismissing, rc) && rc == (0,0)
        (ismissing(gh) || ismissing(ga)) && push!(reasons, "missing_official_score")
        event_defect = any(x -> x in ("missing_is_home", "missing_incident_id", "rescinded_component_incident"), reasons)
        validb = nonneg && bnonneg && clea && bok && !event_defect
        validc = nonneg && cnonneg && clea && cok && !event_defect
        # An alternative hypothesis failing is evidence, not a quarantine. Reconciliation
        # quarantines only when neither convention produces a valid decomposition.
        !validb && !validc && push!(reasons, "component_reconciliation_failed")
        push!(ledger, (m, _value(r,tourn,missing), _value(r,season,missing), gh,ga,C[1],C[2],A[1],A[2],M[1],M[2],ordinary[1],ordinary[2],Ob[1],Ob[2],Oc[1],Oc[2],Yb[1],Yb[2],Yc[1],Yc[2],rb[1],rb[2],rc[1],rc[2],nonneg,bnonneg,cnonneg,clea,bok,cok,validb,validc,join(unique(reasons),";")))
    end
    # One row per reason gives a reviewable tournament/season/reason quarantine summary.
    reason_rows = NamedTuple[]
    for r in eachrow(ledger), reason in split(r.quarantine_reasons, ';')
        isempty(reason) || push!(reason_rows, (tournament_id=r.tournament_id, season=r.season, reason=reason, matches=1))
    end
    quarantine_summary = isempty(reason_rows) ? DataFrame(tournament_id=Any[],season=Any[],reason=String[],matches=Int[]) : combine(groupby(DataFrame(reason_rows), [:tournament_id,:season,:reason]), :matches => sum => :matches)
    summary = combine(groupby(ledger, [:tournament_id,:season]), nrow => :matches, :beneficiary_valid => sum => :beneficiary_valid, :committing_valid => sum => :committing_valid)
    # Informative evidence excludes no-own-goal ties and never chooses a global policy.
    own_goal_hypothesis_evidence = filter(r ->
        (r.own_goal_beneficiary_h + r.own_goal_beneficiary_a + r.own_goal_committing_h + r.own_goal_committing_a > 0) &&
        xor(r.beneficiary_valid, r.committing_valid), ledger)
    own_goal_hypothesis_evidence = isempty(own_goal_hypothesis_evidence) ?
        DataFrame(tournament_id=Any[], season=Any[], matches=Int[], beneficiary_only_valid=Int[], committing_only_valid=Int[]) :
        combine(groupby(own_goal_hypothesis_evidence, [:tournament_id, :season]), nrow => :matches,
            :beneficiary_valid => sum => :beneficiary_only_valid,
            :committing_valid => sum => :committing_only_valid)
    return (ledger=ledger, events=DataFrame(event_rows), summary=summary, own_goal_hypothesis_evidence=own_goal_hypothesis_evidence, quarantine_summary=quarantine_summary,
        diagnostics=(duplicate_incident_ids=sort!(unique(duplicate_ids)), duplicate_incident_count=length(duplicate_ids), missing_incident_id_matches=sort!(unique(missing_incident_ids)), missing_side_events=count(r -> r.note == "missing is_home; not imputed", event_rows), rescinded_component_events=count(r -> r.rescinded && r.classification != :other, event_rows), missing_match_ids=missing_matches, duplicate_match_ids=sort!(unique(duplicate_match_ids))))
end

end # module
