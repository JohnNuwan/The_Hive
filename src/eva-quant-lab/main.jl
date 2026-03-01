#=
EVA Quant-Lab — Moteur de Simulation Quantitative de THE HIVE.

Expert N du système d'experts. Responsable de :
- Optimisation de portefeuille (Markowitz, Mean-Variance).
- Simulation Monte Carlo pour estimation de risque.
- Calcul Value-at-Risk (VaR) et Expected Shortfall (ES).
- Frontière efficiente (Efficient Frontier).
- Stress testing (scénarios adverses).
- Corrélation et covariance de portefeuille.

Architecture :
    - Julia HTTP server (port 8700).
    - Calcul haute performance (BLAS natif Julia).
    - Heartbeat HTTP vers le Core (pas de Redis direct).
=#

using HTTP
using JSON
using Dates
using LinearAlgebra
using Random
using Statistics

println("📈 EVA Quant-Lab (Julia) v1.0 — Initialisation...")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

const PORT = 8701
const SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD",
                 "ETHUSD", "SP500", "NQ100", "DAX40", "CAC40", "AAPL"]
const RISK_FREE_RATE = 0.04  # 4% annuel (2026)
const TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

function json_response(data::Dict; status=200)
    return HTTP.Response(status, ["Content-Type" => "application/json"],
                         body=JSON.json(data))
end

function parse_query(req::HTTP.Request)
    params = Dict{String,String}()
    uri = HTTP.URI(HTTP.target(req))
    query = uri.query
    if !isempty(query)
        for pair in split(query, "&")
            kv = split(pair, "=")
            if length(kv) == 2
                params[kv[1]] = HTTP.unescapeuri(kv[2])
            end
        end
    end
    return params
end

function generate_returns(n_assets::Int, n_days::Int)
    """Génère des rendements simulés réalistes."""
    # Rendements moyens annuels entre -5% et +25%
    mu = (rand(n_assets) .* 0.30 .- 0.05) ./ TRADING_DAYS
    # Volatilité entre 10% et 50% annualisée
    sigma = (rand(n_assets) .* 0.40 .+ 0.10) ./ sqrt(TRADING_DAYS)
    # Matrice de corrélation (semi-définie positive)
    A = randn(n_assets, n_assets)
    corr = A * A'
    corr = corr ./ sqrt.(diag(corr) * diag(corr)')
    L = cholesky(Hermitian(corr)).L
    # Rendements corrélés
    raw = randn(n_days, n_assets)
    returns = raw * L' .* sigma' .+ mu'
    return returns, mu .* TRADING_DAYS, sigma .* sqrt(TRADING_DAYS)
end


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

function health_check(req::HTTP.Request)
    return json_response(Dict(
        "status" => "operational",
        "service" => "quant-lab",
        "version" => "1.0.0",
        "backend" => "Julia $(VERSION)",
        "symbols" => length(SYMBOLS),
        "timestamp" => string(now())
    ))
end


function portfolio_optimize(req::HTTP.Request)
    """Optimisation de portefeuille Markowitz (Mean-Variance)."""
    params = parse_query(req)
    n_assets = parse(Int, get(params, "assets", "10"))
    n_assets = clamp(n_assets, 2, 50)

    start_time = time()

    # Générer rendements simulés
    returns, annual_mu, annual_sigma = generate_returns(n_assets, 500)

    # Matrice de covariance
    cov_matrix = cov(returns) .* TRADING_DAYS

    # Optimisation naïve : poids inversement proportionnels à la variance
    inv_var = 1.0 ./ diag(cov_matrix)
    weights = inv_var ./ sum(inv_var)

    # Rendement et risque du portefeuille
    port_return = dot(weights, annual_mu)
    port_risk = sqrt(weights' * cov_matrix * weights)
    sharpe = (port_return - RISK_FREE_RATE) / port_risk

    elapsed = (time() - start_time) * 1000

    asset_names = SYMBOLS[1:min(n_assets, length(SYMBOLS))]
    if n_assets > length(SYMBOLS)
        for i in (length(SYMBOLS)+1):n_assets
            push!(asset_names, "ASSET_$i")
        end
    end

    return json_response(Dict(
        "assets_count" => n_assets,
        "weights" => Dict(asset_names[i] => round(weights[i], digits=4) for i in 1:n_assets),
        "portfolio_return" => round(port_return * 100, digits=2),
        "portfolio_risk" => round(port_risk * 100, digits=2),
        "sharpe_ratio" => round(sharpe, digits=3),
        "risk_free_rate" => RISK_FREE_RATE,
        "compute_time_ms" => round(elapsed, digits=2),
        "method" => "inverse_variance",
        "status" => "optimized"
    ))
end


function monte_carlo_var(req::HTTP.Request)
    """Simulation Monte Carlo pour Value-at-Risk."""
    params = parse_query(req)
    n_simulations = parse(Int, get(params, "simulations", "10000"))
    n_simulations = clamp(n_simulations, 1000, 100000)
    confidence = parse(Float64, get(params, "confidence", "0.95"))
    initial_value = parse(Float64, get(params, "portfolio_value", "100000"))
    horizon_days = parse(Int, get(params, "horizon", "10"))

    start_time = time()

    # Simulation de rendements log-normaux
    daily_return = 0.0008  # ~20% annuel
    daily_vol = 0.015      # ~24% annuel
    simulated_values = zeros(n_simulations)

    for i in 1:n_simulations
        value = initial_value
        for d in 1:horizon_days
            ret = daily_return + daily_vol * randn()
            value *= exp(ret)
        end
        simulated_values[i] = value
    end

    # PnL
    pnl = simulated_values .- initial_value
    sorted_pnl = sort(pnl)

    # VaR et ES
    var_index = Int(floor((1 - confidence) * n_simulations))
    var_value = -sorted_pnl[max(var_index, 1)]
    es_value = -mean(sorted_pnl[1:max(var_index, 1)])

    elapsed = (time() - start_time) * 1000

    return json_response(Dict(
        "var" => round(var_value, digits=2),
        "expected_shortfall" => round(es_value, digits=2),
        "confidence" => confidence,
        "horizon_days" => horizon_days,
        "simulations" => n_simulations,
        "portfolio_value" => initial_value,
        "mean_pnl" => round(mean(pnl), digits=2),
        "std_pnl" => round(std(pnl), digits=2),
        "worst_case" => round(minimum(pnl), digits=2),
        "best_case" => round(maximum(pnl), digits=2),
        "compute_time_ms" => round(elapsed, digits=2),
    ))
end


function efficient_frontier(req::HTTP.Request)
    """Calcul de la frontière efficiente (20 points)."""
    params = parse_query(req)
    n_assets = parse(Int, get(params, "assets", "5"))
    n_points = parse(Int, get(params, "points", "20"))
    n_assets = clamp(n_assets, 2, 20)
    n_points = clamp(n_points, 5, 50)

    start_time = time()

    returns, annual_mu, annual_sigma = generate_returns(n_assets, 500)
    cov_matrix = cov(returns) .* TRADING_DAYS

    frontier = []
    target_returns = range(minimum(annual_mu), maximum(annual_mu), length=n_points)

    for target_ret in target_returns
        # Approximation : poids interpolés entre min-variance et max-return
        # (Optimisation complète = QP solver, trop lourd pour un serveur HTTP léger)
        inv_var = 1.0 ./ diag(cov_matrix)
        base_weights = inv_var ./ sum(inv_var)

        # Tilt vers l'actif avec le meilleur rendement
        best_asset = argmax(annual_mu)
        alpha = (target_ret - dot(base_weights, annual_mu)) / (annual_mu[best_asset] - dot(base_weights, annual_mu) + 1e-8)
        alpha = clamp(alpha, 0.0, 1.0)

        weights = (1 - alpha) .* base_weights
        weights[best_asset] += alpha
        weights ./= sum(weights)

        port_ret = dot(weights, annual_mu)
        port_risk = sqrt(weights' * cov_matrix * weights)

        push!(frontier, Dict(
            "return" => round(port_ret * 100, digits=2),
            "risk" => round(port_risk * 100, digits=2),
            "sharpe" => round((port_ret - RISK_FREE_RATE) / port_risk, digits=3),
        ))
    end

    elapsed = (time() - start_time) * 1000

    return json_response(Dict(
        "frontier" => frontier,
        "assets_count" => n_assets,
        "points" => n_points,
        "risk_free_rate" => RISK_FREE_RATE,
        "compute_time_ms" => round(elapsed, digits=2),
    ))
end


function stress_test(req::HTTP.Request)
    """Stress testing du portefeuille contre des scénarios adverses."""
    params = parse_query(req)
    portfolio_value = parse(Float64, get(params, "portfolio_value", "100000"))

    start_time = time()

    scenarios = [
        Dict("name" => "Flash Crash (-15% en 1 jour)", "shock" => -0.15),
        Dict("name" => "Bear Market (-30% sur 3 mois)", "shock" => -0.30),
        Dict("name" => "Black Swan (-50%)", "shock" => -0.50),
        Dict("name" => "Taux +200bps", "shock" => -0.08),
        Dict("name" => "Inflation 10%", "shock" => -0.12),
        Dict("name" => "Crise Liquidité", "shock" => -0.20),
        Dict("name" => "Rallye Haussier (+30%)", "shock" => 0.30),
        Dict("name" => "Stagnation (+2%)", "shock" => 0.02),
    ]

    results = []
    for scenario in scenarios
        impact = portfolio_value * scenario["shock"]
        new_value = portfolio_value + impact
        push!(results, Dict(
            "scenario" => scenario["name"],
            "shock_percent" => scenario["shock"] * 100,
            "impact" => round(impact, digits=2),
            "portfolio_after" => round(new_value, digits=2),
            "recovery_needed" => scenario["shock"] < 0 ?
                round(abs(scenario["shock"]) / (1 + scenario["shock"]) * 100, digits=1) : 0.0,
        ))
    end

    elapsed = (time() - start_time) * 1000

    return json_response(Dict(
        "portfolio_value" => portfolio_value,
        "scenarios" => results,
        "scenarios_count" => length(results),
        "worst_case" => round(portfolio_value * (1 + minimum([s["shock"] for s in scenarios])), digits=2),
        "compute_time_ms" => round(elapsed, digits=2),
    ))
end


function correlation_matrix(req::HTTP.Request)
    """Matrice de corrélation entre les actifs."""
    params = parse_query(req)
    n_assets = parse(Int, get(params, "assets", "5"))
    n_assets = clamp(n_assets, 2, length(SYMBOLS))

    start_time = time()
    returns, _, _ = generate_returns(n_assets, 500)
    corr = cor(returns)
    elapsed = (time() - start_time) * 1000

    names = SYMBOLS[1:n_assets]
    matrix = Dict()
    for i in 1:n_assets
        matrix[names[i]] = Dict(names[j] => round(corr[i,j], digits=3) for j in 1:n_assets)
    end

    return json_response(Dict(
        "symbols" => names,
        "correlation" => matrix,
        "compute_time_ms" => round(elapsed, digits=2),
    ))
end


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

const ROUTER = HTTP.Router()
HTTP.register!(ROUTER, "GET", "/health", health_check)
HTTP.register!(ROUTER, "GET", "/quant/optimize", portfolio_optimize)
HTTP.register!(ROUTER, "GET", "/quant/var", monte_carlo_var)
HTTP.register!(ROUTER, "GET", "/quant/frontier", efficient_frontier)
HTTP.register!(ROUTER, "GET", "/quant/stress", stress_test)
HTTP.register!(ROUTER, "GET", "/quant/correlation", correlation_matrix)

println("✅ EVA Quant-Lab — $(length(SYMBOLS)) symboles configurés")
println("🚀 Serveur Julia en écoute sur 0.0.0.0:$(PORT)")
HTTP.serve(ROUTER, "0.0.0.0", PORT)
