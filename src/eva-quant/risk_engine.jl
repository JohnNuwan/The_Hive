# ═══════════════════════════════════════════════════════════════════════════════
# THE QUANT ENGINE (Julia)
# Coprocesseur Mathématique pour les calculs financiers complexes.
#
# Fonctions :
#   1. Monte Carlo VaR (Value at Risk) — 100k simulations en ms
#   2. CVaR (Conditional VaR / Expected Shortfall)
#   3. Black-Scholes (Call/Put pricing)
#   4. Greeks (Delta, Gamma, Vega, Theta)
#   5. Corrélation multi-actifs (matrice de corrélation)
#   6. Kelly Criterion (taille de position optimale)
#   7. Serveur TCP robuste avec gestion d'erreurs
# ═══════════════════════════════════════════════════════════════════════════════

using Random
using Dates
using Sockets
using JSON
using Statistics
using LinearAlgebra

println("╔══════════════════════════════════════════╗")
println("║  🧮 THE QUANT ENGINE (Julia JIT)         ║")
println("║  Coprocesseur Mathématique — THE HIVE     ║")
println("╚══════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MONTE CARLO VALUE AT RISK (VaR)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    calculate_var_monte_carlo(price, volatility, simulations, confidence)

Calcule le VaR à N% de confiance par simulation Monte Carlo (GBM).
Retourne le prix seuil sous lequel la perte se situe avec probabilité (1-confidence).
"""
function calculate_var_monte_carlo(
    current_price::Float64,
    volatility::Float64,
    simulations::Int,
    confidence::Float64=0.95,
    holding_period_days::Int=1,
    drift::Float64=0.05
)
    dt = holding_period_days / 252.0  # Fraction d'année de trading
    final_prices = Vector{Float64}(undef, simulations)

    @inbounds for i in 1:simulations
        shock = randn()
        final_prices[i] = current_price * exp((drift - 0.5 * volatility^2) * dt + volatility * sqrt(dt) * shock)
    end

    sort!(final_prices)
    var_index = max(1, Int(floor(simulations * (1.0 - confidence))))
    var_price = final_prices[var_index]
    var_loss = current_price - var_price

    return Dict(
        "var_price" => var_price,
        "var_loss" => var_loss,
        "var_percent" => (var_loss / current_price) * 100.0,
        "confidence" => confidence,
        "simulations" => simulations
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONDITIONAL VAR (CVaR / Expected Shortfall)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    calculate_cvar(price, volatility, simulations, confidence)

Calcule le CVaR — la perte moyenne dans le pire (1-confidence)% des cas.
Plus conservateur que le VaR classique.
"""
function calculate_cvar(
    current_price::Float64,
    volatility::Float64,
    simulations::Int,
    confidence::Float64=0.95,
    holding_period_days::Int=1
)
    dt = holding_period_days / 252.0
    final_prices = Vector{Float64}(undef, simulations)

    @inbounds for i in 1:simulations
        shock = randn()
        final_prices[i] = current_price * exp((-0.5 * volatility^2) * dt + volatility * sqrt(dt) * shock)
    end

    sort!(final_prices)
    tail_count = max(1, Int(floor(simulations * (1.0 - confidence))))
    tail_prices = final_prices[1:tail_count]
    tail_losses = current_price .- tail_prices

    cvar_loss = mean(tail_losses)

    return Dict(
        "cvar_loss" => cvar_loss,
        "cvar_percent" => (cvar_loss / current_price) * 100.0,
        "worst_case_loss" => maximum(tail_losses),
        "confidence" => confidence,
        "tail_count" => tail_count
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BLACK-SCHOLES (Pricing d'options)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    norm_cdf(x) — CDF de la distribution normale standard
"""
function norm_cdf(x::Float64)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
end

"""
    norm_pdf(x) — PDF de la distribution normale standard
"""
function norm_pdf(x::Float64)
    return exp(-0.5 * x^2) / sqrt(2.0 * π)
end

"""
    black_scholes(S, K, T, r, σ, option_type)

Pricing Black-Scholes pour options européennes.
- S: Prix spot
- K: Prix d'exercice (strike)
- T: Temps à expiration (en années)
- r: Taux sans risque
- σ: Volatilité implicite
- option_type: "call" ou "put"
"""
function black_scholes(
    S::Float64, K::Float64, T::Float64,
    r::Float64, sigma::Float64,
    option_type::String="call"
)
    if T <= 0.0
        # Option expirée
        if option_type == "call"
            return max(S - K, 0.0)
        else
            return max(K - S, 0.0)
        end
    end

    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call"
        price = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else
        price = K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    end

    return Dict(
        "price" => price,
        "d1" => d1,
        "d2" => d2,
        "option_type" => option_type
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GREEKS (Sensibilités)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    calculate_greeks(S, K, T, r, σ)

Calcule les Greeks pour une option call européenne.
"""
function calculate_greeks(
    S::Float64, K::Float64, T::Float64,
    r::Float64, sigma::Float64
)
    if T <= 0.0
        return Dict("delta" => 0.0, "gamma" => 0.0, "vega" => 0.0, "theta" => 0.0, "rho" => 0.0)
    end

    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # Delta : sensibilité au prix du sous-jacent
    delta = norm_cdf(d1)

    # Gamma : taux de changement du delta
    gamma = norm_pdf(d1) / (S * sigma * sqrt(T))

    # Vega : sensibilité à la volatilité (pour 1% de vol)
    vega = S * norm_pdf(d1) * sqrt(T) / 100.0

    # Theta : perte de valeur temporelle (par jour)
    theta = -(S * norm_pdf(d1) * sigma / (2.0 * sqrt(T)) + r * K * exp(-r * T) * norm_cdf(d2)) / 365.0

    # Rho : sensibilité au taux d'intérêt
    rho = K * T * exp(-r * T) * norm_cdf(d2) / 100.0

    return Dict(
        "delta" => delta,
        "gamma" => gamma,
        "vega" => vega,
        "theta" => theta,
        "rho" => rho
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. KELLY CRITERION (Taille de position optimale)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    kelly_criterion(win_rate, avg_win, avg_loss)

Calcule la fraction optimale de capital à risquer (Kelly).
On utilise le Half-Kelly pour la prudence.
"""
function kelly_criterion(win_rate::Float64, avg_win::Float64, avg_loss::Float64)
    if avg_loss == 0.0
        return Dict("full_kelly" => 0.0, "half_kelly" => 0.0, "recommendation" => "No data")
    end

    b = avg_win / abs(avg_loss)  # Ratio gain/perte
    q = 1.0 - win_rate

    full_kelly = (win_rate * b - q) / b
    half_kelly = full_kelly / 2.0  # Plus conservateur

    return Dict(
        "full_kelly" => max(0.0, full_kelly),
        "half_kelly" => max(0.0, half_kelly),
        "win_rate" => win_rate,
        "profit_factor" => b,
        "recommendation" => full_kelly > 0 ? "TRADE" : "NO_EDGE"
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MATRICE DE CORRÉLATION (Multi-actifs)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    correlation_matrix(returns_matrix)

Calcule la matrice de corrélation entre actifs.
Input : matrice NxM (N observations, M actifs)
"""
function correlation_matrix(returns::Matrix{Float64})
    n_assets = size(returns, 2)
    corr = cor(returns)

    # Identifier les paires fortement corrélées (|r| > 0.8)
    high_corr_pairs = []
    for i in 1:n_assets
        for j in (i+1):n_assets
            if abs(corr[i, j]) > 0.8
                push!(high_corr_pairs, Dict("asset_i" => i, "asset_j" => j, "correlation" => corr[i, j]))
            end
        end
    end

    return Dict(
        "matrix" => corr,
        "high_correlation_pairs" => high_corr_pairs,
        "n_assets" => n_assets,
        "warning" => length(high_corr_pairs) > 0 ? "⚠️ Paires fortement corrélées détectées" : "OK"
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# SERVEUR TCP (Interne Docker — Port 9000)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    dispatch(request) — Route la requête vers la bonne fonction
"""
function dispatch(request::Dict)
    func = get(request, "function", "var")

    if func == "var"
        return calculate_var_monte_carlo(
            Float64(get(request, "price", 100.0)),
            Float64(get(request, "volatility", 0.2)),
            Int(get(request, "simulations", 10000)),
            Float64(get(request, "confidence", 0.95)),
        )
    elseif func == "cvar"
        return calculate_cvar(
            Float64(get(request, "price", 100.0)),
            Float64(get(request, "volatility", 0.2)),
            Int(get(request, "simulations", 10000)),
            Float64(get(request, "confidence", 0.95)),
        )
    elseif func == "black_scholes"
        return black_scholes(
            Float64(get(request, "spot", 100.0)),
            Float64(get(request, "strike", 100.0)),
            Float64(get(request, "time_to_expiry", 0.25)),
            Float64(get(request, "risk_free_rate", 0.05)),
            Float64(get(request, "volatility", 0.2)),
            get(request, "option_type", "call"),
        )
    elseif func == "greeks"
        return calculate_greeks(
            Float64(get(request, "spot", 100.0)),
            Float64(get(request, "strike", 100.0)),
            Float64(get(request, "time_to_expiry", 0.25)),
            Float64(get(request, "risk_free_rate", 0.05)),
            Float64(get(request, "volatility", 0.2)),
        )
    elseif func == "kelly"
        return kelly_criterion(
            Float64(get(request, "win_rate", 0.55)),
            Float64(get(request, "avg_win", 100.0)),
            Float64(get(request, "avg_loss", 80.0)),
        )
    else
        return Dict("error" => "Unknown function: $func", "available" => ["var", "cvar", "black_scholes", "greeks", "kelly"])
    end
end

# Lancement du serveur
server = listen(IPv4(0,0,0,0), 9000)
println("🔌 Quant Engine TCP listening on port 9000")
println("   Available functions: var, cvar, black_scholes, greeks, kelly")
println("   Example: {\"function\":\"var\",\"price\":2050.0,\"volatility\":0.15,\"simulations\":100000}")

while true
    conn = accept(server)
    @async begin
        try
            line = readline(conn)
            if isempty(strip(line))
                close(conn)
                continue
            end

            request = JSON.parse(line)
            start_time = time()

            result = dispatch(request)

            elapsed = time() - start_time

            response = Dict(
                "status" => "OK",
                "function" => get(request, "function", "var"),
                "result" => result,
                "compute_time_sec" => elapsed,
                "engine" => "JULIA_JIT_v2"
            )

            println(conn, JSON.json(response))
            println("✅ $(get(request, "function", "?")) computed in $(round(elapsed*1000, digits=2))ms")
        catch e
            if isa(e, JSON.Parser.ParsingException)
                error_resp = Dict("status" => "ERROR", "message" => "Invalid JSON", "engine" => "JULIA_JIT_v2")
                println(conn, JSON.json(error_resp))
                println("❌ JSON parsing error")
            elseif isa(e, Base.IOError)
                println("⚠️ Client disconnected")
            else
                error_resp = Dict("status" => "ERROR", "message" => string(e), "engine" => "JULIA_JIT_v2")
                try
                    println(conn, JSON.json(error_resp))
                catch
                end
                println("❌ Error: $e")
            end
        finally
            try close(conn) catch end
        end
    end
end
