# Build Stage
FROM julia:1.9-alpine

WORKDIR /app

# Installation des dépendances
# Dans un vrai projet, on utiliserait Project.toml avec Pkg
RUN julia -e 'using Pkg; Pkg.add("JSON"); Pkg.add("Dates")'

COPY risk_engine.jl .

CMD ["julia", "risk_engine.jl"]
