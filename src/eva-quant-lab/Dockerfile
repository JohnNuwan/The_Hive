FROM julia:1.10-bookworm

WORKDIR /app

# Install standard quantitative packages
RUN julia -e 'using Pkg; Pkg.add(["DataFrames", "DifferentialEquations", "JuMP", "Ipopt", "Flux", "Redis", "HTTP", "JSON", "Dates"])'

# Copy shared source (for references if needed, although Julia won't use the Python shared lib directly)
# But we can use it to refer to shared proto files
COPY src/shared /app/shared

# Copy eva-quant-lab
COPY src/eva-quant-lab /app/eva-quant-lab

WORKDIR /app/eva-quant-lab

EXPOSE 8700

CMD ["julia", "main.jl"]
