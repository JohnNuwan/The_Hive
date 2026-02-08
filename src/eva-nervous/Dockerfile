# Build Stage
FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY . .
RUN go mod download && \
    go build -o nervous_system main.go

# Run Stage
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/nervous_system .
CMD ["./nervous_system"]
