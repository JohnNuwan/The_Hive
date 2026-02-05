# Build Stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN go mod init eva-nervous && \
    go get github.com/go-redis/redis/v8 && \
    go build -o nervous_system main.go

# Run Stage
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/nervous_system .
CMD ["./nervous_system"]
