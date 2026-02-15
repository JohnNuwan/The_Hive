# Build Stage
FROM golang:1.24-alpine AS builder

# Install protoc and dependencies
RUN apk add --no-cache curl tar

# Install protoc binary
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v25.2/protoc-25.2-linux-x86_64.zip && \
    unzip protoc-25.2-linux-x86_64.zip -d /usr/local && \
    chmod +x /usr/local/bin/protoc

# Install Go plugins for protoc
RUN curl -L https://github.com/protocolbuffers/protobuf-go/releases/download/v1.32.0/protoc-gen-go.v1.32.0.linux.amd64.tar.gz | tar -xz -C /usr/local/bin && \
    chmod +x /usr/local/bin/protoc-gen-go

RUN curl -L https://github.com/grpc/grpc-go/releases/download/cmd/protoc-gen-go-grpc/v1.3.0/protoc-gen-go-grpc.v1.3.0.linux.amd64.tar.gz | tar -xz -C /usr/local/bin && \
    chmod +x /usr/local/bin/protoc-gen-go-grpc

WORKDIR /app

# Copy project root to access shared/proto
COPY . .

# Generate Go code from proto
RUN mkdir -p src/eva-nervous/proto && \
    protoc --proto_path=src/shared/shared/proto \
           --go_out=src/eva-nervous/proto --go_opt=paths=source_relative \
           --go-grpc_out=src/eva-nervous/proto --go-grpc_opt=paths=source_relative \
           src/shared/shared/proto/swarm.proto

WORKDIR /app/src/eva-nervous
# Generate Go code from proto
RUN mkdir -p proto && \
    protoc --proto_path=../shared/shared/proto \
           --go_out=proto --go_opt=paths=source_relative \
           --go-grpc_out=proto --go-grpc_opt=paths=source_relative \
           ../shared/shared/proto/swarm.proto

RUN go build -mod=vendor -o /app/nervous_system main.go

# Run Stage
FROM alpine:latest
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /app/nervous_system .
CMD ["./nervous_system"]
