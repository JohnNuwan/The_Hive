import string

with open('docker-compose.yml', 'rb') as f:
    data = f.read()

# Allowed: Printable ASCII (32-126), Tab (9), LF (10), CR (13)
allowed = set(string.printable.encode('ascii'))
clean_data = bytes([b for b in data if b in allowed])

with open('docker-compose.yml', 'wb') as f:
    f.write(clean_data)

print(f"docker-compose.yml forced to strict ASCII. Reduced from {len(data)} to {len(clean_data)} bytes.")
