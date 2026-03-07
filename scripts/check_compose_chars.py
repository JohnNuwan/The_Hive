with open('docker-compose.yml', 'rb') as f:
    data = f.read()

bad_chars = []
for i, b in enumerate(data):
    # Allow: Tab (9), LF (10), CR (13), and SP (32) to ~ (126)
    # Also allow common UTF-8 start bytes for emojis (>= 128)
    if b < 32 and b not in [9, 10, 13]:
        bad_chars.append((i, b))

if bad_chars:
    print(f"Found {len(bad_chars)} control characters!")
    for i, b in bad_chars[:10]:
        print(f"  Byte {i}: hex {hex(b)}")
else:
    print("No control characters found in docker-compose.yml.")
