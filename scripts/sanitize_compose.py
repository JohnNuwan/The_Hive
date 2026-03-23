import re

with open('docker-compose.yml', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove anything NOT in the standard ASCII range + some common spacing
# This strips emojis and other weird characters
clean = re.sub(r'[^\x00-\x7F]+', '', content)

with open('docker-compose.yml', 'w', encoding='utf-8') as f:
    f.write(clean)

print("docker-compose.yml sanitized: all non-ASCII characters removed.")
