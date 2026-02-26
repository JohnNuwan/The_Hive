import requests

payload = {
    "prompt": "Crée un scraper minimaliste en Python avec requests et BeautifulSoup4 qui extrait les 5 premiers articles de HackerNews. Le script doit imprimer les titres et les liens. Ne fais pas de def_main si ce n'est pas nécessaire.",
    "filename": "hn_scraper.py",
    "language": "python"
}

resp = requests.post("http://localhost:8003/factory/build", json=payload)
print(resp.status_code)
print(resp.text)
