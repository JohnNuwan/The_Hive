import requests
from bs4 import BeautifulSoup

try:
    response = requests.get("https://news.ycombinator.com/")
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

    soup = BeautifulSoup(response.content, "html.parser")

    stories = soup.find_all("a", class_="storylink")

    for i in range(min(5, len(stories))):
        title = stories[i].text
        link = stories[i].get("href")
        print(f"{i+1}. {title} - {link}")

except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête : {e}")
except Exception as e:
    print(f"Une erreur inattendue s'est produite : {e}")