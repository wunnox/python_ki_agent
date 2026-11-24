##############################################
# Name: Beispiel_3_1_OpenAI.py
# Zweck: LLM-Anfrage mit Systemrolle und Benutzerrolle
#        Mit Schleife ohne Gesprächsverlauf
#
# Wichtig:
# - OpenAI Module installieren: python -m pip install --upgrade openai
# - Schlüssel muss auf OS hinterlegt werden
#   - Linux/OSX: export OPENAI_API_KEY="dein-api-schluessel"
#   - Windows: $env:OPENAI_API_KEY = "dein-api-schluessel"
#
##############################################

from time import perf_counter
from openai import OpenAI

# Liest den API-Schlüssel aus OPENAI_API_KEY.
client = OpenAI()

def generate_response(message: str) -> str:
    """Sendet eine Nachricht an die OpenAI-API."""
    response = client.responses.create(
        model="gpt-4.1",
        input=[{"role": "system", "content": "Du bist ein Tierlaute-Experte. Antworte mit höchstens einem Satz oder nenne nur den Laut."},{"role": "user", "content": message}],
        stream=False,
    )

    return response.output_text

while True:
    frage = input("Geben Sie ein Tier ein ('ende' zum Beenden): ").strip()

    if frage.lower() == "ende":
        break

    if not frage:
        continue

    message = (f"Gib einen kurzen Text aus, welchen Laut dieses Tier macht: {frage}")

    start = perf_counter()
    response = generate_response(message)
    end = perf_counter()

    print(response, f"({end - start:.2f} Sekunden)\n")
