##############################################
# Name: Beispiel_3_1_Ollama.py
# Zweck: LLM-Anfrage mit Systemrolle und Benutzerrolle
#        Mit Schleife ohne Gesprächsverlauf
##############################################

from ollama import chat
from time import perf_counter

def generate_response(message) -> str:
    """Call LLM to get response"""
    response = chat(
        model="gemma4:26b",
        stream = False,
        think = False,
        messages=[{"role": "system", "content": "Du bist ein Tierlaute-Experte. Antworte mit höchstens einem Satz oder nenne nur den Laut."},{"role": "user", "content": message}]
    )
    return response["message"]["content"]

while True:
    frage = input("Geben Sie ein Tier ein: ").strip()
    if frage.lower() == "ende":
            break

    message = f"Gib einen kurzen Text aus, welchen Laut dieses Tier macht: {frage}"
    start=perf_counter()
    response = generate_response(message)
    end=perf_counter()
    print(response, f"({end-start:.2f} seconds.)\n")
