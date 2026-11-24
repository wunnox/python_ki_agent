##############################################
# Name: Beispiel_3_2_Ollama.py
# Zweck: LLM-Anfrage mit Systemrolle und Benutzerrolle und einem freien Gesprächsverlauf
#        Mit Schleife ohne Gesprächsverlauf
##############################################

from ollama import chat
import textwrap
from time import perf_counter

def generate_response(message) -> str:
    """Call LLM to get response"""
    response = chat(
        model="gemma4:26b",
        stream = False,
        think = False,
        messages=[{"role": "system", "content": "Du bist ein Tier-Experte. Antworte in kurzen Sätzen."},{"role": "user", "content": message}]
    )
    return response["message"]["content"]

while True:
    frage = input("Fragen Sie etwas zu einem Tier: ").strip()
    if frage.lower() == "ende":
            break

    message = frage
    start=perf_counter()
    response = generate_response(message)
    end=perf_counter()
    print(textwrap.indent(response, "  "), f"({end-start:.2f} seconds.)\n")
