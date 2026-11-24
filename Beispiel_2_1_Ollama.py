##############################################
# Name: Beispiel_2_1_KI.py
# Zweck: LLM-Anfrage nur mit der Rolle "user"
#        Mit Schleife ohne Gesprächsverlauf
##############################################

from ollama import chat
from time import perf_counter

def generate_response(message) -> str:
    """Call LLM to get response"""
    response = chat(
        model="qwen3.5:4b",
        stream = False,
        think = False,
        messages=[{"role": "user", "content": message}]
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
