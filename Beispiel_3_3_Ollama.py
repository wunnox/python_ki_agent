##############################################
# Name: Beispiel_3_3_Ollama.py
# Zweck: Konversation mit System-, Benutzer-
#        und Assistant-Rolle.
# Beenden mit: ende
##############################################

import textwrap
from time import perf_counter
from ollama import chat

def generate_response(messages: list[dict[str, str]]) -> str:
    """Übergibt den Gesprächsverlauf an das LLM."""
    response = chat(
        model="gemma4:26b",
        stream=False,
        think=False,
        messages=messages,
    )

    return response["message"]["content"]



# Die Systemrolle wird einmalig am Anfang gesetzt.
messages = [{"role": "system", "content": "Du bist ein Tier-Experte. Antworte in kurzen Sätzen."}]

while True:
    frage = input("Fragen Sie etwas zu einem Tier (ende = beenden): ").strip()

    if frage.lower() == "ende":
        break

    if not frage:
        continue

    # Benutzerfrage zum Gesprächsverlauf hinzufügen.
    messages.append({"role": "user","content": frage})

    start = perf_counter()
    response = generate_response(messages)
    dauer = perf_counter() - start

    # Antwort ebenfalls speichern, damit sie bei der nächsten
    # Anfrage Teil des Gesprächsverlaufs ist.
    messages.append({"role": "assistant", "content": response})

    print("\nAssistant:")
    print(textwrap.indent(response, "  "))
    print(f"  (Antwortzeit: {dauer:.2f} Sekunden)\n")


print("\n=============================================================")
print("Im Gesprächsverlauf wurden folgende Nachrichten ausgetauscht:")
print("=============================================================\n")
for msg in messages:
    if msg["role"] == "user":
        print()
    print(f"  {msg['role'].capitalize()}: {textwrap.indent(msg['content'], '  ')}")