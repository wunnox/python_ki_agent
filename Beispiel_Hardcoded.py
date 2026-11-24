##############################################
# Name: Beispiel_Hardcoded.py
# Zweck: Ausgabe von Lauten zu bekannten Tieren
#        
##############################################

while True:
    tier = input("Geben Sie ein Tier ein oder Ende: ")
 
    if tier.lower()=="ende":
        break

    if tier.lower() == "hund": laut="Wau"
    elif tier.lower() == "kuh": laut="Muu"
    else: laut="einen unbekannten Laut"

    print(f"Ein {tier} macht {laut}\n")
