#!/usr/bin/env python3
"""Trainierbarer Reflexions-Simulator: Häufigkeiten statt neuronales LLM."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "2.0 / 20.11.2026"
FORMAT_VERSION = 1
MODEL_TYPE = "word_reflection_counts"
START = ""


def tokenisieren(text):
    """Trennt an Leerraum; Satzzeichen bleiben Bestandteil der Wörter."""
    return text.split()


def training_laden(pfad):
    """Liest positionsweise zugeordnete Paare: Aussage : Reflexion."""
    paare = []
    for nummer, zeile in enumerate(pfad.read_text(encoding="utf-8-sig").splitlines(), 1):
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#"):
            continue
        if zeile.count(":") != 1:
            raise ValueError(f"Zeile {nummer}: Genau ein Doppelpunkt erforderlich.")
        quelle, ziel = (tokenisieren(teil) for teil in zeile.split(":"))
        if not quelle or len(quelle) != len(ziel):
            raise ValueError(f"Zeile {nummer}: Beide Seiten brauchen gleich viele Wörter.")
        paare.append((quelle, ziel))
    if not paare:
        raise ValueError("Keine Trainingspaare vorhanden.")
    return paare


def trainieren(paare):
    """Erstellt neue Häufigkeitstabellen; kein inkrementelles Weiterlernen."""
    wortmodell = defaultdict(Counter)
    kontextmodell = defaultdict(Counter)
    for quelle, ziel in paare:
        vorher = START
        for quellwort, zielwort in zip(quelle, ziel):
            wort = quellwort.casefold()
            wortmodell[wort][zielwort] += 1
            kontextmodell[(vorher, wort)][zielwort] += 1
            vorher = wort
    return wortmodell, kontextmodell


def modell_speichern(pfad, modell, anzahl_paare):
    """Speichert Tabellen und Metadaten als lesbares JSON."""
    wortmodell, kontextmodell = modell
    daten = {
        "format_version": FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "training_pairs": anzahl_paare,
        "word_model": {wort: dict(ziele) for wort, ziele in wortmodell.items()},
        "context_model": [
            {"previous": vorher, "current": wort, "targets": dict(ziele)}
            for (vorher, wort), ziele in kontextmodell.items()
        ],
    }
    # Reihenfolge erhalten: Sie entscheidet bei gleichen Häufigkeiten.
    pfad.write_text(json.dumps(daten, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def kandidaten_laden(daten):
    """Prüft eine Kandidatentabelle aus der JSON-Datei."""
    if not isinstance(daten, dict) or not daten:
        raise ValueError("Kandidatentabelle fehlt oder ist leer.")
    for wort, anzahl in daten.items():
        if not isinstance(wort, str) or len(tokenisieren(wort)) != 1:
            raise ValueError("Zielwörter müssen einzelne, nicht leere Tokens sein.")
        if type(anzahl) is not int or anzahl <= 0:
            raise ValueError("Häufigkeiten müssen positive ganze Zahlen sein.")
    return Counter(daten)


def quellwort_pruefen(wort, leer_erlaubt=False):
    if leer_erlaubt and wort == START:
        return
    if (not isinstance(wort, str) or len(tokenisieren(wort)) != 1
            or wort != wort.casefold()):
        raise ValueError("Ungültiges Quellwort im Modell.")


def modell_laden(pfad):
    """Rekonstruiert Counter und Tupelschlüssel, ohne Training auszuführen."""
    daten = json.loads(pfad.read_text(encoding="utf-8"))
    if not isinstance(daten, dict):
        raise ValueError("Das Modell muss ein JSON-Objekt sein.")
    if type(daten.get("format_version")) is not int or daten["format_version"] != FORMAT_VERSION:
        raise ValueError("Nicht unterstützte Modellformat-Version.")
    if daten.get("model_type") != MODEL_TYPE:
        raise ValueError("Nicht unterstützter Modelltyp.")
    anzahl_paare = daten.get("training_pairs")
    if type(anzahl_paare) is not int or anzahl_paare <= 0:
        raise ValueError("Ungültige Anzahl Trainingspaare.")
    woerter = daten.get("word_model")
    kontexte = daten.get("context_model")
    if not isinstance(woerter, dict) or not woerter:
        raise ValueError("Wortmodell fehlt oder ist leer.")
    if not isinstance(kontexte, list) or not kontexte:
        raise ValueError("Kontextmodell fehlt oder ist leer.")
    wortmodell = {}
    for wort, kandidaten in woerter.items():
        quellwort_pruefen(wort)
        wortmodell[wort] = kandidaten_laden(kandidaten)
    kontextmodell = {}
    summen = defaultdict(Counter)
    for eintrag in kontexte:
        if not isinstance(eintrag, dict):
            raise ValueError("Ungültiger Kontexteintrag.")
        vorher, wort = eintrag.get("previous"), eintrag.get("current")
        quellwort_pruefen(vorher, leer_erlaubt=True)
        quellwort_pruefen(wort)
        if wort not in wortmodell or (vorher != START and vorher not in wortmodell):
            raise ValueError("Kontext referenziert ein unbekanntes Quellwort.")
        schluessel = (vorher, wort)
        if schluessel in kontextmodell:
            raise ValueError("Doppelter Kontexteintrag.")
        kandidaten = kandidaten_laden(eintrag.get("targets"))
        kontextmodell[schluessel] = kandidaten
        summen[wort].update(kandidaten)
    if dict(summen) != wortmodell:
        raise ValueError("Wort- und Kontexthäufigkeiten stimmen nicht überein.")
    return (wortmodell, kontextmodell), anzahl_paare


def reflektieren(text, modell):
    """Wählt den häufigsten Kandidaten; unbekannte Wörter werden kopiert."""
    wortmodell, kontextmodell = modell
    ergebnis = []
    vorher = START
    for quellwort in tokenisieren(text):
        wort = quellwort.casefold()
        kontext = (vorher, wort)
        if kontext in kontextmodell:
            kandidaten = kontextmodell[kontext]
            basis = f"Kontext: {vorher or '<START>'} + {wort}"
        elif wort in wortmodell:
            kandidaten = wortmodell[wort]
            basis = f"Nur Quellwort: {wort}"
        else:
            kandidaten = Counter()
            basis = "Unbekannt: unverändert kopiert"
        zielwort, anzahl = kandidaten.most_common(1)[0] if kandidaten else (quellwort, 0)
        ergebnis.append((quellwort, zielwort, anzahl, kandidaten, basis))
        vorher = wort
    return ergebnis


def ausgeben(text, modell, details=False):
    ergebnis = reflektieren(text, modell)
    print("\nReflexion :", " ".join(e[1] for e in ergebnis))
    print("\nHäufigkeit:", " ".join(f"{e[2]}|{e[1]}" for e in ergebnis))
    if details:
        print()
        for quelle, ziel, anzahl, kandidaten, basis in ergebnis:
            print(f"  {quelle!r} -> {ziel!r}: {basis}")
            gesamt = sum(kandidaten.values())
            if gesamt:
                optionen = ", ".join(
                    f"{wort}={zahl}/{gesamt} ({zahl / gesamt:.1%})"
                    for wort, zahl in kandidaten.most_common()
                )
                print(f"    Kandidaten: {optionen}")
            else:
                print("    Kein Trainingsbeleg; Häufigkeit 0.")
    print()


def parser_erstellen():
    basis = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learn", action="store_true",
                        help="Modell neu trainieren, speichern und beenden")
    parser.add_argument("--training", type=Path, default=None,
                        help="Trainingsdatei, nur mit --learn (Standard: neben dem Skript)")
    parser.add_argument("--model", type=Path, default=basis / "llm_model.json",
                        help="Modelldatei (Standard: llm_model.json neben dem Skript)")
    parser.add_argument("--text", help="Einzelne Aussage statt interaktiver Eingabe")
    parser.add_argument("--details", action="store_true", help="Auswahlgrundlagen anzeigen")
    parser.add_argument("--doc", action="store_true", help="Dokumentation anzeigen")
    parser.add_argument("-V", "--version", action="version", version=VERSION)
    return parser


def dokumentation(parser):
    print(f"Reflexions-Simulator {VERSION}")
    print("=" * 72)
    print("""Dieses Modell ist eine Häufigkeitstabelle, kein neuronales LLM.

Lernen:
  python llm_simulator.py --learn --training llm_training.txt
  Die Tabellen werden vollständig neu erstellt und als JSON gespeichert.
  Eine vorhandene Modelldatei wird überschrieben; es wird nicht weitergezählt.

Anwenden:
  python llm_simulator.py
  python llm_simulator.py --text 'Ich bin müde' --details
  Nur die Modelldatei wird geladen. Die Trainingsdatei wird nicht benötigt.
  Änderungen an Trainingsdaten wirken erst nach erneutem --learn.
  Eingaben verändern das Modell nicht.

Trainingsformat:
  Aussage : Reflexion
  UTF-8, ein Paar je Zeile, genau ein Doppelpunkt.
  Leere Zeilen und Zeilen mit # am Anfang werden ignoriert.
  Beide Seiten müssen gleich viele Wörter haben.

Auswahl:
  Vorheriges + aktuelles Quellwort; ersatzweise nur aktuelles Quellwort.
  Der häufigste Zielkandidat gewinnt, bei Gleichstand der zuerst gelernte.
  Unbekannte Wörter werden unverändert kopiert (Häufigkeit 0).
  Quellwörter werden ohne Beachtung der Gross-/Kleinschreibung nachgeschlagen.
  Satzzeichen gehören zum Wort. Keine Wortumstellungen oder freie Generierung.
  Kandidatenprozente sind relative Häufigkeiten, keine Qualitätsgarantie.

Dateipfade:
  Standarddateien liegen neben dem Skript.
  Explizite relative Pfade beziehen sich auf das aktuelle Arbeitsverzeichnis.
""")
    parser.print_help()


def main():
    parser = parser_erstellen()
    args = parser.parse_args()
    if args.doc:
        dokumentation(parser)
        return
    if args.learn and (args.text is not None or args.details):
        parser.error("--learn ist nicht mit --text oder --details kombinierbar.")
    if args.training is not None and not args.learn:
        parser.error("--training ist nur zusammen mit --learn zulässig.")
    try:
        if args.learn:
            training = args.training if args.training is not None else Path(__file__).resolve().with_name("llm_training.txt")
            if training.resolve() == args.model.resolve():
                parser.error("Trainingsdatei und Modelldatei müssen verschieden sein.")
            paare = training_laden(training)
            modell = trainieren(paare)
            modell_speichern(args.model, modell, len(paare))
            print(f"Training: {len(paare)} Paare, {len(modell[0])} Quellwörter, "
                  f"{len(modell[1])} Kontexte")
            print(f"Modell gespeichert: {args.model}")
            return
        if not args.model.exists():
            parser.error(f"Modelldatei fehlt: {args.model}. Zuerst mit --learn erstellen.")
        modell, anzahl_paare = modell_laden(args.model)
    except (OSError, UnicodeError, ValueError, RecursionError) as fehler:
        parser.error(str(fehler))
    print(f"Modell geladen: {args.model}")
    print(f"Trainingsbasis: {anzahl_paare} Paare, {len(modell[0])} Quellwörter")
    if args.text is not None:
        ausgeben(args.text, modell, args.details)
        return
    print("Aussage eingeben; /ende beendet das Programm.")
    while True:
        try:
            text = input("\nAussage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.casefold() == "/ende":
            break
        if text:
            ausgeben(text, modell, args.details)


if __name__ == "__main__":
    main()
