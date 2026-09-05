"""
Micro-service d'orchestration MIDI -> MP3.

Reçoit un fichier MIDI piano (accords jusqu'à 4 notes) et produit un
arrangement enrichi et personnalisable :

- Un ou plusieurs instruments au choix, chacun avec un rôle musical propre
  (mélodie, harmonie, basse/pad, arpège) : trompette, flûte traversière,
  clarinette, clarinette aiguë, saxophone, trombone, tuba, orgue, chœur,
  guitare, guitare basse, guitare électrique, piano grave, piano medium, piano aigu.
- Un style rythmique (pop, ballade, latin, valse, classic, gospel, rnb, blues)
  qui change le pattern de batterie/basse.
- Un ou plusieurs types de roulement de batterie en fin de phrase (caisse
  claire, toms descendants, cymbale, accélération de charleston), combinables
  et alternés au fil du morceau.
- Des notes d'ornement (passages/échappées) pour densifier la ligne mélodique.

Le résultat est rendu directement en MP3 via FluidSynth + lame, et le nom
du fichier reprend celui du MIDI importé (+ "_Orchestrated.mp3").
"""

import io
import os
import shutil
import subprocess
import struct
import tempfile
import wave
from typing import List

import pretty_midi
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

app = FastAPI(title="MIDI Orchestrator")

API_KEY = os.environ.get("ORCHESTRATOR_API_KEY", "change-moi")
SOUNDFONT_PATH = os.environ.get("SOUNDFONT_PATH", "/usr/share/sounds/sf2/FluidR3_GM.sf2")

CHORD_TOLERANCE = 0.05  # secondes

# Notes GM du kit de batterie standard (canal 10)
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42
DRUM_HIHAT_OPEN = 46
DRUM_CRASH = 49
DRUM_TOM_LOW = 45
DRUM_TOM_MID = 47
DRUM_TOM_HIGH = 50

# --------------------------------------------------------------------------
# Registre des instruments disponibles : programme General MIDI + rôle musical
# --------------------------------------------------------------------------
INSTRUMENTS = {
    "trumpet":         {"program": 56, "name": "Trumpet",          "role": "bass_pad"},
    "flute":           {"program": 73, "name": "Flute",            "role": "melody_high"},
    "clarinet":        {"program": 71, "name": "Clarinet",         "role": "harmony"},
    "clarinet_high":   {"program": 71, "name": "Clarinette Aiguë", "role": "harmony_high"},
    "saxophone":       {"program": 65, "name": "Saxophone",        "role": "melody"},
    "trombone":        {"program": 57, "name": "Trombone",         "role": "bass_pad"},
    "tuba":            {"program": 58, "name": "Tuba",             "role": "bass_pad"},
    "organ":           {"program": 19, "name": "Organ",            "role": "melody"},
    "choir":           {"program": 52, "name": "Choir",            "role": "pad_chord"},
    "guitar":          {"program": 25, "name": "Guitar",           "role": "arpeggio"},
    "bass_guitar":     {"program": 32, "name": "Bass Guitar",      "role": "bass_pulse"},
    "electric_guitar": {"program": 29, "name": "Electric Guitar",  "role": "arpeggio"},
    "piano_low":       {"program": 0,  "name": "Piano (grave)",    "role": "bass_pulse"},
    "piano_medium":    {"program": 0,  "name": "Piano (medium)",   "role": "harmony"},
    "piano_high":      {"program": 0,  "name": "Piano (aigu)",     "role": "melody_sparkle"},
}

INSTRUMENT_ALIASES = {
    "band": ["trumpet", "clarinet", "organ"],
    "orchestra": ["trumpet", "clarinet", "organ", "flute", "trombone", "choir"],
    "full": list(INSTRUMENTS.keys()),
    "all": list(INSTRUMENTS.keys()),
}

STYLES = {"pop", "ballad", "latin", "waltz", "classic", "gospel", "rnb", "blues"}
ROLL_TYPES = {"snare", "toms", "crescendo", "double"}
RESPONSE_INSTRUMENTS = {
    "clarinet": 71,
    "flute": 73,
    "guitar": 25,
    "piano_high": 0,
}


def group_notes_into_chords(notes: List[pretty_midi.Note]) -> List[List[pretty_midi.Note]]:
    notes = sorted(notes, key=lambda n: n.start)
    chords = []
    current = []
    for note in notes:
        if not current or note.start - current[0].start <= CHORD_TOLERANCE:
            current.append(note)
        else:
            chords.append(current)
            current = [note]
    if current:
        chords.append(current)
    return chords


def estimate_tempo_from_chords(chords: List[List[pretty_midi.Note]]) -> float:
    """
    pretty_midi.estimate_tempo() est peu fiable sur un fichier composé
    uniquement d'accords plaqués (pas de pulsation régulière à détecter),
    et renvoie souvent une valeur sans rapport avec le morceau. On calcule
    ici un tempo directement à partir de l'écartement réel entre les
    accords du fichier, ce qui reste cohérent quel que soit le tempo
    d'origine.
    """
    if len(chords) < 2:
        return 120.0

    onsets = sorted(min(n.start for n in c) for c in chords)
    intervals = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 0.05]
    if not intervals:
        return 120.0

    intervals.sort()
    median = intervals[len(intervals) // 2]
    bpm = 60.0 / median

    # Ramène le résultat dans une plage musicale plausible (60-180 bpm)
    # en doublant/divisant par deux si besoin, plutôt que de tronquer
    # brutalement une valeur hors plage.
    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm /= 2

    return bpm


def parse_instruments(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()
    if not raw:
        return ["trumpet"]
    if raw in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[raw]
    result = [tok.strip() for tok in raw.split(",") if tok.strip() in INSTRUMENTS]
    return result or ["trumpet"]


def parse_rolls(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()
    if not raw:
        return ["snare"]
    result = [tok.strip() for tok in raw.split(",") if tok.strip() in ROLL_TYPES]
    return result or ["snare"]


def parse_responses(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip() in RESPONSE_INSTRUMENTS]


def clamp_to_range(pitch: int, low: int, high: int) -> int:
    p = pitch
    while p < low:
        p += 12
    while p > high:
        p -= 12
    return max(0, min(127, p))


def build_arpeggio_notes(track: pretty_midi.Instrument, pitches, start: float, end: float, beat: float):
    """Décompose l'accord en pattern rythmique (comping guitare) plutôt qu'un plaqué."""
    base = [n.pitch for n in pitches]
    pattern = [base[0]]
    if len(base) > 1:
        pattern.append(base[min(1, len(base) - 1)])
    pattern.append(base[-1])
    if len(base) > 1:
        pattern.append(base[min(1, len(base) - 1)])

    step = beat / 2  # croches
    t = start
    i = 0
    while t < end:
        note_end = min(t + step * 0.85, end)
        track.notes.append(pretty_midi.Note(velocity=70, pitch=pattern[i % len(pattern)], start=t, end=note_end))
        t += step
        i += 1


def build_solo_track(name: str, chords, tempo: float) -> pretty_midi.Instrument:
    spec = INSTRUMENTS[name]
    role = spec["role"]
    track = pretty_midi.Instrument(program=spec["program"], name=spec["name"])
    beat = 60.0 / max(tempo, 40)

    for chord in chords:
        pitches = sorted(chord, key=lambda n: n.pitch)
        start = min(n.start for n in chord)
        end = max(n.end for n in chord)
        bass = pitches[0]
        melody = pitches[-1]
        inner = pitches[1:-1]

        if role == "melody":
            track.notes.append(pretty_midi.Note(
                velocity=95, pitch=melody.pitch, start=start, end=max(start + 0.3, end - 0.05)
            ))

        elif role == "melody_high":
            p = clamp_to_range(melody.pitch + 12, 72, 96)
            track.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=start, end=end))

        elif role == "melody_sparkle":
            p = clamp_to_range(melody.pitch + 12, 72, 108)
            dur = min(0.25, end - start)
            track.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=start, end=start + dur))

        elif role == "harmony":
            if inner:
                for n in inner:
                    track.notes.append(pretty_midi.Note(velocity=75, pitch=n.pitch, start=start, end=end))
            else:
                track.notes.append(pretty_midi.Note(
                    velocity=70, pitch=max(melody.pitch - 12, 0), start=start, end=end
                ))

        elif role == "harmony_high":
            # Clarinette aiguë : reprend les voix intérieures, transposées vers le registre aigu
            if inner:
                for n in inner:
                    p = clamp_to_range(n.pitch + 12, 72, 96)
                    track.notes.append(pretty_midi.Note(velocity=72, pitch=p, start=start, end=end))
            else:
                p = clamp_to_range(melody.pitch + 12, 72, 96)
                track.notes.append(pretty_midi.Note(velocity=68, pitch=p, start=start, end=end))

        elif role == "bass_pad":
            p = clamp_to_range(bass.pitch - 12, 24, 48)
            track.notes.append(pretty_midi.Note(velocity=65, pitch=p, start=start, end=end))

        elif role == "pad_chord":
            for n in pitches:
                track.notes.append(pretty_midi.Note(velocity=55, pitch=n.pitch, start=start, end=end))

        elif role == "arpeggio":
            build_arpeggio_notes(track, pitches, start, end, beat)

        elif role == "bass_pulse":
            p = clamp_to_range(bass.pitch - 12, 24, 48)
            t = start
            while t < end:
                note_end = min(t + beat * 0.9, end)
                track.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=t, end=note_end))
                t += beat

    return track


# --------------------------------------------------------------------------
# Rythme : batterie + basse, avec un pattern par style et des roulements
# de fin de phrase choisis parmi plusieurs types
# --------------------------------------------------------------------------

def build_fill(roll_type: str, drums: pretty_midi.Instrument, t: float, beat: float):
    if roll_type == "toms":
        toms = [DRUM_TOM_HIGH, DRUM_TOM_MID, DRUM_TOM_LOW, DRUM_TOM_LOW]
        step = beat / 4
        for i, pitch in enumerate(toms):
            st = t + i * step
            drums.notes.append(pretty_midi.Note(velocity=90 + i * 3, pitch=pitch, start=st, end=st + step * 0.85))

    elif roll_type == "crescendo":
        # accélération de charleston qui monte en puissance, ponctuée d'une cymbale
        n_hits = 6
        for i in range(n_hits):
            frac = i / n_hits
            st = t + frac * beat
            vel = 55 + int(frac * 45)
            drums.notes.append(pretty_midi.Note(velocity=vel, pitch=DRUM_HIHAT_OPEN, start=st, end=st + beat / n_hits * 0.8))
        drums.notes.append(pretty_midi.Note(velocity=115, pitch=DRUM_CRASH, start=t + beat * 0.85, end=t + beat))

    elif roll_type == "double":
        # double roulement : 8 double-croches à la caisse claire, coups doublés
        step = beat / 8
        for i in range(8):
            vel = min(65 + (15 if i % 2 == 0 else 0) + i * 3, 127)
            st = t + i * step
            drums.notes.append(pretty_midi.Note(velocity=vel, pitch=DRUM_SNARE, start=st, end=st + step * 0.75))

    else:  # "snare" par défaut
        step = beat / 4
        for i in range(4):
            vel = 70 + i * 10
            st = t + i * step
            drums.notes.append(pretty_midi.Note(velocity=vel, pitch=DRUM_SNARE, start=st, end=st + step * 0.8))


def add_style_beat(drums: pretty_midi.Instrument, t: float, beat: float, beat_i: int, style: str):
    if style == "ballad":
        if beat_i == 0:
            drums.notes.append(pretty_midi.Note(velocity=90, pitch=DRUM_KICK, start=t, end=t + 0.1))
        if beat_i == 2:
            drums.notes.append(pretty_midi.Note(velocity=85, pitch=DRUM_SNARE, start=t, end=t + 0.1))
        drums.notes.append(pretty_midi.Note(velocity=45, pitch=DRUM_HIHAT_CLOSED, start=t, end=t + beat * 0.8))

    elif style == "latin":
        if beat_i == 0:
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=DRUM_KICK, start=t, end=t + 0.1))
        if beat_i == 1:
            drums.notes.append(pretty_midi.Note(velocity=90, pitch=DRUM_KICK, start=t + beat / 2, end=t + beat / 2 + 0.1))
        if beat_i in (1, 3):
            drums.notes.append(pretty_midi.Note(velocity=85, pitch=DRUM_SNARE, start=t, end=t + 0.1))
        drums.notes.append(pretty_midi.Note(velocity=65, pitch=DRUM_HIHAT_CLOSED, start=t, end=t + beat * 0.4))
        drums.notes.append(pretty_midi.Note(velocity=55, pitch=DRUM_HIHAT_CLOSED, start=t + beat / 2, end=t + beat * 0.9))

    elif style == "waltz":
        if beat_i == 0:
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=DRUM_KICK, start=t, end=t + 0.1))
        else:
            drums.notes.append(pretty_midi.Note(velocity=65, pitch=DRUM_HIHAT_CLOSED, start=t, end=t + beat * 0.6))

    else:  # "pop", "classic", "gospel", "rnb", "blues"
        if beat_i in (0, 2):
            drums.notes.append(pretty_midi.Note(velocity=105, pitch=DRUM_KICK, start=t, end=t + 0.1))
        if beat_i in (1, 3):
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=DRUM_SNARE, start=t, end=t + 0.1))
        drums.notes.append(pretty_midi.Note(velocity=70, pitch=DRUM_HIHAT_CLOSED, start=t, end=t + beat * 0.4))
        drums.notes.append(pretty_midi.Note(velocity=55, pitch=DRUM_HIHAT_CLOSED, start=t + beat / 2, end=t + beat * 0.9))


def build_drum_track(total_duration: float, tempo_bpm: float, style: str, rolls: List[str]) -> pretty_midi.Instrument:
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    beat = 60.0 / max(tempo_bpm, 40)
    beats_per_bar = 3 if style == "waltz" else 4

    t = 0.0
    bar_i = 0
    beat_i = 0
    roll_index = 0
    while t < total_duration:
        is_phrase_end = (bar_i % 2 == 1) and (beat_i == beats_per_bar - 1)
        if is_phrase_end:
            chosen = rolls[roll_index % len(rolls)]
            build_fill(chosen, drums, t, beat)
            roll_index += 1
        else:
            add_style_beat(drums, t, beat, beat_i, style)

        t += beat
        beat_i += 1
        if beat_i >= beats_per_bar:
            beat_i = 0
            bar_i += 1

    return drums


def build_bass_track(chords, tempo_bpm: float, style: str = "pop") -> pretty_midi.Instrument:
    bass = pretty_midi.Instrument(program=33, name="Bass")
    beat = 60.0 / max(tempo_bpm, 40)

    for chord in chords:
        pitches = sorted(chord, key=lambda n: n.pitch)
        root = max(pitches[0].pitch - 12, 0)
        fifth = min(root + 7, 127)
        start = min(n.start for n in chord)
        end = max(n.end for n in chord)

        if style == "waltz":
            t = start
            i = 0
            while t < end:
                pitch = root if i % 3 == 0 else fifth
                note_end = min(t + beat * 0.9, end)
                bass.notes.append(pretty_midi.Note(velocity=85, pitch=pitch, start=t, end=note_end))
                t += beat
                i += 1

        elif style == "ballad":
            t = start
            while t < end:
                note_end = min(t + beat * 2 * 0.95, end)
                bass.notes.append(pretty_midi.Note(velocity=75, pitch=root, start=t, end=note_end))
                t += beat * 2

        elif style == "latin":
            t = start
            i = 0
            while t < end:
                pitch = root if i % 2 == 0 else fifth
                note_end = min(t + beat / 2 * 0.85, end)
                bass.notes.append(pretty_midi.Note(velocity=85, pitch=pitch, start=t, end=note_end))
                t += beat / 2
                i += 1

        else:  # "pop" et autres styles non spécialisés
            t = start
            while t < end:
                note_end = min(t + beat * 0.9, end)
                bass.notes.append(pretty_midi.Note(velocity=90, pitch=root, start=t, end=note_end))
                t += beat

    return bass


# --------------------------------------------------------------------------
# Notes d'ornement
# --------------------------------------------------------------------------

def build_ornament_track(melody_notes: List[pretty_midi.Note]) -> pretty_midi.Instrument:
    """Notes de passage / d'échappée entre les notes mélodiques."""
    ornaments = pretty_midi.Instrument(program=68, name="Ornaments")
    for i in range(len(melody_notes) - 1):
        n1, n2 = melody_notes[i], melody_notes[i + 1]
        interval = n2.pitch - n1.pitch
        if abs(interval) >= 3:
            direction = 1 if interval > 0 else -1
            pitch = n1.pitch + direction * 2
            gap = max(n2.start - n1.end, 0)
            dur = min(0.15, gap / 2) if gap else 0.10
            start = max(n1.end, n2.start - 0.15)
            ornaments.notes.append(pretty_midi.Note(
                velocity=55, pitch=pitch, start=start, end=start + dur
            ))
    return ornaments


# --------------------------------------------------------------------------
# RÉPONSES INSTRUMENTALES — courts arpèges harmoniques
# --------------------------------------------------------------------------
#
# Une réponse n'est PAS une nouvelle mélodie.
# Elle doit remplir l'espace entre deux groupes de notes avec un court geste
# harmonique de 3 à 6 notes, directement dérivé des accords concernés.
#
# Règles :
#   - 3 à 6 notes par réponse ;
#   - toutes les notes appartiennent à l'accord courant ou à l'accord suivant,
#     avec priorité aux notes communes ;
#   - la dernière note tombe sur une note consonante de l'accord suivant ;
#   - le motif est placé dans l'espace réel entre les deux phrases ;
#   - aucune réponse si l'espace est trop court ;
#   - vélocité clairement audible mais subordonnée au piano ;
#   - registre spécifique à chaque instrument ;
#   - petits arpèges ascendants/descendants/brisés, avec variations.
#
# Le point essentiel : on ne choisit plus des hauteurs "à côté" de la mélodie.
# On construit le matériau depuis L'HARMONIE RÉELLE du MIDI.

def _response_register(name: str):
    return {
        "clarinet": (55, 88),
        "flute": (67, 98),
        "guitar": (52, 84),
        "piano_high": (72, 105),
    }.get(name, (55, 88))


def _unique_pcs(chord):
    return list(dict.fromkeys(int(n.pitch) % 12 for n in chord))


def _nearest_pitch(pc: int, reference: int, low: int, high: int) -> int:
    candidates = [
        pc + 12 * octave
        for octave in range(-1, 11)
        if low <= pc + 12 * octave <= high
    ]
    return min(candidates, key=lambda p: abs(p - reference)) if candidates else clamp_to_range(reference, low, high)


def _arp_chord_pitches(chord, name: str, reference: int) -> List[int]:
    """Transforme un accord réel du MIDI en notes utilisables par l'arpège."""
    pcs = _unique_pcs(chord)
    if not pcs:
        return []

    low, high = _response_register(name)

    # On conserve jusqu'à 4 degrés réellement présents dans l'accord.
    pitches = []
    for pc in pcs:
        p = _nearest_pitch(pc, reference if not pitches else pitches[-1], low, high)
        if not pitches or p != pitches[-1]:
            pitches.append(p)

    return pitches


def _make_response_arpeggio(
    current_chord,
    next_chord,
    last_melody_pitch: int,
    name: str,
    variation: int,
) -> List[int]:
    """
    Construit un arpège de 3 à 6 notes.

    Les premières notes viennent principalement de l'accord de départ,
    les dernières préparent l'accord suivant. La dernière note est toujours
    choisie dans l'accord d'arrivée.
    """
    low, high = _response_register(name)

    current_pcs = _unique_pcs(current_chord)
    next_pcs = _unique_pcs(next_chord)
    if not current_pcs and not next_pcs:
        return []

    common = [pc for pc in current_pcs if pc in next_pcs]
    arrival = next_pcs or current_pcs
    source = current_pcs or arrival

    # 3 à 6 notes selon l'instrument et la variation.
    count = {
        "clarinet": 4,
        "flute": 4,
        "guitar": 5,
        "piano_high": 4,
    }.get(name, 4)

    # Quelques réponses à 5/6 notes pour éviter la monotonie.
    if variation % 5 == 0:
        count += 1
    if variation % 11 == 0 and name in ("guitar", "piano_high"):
        count += 1
    count = max(3, min(6, count))

    # Matériau : accord courant + notes communes + accord d'arrivée.
    material = []
    for pc in source + common + arrival:
        if pc not in material:
            material.append(pc)

    # Convertit le matériau en notes proches, en privilégiant une progression
    # par degrés d'accord plutôt que des sauts chromatiques arbitraires.
    candidates = [
        _nearest_pitch(pc, last_melody_pitch, low, high)
        for pc in material
    ]

    # La cible finale doit appartenir à l'accord d'arrivée.
    target_pc = (
        common[0] if common
        else arrival[0]
    )
    target = _nearest_pitch(
        target_pc,
        last_melody_pitch + (-7 if variation % 2 == 0 else 7),
        low,
        high,
    )

    # Évite une réponse qui commence exactement sur la dernière note mélodique.
    usable = [p for p in candidates if p != last_melody_pitch]
    if not usable:
        usable = candidates

    # Trois familles de gestes :
    #   0 = montée vers la résolution
    #   1 = descente vers la résolution
    #   2 = arpège brisé (aller-retour)
    family = variation % 3

    if family == 0:
        ordered = sorted(usable, key=lambda p: p)
    elif family == 1:
        ordered = sorted(usable, key=lambda p: p, reverse=True)
    else:
        ascending = sorted(usable)
        ordered = ascending + list(reversed(ascending[:-1]))

    # Choisit des notes distinctes jusqu'à la longueur souhaitée.
    body = []
    for p in ordered:
        if p not in body:
            body.append(p)
        if len(body) >= count - 1:
            break

    # Si l'accord ne fournit que 2–3 hauteurs distinctes, on réutilise une
    # note structurelle à l'octave, jamais une note chromatique étrangère.
    if len(body) < count - 1:
        for p in list(body):
            for candidate in (p + 12, p - 12):
                if low <= candidate <= high and candidate not in body:
                    body.append(candidate)
                    if len(body) >= count - 1:
                        break
            if len(body) >= count - 1:
                break

    body = body[:count - 1]

    # Ordonne le corps pour que la dernière transition vers la cible soit
    # courte et musicale.
    if body:
        body[-1] = min(body, key=lambda p: abs(p - target))

    # Réordonne le corps en respectant la famille de geste.
    if family == 0:
        body = sorted(body)
    elif family == 1:
        body = sorted(body, reverse=True)
    elif len(body) >= 3:
        body = [body[0], body[-1], *body[1:-1]]

    result = body + [target]

    # Élimine les doublons consécutifs et borne strictement le registre.
    clean = []
    for p in result:
        p = clamp_to_range(int(p), low, high)
        if not clean or p != clean[-1]:
            clean.append(p)

    # Garantie absolue : 3–6 notes.
    if len(clean) < 3:
        for pc in arrival:
            p = _nearest_pitch(pc, clean[-1] if clean else last_melody_pitch, low, high)
            if not clean or p != clean[-1]:
                clean.append(p)
            if len(clean) >= 3:
                break

    return clean[:6]


def _response_events(start: float, end: float, count: int, beat: float, name: str):
    """Place précisément l'arpège dans l'espace disponible."""
    available = end - start
    if available < beat * 0.75 or count < 3:
        return []

    # L'arpège occupe ~85% de l'espace, en laissant une respiration finale.
    usable = min(available * 0.88, beat * 2.25)
    step = usable / count

    # Légèrement plus legato pour les vents, plus détaché pour guitare/piano.
    gate = 0.84 if name in ("clarinet", "flute") else 0.68

    return [
        (start + i * step, max(0.055, step * gate))
        for i in range(count)
    ]


def build_response_tracks(
    melody_notes: List[pretty_midi.Note],
    chords,
    tempo_bpm: float,
    responses: List[str],
    style: str = "pop",
) -> dict:
    """
    Génère des réponses sous forme de courts arpèges harmoniques de 3 à 6 notes.

    Une réponse est placée ENTRE la fin d'un groupe de quatre accords et le
    début du groupe suivant. Elle ne déborde donc pas aléatoirement dans la
    phrase suivante.

    La sélection instrumentale reste alternée, mais chaque réponse est
    harmoniquement liée aux accords qui l'encadrent.
    """
    if not responses or not melody_notes or len(chords) < 5:
        return {}

    beat = 60.0 / max(tempo_bpm, 40)

    tracks = {
        name: pretty_midi.Instrument(
            program=RESPONSE_INSTRUMENTS[name],
            name=f"Réponse {name.capitalize()}",
        )
        for name in responses
        if name in RESPONSE_INSTRUMENTS
    }
    if not tracks:
        return {}

    response_index = 0

    # Un groupe = 4 accords. La réponse est entre l'accord 4 et l'accord 5.
    for phrase_end in range(3, min(len(chords) - 1, len(melody_notes) - 1), 4):
        current_chord = chords[phrase_end]
        next_chord = chords[phrase_end + 1]

        last_melody = melody_notes[phrase_end]
        next_phrase_start = min(n.start for n in next_chord)

        # Vrai espace disponible : de la fin de la dernière note jusqu'au
        # début de l'accord suivant. Aucune réponse si l'espace est inexistant.
        start = last_melody.end + min(0.035, beat * 0.04)
        end = next_phrase_start - min(0.035, beat * 0.04)

        if end - start < beat * 0.75:
            continue

        name = responses[response_index % len(responses)]
        variation = response_index
        response_index += 1

        if name not in tracks:
            continue

        arp = _make_response_arpeggio(
            current_chord,
            next_chord,
            last_melody.pitch,
            name,
            variation,
        )

        if len(arp) < 3:
            continue

        events = _response_events(start, end, len(arp), beat, name)
        if len(events) != len(arp):
            continue

        # Volume volontairement audible. Le piano principal est généralement
        # à 70–75 ; les réponses sont donc placées autour de 82–92.
        base_velocity = {
            "clarinet": 88,
            "flute": 86,
            "guitar": 91,
            "piano_high": 84,
        }.get(name, 86)

        # Petite courbe : départ affirmé, résolution légèrement accentuée.
        for i, (pitch, (note_start, duration)) in enumerate(zip(arp, events)):
            frac = i / max(1, len(arp) - 1)
            velocity = int(base_velocity + frac * 7)

            tracks[name].notes.append(
                pretty_midi.Note(
                    velocity=min(110, max(65, velocity)),
                    pitch=clamp_to_range(pitch, *_response_register(name)),
                    start=note_start,
                    end=min(note_start + duration, end),
                )
            )

    return tracks


# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------

def orchestrate(
    pm: pretty_midi.PrettyMIDI,
    instruments: List[str],
    style: str,
    rolls: List[str],
    responses: List[str],
    add_rhythm: bool,
    add_ornaments: bool,
    keep_piano: bool = True,
) -> pretty_midi.PrettyMIDI:
    if not pm.instruments:
        raise ValueError("Aucune piste trouvée dans le fichier MIDI.")

    piano = pm.instruments[0]
    chords = group_notes_into_chords(piano.notes)
    if not chords:
        raise ValueError("Aucune note trouvée dans la piste piano.")

    tempo = estimate_tempo_from_chords(chords)

    total_duration = max(n.end for chord in chords for n in chord)

    all_tracks = {name: build_solo_track(name, chords, tempo) for name in instruments}

    if add_rhythm:
        all_tracks["__bass"] = build_bass_track(chords, tempo, style)
        all_tracks["__drums"] = build_drum_track(total_duration, tempo, style, rolls)

    melody_notes = [sorted(c, key=lambda n: n.pitch)[-1] for c in chords]

    if add_ornaments:
        all_tracks["__ornaments"] = build_ornament_track(melody_notes)

    if responses:
        response_tracks = build_response_tracks(melody_notes, chords, tempo, responses, style)
        for name, track in response_tracks.items():
            all_tracks[f"__response_{name}"] = track

    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    if keep_piano:
        piano.program = 0
        piano.name = "Piano"
        out.instruments.append(piano)

    out.instruments.extend(all_tracks.values())
    return out


def render_to_mp3(pm: pretty_midi.PrettyMIDI) -> bytes:
    if not os.path.exists(SOUNDFONT_PATH):
        raise RuntimeError(f"SoundFont introuvable à {SOUNDFONT_PATH}")

    with tempfile.TemporaryDirectory() as tmp:
        midi_path = os.path.join(tmp, "arrangement.mid")
        wav_path = os.path.join(tmp, "arrangement.wav")
        mp3_path = os.path.join(tmp, "arrangement.mp3")

        pm.write(midi_path)
        if os.path.getsize(midi_path) < 50:
            raise RuntimeError("Fichier MIDI généré anormalement petit/vide avant rendu audio.")

        result_wav = subprocess.run(
            ["fluidsynth", "-ni", SOUNDFONT_PATH, midi_path, "-F", wav_path, "-r", "44100"],
            capture_output=True, timeout=90,
        )
        if result_wav.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(
                f"Échec du rendu WAV (fluidsynth). stderr: {result_wav.stderr.decode(errors='ignore')[:500]}"
            )

        result_mp3 = subprocess.run(
            ["lame", "-b", "192", "-q", "2", wav_path, mp3_path],
            capture_output=True, timeout=60,
        )
        if result_mp3.returncode != 0 or not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1000:
            raise RuntimeError(
                f"Échec de l'encodage MP3 (lame). stderr: {result_mp3.stderr.decode(errors='ignore')[:500]}"
            )

        with open(mp3_path, "rb") as f:
            data = f.read()

        if not (data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)):
            raise RuntimeError("Le fichier encodé ne ressemble pas à un MP3 valide (en-tête inattendu).")

        return data


def safe_output_basename(original_filename: str) -> str:
    base = os.path.splitext(original_filename or "orchestration")[0]
    base = "".join(c for c in base if c.isalnum() or c in (" ", "-", "_")).strip()
    return (base or "orchestration") + "_Orchestrated"


@app.post("/orchestrate")
async def orchestrate_endpoint(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
    instrument: str = "trumpet",
    style: str = "pop",
    rolls: str = "snare",              # ex: "snare,toms,crescendo,double"
    responses: str = "",               # ex: "clarinet,flute,guitar,piano_high"
    add_rhythm: bool = False,
    add_ornaments: bool = False,
    keep_piano: bool = True,
    format: str = "mp3",
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    instruments = parse_instruments(instrument)
    roll_list = parse_rolls(rolls)
    response_list = parse_responses(responses)
    style = style.strip().lower() if style.strip().lower() in STYLES else "pop"

    if instrument.strip().lower() in ("band", "orchestra", "full", "all"):
        add_rhythm = True

    original_name = file.filename or "orchestration.mid"
    out_basename = safe_output_basename(original_name)

    raw = await file.read()
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fichier MIDI invalide: {e}")

    try:
        detected_tempo = estimate_tempo_from_chords(group_notes_into_chords(pm.instruments[0].notes))
    except Exception:
        detected_tempo = 120.0

    try:
        result = orchestrate(
            pm,
            instruments=instruments,
            style=style,
            rolls=roll_list,
            responses=response_list,
            add_rhythm=add_rhythm,
            add_ornaments=add_ornaments,
            keep_piano=keep_piano,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'orchestration: {e}")

    if format == "midi":
        buf = io.BytesIO()
        result.write(buf)
        data = buf.getvalue()
        return Response(
            content=data,
            media_type="audio/midi",
            headers={
                "Content-Disposition": f'attachment; filename="{out_basename}.mid"',
                "X-Detected-Tempo": f"{detected_tempo:.1f}",
            },
        )

    try:
        mp3_bytes = render_to_mp3(result)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Le rendu audio a dépassé le temps imparti.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de rendu audio: {e}")

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(mp3_bytes)),
            "Content-Disposition": f'inline; filename="{out_basename}.mp3"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "X-Detected-Tempo": f"{detected_tempo:.1f}",
        },
    )


@app.get("/instruments")
async def list_instruments():
    return {
        "instruments": {k: v["name"] for k, v in INSTRUMENTS.items()},
        "aliases": list(INSTRUMENT_ALIASES.keys()),
        "styles": sorted(STYLES),
        "rolls": sorted(ROLL_TYPES),
        "responses": sorted(RESPONSE_INSTRUMENTS.keys()),
    }


def _test_lame_encoding() -> dict:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "silence.wav")
            mp3_path = os.path.join(tmp, "silence.mp3")
            with wave.open(wav_path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(struct.pack("<4410h", *([0] * 4410)))
            result = subprocess.run(
                ["lame", "-b", "192", "-q", "2", wav_path, mp3_path],
                capture_output=True, timeout=15,
            )
            mp3_size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0
            return {
                "ok": result.returncode == 0 and mp3_size > 0,
                "returncode": result.returncode,
                "mp3_bytes": mp3_size,
                "stderr": result.stderr.decode(errors="ignore")[:300],
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
async def health():
    soundfont_ok = os.path.exists(SOUNDFONT_PATH)
    return {
        "status": "ok",
        "soundfont_found": soundfont_ok,
        "soundfont_path": SOUNDFONT_PATH,
        "fluidsynth_found": shutil.which("fluidsynth") is not None,
        "lame_found": shutil.which("lame") is not None,
        "lame_encode_test": _test_lame_encoding(),
    }
