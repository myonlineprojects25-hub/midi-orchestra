"""
Micro-service d'orchestration MIDI -> MP3.

Deux types de fichiers d'entrée sont supportés :

Fichiers SATB à 4 pistes nommées (Soprano/Alto/Tnor/Basse), comme les
cantiques classiques : chaque voix est monophonique et continue. Dans ce
cas, les instruments choisis DOUBLENT la vraie voix correspondante
(Soprano pour les rôles mélodiques, Alto pour l'harmonie, Basse pour les
rôles de basse), avec son rythme et ses hauteurs réels — pas une
reconstruction synthétique à partir d'accords plaqués.

Fichiers piano à accords plaqués (une seule piste, jusqu'à 4 notes par
accord) : comportement historique, la mélodie/harmonie/basse sont
déduites de l'empilement des notes de chaque accord.

Dans les deux cas, le résultat est enrichi avec :

un style rythmique (pop, ballade, latin, valse, classic, gospel, rnb, blues),

des roulements de batterie en fin de phrase (caisse claire, toms,
crescendo, double roulement), combinables,

des notes d'ornement (passages/échappées),

des réponses instrumentales : de courts arpèges qui suivent la gamme
diatonique du morceau (tonalité estimée via Krumhansl-Schmuckler),
placés uniquement aux vraies frontières de phrase détectées (silence
réel ou accord tenu), pas à intervalles mécaniques fixes.

Le résultat est rendu directement en MP3 via FluidSynth + lame, et le nom
du fichier reprend celui du MIDI importé (+ "_Orchestrated.mp3").
"""

import io
import os
import shutil
import subprocess
import struct
import tempfile
import unicodedata
import wave
from typing import Dict, List, Optional, Tuple

import pretty_midi
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

app = FastAPI(title="MIDI Orchestrator")

API_KEY = os.environ.get("ORCHESTRATOR_API_KEY", "change-moi")
SOUNDFONT_PATH = os.environ.get("SOUNDFONT_PATH", "/usr/share/sounds/sf2/FluidR3_GM.sf2")

CHORD_TOLERANCE = 0.05  # secondes

Notes GM du kit de batterie standard (canal 10)

DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42
DRUM_HIHAT_OPEN = 46
DRUM_CRASH = 49
DRUM_TOM_LOW = 45
DRUM_TOM_MID = 47
DRUM_TOM_HIGH = 50

--------------------------------------------------------------------------

Registre des instruments disponibles : programme General MIDI + rôle musical

--------------------------------------------------------------------------

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

Associe le rôle d'un instrument à la voix SATB réelle qu'il doit doubler

quand le fichier d'entrée fournit des voix nommées. Seuls les instruments

au timbre réellement grave restent dédiés à la voix de Basse seule — tous

les autres rôles (mélodie/harmonie) jouent désormais l'accord complet à 4

notes réelles via build_solo_track, pour ne jamais restreindre un

instrument à une seule voix isolée.

VOICE_FOR_ROLE = {
"bass_pad": "bass",
"bass_pulse": "bass",
}

STYLES = {"pop", "ballad", "latin", "waltz", "classic", "gospel", "rnb", "blues"}
ROLL_TYPES = {"snare", "toms", "crescendo", "double"}
RESPONSE_INSTRUMENTS = {
"clarinet": 71,
"flute": 73,
"guitar": 25,
"piano_high": 0,
}

--------------------------------------------------------------------------

DÉTECTION SATB (Soprano / Alto / Tenor / Basse)

--------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
if not name:
return ""
nfkd = unicodedata.normalize("NFKD", name)
ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
return ascii_name.strip().lower()

def detect_satb_voices(pm: pretty_midi.PrettyMIDI) -> Dict[str, List[pretty_midi.Note]]:
"""
Détecte une structure à voix nommées (Soprano/Alto/Tenor/Basse), comme
c'est l'usage classique pour les cantiques à 4 voix exportés depuis un
logiciel de notation. Renvoie {voix: [Note, ...]} si au moins 2 voix
sont identifiées par leur nom de piste, sinon {} (on retombe alors sur
le comportement "piano à accords").
"""
patterns = {
"soprano": ["soprano", "sop"],
"alto": ["alto", "alt"],
"tenor": ["tenor", "tnor", "ten"],
"bass": ["bass", "basse", "bas"],
}
voices: Dict[str, List[pretty_midi.Note]] = {}
for inst in pm.instruments:
name = _normalize_name(getattr(inst, "name", ""))
if not name or not inst.notes:
continue
for voice, keys in patterns.items():
if voice in voices:
continue
if any(k in name for k in keys):
voices[voice] = list(inst.notes)
break
return voices if len(voices) >= 2 else {}

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
d'accords plaqués ou de voix chorale sans pulsation percussive à
détecter. On calcule ici un tempo directement à partir de l'écartement
médian réel entre les attaques d'accords du fichier.
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

step = beat / 2
t = start
i = 0
while t < end:
    note_end = min(t + step * 0.85, end)
    track.notes.append(pretty_midi.Note(velocity=92, pitch=pattern[i % len(pattern)], start=t, end=note_end))
    t += step
    i += 1

def build_solo_track(name: str, chords, tempo: float) -> pretty_midi.Instrument:
"""
Construit la piste d'un instrument en dérivant mélodie/harmonie/basse
de l'empilement des accords. Utilisé quand aucune voix SATB nommée
n'est disponible (fichier piano à accords plaqués classique).
"""
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

    # Notes distinctes de l'accord (dédoublonnées par hauteur), pour que
    # chaque instrument mélodique/harmonique porte bien les VRAIES 4
    # notes de l'harmonie (Soprano+Alto+Tenor+Basse quand elles existent)
    # à volume comparable, plutôt qu'une seule voix isolée.
    distinct_pitches = []
    seen_pitches = set()
    for n in pitches:
        if n.pitch not in seen_pitches:
            seen_pitches.add(n.pitch)
            distinct_pitches.append(n)

    if role == "melody":
        for n in distinct_pitches:
            track.notes.append(pretty_midi.Note(
                velocity=95, pitch=n.pitch, start=start, end=max(start + 0.3, end - 0.05)
            ))

    elif role == "melody_high":
        for n in distinct_pitches:
            p = clamp_to_range(n.pitch + 12, 72, 96)
            track.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=start, end=end))

    elif role == "melody_sparkle":
        dur = min(0.25, end - start)
        for n in distinct_pitches:
            p = clamp_to_range(n.pitch + 12, 72, 108)
            track.notes.append(pretty_midi.Note(velocity=92, pitch=p, start=start, end=start + dur))

    elif role == "harmony":
        for n in distinct_pitches:
            track.notes.append(pretty_midi.Note(velocity=88, pitch=n.pitch, start=start, end=end))

    elif role == "harmony_high":
        for n in distinct_pitches:
            p = clamp_to_range(n.pitch + 12, 72, 96)
            track.notes.append(pretty_midi.Note(velocity=88, pitch=p, start=start, end=end))

    elif role == "bass_pad":
        p = clamp_to_range(bass.pitch - 12, 24, 48)
        track.notes.append(pretty_midi.Note(velocity=86, pitch=p, start=start, end=end))

    elif role == "pad_chord":
        # Toutes les notes réelles de l'accord, à bon volume : plus
        # besoin de l'alléger, le MIDI original n'est plus dans la
        # sortie pour qu'il faille s'en distinguer.
        for n in distinct_pitches:
            p = clamp_to_range(n.pitch + 12, 60, 96)
            track.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=start, end=end))

    elif role == "arpeggio":
        build_arpeggio_notes(track, pitches, start, end, beat)

    elif role == "bass_pulse":
        p = clamp_to_range(bass.pitch - 12, 24, 48)
        t = start
        while t < end:
            note_end = min(t + beat * 0.9, end)
            track.notes.append(pretty_midi.Note(velocity=88, pitch=p, start=t, end=note_end))
            t += beat

return track

def build_voice_double_track(name: str, voice_notes: List[pretty_midi.Note]) -> pretty_midi.Instrument:
"""
Double une VRAIE voix (Soprano/Alto/Basse) avec le timbre choisi, en
conservant son rythme et ses hauteurs réels — pas une reconstruction
synthétique à partir d'un empilement d'accords. C'est le chemin utilisé
quand le fichier d'entrée a des pistes SATB nommées.
"""
spec = INSTRUMENTS[name]
role = spec["role"]
track = pretty_midi.Instrument(program=spec["program"], name=spec["name"])

for n in sorted(voice_notes, key=lambda x: x.start):
    pitch = n.pitch
    end = n.end

    if role in ("melody_high", "harmony_high"):
        pitch = clamp_to_range(pitch + 12, 72, 96)
    elif role == "melody_sparkle":
        pitch = clamp_to_range(pitch + 12, 72, 108)
        end = min(end, n.start + 0.25)
    elif role in ("bass_pad", "bass_pulse"):
        pitch = clamp_to_range(pitch - 12, 24, 48)

    track.notes.append(pretty_midi.Note(
        velocity=max(80, min(115, n.velocity or 88)),
        pitch=pitch,
        start=n.start,
        end=max(end, n.start + 0.05),
    ))

return track

--------------------------------------------------------------------------

Rythme : batterie + basse, avec un pattern par style et des roulements

de fin de phrase choisis parmi plusieurs types

--------------------------------------------------------------------------

def build_fill(roll_type: str, drums: pretty_midi.Instrument, t: float, beat: float):
if roll_type == "toms":
toms = [DRUM_TOM_HIGH, DRUM_TOM_MID, DRUM_TOM_LOW, DRUM_TOM_LOW]
step = beat / 4
for i, pitch in enumerate(toms):
st = t + i * step
drums.notes.append(pretty_midi.Note(velocity=90 + i * 3, pitch=pitch, start=st, end=st + step * 0.85))

elif roll_type == "crescendo":
    n_hits = 6
    for i in range(n_hits):
        frac = i / n_hits
        st = t + frac * beat
        vel = 55 + int(frac * 45)
        drums.notes.append(pretty_midi.Note(velocity=vel, pitch=DRUM_HIHAT_OPEN, start=st, end=st + beat / n_hits * 0.8))
    drums.notes.append(pretty_midi.Note(velocity=115, pitch=DRUM_CRASH, start=t + beat * 0.85, end=t + beat))

elif roll_type == "double":
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

    else:
        t = start
        while t < end:
            note_end = min(t + beat * 0.9, end)
            bass.notes.append(pretty_midi.Note(velocity=90, pitch=root, start=t, end=note_end))
            t += beat

return bass

def build_ornament_track(melody_notes: List[pretty_midi.Note], skip_indices=frozenset()) -> pretty_midi.Instrument:
"""
Notes de passage / d'échappée entre les notes mélodiques.
skip_indices évite de placer un ornement là où une réponse
instrumentale a déjà été réservée.
"""
ornaments = pretty_midi.Instrument(program=68, name="Ornaments")
for i in range(len(melody_notes) - 1):
if i in skip_indices:
continue
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

--------------------------------------------------------------------------

ESTIMATION DE TONALITÉ (Krumhansl-Schmuckler)

--------------------------------------------------------------------------

_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def estimate_key(notes: List[pretty_midi.Note]) -> Tuple[int, str]:
"""Renvoie (tonic_pitch_class, mode) où mode est 'major' ou 'minor'."""
weights = [0.0] * 12
for n in notes:
weights[n.pitch % 12] += max(n.end - n.start, 0.05)

if sum(weights) == 0:
    return 0, "major"

best_score, best_tonic, best_mode = None, 0, "major"
for tonic in range(12):
    for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        score = sum(weights[pc] * profile[(pc - tonic) % 12] for pc in range(12))
        if best_score is None or score > best_score:
            best_score, best_tonic, best_mode = score, tonic, mode

return best_tonic, best_mode

def _diatonic_degrees(tonic_pc: int, mode: str) -> List[int]:
# Mineur harmonique (7e degré haussé) pour une vraie sensible qui résout.
intervals = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 11]
return [(tonic_pc + i) % 12 for i in intervals]

def _scale_pitches(tonic_pc: int, mode: str, low: int, high: int) -> List[int]:
degrees = set(_diatonic_degrees(tonic_pc, mode))
return sorted(p for p in range(low, high + 1) if p % 12 in degrees)

def _nearest_index(pitches: List[int], value: int) -> int:
return min(range(len(pitches)), key=lambda i: abs(pitches[i] - value))

--------------------------------------------------------------------------

DÉTECTION DE VRAIES FRONTIÈRES DE PHRASE

--------------------------------------------------------------------------

def detect_phrase_boundaries(chords) -> List[int]:
"""
Une frontière de phrase est un endroit musicalement réel où placer une
réponse : un vrai silence avant l'accord suivant, ou un accord
d'ARRIVÉE nettement plus tenu que la normale locale (cadence). Repli
sur un calage toutes les 4 accords seulement si le morceau n'a aucun
de ces indices (legato très régulier sans aucune respiration).
"""
if len(chords) < 3:
return []

harmonic_durations = []
for i, c in enumerate(chords):
    start = min(n.start for n in c)
    if i + 1 < len(chords):
        next_start = min(n.start for n in chords[i + 1])
    else:
        next_start = max(n.end for n in c)
    harmonic_durations.append(max(next_start - start, 0.01))

sorted_durs = sorted(harmonic_durations)
median_dur = sorted_durs[len(sorted_durs) // 2]

boundaries = []
n = len(chords)
for i in range(n - 1):
    end_ = max(x.end for x in chords[i])
    next_start = min(x.start for x in chords[i + 1])
    gap = next_start - end_
    # Cas normal : l'accord de DÉPART (i) est tenu plus longtemps que la
    # normale (allongement agogique classique en fin de phrase interne).
    held_departure = harmonic_durations[i] > median_dur * 1.6
    # Cas particulier : seule la toute dernière transition doit aussi
    # vérifier l'accord d'ARRIVÉE, pour capter un accord final sustenu
    # qui n'a lui-même pas de transition sortante à mesurer.
    held_final_arrival = (i == n - 2) and (harmonic_durations[i + 1] > median_dur * 1.6)
    if gap > 0.12 or held_departure or held_final_arrival:
        boundaries.append(i)

# Repli uniquement si AUCUNE frontière réelle n'a été trouvée (silence
# ou accord tenu) — un cantique bien composé a naturellement peu de
# phrases (3-6 sur un morceau court), ce n'est pas un signe d'échec.
if not boundaries:
    boundaries = list(range(3, len(chords) - 1, 4))

return boundaries

--------------------------------------------------------------------------

RÉPONSES INSTRUMENTALES — mouvement par degrés de la gamme, placées

uniquement aux frontières de phrase détectées

--------------------------------------------------------------------------

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

def _make_diatonic_response(
scale_pitches: List[int],
start_pitch: int,
target_pitch: int,
count: int,
) -> List[int]:
"""
Construit une ligne de 'count' notes qui se déplace PAR DEGRÉS DE LA
GAMME entre start_pitch et target_pitch (inclus en dernier) — une
vraie conduite des voix diatonique.
"""
if not scale_pitches or count < 1:
return []

i0 = _nearest_index(scale_pitches, start_pitch)
i1 = _nearest_index(scale_pitches, target_pitch)

if i0 == i1:
    i1 = i0 + 1 if i0 + 1 < len(scale_pitches) else max(0, i0 - 1)

step = 1 if i1 > i0 else -1
indices = list(range(i0, i1 + step, step))

if len(indices) > count:
    chosen = [indices[0]]
    inner = indices[1:-1]
    if inner and count > 2:
        pick_step = max(1, len(inner) // max(1, count - 2))
        chosen += inner[::pick_step][: count - 2]
    chosen.append(indices[-1])
    indices = chosen[:count]

elif len(indices) < count:
    extra_needed = count - len(indices)
    prefix = []
    cur = i0
    direction = -step
    for _ in range(extra_needed):
        cur = max(0, min(len(scale_pitches) - 1, cur + direction))
        prefix.append(cur)
    indices = list(reversed(prefix)) + indices

return [scale_pitches[i] for i in indices[:count]]

def _response_events(start: float, end: float, count: int, beat: float, name: str):
available = end - start
if available <= 0 or count < 1:
return []

usable = min(available * 0.92, beat * 2.25)
usable = max(usable, min(available, beat * 0.5))
step = usable / count

gate = 0.84 if name in ("clarinet", "flute") else 0.68

return [
    (start + i * step, max(0.06, step * gate))
    for i in range(count)
]

def build_response_tracks(
melody_notes: List[pretty_midi.Note],
chords,
tempo_bpm: float,
responses: List[str],
tonic_pc: int,
mode: str,
) -> Tuple[dict, set]:
reserved_indices = set()

if not responses or not melody_notes or len(chords) < 3:
    return {}, reserved_indices

boundaries = detect_phrase_boundaries(chords)
if not boundaries:
    return {}, reserved_indices

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
    return {}, reserved_indices

response_index = 0

for boundary in boundaries:
    if boundary + 1 >= len(chords) or boundary >= len(melody_notes):
        continue

    reserved_indices.add(boundary)

    current_chord = chords[boundary]
    next_chord = chords[boundary + 1]
    last_melody = melody_notes[boundary]
    next_phrase_start = min(n.start for n in next_chord)

    end = next_phrase_start - min(0.03, beat * 0.03)
    window_len = max(beat * 0.9, min(beat * 2.25, end - last_melody.start))
    start = max(last_melody.start, end - window_len)

    if end - start < beat * 0.5:
        continue

    name = responses[response_index % len(responses)]
    response_index += 1
    if name not in tracks:
        continue

    low, high = _response_register(name)
    scale_pitches = _scale_pitches(tonic_pc, mode, low, high)
    if not scale_pitches:
        continue

    next_pcs = _unique_pcs(next_chord) or _unique_pcs(current_chord)
    target = _nearest_pitch(next_pcs[0], last_melody.pitch, low, high)
    diatonic_set = set(_diatonic_degrees(tonic_pc, mode))
    for pc in next_pcs:
        candidate = _nearest_pitch(pc, last_melody.pitch, low, high)
        if candidate % 12 in diatonic_set:
            target = candidate
            break

    count = {"clarinet": 4, "flute": 4, "guitar": 5, "piano_high": 4}.get(name, 4)
    if response_index % 5 == 0:
        count += 1
    count = max(3, min(6, count))

    arp = _make_diatonic_response(scale_pitches, last_melody.pitch, target, count)
    if len(arp) < 3:
        continue

    events = _response_events(start, end, len(arp), beat, name)
    if len(events) != len(arp):
        continue

    base_velocity = {"clarinet": 100, "flute": 98, "guitar": 104, "piano_high": 96}.get(name, 98)

    for i, (pitch, (note_start, duration)) in enumerate(zip(arp, events)):
        frac = i / max(1, len(arp) - 1)
        velocity = int(base_velocity + frac * 10)
        tracks[name].notes.append(
            pretty_midi.Note(
                velocity=min(127, max(80, velocity)),
                pitch=clamp_to_range(pitch, low, high),
                start=note_start,
                end=min(note_start + duration, end),
            )
        )

return tracks, reserved_indices

--------------------------------------------------------------------------

Assemblage

--------------------------------------------------------------------------

def _scale_velocity(notes: List[pretty_midi.Note], factor: float):
for n in notes:
n.velocity = max(1, min(127, int(round((n.velocity or 80) * factor))))

def _set_pan(instrument: pretty_midi.Instrument, pan_value: int):
"""pan_value: 0 (gauche) - 64 (centre) - 127 (droite), via CC10."""
try:
instrument.control_changes.append(
pretty_midi.ControlChange(number=10, value=max(0, min(127, pan_value)), time=0.0)
)
except Exception:
pass  # certains environnements de test n'ont pas ControlChange, sans conséquence

Panoramique déterministe par instrument, pour que des voix ajoutées aux

hauteurs proches de la piste originale restent distinctes à l'oreille

même quand le volume seul ne suffit pas à les séparer.

_PAN_BY_NAME = {
"trumpet": 100, "flute": 30, "clarinet": 40, "clarinet_high": 20,
"saxophone": 92, "trombone": 105, "tuba": 112, "organ": 64,
"choir": 64, "guitar": 25, "bass_guitar": 64, "electric_guitar": 35,
"piano_low": 64, "piano_medium": 64, "piano_high": 50,
}

def orchestrate(
pm: pretty_midi.PrettyMIDI,
instruments: List[str],
style: str,
rolls: List[str],
responses: List[str],
add_rhythm: bool,
add_ornaments: bool,
keep_piano: bool = False,
) -> pretty_midi.PrettyMIDI:
if not pm.instruments:
raise ValueError("Aucune piste trouvée dans le fichier MIDI.")

satb = detect_satb_voices(pm)

if satb:
    combined_notes = [n for notes in satb.values() for n in notes]
    chords = group_notes_into_chords(combined_notes)
else:
    piano = pm.instruments[0]
    combined_notes = piano.notes
    chords = group_notes_into_chords(piano.notes)

if not chords:
    raise ValueError("Aucune note trouvée dans le fichier.")

tempo = estimate_tempo_from_chords(chords)
total_duration = max(n.end for chord in chords for n in chord)

all_tracks = {}
for name in instruments:
    role = INSTRUMENTS[name]["role"]
    voice_key = VOICE_FOR_ROLE.get(role)
    if satb and voice_key and voice_key in satb and satb[voice_key]:
        all_tracks[name] = build_voice_double_track(name, satb[voice_key])
    else:
        all_tracks[name] = build_solo_track(name, chords, tempo)

if add_rhythm:
    all_tracks["__bass"] = build_bass_track(chords, tempo, style)
    all_tracks["__drums"] = build_drum_track(total_duration, tempo, style, rolls)

melody_notes = [sorted(c, key=lambda n: n.pitch)[-1] for c in chords]

reserved_indices = set()
if responses:
    tonic_pc, mode = estimate_key(combined_notes)
    response_tracks, reserved_indices = build_response_tracks(melody_notes, chords, tempo, responses, tonic_pc, mode)
    for name, track in response_tracks.items():
        all_tracks[f"__response_{name}"] = track

if add_ornaments:
    all_tracks["__ornaments"] = build_ornament_track(melody_notes, skip_indices=reserved_indices)

# Panoramique déterministe par instrument ajouté, pour rester distinct
# de la piste originale même quand les hauteurs se recoupent.
for track_name, track in all_tracks.items():
    pan = _PAN_BY_NAME.get(track_name)
    if pan is None and track_name.startswith("__response_"):
        pan = _PAN_BY_NAME.get(track_name.replace("__response_", ""), 64)
    _set_pan(track, 64 if pan is None else pan)

out = pretty_midi.PrettyMIDI(initial_tempo=tempo)

if keep_piano:
    if satb:
        for voice_name, notes in satb.items():
            v = pretty_midi.Instrument(program=0, name=voice_name.capitalize())
            v.notes = sorted(notes, key=lambda n: n.start)
            # Volume réduit : la piste originale doit servir de
            # fondation discrète, pas rivaliser à égalité avec les
            # instruments ajoutés (c'est ça qui créait l'effet "fondu").
            _scale_velocity(v.notes, 0.62)
            _set_pan(v, 64)
            out.instruments.append(v)
    else:
        piano = pm.instruments[0]
        piano.program = 0
        piano.name = "Piano"
        _scale_velocity(piano.notes, 0.62)
        _set_pan(piano, 64)
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
rolls: str = "snare",
responses: str = "",
add_rhythm: bool = False,
add_ornaments: bool = False,
output_filename: str = "",
keep_piano: bool = False,
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

# Utilise output_filename s'il est fourni ET non vide, sinon traitement existant
if output_filename and output_filename.strip():
    out_basename = output_filename.replace(".mp3", "").replace(".mid", "").strip()
else:
    out_basename = safe_output_basename(original_name)

raw = await file.read()
try:
    pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
except Exception as e:
    raise HTTPException(status_code=400, detail=f"Fichier MIDI invalide: {e}")

try:
    satb_probe = detect_satb_voices(pm)
    probe_notes = [n for notes in satb_probe.values() for n in notes] if satb_probe else pm.instruments[0].notes
    detected_tempo = estimate_tempo_from_chords(group_notes_into_chords(probe_notes))
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
