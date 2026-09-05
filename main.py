"""
MIDI-ORCHESTRA — moteur d'orchestration MIDI -> MP3

Principe musical de cette version
---------------------------------
Le MIDI importé est considéré comme une partition à 4 voix
(Soprano / Alto / Ténor / Basse) contenue dans une seule piste.

Le moteur NE RÉDUIT PLUS chaque groupe de 4 notes à sa seule note
supérieure. Les quatre voix sont conservées comme matière musicale
source pendant toute l'orchestration.

Chaque instrument sélectionné reçoit une PARTIE COMPLÈTE sur toute
la durée du morceau, construite à partir de la partition source :

- instruments mélodiques : ligne soprano, avec articulation musicale ;
- instruments harmoniques : voix intérieures / voicing complet selon
  le registre de l'instrument ;
- instruments graves : basse + mouvements de fondamentale/quinte ;
- chœur / orgue : voicing complet des 4 voix ;
- guitare : arpèges construits à partir des quatre voix ;
- pianos : voicing complet adapté au registre demandé.

Les enrichissements (rythme, ornements) et les réponses instrumentales
sont des couches ADDITIONNELLES. Ils ne remplacent jamais la matière
musicale source.
"""

import io
import os
import shutil
import subprocess
import struct
import tempfile
import wave
from typing import List, Dict, Tuple

import pretty_midi
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response


app = FastAPI(title="MIDI Orchestrator")

API_KEY = os.environ.get("ORCHESTRATOR_API_KEY", "change-moi")
SOUNDFONT_PATH = os.environ.get(
    "SOUNDFONT_PATH",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
)

# Les quatre voix d'un accord sont normalement quasi simultanées.
# On reste volontairement strict pour ne pas fusionner deux accords
# successifs joués rapidement.
CHORD_TOLERANCE = 0.05


# --------------------------------------------------------------------------
# Batterie GM
# --------------------------------------------------------------------------

DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42
DRUM_HIHAT_OPEN = 46
DRUM_CRASH = 49
DRUM_TOM_LOW = 45
DRUM_TOM_MID = 47
DRUM_TOM_HIGH = 50


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

INSTRUMENTS = {
    "trumpet":         {"program": 56, "name": "Trumpet",          "role": "melody_harmonic"},
    "flute":           {"program": 73, "name": "Flute",            "role": "melody_high"},
    "clarinet":        {"program": 71, "name": "Clarinet",         "role": "harmony"},
    "clarinet_high":   {"program": 71, "name": "Clarinette Aiguë", "role": "harmony_high"},
    "saxophone":       {"program": 65, "name": "Saxophone",        "role": "melody_harmonic"},
    "trombone":        {"program": 57, "name": "Trombone",         "role": "bass_harmony"},
    "tuba":            {"program": 58, "name": "Tuba",             "role": "bass"},
    "organ":           {"program": 19, "name": "Organ",            "role": "full_chord"},
    "choir":           {"program": 52, "name": "Choir",            "role": "full_chord"},
    "guitar":          {"program": 25, "name": "Guitar",           "role": "arpeggio"},
    "bass_guitar":     {"program": 32, "name": "Bass Guitar",      "role": "bass_pulse"},
    "electric_guitar": {"program": 29, "name": "Electric Guitar",  "role": "arpeggio"},
    "piano_low":       {"program": 0,  "name": "Piano (grave)",    "role": "low_voicing"},
    "piano_medium":    {"program": 0,  "name": "Piano (medium)",   "role": "full_voicing_medium"},
    "piano_high":      {"program": 0,  "name": "Piano (aigu)",     "role": "high_voicing"},
}

INSTRUMENT_ALIASES = {
    "band": ["trumpet", "clarinet", "organ"],
    "orchestra": [
        "trumpet", "clarinet", "organ", "flute", "trombone", "choir"
    ],
    "full": list(INSTRUMENTS.keys()),
    "all": list(INSTRUMENTS.keys()),
}

STYLES = {
    "pop", "ballad", "latin", "waltz",
    "classic", "gospel", "rnb", "blues",
}

ROLL_TYPES = {"snare", "toms", "crescendo", "double"}

RESPONSE_INSTRUMENTS = {
    "clarinet": 71,
    "flute": 73,
    "guitar": 25,
    "piano_high": 0,
}


# --------------------------------------------------------------------------
# Lecture / analyse de la partition source
# --------------------------------------------------------------------------

def group_notes_into_chords(notes: List[pretty_midi.Note]) -> List[List[pretty_midi.Note]]:
    """Regroupe les notes par onset pour former les événements harmoniques."""
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    groups = []
    for note in notes:
        if not groups or note.start - min(n.start for n in groups[-1]) > CHORD_TOLERANCE:
            groups.append([note])
        else:
            groups[-1].append(note)
    for group in groups:
        group.sort(key=lambda n: (n.pitch, n.start))
    return groups


def collect_satb_chords(pm: pretty_midi.PrettyMIDI) -> List[List[pretty_midi.Note]]:
    """Lit TOUTES les pistes musicales et reconstruit les groupes SATB."""
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)

    if not notes:
        return []

    groups = group_notes_into_chords(notes)
    result = []
    for group in groups:
        # NE PAS dédupliquer les hauteurs : deux voix SATB peuvent
        # volontairement chanter/jouer la même hauteur à l'octave ou à
        # l'unisson. Chaque piste source représente une vraie voix.
        group = sorted(group, key=lambda x: x.pitch)

        # Le format attendu est jusqu'à quatre voix. Pour le MIDI SATB,
        # on conserve exactement les quatre notes présentes sur chaque onset.
        if len(group) > 4:
            group = [group[0], group[1], group[-2], group[-1]]

        result.append(group)
    return result


def estimate_tempo_from_chords(
    chords: List[List[pretty_midi.Note]],
) -> float:
    if len(chords) < 2:
        return 120.0

    onsets = sorted(chord_start(c) for c in chords)
    intervals = [
        b - a for a, b in zip(onsets, onsets[1:])
        if b - a > 0.05
    ]

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


def build_source_analysis(chords: List[List[pretty_midi.Note]]) -> List[Dict]:
    """
    Crée une représentation stable de la partition source.

    Cette structure est utilisée par les couches d'orchestration sans
    jamais remplacer les quatre voix originales.
    """
    analysis = []

    for index, chord in enumerate(chords):
        notes = normalize_four_voice_chord(chord)

        analysis.append({
            "index": index,
            "start": chord_start(chord),
            "end": chord_end(chord),
            "notes": notes,
            "pitches": [n.pitch for n in notes],
            "pitch_classes": chord_pitch_classes(chord),
            "bass": chord_bass(chord),
            "soprano": chord_soprano(chord),
            "inner": chord_inner_notes(chord),
        })

    return analysis


# --------------------------------------------------------------------------
# Paramètres
# --------------------------------------------------------------------------

def parse_instruments(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()

    if not raw:
        return ["trumpet"]

    if raw in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[raw]

    result = [
        tok.strip()
        for tok in raw.split(",")
        if tok.strip() in INSTRUMENTS
    ]

    return result or ["trumpet"]


def parse_rolls(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()

    if not raw:
        return ["snare"]

    result = [
        tok.strip()
        for tok in raw.split(",")
        if tok.strip() in ROLL_TYPES
    ]

    return result or ["snare"]


def parse_responses(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()

    if not raw:
        return []

    return [
        tok.strip()
        for tok in raw.split(",")
        if tok.strip() in RESPONSE_INSTRUMENTS
    ]


def clamp_to_range(pitch: int, low: int, high: int) -> int:
    p = int(pitch)

    while p < low:
        p += 12

    while p > high:
        p -= 12

    return max(0, min(127, p))


# --------------------------------------------------------------------------
# Construction des parties instrumentales COMPLÈTES
# --------------------------------------------------------------------------

def add_note(
    track: pretty_midi.Instrument,
    pitch: int,
    start: float,
    end: float,
    velocity: int,
    low: int = 0,
    high: int = 127,
):
    if end <= start:
        return

    track.notes.append(
        pretty_midi.Note(
            velocity=max(1, min(127, int(velocity))),
            pitch=clamp_to_range(pitch, low, high),
            start=float(start),
            end=float(end),
        )
    )


def add_full_voicing(
    track: pretty_midi.Instrument,
    chord: List[pretty_midi.Note],
    low: int,
    high: int,
    velocity: int,
    transpose_octaves: int = 0,
):
    """
    Joue toutes les voix disponibles du MIDI source.

    Les durées originales de chaque note sont conservées autant que possible.
    """
    for note in normalize_four_voice_chord(chord):
        pitch = note.pitch + transpose_octaves * 12
        add_note(
            track,
            pitch,
            note.start,
            note.end,
            velocity,
            low,
            high,
        )


def add_soprano_line(
    track: pretty_midi.Instrument,
    chords: List[List[pretty_midi.Note]],
    low: int,
    high: int,
    velocity: int,
    octave_shift: int = 0,
):
    """
    Partie complète mélodique : la soprano de chaque événement, sur tout
    le morceau. Ce n'est plus une réduction de la partition pour l'ensemble
    de l'orchestration : c'est une VOIX instrumentale explicite.
    """
    for chord in chords:
        notes = normalize_four_voice_chord(chord)
        if not notes:
            continue

        source = notes[-1]
        start = source.start
        end = source.end

        pitch = source.pitch + octave_shift * 12

        add_note(
            track,
            pitch,
            start,
            max(start + 0.08, end),
            velocity,
            low,
            high,
        )


def add_inner_voice_part(
    track: pretty_midi.Instrument,
    chords: List[List[pretty_midi.Note]],
    low: int,
    high: int,
    velocity: int,
    use_both_inner: bool = True,
):
    """
    Partie harmonique complète basée sur Alto + Ténor.

    Les voix intérieures originales sont conservées au lieu de fabriquer
    une note artificielle à partir de la soprano.
    """
    for chord in chords:
        inner = chord_inner_notes(chord)

        if not inner:
            notes = normalize_four_voice_chord(chord)
            inner = notes[-1:] if notes else []

        if not use_both_inner and inner:
            inner = [inner[0]]

        for note in inner:
            add_note(
                track,
                note.pitch,
                note.start,
                note.end,
                velocity,
                low,
                high,
            )


def add_bass_part(
    track: pretty_midi.Instrument,
    chords: List[List[pretty_midi.Note]],
    low: int,
    high: int,
    velocity: int,
):
    """
    Partie basse complète : la vraie voix de basse du MIDI est prioritaire.
    """
    for chord in chords:
        notes = normalize_four_voice_chord(chord)
        if not notes:
            continue

        bass = notes[0]

        add_note(
            track,
            bass.pitch,
            bass.start,
            bass.end,
            velocity,
            low,
            high,
        )


def add_bass_pulse_part(
    track: pretty_midi.Instrument,
    chords: List[List[pretty_midi.Note]],
    tempo: float,
    low: int,
    high: int,
    velocity: int,
):
    """
    Partie basse complète et rythmique, mais toujours dérivée de la vraie
    basse de chaque accord source.
    """
    beat = 60.0 / max(tempo, 40)

    for chord in chords:
        start = chord_start(chord)
        end = chord_end(chord)
        bass = chord_bass(chord)

        t = start
        beat_index = 0

        while t < end:
            duration = min(beat * 0.82, end - t)

            pitch = bass
            if beat_index % 4 in (1, 3):
                pitch = bass + 7

            add_note(
                track,
                pitch,
                t,
                t + duration,
                velocity,
                low,
                high,
            )

            t += beat
            beat_index += 1


def add_arpeggio_from_source(
    track: pretty_midi.Instrument,
    chords: List[List[pretty_midi.Note]],
    tempo: float,
    low: int,
    high: int,
    velocity: int,
):
    """
    Arpège complet dérivé des QUATRE voix du MIDI source.

    L'ordre est pensé pour garder la basse comme point d'ancrage tout en
    faisant entendre les notes supérieures de l'accord.
    """
    beat = 60.0 / max(tempo, 40)

    for chord in chords:
        notes = normalize_four_voice_chord(chord)
        if not notes:
            continue

        pitches = [n.pitch for n in notes]

        # Séquence source : grave -> intérieur -> aigu -> intérieur.
        if len(pitches) >= 4:
            pattern = [
                pitches[0],
                pitches[1],
                pitches[-1],
                pitches[2],
            ]
        elif len(pitches) == 3:
            pattern = [pitches[0], pitches[1], pitches[2], pitches[1]]
        elif len(pitches) == 2:
            pattern = [pitches[0], pitches[1]]
        else:
            pattern = pitches

        start = chord_start(chord)
        end = chord_end(chord)

        step = max(beat / 2, 0.08)
        t = start
        i = 0

        while t < end:
            note_end = min(t + step * 0.78, end)

            add_note(
                track,
                pattern[i % len(pattern)],
                t,
                note_end,
                velocity,
                low,
                high,
            )

            t += step
            i += 1


def _add_four_voice_voicing(
    track: pretty_midi.Instrument,
    chord: List[pretty_midi.Note],
    low: int,
    high: int,
    velocity_scale: float,
):
    """Joue les quatre voix du groupe simultanément."""
    voices = sorted(chord, key=lambda n: n.pitch)[-4:]
    pitches = []

    for n in voices:
        p = clamp_to_range(int(n.pitch), low, high)
        while pitches and p <= pitches[-1] and p + 12 <= high:
            p += 12
        pitches.append(p)

    for p, n in zip(pitches, voices):
        if n.end > n.start:
            track.notes.append(pretty_midi.Note(
                velocity=max(35, min(127, int(n.velocity * velocity_scale))),
                pitch=p,
                start=n.start,
                end=n.end,
            ))


def build_solo_track(name: str, chords, tempo: float) -> pretty_midi.Instrument:
    """
    Chaque instrument reçoit maintenant les QUATRE VOIX simultanément.
    Aucun instrument n'est réduit à la soprano, à la basse ou aux voix
    intérieures. Le registre est adapté à l'instrument, mais la polyphonie
    SATB est conservée.
    """
    spec = INSTRUMENTS[name]
    track = pretty_midi.Instrument(program=spec["program"], name=spec["name"])

    ranges = {
        "trumpet": (48, 84),
        "flute": (67, 100),
        "clarinet": (50, 91),
        "clarinet_high": (67, 103),
        "saxophone": (52, 88),
        "trombone": (40, 76),
        "tuba": (28, 60),
        "organ": (36, 96),
        "choir": (36, 96),
        "guitar": (40, 88),
        "bass_guitar": (28, 60),
        "electric_guitar": (40, 88),
        "piano_low": (28, 60),
        "piano_medium": (48, 88),
        "piano_high": (60, 108),
    }
    low, high = ranges.get(name, (36, 96))

    scales = {
        "choir": .85, "organ": .80, "piano_low": .90,
        "piano_medium": .95, "piano_high": .90,
        "bass_guitar": .90, "tuba": .90, "trombone": .92,
        "trumpet": .92, "flute": .90, "clarinet": .92,
        "clarinet_high": .90, "saxophone": .92, "guitar": .90,
        "electric_guitar": .92,
    }
    scale = scales.get(name, .90)

    for chord in chords:
        _add_four_voice_voicing(track, chord, low, high, scale)

    return track


# --------------------------------------------------------------------------
# Rythme
# --------------------------------------------------------------------------

def build_fill(
    roll_type: str,
    drums: pretty_midi.Instrument,
    t: float,
    beat: float,
):
    if roll_type == "toms":
        toms = [
            DRUM_TOM_HIGH,
            DRUM_TOM_MID,
            DRUM_TOM_LOW,
            DRUM_TOM_LOW,
        ]
        step = beat / 4

        for i, pitch in enumerate(toms):
            st = t + i * step
            drums.notes.append(
                pretty_midi.Note(
                    velocity=90 + i * 3,
                    pitch=pitch,
                    start=st,
                    end=st + step * 0.85,
                )
            )

    elif roll_type == "crescendo":
        n_hits = 6

        for i in range(n_hits):
            frac = i / n_hits
            st = t + frac * beat
            vel = 55 + int(frac * 45)

            drums.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=DRUM_HIHAT_OPEN,
                    start=st,
                    end=st + beat / n_hits * 0.8,
                )
            )

        drums.notes.append(
            pretty_midi.Note(
                velocity=115,
                pitch=DRUM_CRASH,
                start=t + beat * 0.85,
                end=t + beat,
            )
        )

    elif roll_type == "double":
        step = beat / 8

        for i in range(8):
            vel = min(
                65 + (15 if i % 2 == 0 else 0) + i * 3,
                127,
            )
            st = t + i * step

            drums.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=DRUM_SNARE,
                    start=st,
                    end=st + step * 0.75,
                )
            )

    else:
        step = beat / 4

        for i in range(4):
            vel = 70 + i * 10
            st = t + i * step

            drums.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=DRUM_SNARE,
                    start=st,
                    end=st + step * 0.8,
                )
            )


def add_style_beat(
    drums: pretty_midi.Instrument,
    t: float,
    beat: float,
    beat_i: int,
    style: str,
):
    if style == "ballad":
        if beat_i == 0:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=90,
                    pitch=DRUM_KICK,
                    start=t,
                    end=t + 0.1,
                )
            )

        if beat_i == 2:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=85,
                    pitch=DRUM_SNARE,
                    start=t,
                    end=t + 0.1,
                )
            )

        drums.notes.append(
            pretty_midi.Note(
                velocity=45,
                pitch=DRUM_HIHAT_CLOSED,
                start=t,
                end=t + beat * 0.8,
            )
        )

    elif style == "latin":
        if beat_i == 0:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=DRUM_KICK,
                    start=t,
                    end=t + 0.1,
                )
            )

        if beat_i == 1:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=90,
                    pitch=DRUM_KICK,
                    start=t + beat / 2,
                    end=t + beat / 2 + 0.1,
                )
            )

        if beat_i in (1, 3):
            drums.notes.append(
                pretty_midi.Note(
                    velocity=85,
                    pitch=DRUM_SNARE,
                    start=t,
                    end=t + 0.1,
                )
            )

        drums.notes.append(
            pretty_midi.Note(
                velocity=65,
                pitch=DRUM_HIHAT_CLOSED,
                start=t,
                end=t + beat * 0.4,
            )
        )

        drums.notes.append(
            pretty_midi.Note(
                velocity=55,
                pitch=DRUM_HIHAT_CLOSED,
                start=t + beat / 2,
                end=t + beat * 0.9,
            )
        )

    elif style == "waltz":
        if beat_i == 0:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=DRUM_KICK,
                    start=t,
                    end=t + 0.1,
                )
            )
        else:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=65,
                    pitch=DRUM_HIHAT_CLOSED,
                    start=t,
                    end=t + beat * 0.6,
                )
            )

    else:
        if beat_i in (0, 2):
            drums.notes.append(
                pretty_midi.Note(
                    velocity=105,
                    pitch=DRUM_KICK,
                    start=t,
                    end=t + 0.1,
                )
            )

        if beat_i in (1, 3):
            drums.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=DRUM_SNARE,
                    start=t,
                    end=t + 0.1,
                )
            )

        drums.notes.append(
            pretty_midi.Note(
                velocity=70,
                pitch=DRUM_HIHAT_CLOSED,
                start=t,
                end=t + beat * 0.4,
            )
        )

        drums.notes.append(
            pretty_midi.Note(
                velocity=55,
                pitch=DRUM_HIHAT_CLOSED,
                start=t + beat / 2,
                end=t + beat * 0.9,
            )
        )


def build_drum_track(
    total_duration: float,
    tempo_bpm: float,
    style: str,
    rolls: List[str],
) -> pretty_midi.Instrument:
    drums = pretty_midi.Instrument(
        program=0,
        is_drum=True,
        name="Drums",
    )

    beat = 60.0 / max(tempo_bpm, 40)
    beats_per_bar = 3 if style == "waltz" else 4

    t = 0.0
    bar_i = 0
    beat_i = 0
    roll_index = 0

    while t < total_duration:
        is_phrase_end = (
            bar_i % 2 == 1
            and beat_i == beats_per_bar - 1
        )

        if is_phrase_end and rolls:
            chosen = rolls[roll_index % len(rolls)]
            build_fill(chosen, drums, t, beat)
            roll_index += 1
        else:
            add_style_beat(
                drums, t, beat, beat_i, style
            )

        t += beat
        beat_i += 1

        if beat_i >= beats_per_bar:
            beat_i = 0
            bar_i += 1

    return drums


def build_bass_track(
    chords: List[List[pretty_midi.Note]],
    tempo_bpm: float,
    style: str = "pop",
) -> pretty_midi.Instrument:
    """
    Basse d'accompagnement dérivée de la vraie voix de basse source.
    """
    bass = pretty_midi.Instrument(
        program=33,
        name="Bass",
    )

    beat = 60.0 / max(tempo_bpm, 40)

    for chord in chords:
        notes = normalize_four_voice_chord(chord)
        if not notes:
            continue

        source_bass = notes[0]
        root = source_bass.pitch - 12
        fifth = root + 7

        start = chord_start(chord)
        end = chord_end(chord)

        if style == "waltz":
            t = start
            i = 0

            while t < end:
                pitch = root if i % 3 == 0 else fifth
                note_end = min(t + beat * 0.9, end)

                add_note(
                    bass, pitch, t, note_end,
                    86, 24, 52,
                )

                t += beat
                i += 1

        elif style == "ballad":
            t = start

            while t < end:
                note_end = min(t + beat * 1.9, end)

                add_note(
                    bass, root, t, note_end,
                    78, 24, 52,
                )

                t += beat * 2

        elif style == "latin":
            t = start
            i = 0

            while t < end:
                pitch = root if i % 2 == 0 else fifth
                note_end = min(t + beat * 0.4, end)

                add_note(
                    bass, pitch, t, note_end,
                    86, 24, 52,
                )

                t += beat / 2
                i += 1

        else:
            t = start

            while t < end:
                note_end = min(t + beat * 0.9, end)

                add_note(
                    bass, root, t, note_end,
                    90, 24, 52,
                )

                t += beat

    return bass


# --------------------------------------------------------------------------
# Ornements — couche indépendante
# --------------------------------------------------------------------------

def build_ornament_track(
    melody_notes: List[pretty_midi.Note],
    skip_indices=frozenset(),
) -> pretty_midi.Instrument:
    ornaments = pretty_midi.Instrument(
        program=68,
        name="Ornaments",
    )

    for i in range(len(melody_notes) - 1):
        if i in skip_indices:
            continue

        n1 = melody_notes[i]
        n2 = melody_notes[i + 1]

        interval = n2.pitch - n1.pitch

        if abs(interval) >= 3:
            direction = 1 if interval > 0 else -1
            pitch = n1.pitch + direction * 2

            gap = max(n2.start - n1.end, 0)
            dur = min(0.15, gap / 2) if gap else 0.10
            start = max(n1.end, n2.start - 0.15)

            ornaments.notes.append(
                pretty_midi.Note(
                    velocity=55,
                    pitch=pitch,
                    start=start,
                    end=start + dur,
                )
            )

    return ornaments


# --------------------------------------------------------------------------
# Réponses instrumentales
# --------------------------------------------------------------------------

def _response_register(name: str) -> Tuple[int, int]:
    return {
        "clarinet": (55, 88),
        "flute": (67, 98),
        "guitar": (52, 84),
        "piano_high": (72, 105),
    }.get(name, (55, 88))


def _unique_pcs(chord: List[pretty_midi.Note]) -> List[int]:
    result = []

    for note in normalize_four_voice_chord(chord):
        pc = note.pitch % 12

        if pc not in result:
            result.append(pc)

    return result


def _nearest_pitch(
    pc: int,
    reference: int,
    low: int,
    high: int,
) -> int:
    candidates = [
        pc + 12 * octave
        for octave in range(-1, 11)
        if low <= pc + 12 * octave <= high
    ]

    if not candidates:
        return clamp_to_range(reference, low, high)

    return min(
        candidates,
        key=lambda p: abs(p - reference),
    )


def _make_response_arpeggio(
    current_chord: List[pretty_midi.Note],
    next_chord: List[pretty_midi.Note],
    last_melody_pitch: int,
    name: str,
    variation: int,
) -> List[int]:
    """
    Réponse de 3 à 6 notes construite avec les notes réelles des deux
    événements harmoniques.

    Priorité :
    1. notes communes ;
    2. notes de l'accord suivant ;
    3. notes de l'accord actuel.

    La dernière note vise volontairement l'accord suivant.
    """
    low, high = _response_register(name)

    current_pcs = _unique_pcs(current_chord)
    next_pcs = _unique_pcs(next_chord)

    if not current_pcs and not next_pcs:
        return []

    common = [
        pc for pc in current_pcs
        if pc in next_pcs
    ]

    arrival = next_pcs or current_pcs
    source = current_pcs or arrival

    count = {
        "clarinet": 4,
        "flute": 4,
        "guitar": 5,
        "piano_high": 4,
    }.get(name, 4)

    if variation % 5 == 0:
        count += 1

    if variation % 11 == 0 and name in (
        "guitar", "piano_high"
    ):
        count += 1

    count = max(3, min(6, count))

    material = []

    # Construire une matière à partir des deux accords.
    for pc in source + common + arrival:
        if pc not in material:
            material.append(pc)

    candidates = [
        _nearest_pitch(
            pc,
            last_melody_pitch,
            low,
            high,
        )
        for pc in material
    ]

    if common:
        target_pc = common[0]
    else:
        target_pc = arrival[0]

    target = _nearest_pitch(
        target_pc,
        last_melody_pitch + (
            -7 if variation % 2 == 0 else 7
        ),
        low,
        high,
    )

    usable = [
        p for p in candidates
        if p != last_melody_pitch
    ]

    if not usable:
        usable = candidates

    family = variation % 3

    if family == 0:
        ordered = sorted(usable)

    elif family == 1:
        ordered = sorted(
            usable,
            reverse=True,
        )

    else:
        ascending = sorted(usable)
        ordered = (
            ascending
            + list(reversed(ascending[:-1]))
        )

    body = []

    for p in ordered:
        if p not in body:
            body.append(p)

        if len(body) >= count - 1:
            break

    # Compléter par octave si nécessaire, sans introduire de nouvelle
    # classe de hauteur étrangère à la matière source.
    if len(body) < count - 1:
        for p in list(body):
            for candidate in (p + 12, p - 12):
                if (
                    low <= candidate <= high
                    and candidate not in body
                ):
                    body.append(candidate)

                    if len(body) >= count - 1:
                        break

            if len(body) >= count - 1:
                break

    body = body[:count - 1]

    if body:
        body[-1] = min(
            body,
            key=lambda p: abs(p - target),
        )

    if family == 0:
        body = sorted(body)

    elif family == 1:
        body = sorted(
            body,
            reverse=True,
        )

    elif len(body) >= 3:
        body = [
            body[0],
            body[-1],
            *body[1:-1],
        ]

    result = body + [target]

    clean = []

    for p in result:
        p = clamp_to_range(
            int(p),
            low,
            high,
        )

        if not clean or p != clean[-1]:
            clean.append(p)

    return clean[:6]


def _response_events(
    start: float,
    end: float,
    count: int,
    beat: float,
    name: str,
):
    available = end - start

    if available <= 0 or count < 1:
        return []

    usable = min(
        available * 0.92,
        beat * 2.25,
    )

    usable = max(
        usable,
        min(available, beat * 0.5),
    )

    step = usable / count

    gate = (
        0.84
        if name in ("clarinet", "flute")
        else 0.68
    )

    return [
        (
            start + i * step,
            max(0.06, step * gate),
        )
        for i in range(count)
    ]


def build_response_tracks(
    melody_notes: List[pretty_midi.Note],
    chords: List[List[pretty_midi.Note]],
    tempo_bpm: float,
    responses: List[str],
    style: str = "pop",
) -> tuple:
    """
    Réponses instrumentales indépendantes des instruments principaux.

    Une réponse intervient à chaque fin de phrase de quatre événements
    harmoniques. Elle est constituée de 3 à 6 notes provenant de la
    matière harmonique réelle, avec une résolution vers l'événement suivant.
    """
    reserved_indices = set()

    if (
        not responses
        or not melody_notes
        or len(chords) < 5
    ):
        return {}, reserved_indices

    beat = 60.0 / max(
        tempo_bpm,
        40,
    )

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

    # Une phrase = quatre événements harmoniques.
    for phrase_end in range(
        3,
        min(
            len(chords) - 1,
            len(melody_notes) - 1,
        ),
        4,
    ):
        reserved_indices.add(phrase_end)

        current_chord = chords[phrase_end]
        next_chord = chords[phrase_end + 1]

        last_melody = melody_notes[phrase_end]
        next_phrase_start = chord_start(next_chord)

        # La réponse se termine juste avant la prochaine attaque.
        end = next_phrase_start - min(
            0.03,
            beat * 0.03,
        )

        window_len = max(
            beat * 0.9,
            min(
                beat * 2.25,
                end - last_melody.start,
            ),
        )

        start = max(
            last_melody.start,
            end - window_len,
        )

        if end - start < beat * 0.5:
            continue

        name = responses[
            response_index % len(responses)
        ]
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

        events = _response_events(
            start,
            end,
            len(arp),
            beat,
            name,
        )

        if len(events) != len(arp):
            continue

        base_velocity = {
            "clarinet": 100,
            "flute": 98,
            "guitar": 104,
            "piano_high": 96,
        }.get(name, 98)

        for i, (pitch, event) in enumerate(
            zip(arp, events)
        ):
            note_start, duration = event
            frac = i / max(
                1,
                len(arp) - 1,
            )

            velocity = int(
                base_velocity + frac * 10
            )

            add_note(
                tracks[name],
                pitch,
                note_start,
                min(
                    note_start + duration,
                    end,
                ),
                velocity,
                *_response_register(name),
            )

    return tracks, reserved_indices


# --------------------------------------------------------------------------
# Assemblage final
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

    # Le MIDI source peut contenir Soprano, Alto, Ténor et Basse sur quatre
    # pistes séparées : toutes les pistes musicales sont donc utilisées.
    chords = collect_satb_chords(pm)
    if not chords:
        raise ValueError("Aucune note musicale trouvée dans le fichier MIDI.")

    tempo = estimate_tempo_from_chords(chords)
    total_duration = max(n.end for chord in chords for n in chord)

    # Tous les instruments sélectionnés reçoivent les 4 voix simultanément.
    all_tracks = {
        name: build_solo_track(name, chords, tempo)
        for name in instruments
    }

    if add_rhythm:
        all_tracks["__bass"] = build_bass_track(chords, tempo, style)
        all_tracks["__drums"] = build_drum_track(
            total_duration, tempo, style, rolls
        )

    # Les enrichissements utilisent la voix supérieure uniquement comme
    # référence mélodique. Ils ne modifient pas les parties SATB principales.
    melody_notes = [
        sorted(chord, key=lambda n: n.pitch)[-1]
        for chord in chords
        if chord
    ]

    reserved_indices = set()
    if responses:
        response_tracks, reserved_indices = build_response_tracks(
            melody_notes, chords, tempo, responses, style
        )
        for name, track in response_tracks.items():
            all_tracks[f"__response_{name}"] = track

    if add_ornaments:
        all_tracks["__ornaments"] = build_ornament_track(
            melody_notes,
            skip_indices=reserved_indices,
        )

    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    if keep_piano:
        for source in pm.instruments:
            if source.is_drum:
                continue
            copy_track = pretty_midi.Instrument(
                program=source.program,
                name=source.name or "Source",
            )
            for n in source.notes:
                copy_track.notes.append(pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start,
                    end=n.end,
                ))
            out.instruments.append(copy_track)

    out.instruments.extend(all_tracks.values())
    return out


def render_to_mp3(
    pm: pretty_midi.PrettyMIDI,
) -> bytes:
    if not os.path.exists(SOUNDFONT_PATH):
        raise RuntimeError(
            f"SoundFont introuvable à {SOUNDFONT_PATH}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        midi_path = os.path.join(
            tmp,
            "arrangement.mid",
        )
        wav_path = os.path.join(
            tmp,
            "arrangement.wav",
        )
        mp3_path = os.path.join(
            tmp,
            "arrangement.mp3",
        )

        pm.write(midi_path)

        if os.path.getsize(midi_path) < 50:
            raise RuntimeError(
                "Fichier MIDI généré anormalement petit/vide avant rendu audio."
            )

        result_wav = subprocess.run(
            [
                "fluidsynth",
                "-ni",
                SOUNDFONT_PATH,
                midi_path,
                "-F",
                wav_path,
                "-r",
                "44100",
            ],
            capture_output=True,
            timeout=90,
        )

        if (
            result_wav.returncode != 0
            or not os.path.exists(wav_path)
            or os.path.getsize(wav_path) < 1000
        ):
            raise RuntimeError(
                "Échec du rendu WAV (fluidsynth). stderr: "
                + result_wav.stderr.decode(
                    errors="ignore"
                )[:500]
            )

        result_mp3 = subprocess.run(
            [
                "lame",
                "-b",
                "192",
                "-q",
                "2",
                wav_path,
                mp3_path,
            ],
            capture_output=True,
            timeout=60,
        )

        if (
            result_mp3.returncode != 0
            or not os.path.exists(mp3_path)
            or os.path.getsize(mp3_path) < 1000
        ):
            raise RuntimeError(
                "Échec de l'encodage MP3 (lame). stderr: "
                + result_mp3.stderr.decode(
                    errors="ignore"
                )[:500]
            )

        with open(mp3_path, "rb") as f:
            data = f.read()

        if not (
            data[:3] == b"ID3"
            or (
                len(data) >= 2
                and data[0] == 0xFF
                and (data[1] & 0xE0) == 0xE0
            )
        ):
            raise RuntimeError(
                "Le fichier encodé ne ressemble pas à un MP3 valide."
            )

        return data


# --------------------------------------------------------------------------
# Utilitaires API
# --------------------------------------------------------------------------

def safe_output_basename(
    original_filename: str,
) -> str:
    base = os.path.splitext(
        original_filename or "orchestration"
    )[0]

    base = "".join(
        c
        for c in base
        if c.isalnum()
        or c in (" ", "-", "_")
    ).strip()

    return (
        (base or "orchestration")
        + "_Orchestrated"
    )


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

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
    keep_piano: bool = True,
    format: str = "mp3",
):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide",
        )

    instruments = parse_instruments(
        instrument
    )
    roll_list = parse_rolls(rolls)
    response_list = parse_responses(
        responses
    )

    style = (
        style.strip().lower()
        if style.strip().lower() in STYLES
        else "pop"
    )

    if instrument.strip().lower() in (
        "band",
        "orchestra",
        "full",
        "all",
    ):
        add_rhythm = True

    original_name = (
        file.filename
        or "orchestration.mid"
    )

    out_basename = safe_output_basename(
        original_name
    )

    raw = await file.read()

    try:
        pm = pretty_midi.PrettyMIDI(
            io.BytesIO(raw)
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier MIDI invalide: {e}",
        )

    if not pm.instruments:
        raise HTTPException(
            status_code=400,
            detail="Le fichier MIDI ne contient aucune piste.",
        )

    try:
        source_chords = group_notes_into_chords(
            pm.instruments[0].notes
        )

        detected_tempo = (
            estimate_tempo_from_chords(
                source_chords
            )
        )

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
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'orchestration: {e}",
        )

    if format.strip().lower() == "midi":
        buf = io.BytesIO()
        result.write(buf)
        data = buf.getvalue()

        return Response(
            content=data,
            media_type="audio/midi",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{out_basename}.mid"',
                "X-Detected-Tempo":
                    f"{detected_tempo:.1f}",
            },
        )

    try:
        mp3_bytes = render_to_mp3(result)

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Le rendu audio a dépassé le temps imparti.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de rendu audio: {e}",
        )

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(mp3_bytes)),
            "Content-Disposition":
                f'inline; filename="{out_basename}.mp3"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "X-Detected-Tempo":
                f"{detected_tempo:.1f}",
        },
    )


@app.get("/instruments")
async def list_instruments():
    return {
        "instruments": {
            k: v["name"]
            for k, v in INSTRUMENTS.items()
        },
        "aliases": list(
            INSTRUMENT_ALIASES.keys()
        ),
        "styles": sorted(STYLES),
        "rolls": sorted(ROLL_TYPES),
        "responses": sorted(
            RESPONSE_INSTRUMENTS.keys()
        ),
    }


def _test_lame_encoding() -> dict:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(
                tmp,
                "silence.wav",
            )
            mp3_path = os.path.join(
                tmp,
                "silence.mp3",
            )

            with wave.open(
                wav_path,
                "w",
            ) as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(
                    struct.pack(
                        "<4410h",
                        *([0] * 4410),
                    )
                )

            result = subprocess.run(
                [
                    "lame",
                    "-b",
                    "192",
                    "-q",
                    "2",
                    wav_path,
                    mp3_path,
                ],
                capture_output=True,
                timeout=15,
            )

            mp3_size = (
                os.path.getsize(mp3_path)
                if os.path.exists(mp3_path)
                else 0
            )

            return {
                "ok":
                    result.returncode == 0
                    and mp3_size > 0,
                "returncode":
                    result.returncode,
                "mp3_bytes":
                    mp3_size,
                "stderr":
                    result.stderr.decode(
                        errors="ignore"
                    )[:300],
            }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


@app.get("/health")
async def health():
    soundfont_ok = os.path.exists(
        SOUNDFONT_PATH
    )

    return {
        "status": "ok",
        "soundfont_found":
            soundfont_ok,
        "soundfont_path":
            SOUNDFONT_PATH,
        "fluidsynth_found":
            shutil.which("fluidsynth") is not None,
        "lame_found":
            shutil.which("lame") is not None,
        "lame_encode_test":
            _test_lame_encoding(),
    }
