"""
Micro-service d'orchestration MIDI -> MP3.

Deux types de fichiers d'entrée sont supportés :

1. Fichiers SATB à 4 pistes nommées (Soprano/Alto/Tnor/Basse), comme les
   cantiques classiques : chaque voix est monophonique et continue. Dans ce
   cas, les instruments choisis DOUBLENT la vraie voix correspondante
   (Soprano pour les rôles mélodiques, Alto pour l'harmonie, Basse pour les
   rôles de basse), avec son rythme et ses hauteurs réels — pas une
   reconstruction synthétique à partir d'accords plaqués.

2. Fichiers piano à accords plaqués (une seule piste, jusqu'à 4 notes par
   accord) : comportement historique, la mélodie/harmonie/basse sont
   déduites de l'empilement des notes de chaque accord.

Dans les deux cas, le résultat est enrichi avec :
- un style rythmique (pop, ballade, latin, valse, classic, gospel, rnb, blues),
  appliqué à la batterie ET à la basse, sur la vraie grille de temps du
  morceau (voir resolve_pulse) — uniquement quand "Rythme" est activé,
- des roulements de batterie en fin de phrase (caisse claire, toms,
  crescendo, double roulement), combinables,
- des notes d'ornement (passages/échappées),
- des réponses instrumentales : de courtes lignes de croches calées sur la
  grille de temps, qui suivent la gamme du morceau (tonalité estimée via
  Krumhansl-Schmuckler), restent dans l'accord sur les temps et rejoignent
  la phrase suivante ; posées uniquement dans un silence ou sur une note
  tenue en fin de phrase (jamais par-dessus une mélodie qui avance).

Le résultat est rendu directement en MP3 via FluidSynth + lame, et le nom
du fichier reprend celui du MIDI importé, suivi du ou des instruments
choisis par l'utilisateur (ex: "H322 Piano medium - Clarinette.mp3").
"""

import bisect
import io
import logging
import math
import os
import shutil
import subprocess
import struct
import tempfile
import traceback
import unicodedata
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import pretty_midi
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

logger = logging.getLogger("midi_orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MIDI Orchestrator")


@app.on_event("startup")
async def warm_up_librosa():
    """
    librosa.yin s'appuie sur numba, qui compile son code au tout premier
    appel (JIT). Sans ce réchauffement, ce coût de compilation serait payé
    par le tout premier utilisateur réel — potentiellement suffisant à lui
    seul pour dépasser le timeout de la passerelle devant l'hébergement.
    On l'exécute donc une fois ici, sur un signal factice minuscule,
    pendant le démarrage du conteneur plutôt que pendant une requête.
    """
    try:
        import time
        import numpy as np
        import librosa

        t0 = time.monotonic()
        dummy = np.random.randn(16000).astype(np.float32)  # 1 seconde de bruit à 16kHz
        librosa.yin(dummy, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=16000, hop_length=512)
        librosa.feature.rms(y=dummy, hop_length=512)
        logger.info("Réchauffement librosa/numba terminé en %.1fs", time.monotonic() - t0)
    except Exception as e:
        logger.warning("Échec du réchauffement librosa (non bloquant): %s", e)

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
DRUM_SIDE_STICK = 37
DRUM_CLAP = 39
DRUM_RIDE = 51
DRUM_TAMBOURINE = 54

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
    "choir":           {"program": 52, "name": "Chorale",          "role": "pad_chord"},
    "guitar":          {"program": 25, "name": "Guitare Acoustique", "role": "arpeggio"},
    "classical_guitar": {"program": 24, "name": "Guitare Classique", "role": "arpeggio"},
    "bass_guitar":     {"program": 32, "name": "Bass Guitar",      "role": "bass_pulse"},
    "electric_guitar": {"program": 29, "name": "Electric Guitar",  "role": "arpeggio"},
    "piano_low":       {"program": 0,  "name": "Piano (grave)",    "role": "bass_pulse"},
    "piano_medium":    {"program": 0,  "name": "Piano (medium)",   "role": "harmony"},
    "piano_high":      {"program": 0,  "name": "Piano (aigu)",     "role": "melody_sparkle"},
    # Synthés ajoutés à la demande de l'utilisateur : aucun des 4 ne
    # correspondait à un instrument déjà présent, les choix de programme
    # GM et de rôle musical sont donc les miens (à ajuster si besoin).
    # "warm"/"choir"/"halo" sont des nappes soutenues du groupe GM
    # "Synth Pad" (89-96) : même rôle que "choir" (pad_chord). "lead" est
    # une voix mélodique du groupe GM "Synth Lead" (81-88), destinée à
    # porter la mélodie au premier plan comme "flute" (melody_high).
    "synth_choir":     {"program": 91, "name": "Synth Choir",      "role": "pad_chord"},
    "synth_halo":      {"program": 94, "name": "Synth Halo",       "role": "pad_chord"},
    "synth_lead":      {"program": 81, "name": "Synth Lead",       "role": "melody_high"},
    "synth_warm":      {"program": 89, "name": "Synth Warm",       "role": "pad_chord"},
    # Cordes/vents/voix orchestrales classiques. Note sur "pedal_steel" :
    # le General MIDI standard (128 programmes) n'a PAS de patch dédié à la
    # pedal steel guitar (le groupe Guitare ne contient que nylon/steel/
    # jazz/clean/muted/overdriven/distortion/harmonics). "Electric Guitar
    # (clean)" est le substitut le plus proche disponible dans une
    # SoundFont GM standard comme FluidR3_GM — le son ne sera pas
    # identique à une vraie pedal steel (pas de glissando/bend), à ajuster
    # si une SoundFont avec un patch plus adapté est utilisée.
    "pedal_steel":     {"program": 26, "name": "Pedal Steel",      "role": "melody"},
    "harp":            {"program": 46, "name": "Harp",             "role": "arpeggio"},
    "violin":          {"program": 40, "name": "Violin",           "role": "melody"},
    "violoncelle":     {"program": 42, "name": "Violoncelle",      "role": "harmony"},
    "opera":           {"program": 53, "name": "Opera",            "role": "pad_chord"},
    "harmonica":       {"program": 22, "name": "Harmonica",        "role": "melody"},
    # Instruments MÉLODIQUES SOLO : ils ne jouent QUE la ligne mélodique
    # (voix de Soprano réelle quand le fichier est SATB, note la plus aiguë
    # de chaque accord sinon) — jamais l'accord complet. Chacun est transposé
    # et contraint dans le registre confortable de son instrument (voir
    # SOLO_SETTINGS). Mêmes programmes GM que leurs homonymes "harmoniques".
    "flute_solo":         {"program": 73, "name": "Flûte traversière solo", "role": "solo_melody"},
    "clarinet_solo":      {"program": 71, "name": "Clarinette solo",        "role": "solo_melody"},
    "clarinet_high_solo": {"program": 71, "name": "Clarinette aiguë solo",  "role": "solo_melody"},
    "saxophone_solo":     {"program": 65, "name": "Saxophone solo",         "role": "solo_melody"},
    # Saxophone à sourdine : le General MIDI n'a PAS de patch de saxophone
    # avec sourdine. On part du sax alto (65) et on reproduit l'effet de
    # sourdine par le jeu : vélocité basse (la SoundFont assombrit alors le
    # timbre via son modulateur vélocité -> filtre), volume de canal relevé
    # (CC7) pour que la note reste audible malgré cette vélocité basse, et
    # réverbération/chorus réduits (CC91/CC93) pour un son sec et resserré.
    "saxophone_muted_solo": {"program": 65, "name": "Saxophone à sourdine solo", "role": "solo_melody"},
    "trombone_solo":      {"program": 57, "name": "Trombone solo",          "role": "solo_melody"},
}

# Réglages des instruments solo : décalage d'octave appliqué à la mélodie
# (en demi-tons), registre jouable (bas, haut) et vélocité de base.
SOLO_SETTINGS = {
    "flute_solo":         {"shift": 12,  "range": (72, 96), "velocity": 96},
    "clarinet_solo":      {"shift": 0,   "range": (55, 88), "velocity": 96},
    "clarinet_high_solo": {"shift": 12,  "range": (72, 96), "velocity": 94},
    "saxophone_solo":     {"shift": 0,   "range": (55, 88), "velocity": 98},
    "saxophone_muted_solo": {"shift": 0, "range": (55, 88), "velocity": 70,
                             "cc": {7: 115, 91: 20, 93: 0}, "fixed_velocity": True},
    "trombone_solo":      {"shift": -12, "range": (40, 72), "velocity": 96},
}

INSTRUMENT_ALIASES = {
    "band": ["trumpet", "clarinet", "organ"],
    "orchestra": ["trumpet", "clarinet", "organ", "flute", "trombone", "choir"],
    "full": list(INSTRUMENTS.keys()),
    "all": list(INSTRUMENTS.keys()),
}

# Associe le rôle d'un instrument à la voix SATB réelle qu'il doit doubler
# quand le fichier d'entrée fournit des voix nommées. Seuls les instruments
# au timbre réellement grave restent dédiés à la voix de Basse seule — tous
# les autres rôles (mélodie/harmonie) jouent désormais l'accord complet à 4
# notes réelles via build_solo_track, pour ne jamais restreindre un
# instrument à une seule voix isolée.
VOICE_FOR_ROLE = {
    "bass_pad": "bass",
    "bass_pulse": "bass",
    "solo_melody": "soprano",
}

STYLES = {"pop", "ballad", "latin", "waltz", "classic", "gospel", "rnb", "blues"}
ROLL_TYPES = {"snare", "toms", "crescendo", "double"}
RESPONSE_INSTRUMENTS = {
    "clarinet": 71,
    "flute": 73,
    "guitar": 25,
    # Mêmes caractéristiques de réponse que "guitar" (registre, nombre de
    # notes, vélocité — voir _response_register, build_response_tracks) ;
    # seul le programme General MIDI change (24 = guitare nylon, contre
    # 25 = guitare acier pour "guitar"), pour le même son de corde que la
    # piste principale "classical_guitar" dans INSTRUMENTS.
    "classical_guitar": 24,
    "piano_high": 0,
}

# --------------------------------------------------------------------------
# Nom de piste MIDI "sûr" pour l'encodage
# --------------------------------------------------------------------------
# mido (utilisé en interne par pretty_midi pour écrire les méta-événements,
# dont le nom de piste) encode ces textes en Latin-1, qui ne couvre que les
# points de code Unicode 0-255. Le "œ" français (U+0153) est hors de cette
# plage et fait planter l'écriture du fichier MIDI (UnicodeEncodeError) —
# et ce, même si le nom d'affichage utilisateur (fichier de sortie,
# interface) doit lui rester inchangé, "œ" compris.
_MIDI_NAME_REPLACEMENTS = {
    "œ": "oe", "Œ": "OE",
    "æ": "ae", "Æ": "AE",
}


def midi_safe_track_name(name: str) -> str:
    """
    Convertit un nom d'instrument (potentiellement hors Latin-1, ex: "Chœur" ou "Cœur")
    en un nom de piste MIDI sûr pour mido. Ne sert QUE pour le nom interne
    de la piste MIDI — le nom d'affichage utilisateur (fichier de sortie,
    interface web) n'est jamais modifié par cette fonction.
    """
    safe = name
    for src, dst in _MIDI_NAME_REPLACEMENTS.items():
        safe = safe.replace(src, dst)
    try:
        safe.encode("latin-1")
    except UnicodeEncodeError:
        # Filet de sécurité pour tout autre caractère hors Latin-1 non prévu
        # ci-dessus : on ne veut jamais planter l'écriture du MIDI pour un
        # simple nom de piste.
        safe = safe.encode("latin-1", errors="replace").decode("latin-1")
    return safe


# --------------------------------------------------------------------------
# DÉTECTION SATB (Soprano / Alto / Tenor / Basse)
# --------------------------------------------------------------------------

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


# Vélocité de la note principale de l'arpège de la guitare acoustique
# (voir build_arpeggio_notes). Réutilisée telle quelle pour la guitare
# basse, à la demande explicite de l'utilisateur, afin que les deux
# guitares restent au même volume l'une par rapport à l'autre.
GUITAR_PRINCIPAL_VELOCITY = 104


def build_arpeggio_notes(track: pretty_midi.Instrument, pitches, start: float, end: float, beat: float):
    """
    Décompose l'accord en pattern rythmique (comping guitare) plutôt qu'un
    plaqué.

    La note principale (`base[-1]`, la plus aiguë de l'accord — c'est-à-dire
    la voix mélodique quand les données proviennent d'un SATB) est toujours
    placée en TÊTE du motif (index 0) et reçoit une vélocité plus marquée.

    Avant ce correctif, le motif plaçait la note principale en 3e position
    (`[basse, intérieure, MÉLODIE, intérieure]`), ce qui la rendait
    silencieuse dès que le segment harmonique était trop court pour boucler
    jusque-là. Sur un fichier SATB, les voix bougent indépendamment
    (notes de passage, suspensions) : `group_notes_into_chords` crée alors
    un nouvel accord à chaque mouvement de voix, ce qui produit de très
    nombreux segments ne durant qu'un demi-temps — assez pour une seule
    itération du motif. Avec l'ancien ordre, cette unique itération jouait
    systématiquement la note la MOINS importante (la basse, garantie en
    position 0) et sacrifiait la mélodie. En mettant la mélodie en position
    0, elle est désormais garantie dès la première itération, quelle que
    soit la durée du segment.
    """
    base = [n.pitch for n in pitches]
    principal = base[-1]
    inner = base[min(1, len(base) - 1)]
    bass = base[0]
    pattern = [principal, inner, bass, inner]
    velocities = [GUITAR_PRINCIPAL_VELOCITY, 88, 88, 88]

    step = beat / 2

    # Durée minimale pour qu'une note du motif reste musicalement utile.
    # En dessous, on la considère comme un simple reliquat de découpage
    # plutôt qu'une vraie note voulue.
    #
    # Pourquoi c'est nécessaire : la boucle ci-dessous avance par pas fixes
    # de `step` et clippe la dernière itération à `end`, quel que soit le
    # temps qu'il reste. Au milieu d'un morceau, cette dernière note
    # tronquée est inaudible car la note "principal" du segment suivant
    # démarre juste après, à l'instant précis où `end` (= `arp_end`) a été
    # prolongé pour rejoindre ce segment suivant. Mais en fin de phrase
    # réelle (silence avant la phrase suivante) ou en toute fin de morceau
    # (aucun segment suivant, voir l'appelant), rien ne masque plus cette
    # note tronquée : elle ressort comme une petite défaillance audible
    # (note quasi-nulle ou coupée trop court). On la saute donc simplement
    # plutôt que de la jouer clippée.
    MIN_ARPEGGIO_NOTE_DURATION = step * 0.25

    t = start
    i = 0
    while t < end:
        note_end = min(t + step * 0.85, end)
        # La toute première note (i == 0, la mélodie/note principale) doit
        # TOUJOURS jouer, même sur un segment très court — c'est la
        # garantie décrite plus haut. Le garde-fou anti-troncature ne
        # s'applique donc qu'aux notes suivantes du motif (i >= 1), là où
        # sauter une note trop courte ne coûte qu'un battement de comping
        # de moindre importance plutôt que la mélodie.
        if i > 0 and note_end - t < MIN_ARPEGGIO_NOTE_DURATION:
            break
        idx = i % len(pattern)
        track.notes.append(pretty_midi.Note(velocity=velocities[idx], pitch=pattern[idx], start=t, end=note_end))
        t += step
        i += 1


def _apply_solo_controllers(track: pretty_midi.Instrument, name: str):
    """Pose en début de piste les contrôleurs MIDI propres à un solo (ex: sourdine)."""
    cfg = SOLO_SETTINGS.get(name, {})
    for number, value in cfg.get("cc", {}).items():
        track.control_changes.append(
            pretty_midi.ControlChange(number=number, value=max(0, min(127, value)), time=0.0)
        )


def build_solo_track(name: str, chords, tempo: float) -> pretty_midi.Instrument:
    """
    Construit la piste d'un instrument en dérivant mélodie/harmonie/basse
    de l'empilement des accords. Utilisé quand aucune voix SATB nommée
    n'est disponible (fichier piano à accords plaqués classique).
    """
    spec = INSTRUMENTS[name]
    role = spec["role"]
    track = pretty_midi.Instrument(program=spec["program"], name=midi_safe_track_name(spec["name"]))
    beat = 60.0 / max(tempo, 40)
    _apply_solo_controllers(track, name)

    for idx, chord in enumerate(chords):
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

        if role == "solo_melody":
            # Ligne mélodique seule : uniquement la note la plus aiguë de
            # l'accord, jouée dans le registre propre à l'instrument.
            cfg = SOLO_SETTINGS[name]
            low, high = cfg["range"]
            p = clamp_to_range(melody.pitch + cfg["shift"], low, high)
            note_end = max(melody.end - 0.03, melody.start + 0.1)
            track.notes.append(pretty_midi.Note(
                velocity=cfg["velocity"], pitch=p, start=melody.start, end=note_end
            ))

        elif role == "melody":
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
            # L'arpège doit rester continu même quand le MIDI source a une
            # pause ou un changement de phrase à cet endroit (silence entre
            # la fin réelle de l'accord et le début du suivant) : on
            # prolonge donc son point d'arrêt jusqu'au début de l'accord
            # suivant plutôt que jusqu'à la fin réelle des notes en cours.
            # Pour le tout dernier accord, il n'y a pas de "suivant" à
            # rejoindre : on garde alors sa propre fin.
            if idx + 1 < len(chords):
                next_start = min(n.start for n in chords[idx + 1])
                arp_end = max(end, next_start)
            else:
                arp_end = end
            build_arpeggio_notes(track, pitches, start, arp_end, beat)

        elif role == "bass_pulse":
            p = clamp_to_range(bass.pitch - 12, 24, 48)
            # La guitare basse doit sonner au même volume que la note
            # principale de la guitare acoustique ; les autres instruments
            # à pulsation grave (ex: piano grave) gardent leur vélocité
            # d'origine, inchangée.
            velocity = GUITAR_PRINCIPAL_VELOCITY if name == "bass_guitar" else 88
            t = start
            while t < end:
                note_end = min(t + beat * 0.9, end)
                track.notes.append(pretty_midi.Note(velocity=velocity, pitch=p, start=t, end=note_end))
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
    track = pretty_midi.Instrument(program=spec["program"], name=midi_safe_track_name(spec["name"]))
    _apply_solo_controllers(track, name)

    for n in sorted(voice_notes, key=lambda x: x.start):
        pitch = n.pitch
        end = n.end

        if role == "solo_melody":
            cfg = SOLO_SETTINGS[name]
            low, high = cfg["range"]
            pitch = clamp_to_range(pitch + cfg["shift"], low, high)
            end = max(end - 0.03, n.start + 0.1)
            if cfg.get("fixed_velocity"):
                # Sourdine : vélocité fixe, pour garder un timbre constant
                # quelle que soit la dynamique de la voix de Soprano source.
                velocity = cfg["velocity"]
            else:
                velocity = max(cfg["velocity"] - 10, min(cfg["velocity"] + 15, n.velocity or cfg["velocity"]))
            track.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=n.start, end=end,
            ))
            continue

        if role in ("melody_high", "harmony_high"):
            pitch = clamp_to_range(pitch + 12, 72, 96)
        elif role == "melody_sparkle":
            pitch = clamp_to_range(pitch + 12, 72, 108)
            end = min(end, n.start + 0.25)
        elif role in ("bass_pad", "bass_pulse"):
            pitch = clamp_to_range(pitch - 12, 24, 48)

        if name == "bass_guitar":
            # Même volume fixe que la note principale de la guitare
            # acoustique, plutôt que la vélocité dynamique héritée de la
            # voix de Basse d'origine — pour que les deux guitares restent
            # équilibrées entre elles quelle que soit la dynamique du
            # chant choral source. Les autres instruments qui doublent une
            # voix (trompette, trombone, tuba, piano grave...) gardent leur
            # comportement dynamique d'origine, inchangé.
            velocity = GUITAR_PRINCIPAL_VELOCITY
        else:
            velocity = max(80, min(115, n.velocity or 88))

        track.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=n.start,
            end=max(end, n.start + 0.05),
        ))

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


# --------------------------------------------------------------------------
# PULSATION : tempo réel, grille de temps et position des temps forts
#
# Batterie, basse et réponses instrumentales s'appuient toutes sur CETTE
# grille (et non plus sur un simple "écart médian entre accords"), pour que
# leurs notes tombent sur les vrais temps du morceau.
# --------------------------------------------------------------------------

@dataclass
class Pulse:
    tempo: float   # battements par minute
    beat: float    # durée d'un temps, en secondes
    phi: float     # instant du temps k = 0, en secondes
    p: int         # les temps k tels que k ≡ p (mod bpb) sont des temps 1
    bpb: int       # temps par mesure

    def time(self, k: int) -> float:
        return self.phi + k * self.beat

    def beat_index(self, k: int) -> int:
        return (k - self.p) % self.bpb

    def bar(self, k: int) -> int:
        return (k - self.p) // self.bpb

    def nearest_beat(self, t: float) -> Tuple[int, float]:
        x = (t - self.phi) / self.beat
        k = int(round(x))
        return k, abs(x - k) * self.beat

    def on_beat(self, t: float, tol: float = 0.06) -> bool:
        return self.nearest_beat(t)[1] <= tol

    def is_downbeat(self, t: float, tol: float = 0.06) -> bool:
        k, dist = self.nearest_beat(t)
        return dist <= tol and self.beat_index(k) == 0


class _NoteIndex:
    """Retrouve rapidement les notes qui sonnent à un instant donné."""

    def __init__(self, notes):
        self.notes = sorted(notes, key=lambda n: n.start)
        self.starts = [n.start for n in self.notes]

    def sounding(self, t: float, lead: float = 0.03):
        i = bisect.bisect_right(self.starts, t + lead)
        return [n for n in self.notes[:i] if n.end > t + 0.02]


def _grid_alignment(onsets: List[float], bpm: float) -> Tuple[float, float]:
    """Meilleur (score, phase) d'alignement des attaques sur une grille au tempo `bpm`."""
    beat = 60.0 / bpm
    tol = max(0.03, min(0.07, beat * 0.1))
    best_score, best_phi = -1.0, 0.0
    for ph in sorted({round(o % beat, 3) for o in onsets[:16]}):
        s = 0.0
        for o in onsets:
            x = (o - ph) / beat
            if abs(x - round(x)) * beat <= tol:
                s += 1.0                                   # sur le temps
            elif abs(x * 2 - round(x * 2)) * beat / 2 <= tol:
                s += 0.8                                   # sur la croche
            elif abs(x * 4 - round(x * 4)) * beat / 4 <= tol:
                s += 0.4                                   # sur la double-croche
        s /= len(onsets)
        if s > best_score + 1e-9:
            best_score, best_phi = s, ph
    return best_score, best_phi


def _pick_downbeat(chords, beat: float, phi: float, bpb: int) -> int:
    """
    Choisit lequel des `bpb` temps est le temps 1. Indice principal : l'harmonie
    change surtout au début des mesures (changement de note de basse sur un
    temps fort). Par défaut, le tout premier accord est un temps 1 (pas
    d'anacrouse), sauf indice net du contraire.
    """
    tol = max(0.03, min(0.07, beat * 0.1))
    weights = [1.0, 0.4, 0.4] if bpb == 3 else [1.0, 0.3, 0.7, 0.3]
    scores = [0.0] * bpb
    prev_bass = None
    first_k = None
    for c in chords:
        onset = min(n.start for n in c)
        bass_pc = min(c, key=lambda n: n.pitch).pitch % 12
        changed = prev_bass is None or bass_pc != prev_bass
        prev_bass = bass_pc
        x = (onset - phi) / beat
        k = int(round(x))
        if abs(x - k) * beat > tol:
            continue
        if first_k is None:
            first_k = k
        w = 1.0 if changed else 0.15
        for p in range(bpb):
            scores[p] += w * weights[(k - p) % bpb]
    if first_k is None:
        return 0
    scores[first_k % bpb] *= 1.25
    return max(range(bpb), key=lambda p: scores[p])


def resolve_pulse(pm: pretty_midi.PrettyMIDI, chords, bpb: int = 4) -> Pulse:
    """
    Détermine le tempo de pulsation, la phase de la grille et le temps 1.
    Le tempo inscrit dans le fichier MIDI est préféré quand les attaques
    s'alignent bien dessus ; sinon on retombe sur l'estimation par accords
    (et ses multiples/sous-multiples).
    """
    onsets = sorted({round(min(n.start for n in c), 4) for c in chords})
    est = estimate_tempo_from_chords(chords)

    if len(onsets) < 3:
        return Pulse(est, 60.0 / est, 0.0, 0, bpb)

    file_tempo = None
    try:
        _, tempi = pm.get_tempo_changes()
        if len(tempi):
            file_tempo = float(tempi[0])
    except Exception:
        file_tempo = None

    bases = [b for b in (file_tempo, est) if b and b > 0]
    candidates: List[float] = []
    for mult in (1.0, 2.0, 0.5):
        for b in bases:
            v = b * mult
            if 50.0 <= v <= 200.0 and not any(abs(v - c) < 0.5 for c in candidates):
                candidates.append(v)
    if not candidates:
        candidates = [est]

    results = [(bpm,) + _grid_alignment(onsets, bpm) for bpm in candidates]
    best = max(r[1] for r in results)
    bpm, _, phi = next(r for r in results if r[1] >= 0.85 * best)

    beat = 60.0 / bpm
    return Pulse(bpm, beat, phi, _pick_downbeat(chords, beat, phi, bpb), bpb)


# --------------------------------------------------------------------------
# Motifs par style. Chaque motif décrit, pour un temps donné de la mesure,
# les frappes (décalage en temps, note, vélocité, durée en temps).
# --------------------------------------------------------------------------

def _drum_events(style: str, idx: int):
    K, S, HC = DRUM_KICK, DRUM_SNARE, DRUM_HIHAT_CLOSED
    ev = []

    if style == "waltz":
        if idx == 0:
            ev.append((0, K, 100, 0.25))
        else:
            ev.append((0, HC, 65, 0.6))

    elif style == "ballad":
        if idx == 0:
            ev.append((0, K, 90, 0.25))
        if idx == 2:
            ev.append((0, S, 85, 0.25))
        ev.append((0, HC, 45, 0.8))

    elif style == "latin":
        if idx == 0:
            ev.append((0, K, 100, 0.25))
        if idx == 1:
            ev.append((0.5, K, 90, 0.25))
        if idx in (1, 3):
            ev.append((0, S, 85, 0.25))
        ev += [(0, HC, 65, 0.4), (0.5, HC, 55, 0.4)]

    elif style == "classic":
        # Sobre : pas de caisse claire, grosse caisse douce sur le temps 1,
        # ride très légère sur chaque temps.
        if idx == 0:
            ev.append((0, K, 70, 0.25))
        ev.append((0, DRUM_RIDE, 38, 0.9))

    elif style == "gospel":
        # Contretemps marqués : caisse claire + battements de mains sur 2 et 4,
        # tambourin en croches.
        if idx in (0, 2):
            ev.append((0, K, 100, 0.25))
        if idx == 2:
            ev.append((0.5, K, 80, 0.25))
        if idx in (1, 3):
            ev += [(0, S, 95, 0.25), (0, DRUM_CLAP, 90, 0.25)]
        ev += [(0, DRUM_TAMBOURINE, 75, 0.3), (0.5, DRUM_TAMBOURINE, 55, 0.3)]

    elif style == "rnb":
        # Grosse caisse syncopée, charleston en doubles-croches.
        if idx == 0:
            ev.append((0, K, 100, 0.25))
        if idx == 1:
            ev += [(0, S, 92, 0.25), (0.5, K, 82, 0.25)]
        if idx == 2:
            ev.append((0.5, K, 90, 0.25))
        if idx == 3:
            ev.append((0, S, 92, 0.25))
        ev += [(0, HC, 70, 0.2), (0.25, HC, 35, 0.2), (0.5, HC, 55, 0.2), (0.75, HC, 35, 0.2)]

    elif style == "blues":
        # Shuffle ternaire : ride sur le temps et sur le dernier tiers du temps.
        if idx in (0, 2):
            ev.append((0, K, 100, 0.25))
        if idx in (1, 3):
            ev.append((0, S, 95, 0.25))
        ev += [(0, DRUM_RIDE, 80, 0.5), (2.0 / 3.0, DRUM_RIDE, 55, 0.3)]

    else:  # "pop"
        if idx in (0, 2):
            ev.append((0, K, 105, 0.25))
        if idx in (1, 3):
            ev.append((0, S, 100, 0.25))
        ev += [(0, HC, 70, 0.4), (0.5, HC, 55, 0.4)]

    return ev


def phrase_end_bars(line, pulse: Pulse) -> set:
    """Numéros des mesures qui CLÔTURENT une phrase (dernière mesure avant la phrase suivante)."""
    bars = set()
    for i in detect_phrase_boundaries(line, pulse):
        x = (line[i].end - 0.05 - pulse.phi) / pulse.beat
        bars.add(pulse.bar(math.floor(x + 1e-6)))
    return bars


def _snare_phrase_end_events(roll_type: str, idx: int, bpb: int):
    """
    Mesure de fin de phrase pour les roulements "caisse claire" et "double" :
    les TEMPS FORTS (1 et 3 en 4/4, 1 en 3/4) reçoivent un coup de caisse claire
    accentué ; tous les autres coups tombent sur les TEMPS FAIBLES, plus
    légers, le dernier temps montant en roulement vers le temps fort suivant.
    """
    S = DRUM_SNARE
    strong = (0,) if bpb == 3 else (0, 2)

    if idx in strong:
        return [(0, S, 112 if idx == 0 else 105, 0.25)]

    if idx == bpb - 1:  # dernier temps (faible) : le roulement monte
        if roll_type == "double":
            return [(j / 8.0, S, min(127, 55 + j * 6), 0.1) for j in range(8)]
        return [(j / 4.0, S, 60 + j * 8, 0.12) for j in range(4)]

    if roll_type == "double":
        return [(j / 4.0, S, 52, 0.1) for j in range(4)]
    return [(0, S, 58, 0.2), (0.5, S, 52, 0.2)]


def build_drum_track(
    total_duration: float,
    pulse: Pulse,
    style: str,
    rolls: List[str],
    fill_bars: Optional[set] = None,
) -> pretty_midi.Instrument:
    """
    `fill_bars` : mesures de fin de phrase réellement détectées dans la mélodie.
    À défaut (mélodie sans respiration), repli sur une mesure sur deux.
    """
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    rolls = rolls or ["snare"]
    explicit = bool(fill_bars)
    roll_index = 0
    chosen_by_bar: Dict[int, str] = {}

    def put(t, events):
        for off, pitch, vel, length in events:
            st = t + off * pulse.beat
            drums.notes.append(pretty_midi.Note(
                velocity=min(127, vel), pitch=pitch, start=st, end=st + max(0.06, length * pulse.beat),
            ))

    k = 0
    while True:
        t = pulse.time(k)
        if t >= total_duration:
            break

        idx = pulse.beat_index(k)
        bar = pulse.bar(k)
        in_fill = (bar in fill_bars) if explicit else (bar >= 0 and bar % 2 == 1)

        if in_fill and bar not in chosen_by_bar:
            chosen_by_bar[bar] = rolls[roll_index % len(rolls)]
            roll_index += 1

        roll = chosen_by_bar.get(bar) if in_fill else None
        base = _drum_events(style, idx)

        if roll in ("snare", "double"):
            keep = [e for e in base if e[1] not in (DRUM_SNARE, DRUM_CLAP)]
            put(t, keep + _snare_phrase_end_events(roll, idx, pulse.bpb))
        elif roll in ("toms", "crescendo") and idx == pulse.bpb - 1:
            build_fill(roll, drums, t, pulse.beat)
        else:
            put(t, base)

        # Atterrissage : coup accentué sur le temps fort qui suit le roulement.
        if idx == 0 and chosen_by_bar.get(bar - 1) in ("snare", "double"):
            put(t, [(0, DRUM_SNARE, 115, 0.25)])

        k += 1

    return drums


def _bass_events(style: str, idx: int):
    """(décalage en temps, degré, durée en temps ou 'bar', vélocité)."""
    if style == "waltz":
        return [(0, "R" if idx == 0 else "5", 0.9, 85)]
    if style == "ballad":
        return [(0, "R", 1.9, 75)] if idx in (0, 2) else []
    if style == "latin":
        return [(0, "R", 0.42, 85), (0.5, "5", 0.42, 85)]
    if style == "classic":
        return [(0, "R", "bar", 72)] if idx == 0 else []
    if style == "gospel":
        return {0: [(0, "R", 1.0, 95)], 1: [(0.5, "R", 0.45, 78)],
                2: [(0, "5", 1.0, 88)], 3: [(0, "T", 0.9, 80)]}.get(idx, [])
    if style == "rnb":
        return {0: [(0, "R", 0.9, 95)], 1: [(0.5, "R", 0.4, 78)],
                2: [(0, "5", 0.9, 88)], 3: [(0.5, "R", 0.4, 80)]}.get(idx, [])
    if style == "blues":
        # Basse "qui marche" : fondamentale, tierce, quinte, sixte.
        return [(0, {0: "R", 1: "T", 2: "5", 3: "6"}.get(idx, "R"), 0.95, 88)]
    return [(0, "R", 0.9, 90)]  # pop


def build_bass_track(chords, pulse: Pulse, style: str = "pop") -> pretty_midi.Instrument:
    bass = pretty_midi.Instrument(program=33, name="Bass")
    all_notes = [n for c in chords for n in c]
    if not all_notes:
        return bass

    index = _NoteIndex(all_notes)
    end_total = max(n.end for n in all_notes)

    k = 0
    while True:
        t0 = pulse.time(k)
        if t0 >= end_total:
            break

        idx = pulse.beat_index(k)
        for off, degree, length, vel in _bass_events(style, idx):
            t = t0 + off * pulse.beat
            sounding = index.sounding(t)
            if not sounding:
                continue

            lowest = min(sounding, key=lambda n: n.pitch).pitch
            root = lowest - 12
            while root < 28:
                root += 12
            while root > 55:
                root -= 12

            pcs = {n.pitch % 12 for n in sounding}
            if degree == "R":
                pitch = root
            elif degree == "5":
                pitch = root + 7
            elif degree == "6":
                pitch = root + 9
            else:  # "T" : tierce mineure si l'accord la contient, majeure sinon
                pitch = root + (3 if (lowest + 3) % 12 in pcs else 4)

            chord_end = max(n.end for n in sounding)
            span = pulse.beat * pulse.bpb * 0.98 if length == "bar" else length * pulse.beat
            end = min(t + span, chord_end)
            if end - t < 0.05:
                continue

            bass.notes.append(pretty_midi.Note(velocity=vel, pitch=min(pitch, 127), start=t, end=end))
        k += 1

    return bass


def build_ornament_track(
    melody_notes: List[pretty_midi.Note],
    skip_indices=frozenset(),
    skip_windows=(),
) -> pretty_midi.Instrument:
    """
    Notes de passage / d'échappée entre les notes mélodiques.
    `skip_indices` évite de placer un ornement là où une réponse
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
            if any(w0 <= start <= w1 for (w0, w1) in skip_windows):
                continue
            ornaments.notes.append(pretty_midi.Note(
                velocity=55, pitch=pitch, start=start, end=start + dur
            ))
    return ornaments


# --------------------------------------------------------------------------
# ESTIMATION DE TONALITÉ (Krumhansl-Schmuckler)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# DÉTECTION DE VRAIES FRONTIÈRES DE PHRASE
# --------------------------------------------------------------------------

def detect_phrase_boundaries(line, pulse: Pulse) -> List[int]:
    """
    Renvoie les indices `i` de la ligne mélodique après lesquels une phrase se
    termine réellement, c'est-à-dire là où la mélodie laisse de la place :

    - un vrai silence (au moins un demi-temps) avant la note suivante ; ou
    - une note nettement tenue (au moins 1,75 temps ET plus longue que la
      normale du morceau) dont la fin tombe sur un début de mesure.

    Deux frontières retenues sont séparées d'au moins 6 temps. Si le morceau
    n'offre aucun de ces espaces (mélodie legato sans respiration), il n'y a
    PAS de frontière : mieux vaut aucune réponse qu'une réponse posée par-dessus
    la mélodie.
    """
    if len(line) < 3:
        return []

    beat = pulse.beat
    iois = sorted(max(line[i + 1].start - line[i].start, 0.01) for i in range(len(line) - 1))
    median = iois[len(iois) // 2]

    boundaries: List[int] = []
    last_start = -1e9
    for i in range(len(line) - 1):
        cur, nxt = line[i], line[i + 1]
        gap = nxt.start - cur.end
        ioi = nxt.start - cur.start

        is_rest = gap >= 0.5 * beat
        is_held = ioi >= max(1.75 * beat, 1.6 * median) and pulse.is_downbeat(nxt.start)

        if (is_rest or is_held) and cur.start - last_start >= 6 * beat:
            boundaries.append(i)
            last_start = cur.start

    return boundaries


# --------------------------------------------------------------------------
# RÉPONSES INSTRUMENTALES — mouvement par degrés de la gamme, placées
# uniquement aux frontières de phrase détectées
# --------------------------------------------------------------------------

def _response_register(name: str):
    return {
        "clarinet": (55, 88),
        "flute": (67, 98),
        "guitar": (52, 84),
        "classical_guitar": (52, 84),
        "piano_high": (72, 105),
    }.get(name, (55, 88))


def _line_indices(i_start: int, i_end: int, count: int, n_scale: int) -> List[int]:
    """
    Indices de gamme d'une ligne de `count` notes qui ARRIVE sur `i_end` en
    avançant par degrés conjoints (ou par petits sauts si la distance est grande).
    """
    d = abs(i_end - i_start)
    direction = 1 if i_end >= i_start else -1
    if d >= count - 1:
        raw = [round(i_start + direction * d * j / (count - 1)) for j in range(count)]
    else:
        raw = [i_end - direction * (count - 1 - j) for j in range(count)]
    return [max(0, min(n_scale - 1, i)) for i in raw]


def _snap_to_chord(pitch: int, pcs, low: int, high: int, avoid=()) -> int:
    """Note de l'accord la plus proche de `pitch`, en évitant les répétitions (`avoid`)."""
    candidates = [p for p in range(low, high + 1) if p % 12 in pcs]
    if not candidates:
        return pitch
    # On évite la répétition seulement si une autre note de l'accord est proche
    # (≤ 5 demi-tons) ; sinon on préfère répéter plutôt que de faire un grand saut.
    preferred = [p for p in candidates if p not in avoid and abs(p - pitch) <= 5]
    pool = preferred or candidates
    return min(pool, key=lambda p: abs(p - pitch))


def build_response_tracks(
    line,
    all_notes,
    pulse: Pulse,
    responses: List[str],
    tonic_pc: int,
    mode: str,
) -> Tuple[dict, list]:
    """
    Réponses instrumentales : courtes lignes de croches, calées sur la grille
    des temps, posées UNIQUEMENT là où la mélodie laisse de la place.

    - Dans un silence : la réponse démarre sur la première croche qui suit la
      fin de la phrase et finit avant l'entrée suivante.
    - Sur une note tenue : elle occupe la fin de la tenue, calée sur l'attaque
      suivante ; ses notes tombant sur un temps appartiennent à l'accord qui
      sonne, les autres sont des notes de passage de la gamme.
    - Dans tous les cas la ligne avance par degrés et ARRIVE à une note voisine
      de la première note de la phrase suivante, pour s'enchaîner avec elle.
    """
    reserved: list = []
    if not responses or len(line) < 3:
        return {}, reserved

    boundaries = detect_phrase_boundaries(line, pulse)
    if not boundaries:
        return {}, reserved

    tracks = {
        name: pretty_midi.Instrument(
            program=RESPONSE_INSTRUMENTS[name],
            name=f"Réponse {name.capitalize()}",
        )
        for name in responses
        if name in RESPONSE_INSTRUMENTS
    }
    if not tracks:
        return {}, reserved

    beat = pulse.beat
    half = beat / 2.0
    index = _NoteIndex(all_notes)
    response_index = 0

    for i in boundaries:
        cur, nxt = line[i], line[i + 1]
        gap = nxt.start - cur.end

        # Croches d'abord ; doubles-croches si l'espace est trop court pour
        # y loger au moins trois croches.
        start, count, unit = 0.0, 0, half
        for unit in (half, beat / 4.0):
            if gap >= 0.5 * beat:
                k = math.ceil((cur.end - pulse.phi) / unit - 1e-6)
                start = pulse.phi + k * unit
                count = min(4, int(((nxt.start - 0.1 * beat) - start) / unit + 1e-6))
            else:
                held = nxt.start - cur.start
                count = min(4, int((held - 0.5 * beat) / unit + 1e-6))
                start = nxt.start - count * unit
            if count >= 3:
                break
        if count < 3:
            continue

        name = responses[response_index % len(responses)]
        response_index += 1
        if name not in tracks:
            continue

        low, high = _response_register(name)
        scale = _scale_pitches(tonic_pc, mode, low, high)
        if len(scale) < 4:
            continue

        i_cur = _nearest_index(scale, cur.pitch)
        i_nxt = _nearest_index(scale, nxt.pitch)
        # On arrive à côté de la note suivante, du côté d'où l'on vient.
        i_end = i_nxt + (1 if i_cur > i_nxt else -1)
        i_end = max(0, min(len(scale) - 1, i_end))
        pitches = [scale[j] for j in _line_indices(i_cur, i_end, count, len(scale))]

        harmony_at_end = {n.pitch % 12 for n in index.sounding(cur.start + 0.05)}
        base_velocity = {"clarinet": 100, "flute": 98, "guitar": 104,
                         "classical_guitar": 104, "piano_high": 96}.get(name, 98)
        gate = 0.84 if name in ("clarinet", "flute") else 0.68
        limit = nxt.start - 0.03

        prev_pitch = None
        for j, pitch in enumerate(pitches):
            t = start + j * unit
            # Les notes sur un temps (sauf la note d'arrivée) restent dans l'accord,
            # sans répéter la note précédente ni doubler la note d'arrivée.
            if j < count - 1 and pulse.on_beat(t, tol=0.05):
                pcs = {n.pitch % 12 for n in index.sounding(t)} or harmony_at_end
                if pcs:
                    avoid = {prev_pitch}
                    if j == count - 2:
                        avoid.add(pitches[-1])
                    pitch = _snap_to_chord(pitch, pcs, low, high, avoid)
            prev_pitch = pitch

            end = min(t + unit * gate, limit)
            if end - t < 0.05:
                continue

            frac = j / max(1, count - 1)
            velocity = int(base_velocity + frac * 10)
            tracks[name].notes.append(pretty_midi.Note(
                velocity=min(127, max(80, velocity)),
                pitch=clamp_to_range(pitch, low, high),
                start=t,
                end=end,
            ))

        reserved.append((start - 0.05, start + count * unit))

    return {k: v for k, v in tracks.items() if v.notes}, reserved


# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------

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


# Panoramique déterministe par instrument, pour que des voix ajoutées aux
# hauteurs proches de la piste originale restent distinctes à l'oreille
# même quand le volume seul ne suffit pas à les séparer.
_PAN_BY_NAME = {
    "trumpet": 100, "flute": 30, "clarinet": 40, "clarinet_high": 20,
    "saxophone": 92, "trombone": 105, "tuba": 112, "organ": 64,
    "choir": 64, "guitar": 25, "classical_guitar": 103, "bass_guitar": 64, "electric_guitar": 35,
    "piano_low": 64, "piano_medium": 64, "piano_high": 50,
    "synth_choir": 78, "synth_halo": 50, "synth_lead": 88, "synth_warm": 64,
    "flute_solo": 30, "clarinet_solo": 40, "clarinet_high_solo": 20,
    "saxophone_solo": 92, "saxophone_muted_solo": 84, "trombone_solo": 105,
    "pedal_steel": 45, "harp": 58, "violin": 35, "violoncelle": 95, "opera": 64, "harmonica": 108,
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
    # Grille de temps réelle du morceau, partagée par la batterie, la basse
    # et les réponses instrumentales.
    pulse = resolve_pulse(pm, chords, 3 if style == "waltz" else 4)

    all_tracks = {}
    for name in instruments:
        role = INSTRUMENTS[name]["role"]
        voice_key = VOICE_FOR_ROLE.get(role)
        if satb and voice_key and voice_key in satb and satb[voice_key]:
            all_tracks[name] = build_voice_double_track(name, satb[voice_key])
        else:
            all_tracks[name] = build_solo_track(name, chords, tempo)

    melody_notes = [sorted(c, key=lambda n: n.pitch)[-1] for c in chords]

    # Ligne mélodique à suivre : la vraie voix de Soprano d'un fichier à voix
    # nommées (et non le "plus haut son de chaque accord", faussé dès que les
    # voix n'ont pas le même rythme), sinon le haut de chaque accord.
    if satb and satb.get("soprano"):
        line = sorted(satb["soprano"], key=lambda n: n.start)
    else:
        line = melody_notes

    if add_rhythm:
        all_tracks["__bass"] = build_bass_track(chords, pulse, style)
        all_tracks["__drums"] = build_drum_track(
            total_duration, pulse, style, rolls, fill_bars=phrase_end_bars(line, pulse)
        )

    reserved_windows: list = []
    if responses:
        tonic_pc, mode = estimate_key(combined_notes)
        response_tracks, reserved_windows = build_response_tracks(
            line, combined_notes, pulse, responses, tonic_pc, mode
        )
        for name, track in response_tracks.items():
            all_tracks[f"__response_{name}"] = track

    if add_ornaments:
        all_tracks["__ornaments"] = build_ornament_track(melody_notes, skip_windows=reserved_windows)

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


# --------------------------------------------------------------------------
# TRANSCRIPTION MP3 -> MIDI (mélodie monophonique)
# --------------------------------------------------------------------------
#
# Conçu pour une entrée à une seule voix (chant fredonné, instrument
# monophonique, ligne d'arpège) sans harmonisation.
#
# Utilise librosa.yin (YIN simple) plutôt que pyin (YIN probabiliste) :
# pyin est plus précis mais nettement plus lent, et l'infrastructure
# d'hébergement (reverse-proxy openresty devant WordPress) coupe les
# requêtes trop longues avec un 504 Gateway Timeout — un temps de calcul
# court est donc une contrainte dure, pas juste une question de confort.
# Le voisement (silence vs note) est déterminé nous-mêmes via l'énergie
# RMS, puisque yin (contrairement à pyin) ne fournit pas de probabilité
# de voisement intégrée.

MIN_NOTE_DURATION = 0.08  # secondes — filtre les détections trop courtes (bruit/artefacts)


def decode_mp3_to_wav(mp3_path: str, wav_path: str):
    result = subprocess.run(
        ["lame", "--decode", mp3_path, wav_path],
        capture_output=True, timeout=90,
    )
    if result.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        raise RuntimeError(
            f"Échec du décodage MP3 (lame --decode). stderr: {result.stderr.decode(errors='ignore')[:500]}"
        )


def _sliding_nanmedian(values, half_window: int):
    """Médiane glissante tolérante aux NaN (fenêtre de 2*half_window+1 trames)."""
    import numpy as np
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        w = values[max(0, i - half_window): i + half_window + 1]
        w = w[~np.isnan(w)]
        if w.size:
            out[i] = np.median(w)
    return out


def transcribe_monophonic_melody(wav_path: str) -> pretty_midi.PrettyMIDI:
    """
    Transcrit la mélodie principale d'un fichier audio en UNE SEULE suite de
    notes (monophonique, jamais deux notes simultanées), jouée au piano.

    Priorités : fidélité des hauteurs d'abord, puis du rythme.
      1. Détection de f0 par YIN (rapide, compatible timeout passerelle).
      2. Correction d'accordage global : si l'enregistrement est désaccordé
         (ex. +30 cents), on recale avant d'arrondir, sinon chaque note
         risque de tomber sur le mauvais demi-ton.
      3. Correction des erreurs d'octave de YIN (sauts de +/-12 demi-tons
         isolés) par comparaison à la médiane locale.
      4. Lissage médian, puis segmentation avec hystérésis : une note ne
         change que si la hauteur s'écarte nettement (vibrato, glissandi
         légers et gigue ne créent plus de fausses notes).
      5. Nettoyage : notes trop courtes supprimées, trous minuscules entre
         deux mêmes hauteurs recollés, chevauchements interdits.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("Fichier audio vide ou illisible.")

    # Normalisation du niveau pour que les seuils d'énergie soient stables.
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    hop_length = 256          # ~16 ms : meilleure résolution temporelle
    frame_length = 2048       # ~128 ms : reste fiable jusqu'à C2

    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    frame_dt = hop_length / sr

    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)[0]
    rms = rms[: len(times)] if len(rms) >= len(times) else np.pad(rms, (0, len(times) - len(rms)))

    # --- Voisement : seuil relatif au niveau typique (95e centile) plutôt
    # qu'au pic absolu, pour qu'un seul transitoire fort ne rende pas muettes
    # les notes douces.
    ref_rms = float(np.percentile(rms, 95)) if rms.size else 0.0
    if ref_rms <= 0:
        raise ValueError("Aucune mélodie monophonique détectée dans ce fichier.")
    voiced = rms > ref_rms * 0.10

    midi_f = np.full(len(f0), np.nan)
    ok = voiced & np.isfinite(f0) & (f0 > 0)
    midi_f[ok] = librosa.hz_to_midi(f0[ok])

    if not np.any(ok):
        raise ValueError(
            "Aucune mélodie monophonique détectée dans ce fichier — vérifiez qu'il "
            "s'agit bien d'une voix/mélodie seule, sans harmonisation ni accompagnement dense."
        )

    # --- Correction d'accordage global (moyenne circulaire des écarts au
    # demi-ton le plus proche, pondérée par l'énergie).
    valid = ~np.isnan(midi_f)
    devs = midi_f[valid] - np.round(midi_f[valid])
    w = rms[valid]
    ang = 2 * np.pi * devs
    offset = np.arctan2(np.sum(w * np.sin(ang)), np.sum(w * np.cos(ang))) / (2 * np.pi)
    midi_f -= float(np.clip(offset, -0.5, 0.5))

    # --- Correction des erreurs d'octave (sauts YIN isolés).
    local_med = _sliding_nanmedian(midi_f, 6)
    for i in np.where(~np.isnan(midi_f))[0]:
        if not np.isnan(local_med[i]):
            diff = local_med[i] - midi_f[i]
            if abs(diff) > 9:
                midi_f[i] += 12 * round(diff / 12)

    # --- Lissage médian (supprime la gigue image par image).
    midi_f = _sliding_nanmedian(np.where(np.isnan(midi_f), np.nan, midi_f), 2)
    midi_f[~voiced] = np.nan

    # --- Segmentation avec hystérésis.
    HYST = 0.65  # demi-tons d'écart requis pour changer de note
    notes_raw = []  # (start_idx, end_idx_exclusive, pitch, rms_mean)
    cur_pitch = None
    cur_start = 0
    cur_rms = []

    def close(end_idx):
        nonlocal cur_pitch, cur_rms
        if cur_pitch is not None:
            notes_raw.append((cur_start, end_idx, cur_pitch, float(np.mean(cur_rms))))
        cur_pitch = None
        cur_rms = []

    for i, p in enumerate(midi_f):
        if np.isnan(p):
            close(i)
            continue
        if cur_pitch is None:
            cur_pitch, cur_start, cur_rms = int(round(p)), i, [rms[i]]
        elif abs(p - cur_pitch) < HYST:
            cur_rms.append(rms[i])
        else:
            close(i)
            cur_pitch, cur_start, cur_rms = int(round(p)), i, [rms[i]]
    close(len(midi_f))

    # --- Nettoyage : durée minimale, recollage des micro-trous, monophonie.
    min_frames = max(1, int(round(MIN_NOTE_DURATION / frame_dt)))
    notes_raw = [n for n in notes_raw if (n[1] - n[0]) >= min_frames]

    merged = []
    for s, e, p, r in notes_raw:
        if merged:
            ps, pe, pp, pr = merged[-1]
            if pp == p and (s - pe) * frame_dt <= 0.04:
                merged[-1] = (ps, e, pp, (pr + r) / 2)
                continue
        merged.append((s, e, p, r))

    if not merged:
        raise ValueError(
            "Aucune mélodie monophonique détectée dans ce fichier — vérifiez qu'il "
            "s'agit bien d'une voix/mélodie seule, sans harmonisation ni accompagnement dense."
        )

    notes = []
    for idx, (s, e, p, r) in enumerate(merged):
        start = float(times[s])
        end = float(times[e - 1] + frame_dt)
        if idx + 1 < len(merged):
            end = min(end, float(times[merged[idx + 1][0]]))  # jamais de chevauchement
        if end - start < 0.03:
            continue
        velocity = int(np.clip(45 + (r / ref_rms) * 55, 50, 120))
        notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=int(np.clip(p, 21, 108)),
            start=start,
            end=end,
        ))

    if not notes:
        raise ValueError(
            "Aucune mélodie monophonique détectée dans ce fichier — vérifiez qu'il "
            "s'agit bien d'une voix/mélodie seule, sans harmonisation ni accompagnement dense."
        )

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=0, name="Melody")  # piano acoustique
    melody.notes = notes
    pm.instruments.append(melody)
    return pm


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


# Caractères strictement interdits dans un nom de fichier (séparateurs de
# chemin, guillemets...). On ne restreint plus aux caractères ASCII/isalnum :
# tous les caractères Unicode (accents, œ, etc.) sont conservés tels quels.
_FORBIDDEN_FILENAME_CHARS = set('/\\:*?"<>|')

# Caractères de contrôle (CR, LF, tabulation, etc. — points de code U+0000
# à U+001F, plus U+007F). Sans lien avec "Chœur" lui-même, mais ce filtre
# devient nécessaire maintenant que l'en-tête Content-Disposition construit
# ici est relayé TEL QUEL par n8n puis par le proxy PHP, sans plus jamais
# être reconstruit en aval : un nom de fichier MIDI importé contenant un
# retour à la ligne se propagerait alors littéralement dans l'en-tête HTTP
# final (au mieux un rejet par le serveur ASGI, au pire un début
# d'injection d'en-tête). On l'assainit donc ici, à la source, une bonne
# fois pour toutes.
_CONTROL_CHARS = {chr(c) for c in range(0x00, 0x20)} | {chr(0x7F)}


def build_output_basename(original_filename: str, instrument_names: List[str]) -> str:
    """
    Construit le nom du fichier de sortie : nom du MIDI importé, suivi du
    ou des instruments choisis par l'utilisateur dans le pré-filtrage,
    séparés par " - " quand il y en a plusieurs.

    Exemple : fichier "H322.mid" + instruments ["piano_medium", "clarinet"]
    -> "H322 Piano (medium) - Clarinet".

    Le nom garde ses caractères Unicode d'origine (accents, œ, etc.) ; il
    est stocké et manipulé en str Python, donc nativement en UTF-8. Seul
    l'encodage de l'en-tête HTTP Content-Disposition doit être géré à part
    (voir `content_disposition_header`), car cet en-tête n'accepte pas
    l'UTF-8 brut. Les caractères de contrôle (CR/LF compris) sont en
    revanche toujours retirés ici, car eux n'ont pas leur place dans un nom
    de fichier ni dans un en-tête HTTP, quel que soit l'encodage.
    """
    base = os.path.splitext(original_filename or "orchestration")[0].strip()
    if not base:
        base = "orchestration"

    labels = [INSTRUMENTS[name]["name"] for name in instrument_names if name in INSTRUMENTS]
    suffix = " - ".join(labels)

    full = f"{base} {suffix}" if suffix else base
    full = "".join(
        c for c in full
        if c not in _FORBIDDEN_FILENAME_CHARS and c not in _CONTROL_CHARS
    ).strip()

    return full or "orchestration"


def content_disposition_header(disposition: str, filename: str) -> str:
    """
    Construit un en-tête Content-Disposition qui préserve correctement les
    caractères Unicode du nom de fichier (accents, œ, etc.).

    Le protocole HTTP impose que les valeurs d'en-tête soient encodables en
    Latin-1. On ne peut donc jamais y placer le texte Unicode complet tel
    quel (ex: "œ" provoquerait une UnicodeEncodeError à l'encodage Latin-1).

    On fournit donc DEUX représentations, pour couvrir aussi bien les
    clients stricts que les clients simplistes qui ignorent `filename*` :

    - `filename="..."` : les octets UTF-8 du nom, réinterprétés un par un
      comme des caractères Latin-1 (technique standard, utilisée par
      Flask/Werkzeug et Django). Chaque octet UTF-8 (0-255) a un point de
      code Latin-1 valide, donc cette conversion ne lève jamais d'erreur
      d'encodage, et les octets qui transitent réellement sur le réseau
      sont bien les octets UTF-8 d'origine — la plupart des navigateurs
      et des clients HTTP décodent alors correctement le nom en UTF-8,
      même sans connaître la RFC 5987.
    - `filename*=UTF-8''...` (RFC 5987) : la version pourcent-encodée, pour
      les clients strictement conformes qui privilégient ce paramètre.

    `filename` est supposé déjà assaini de tout caractère de contrôle par
    `build_output_basename` — cette fonction ne le revérifie pas.
    """
    utf8_bytes = filename.encode("utf-8")
    mojibake_filename = utf8_bytes.decode("latin-1")  # 1 octet -> 1 point de code (0-255), toujours valide
    quoted_filename = mojibake_filename.replace('"', "'")
    encoded_utf8 = quote(filename, safe="")
    return f'{disposition}; filename="{quoted_filename}"; filename*=UTF-8\'\'{encoded_utf8}'


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
    # Le MIDI importé ne doit jamais être audible en sortie par défaut —
    # seuls les instruments choisis doivent s'entendre. keep_piano reste
    # disponible pour un usage API explicite, mais n'est pas exposé dans le
    # formulaire et ne doit pas être activé par défaut.
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
    out_basename = build_output_basename(original_name, instruments)

    raw = await file.read()
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    except Exception as e:
        logger.error("Échec de lecture du MIDI '%s': %s\n%s", original_name, e, traceback.format_exc())
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
        logger.error(
            "Échec d'orchestration '%s' (instruments=%s, style=%s): %s\n%s",
            original_name, instruments, style, e, traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=f"Erreur d'orchestration: {e}")

    if format == "midi":
        buf = io.BytesIO()
        result.write(buf)
        data = buf.getvalue()
        return Response(
            content=data,
            media_type="audio/midi",
            headers={
                "Content-Disposition": content_disposition_header("attachment", f"{out_basename}.mid"),
                "X-Detected-Tempo": f"{detected_tempo:.1f}",
            },
        )

    try:
        mp3_bytes = render_to_mp3(result)
    except subprocess.TimeoutExpired:
        logger.error("Timeout du rendu audio pour '%s'", original_name)
        raise HTTPException(status_code=504, detail="Le rendu audio a dépassé le temps imparti.")
    except Exception as e:
        logger.error("Échec de rendu audio pour '%s': %s\n%s", original_name, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur de rendu audio: {e}")

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(mp3_bytes)),
            "Content-Disposition": content_disposition_header("inline", f"{out_basename}.mp3"),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "X-Detected-Tempo": f"{detected_tempo:.1f}",
        },
    )


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
    format: str = "mp3",  # "mp3" pour écouter un aperçu, "midi" pour télécharger le fichier
):
    """
    Convertit un MP3 contenant une mélodie monophonique (voix fredonnée,
    instrument seul, ligne d'arpège — sans harmonisation ni paroles denses)
    en un fichier MIDI à une seule piste. Ce MIDI peut ensuite être réimporté
    tel quel dans /orchestrate, exactement comme un fichier MIDI piano
    classique (une seule piste, pas de voix SATB nommées).
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    original_name = file.filename or "melodie.mp3"
    out_basename = build_output_basename(original_name, [])
    out_basename = f"{out_basename} Transcription"

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, "input.mp3")
        wav_path = os.path.join(tmp, "input.wav")
        with open(mp3_path, "wb") as f:
            f.write(raw)

        try:
            decode_mp3_to_wav(mp3_path, wav_path)
        except Exception as e:
            logger.error("Échec du décodage MP3 '%s': %s\n%s", original_name, e, traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Fichier MP3 invalide ou illisible: {e}")

        try:
            pm = transcribe_monophonic_melody(wav_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error("Échec de la transcription '%s': %s\n%s", original_name, e, traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Erreur de transcription: {e}")

    if format == "midi":
        buf = io.BytesIO()
        pm.write(buf)
        data = buf.getvalue()
        return Response(
            content=data,
            media_type="audio/midi",
            headers={
                "Content-Disposition": content_disposition_header("attachment", f"{out_basename}.mid"),
            },
        )

    try:
        mp3_bytes = render_to_mp3(pm)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Le rendu audio a dépassé le temps imparti.")
    except Exception as e:
        logger.error("Échec du rendu de l'aperçu '%s': %s\n%s", original_name, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur de rendu audio: {e}")

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(mp3_bytes)),
            "Content-Disposition": content_disposition_header("inline", f"{out_basename}_preview.mp3"),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
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
    try:
        import librosa  # noqa: F401
        librosa_ok = True
    except Exception:
        librosa_ok = False
    return {
        "status": "ok",
        "soundfont_found": soundfont_ok,
        "soundfont_path": SOUNDFONT_PATH,
        "fluidsynth_found": shutil.which("fluidsynth") is not None,
        "lame_found": shutil.which("lame") is not None,
        "lame_encode_test": _test_lame_encoding(),
        "librosa_found": librosa_ok,
    }
