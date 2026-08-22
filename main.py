"""
Micro-service d'orchestration MIDI -> MP3.

Reçoit un fichier MIDI piano (accords jusqu'à 4 notes) et produit un
arrangement enrichi et personnalisable :

- Un ou plusieurs instruments au choix, chacun avec un rôle musical propre
  (mélodie, harmonie, basse/pad, arpège) : trompette, flûte traversière,
  clarinette, trombone, tuba, orgue, chœur, guitare, guitare basse,
  guitare électrique, piano grave, piano aigu.
- Un style rythmique (pop, ballade, latin, valse) qui change le pattern
  de batterie/basse.
- Un ou plusieurs types de roulement de batterie en fin de phrase (caisse
  claire, toms descendants, cymbale, accélération de charleston), combinables
  et alternés au fil du morceau.
- Un canon (écho mélodique décalé) et des notes d'ornement (passages/
  échappées) pour densifier la ligne mélodique.

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
    "trumpet":         {"program": 56, "name": "Trumpet",           "role": "bass_pad"},
    "flute":           {"program": 73, "name": "Flute",             "role": "melody_high"},
    "clarinet":        {"program": 71, "name": "Clarinet",          "role": "harmony"},
    "trombone":        {"program": 57, "name": "Trombone",          "role": "bass_pad"},
    "tuba":            {"program": 58, "name": "Tuba",              "role": "bass_pad"},
    "organ":           {"program": 19, "name": "Organ",             "role": "melody"},
    "choir":           {"program": 52, "name": "Choir",             "role": "pad_chord"},
    "guitar":          {"program": 25, "name": "Guitar",            "role": "arpeggio"},
    "bass_guitar":     {"program": 32, "name": "Bass Guitar",       "role": "bass_pulse"},
    "electric_guitar": {"program": 29, "name": "Electric Guitar",  "role": "arpeggio"},
    "piano_low":       {"program": 0,  "name": "Piano (grave)",     "role": "bass_pulse"},
    "piano_high":      {"program": 0,  "name": "Piano (aigu)",      "role": "melody_sparkle"},
}

INSTRUMENT_ALIASES = {
    "band": ["trumpet", "clarinet", "organ"],
    "orchestra": ["trumpet", "clarinet", "organ", "flute", "trombone", "choir"],
    "full": list(INSTRUMENTS.keys()),
    "all": list(INSTRUMENTS.keys()),
}

STYLES = {"pop", "ballad", "latin", "waltz", "classic", "gospel", "rnb", "blues"}
FILL_TYPES = {"snare_roll", "tom_roll", "cymbal_swell", "hihat_build"}


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


def parse_instruments(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()
    if not raw:
        return ["trumpet"]
    if raw in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[raw]
    result = [tok.strip() for tok in raw.split(",") if tok.strip() in INSTRUMENTS]
    return result or ["trumpet"]


def parse_fill_types(raw: str) -> List[str]:
    raw = (raw or "").strip().lower()
    if not raw:
        return ["snare_roll"]
    result = [tok.strip() for tok in raw.split(",") if tok.strip() in FILL_TYPES]
    return result or ["snare_roll"]


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

def build_fill(fill_type: str, drums: pretty_midi.Instrument, t: float, beat: float):
    if fill_type == "tom_roll":
        toms = [DRUM_TOM_HIGH, DRUM_TOM_MID, DRUM_TOM_LOW, DRUM_TOM_LOW]
        step = beat / 4
        for i, pitch in enumerate(toms):
            st = t + i * step
            drums.notes.append(pretty_midi.Note(velocity=90 + i * 3, pitch=pitch, start=st, end=st + step * 0.85))

    elif fill_type == "cymbal_swell":
        drums.notes.append(pretty_midi.Note(velocity=110, pitch=DRUM_CRASH, start=t, end=t + beat * 0.95))
        drums.notes.append(pretty_midi.Note(velocity=105, pitch=DRUM_KICK, start=t, end=t + 0.1))

    elif fill_type == "hihat_build":
        # charleston qui accélère : 2 puis 4 puis 8èmes
        n_hits = 6
        for i in range(n_hits):
            frac = i / n_hits
            st = t + frac * beat
            vel = 55 + int(frac * 45)
            drums.notes.append(pretty_midi.Note(velocity=vel, pitch=DRUM_HIHAT_OPEN, start=st, end=st + beat / n_hits * 0.8))

    else:  # "snare_roll" par défaut
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


def build_drum_track(total_duration: float, tempo_bpm: float, style: str, fill_types: List[str]) -> pretty_midi.Instrument:
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    beat = 60.0 / max(tempo_bpm, 40)
    beats_per_bar = 3 if style == "waltz" else 4

    t = 0.0
    bar_i = 0
    beat_i = 0
    fill_index = 0
    while t < total_duration:
        is_phrase_end = (bar_i % 4 == 3) and (beat_i == beats_per_bar - 1)
        if is_phrase_end:
            chosen = fill_types[fill_index % len(fill_types)]
            build_fill(chosen, drums, t, beat)
            fill_index += 1
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

        else:  # "pop"
            t = start
            while t < end:
                note_end = min(t + beat * 0.9, end)
                bass.notes.append(pretty_midi.Note(velocity=90, pitch=root, start=t, end=note_end))
                t += beat

    return bass


# --------------------------------------------------------------------------
# Notes d’ornement
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------

def orchestrate(
    pm: pretty_midi.PrettyMIDI,
    instruments: List[str],
    style: str,
    fill_types: List[str],
    add_rhythm: bool,
    add_canon: bool,
    add_ornaments: bool,
    keep_piano: bool = True,
) -> pretty_midi.PrettyMIDI:
    if not pm.instruments:
        raise ValueError("Aucune piste trouvée dans le fichier MIDI.")

    piano = pm.instruments[0]
    chords = group_notes_into_chords(piano.notes)
    if not chords:
        raise ValueError("Aucune note trouvée dans la piste piano.")

    tempo = pm.estimate_tempo() if pm.get_tempo_changes()[0].size else 120.0
    if not (40 < tempo < 240):
        tempo = 120.0

    total_duration = max(n.end for chord in chords for n in chord)

    all_tracks = {name: build_solo_track(name, chords, tempo) for name in instruments}

    if add_rhythm:
        all_tracks["__bass"] = build_bass_track(chords, tempo, style)
        all_tracks["__drums"] = build_drum_track(total_duration, tempo, style, fill_types)

    melody_notes = [sorted(c, key=lambda n: n.pitch)[-1] for c in chords]

    if add_ornaments:
        all_tracks["__ornaments"] = build_ornament_track(melody_notes)

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

        # Garde-fou : un MP3 valide commence par un tag ID3 ("ID3") ou une
        # synchro de frame MPEG (0xFFEx/0xFFFx). Sinon, ce n'est pas un MP3 exploitable.
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
    fill_types: str = "snare_roll",   # ex: "snare_roll,tom_roll,cymbal_swell"
    add_rhythm: bool = False,
    add_ornaments: bool = False,
    keep_piano: bool = True,
    format: str = "mp3",
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    instruments = parse_instruments(instrument)
    fills = parse_fill_types(fill_types)
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
        result = orchestrate(
            pm,
            instruments=instruments,
            style=style,
            fill_types=fills,
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
            headers={"Content-Disposition": f'attachment; filename="{out_basename}.mid"'},
        )

    try:
        mp3_bytes = render_to_mp3(result)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Le rendu audio a dépassé le temps imparti.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de rendu audio: {e}")

    # Response (pas StreamingResponse) : le corps est envoyé d'un bloc avec un
    # Content-Length explicite, plus fiable qu'un flux chunké à travers un proxy.
    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(mp3_bytes)),
            "Content-Disposition": f'inline; filename="{out_basename}.mp3"',
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
        "fill_types": sorted(FILL_TYPES),
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
                w.writeframes(struct.pack("<4410h", *([0] * 4410)))  # 0.1s de silence
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
