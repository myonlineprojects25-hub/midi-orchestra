"""
Micro-service d'orchestration MIDI -> MP3.
Reçoit un fichier MIDI piano (accords jusqu'à 4 notes), ajoute UN instrument
au choix (clarinette, trompette ou orgue) en gardant la ligne mélodique du
piano, puis rend le résultat directement en MP3 prêt à écouter.
"""

import io
import os
import subprocess
import tempfile
from typing import List

import pretty_midi
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="MIDI Orchestrator")

API_KEY = os.environ.get("ORCHESTRATOR_API_KEY", "change-moi")
SOUNDFONT_PATH = os.environ.get("SOUNDFONT_PATH", "/usr/share/sounds/sf2/FluidR3_GM.sf2")

# Programmes General MIDI (0-indexés)
PROGRAM_CLARINET = 71
PROGRAM_TRUMPET = 56
PROGRAM_ORGAN = 19  # Church Organ
PROGRAM_PIANO = 0

CHORD_TOLERANCE = 0.05  # secondes
ALLOWED_INSTRUMENTS = {"clarinet", "trumpet", "organ", "all"}


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


def orchestrate(pm: pretty_midi.PrettyMIDI, instrument: str, keep_piano: bool = True) -> pretty_midi.PrettyMIDI:
    if not pm.instruments:
        raise ValueError("Aucune piste trouvée dans le fichier MIDI.")

    piano = pm.instruments[0]
    chords = group_notes_into_chords(piano.notes)

    tracks = {}
    if instrument in ("trumpet", "all"):
        tracks["trumpet"] = pretty_midi.Instrument(program=PROGRAM_TRUMPET, name="Trumpet")
    if instrument in ("clarinet", "all"):
        tracks["clarinet"] = pretty_midi.Instrument(program=PROGRAM_CLARINET, name="Clarinet")
    if instrument in ("organ", "all"):
        tracks["organ"] = pretty_midi.Instrument(program=PROGRAM_ORGAN, name="Organ")

    for chord in chords:
        pitches = sorted(chord, key=lambda n: n.pitch)
        start = min(n.start for n in chord)
        end = max(n.end for n in chord)

        bass = pitches[0]
        melody = pitches[-1]
        inner = pitches[1:-1]

        if "trumpet" in tracks:
            # Porte la ligne mélodique (note la plus aiguë), articulée
            tracks["trumpet"].notes.append(pretty_midi.Note(
                velocity=95, pitch=melody.pitch, start=start, end=max(start + 0.3, end - 0.05)
            ))

        if "clarinet" in tracks:
            if inner:
                for n in inner:
                    tracks["clarinet"].notes.append(pretty_midi.Note(
                        velocity=75, pitch=n.pitch, start=start, end=end
                    ))
            else:
                # Accord à 2 notes seulement : la clarinette double la mélodie une octave plus bas
                tracks["clarinet"].notes.append(pretty_midi.Note(
                    velocity=70, pitch=max(melody.pitch - 12, 0), start=start, end=end
                ))

        if "organ" in tracks:
            # Tient la basse, une octave plus bas, en fond soutenu
            tracks["organ"].notes.append(pretty_midi.Note(
                velocity=65, pitch=max(bass.pitch - 12, 0), start=start, end=end
            ))

    out = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo() if pm.get_tempo_changes()[0].size else 120)

    if keep_piano:
        piano.program = PROGRAM_PIANO
        piano.name = "Piano"
        out.instruments.append(piano)

    out.instruments.extend(tracks.values())
    return out


def render_to_mp3(pm: pretty_midi.PrettyMIDI) -> bytes:
    if not os.path.exists(SOUNDFONT_PATH):
        raise RuntimeError(f"SoundFont introuvable à {SOUNDFONT_PATH}")

    with tempfile.TemporaryDirectory() as tmp:
        midi_path = os.path.join(tmp, "arrangement.mid")
        wav_path = os.path.join(tmp, "arrangement.wav")
        mp3_path = os.path.join(tmp, "arrangement.mp3")

        pm.write(midi_path)

        subprocess.run(
            ["fluidsynth", "-ni", SOUNDFONT_PATH, midi_path, "-F", wav_path, "-r", "44100"],
            check=True, capture_output=True, timeout=60,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path],
            check=True, capture_output=True, timeout=60,
        )

        with open(mp3_path, "rb") as f:
            return f.read()


@app.post("/orchestrate")
async def orchestrate_endpoint(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
    instrument: str = "trumpet",
    keep_piano: bool = True,
    format: str = "mp3",  # "mp3" ou "midi"
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    if instrument not in ALLOWED_INSTRUMENTS:
        raise HTTPException(status_code=400, detail=f"instrument doit être l'un de: {sorted(ALLOWED_INSTRUMENTS)}")

    raw = await file.read()
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fichier MIDI invalide: {e}")

    try:
        result = orchestrate(pm, instrument=instrument, keep_piano=keep_piano)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'orchestration: {e}")

    if format == "midi":
        buf = io.BytesIO()
        result.write(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/midi",
            headers={"Content-Disposition": "attachment; filename=orchestrated.mid"},
        )

    try:
        mp3_bytes = render_to_mp3(result)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Erreur de rendu audio: {e.stderr.decode(errors='ignore')[:500]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de rendu audio: {e}")

    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=orchestrated.mp3"},
    )


@app.get("/health")
async def health():
    soundfont_ok = os.path.exists(SOUNDFONT_PATH)
    return {"status": "ok", "soundfont_found": soundfont_ok, "soundfont_path": SOUNDFONT_PATH}
