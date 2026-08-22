"""
Micro-service d'orchestration MIDI.
Reçoit un fichier MIDI piano (accords jusqu'à 4 notes) et renvoie
un fichier MIDI multi-pistes avec clarinette, trompette et orgue.
"""

import io
import os
from typing import List

import pretty_midi
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="MIDI Orchestrator")

# Clé API simple pour éviter les appels non autorisés (mets la tienne en variable d'env sur EasyPanel)
API_KEY = os.environ.get("ORCHESTRATOR_API_KEY", "change-moi")

# Programmes General MIDI (0-indexés)
PROGRAM_CLARINET = 71
PROGRAM_TRUMPET = 56
PROGRAM_ORGAN = 19  # Church Organ ; essaie aussi 16 (Drawbar Organ) ou 20 (Reed Organ)
PROGRAM_PIANO = 0

CHORD_TOLERANCE = 0.05  # secondes : notes considérées comme un même accord si elles démarrent à < 50ms d'écart


def group_notes_into_chords(notes: List[pretty_midi.Note]) -> List[List[pretty_midi.Note]]:
    """Regroupe les notes qui démarrent quasi en même temps (= un accord joué à la main)."""
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


def orchestrate(pm: pretty_midi.PrettyMIDI, keep_piano: bool = True) -> pretty_midi.PrettyMIDI:
    if not pm.instruments:
        raise ValueError("Aucune piste trouvée dans le fichier MIDI.")

    piano = pm.instruments[0]
    chords = group_notes_into_chords(piano.notes)

    organ = pretty_midi.Instrument(program=PROGRAM_ORGAN, name="Organ")
    trumpet = pretty_midi.Instrument(program=PROGRAM_TRUMPET, name="Trumpet")
    clarinet = pretty_midi.Instrument(program=PROGRAM_CLARINET, name="Clarinet")

    for chord in chords:
        pitches = sorted(chord, key=lambda n: n.pitch)
        start = min(n.start for n in chord)
        end = max(n.end for n in chord)

        bass = pitches[0]
        melody = pitches[-1]
        inner = pitches[1:-1]  # 0, 1 ou 2 notes selon la taille de l'accord

        # Orgue : tient la basse, une octave plus bas pour un fond chaud et soutenu
        organ.notes.append(pretty_midi.Note(
            velocity=70, pitch=max(bass.pitch - 12, 0), start=start, end=end
        ))

        # Trompette : porte la mélodie (note la plus aiguë), un peu plus fort et articulée
        trumpet.notes.append(pretty_midi.Note(
            velocity=95, pitch=melody.pitch, start=start, end=max(start + 0.3, end - 0.05)
        ))

        # Clarinette : joue les voix intermédiaires (le "liant" harmonique)
        for n in inner:
            clarinet.notes.append(pretty_midi.Note(
                velocity=75, pitch=n.pitch, start=start, end=end
            ))

    out = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo() if pm.get_tempo_changes()[0].size else 120)

    if keep_piano:
        piano.program = PROGRAM_PIANO
        piano.name = "Piano"
        out.instruments.append(piano)

    out.instruments.extend([organ, trumpet, clarinet])
    return out


@app.post("/orchestrate")
async def orchestrate_endpoint(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
    keep_piano: bool = True,
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    raw = await file.read()
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fichier MIDI invalide: {e}")

    try:
        result = orchestrate(pm, keep_piano=keep_piano)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'orchestration: {e}")

    buf = io.BytesIO()
    result.write(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/midi",
        headers={"Content-Disposition": "attachment; filename=orchestrated.mid"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
