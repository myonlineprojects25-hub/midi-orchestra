"""
Micro-service d'orchestration MIDI -> MP3.

Reçoit un fichier MIDI piano (accords jusqu'à 4 notes) et produit un
arrangement enrichi :
- un ou plusieurs instruments solistes (clarinette, trompette, orgue),
  ou un mode "band"/"orchestra" qui les combine tous + rythme,
- une section rythmique optionnelle (basse + batterie calées sur le tempo),
- un canon (écho mélodique décalé sur un autre instrument),
- des notes d'ornement (notes de passage / d'échappée entre les notes
  de la ligne mélodique, pour densifier l'écriture).

Le résultat est rendu directement en MP3 via FluidSynth + ffmpeg.
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
PROGRAM_ORGAN = 19       # Church Organ
PROGRAM_PIANO = 0
PROGRAM_BASS = 33        # Electric Bass (finger)
PROGRAM_FLUTE = 73       # utilisée pour le canon
PROGRAM_OBOE = 68        # utilisée pour les notes d'ornement

# Notes GM du kit de batterie standard (canal 10)
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42

CHORD_TOLERANCE = 0.05  # secondes
SOLO_INSTRUMENTS = {"trumpet": PROGRAM_TRUMPET, "clarinet": PROGRAM_CLARINET, "organ": PROGRAM_ORGAN}


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
    if raw in ("band", "orchestra", "all"):
        return ["trumpet", "clarinet", "organ"]
    result = [tok.strip() for tok in raw.split(",") if tok.strip() in SOLO_INSTRUMENTS]
    return result or ["trumpet"]


def build_solo_tracks(chords, instruments: List[str]) -> dict:
    tracks = {}
    for name in instruments:
        tracks[name] = pretty_midi.Instrument(program=SOLO_INSTRUMENTS[name], name=name.capitalize())

    for chord in chords:
        pitches = sorted(chord, key=lambda n: n.pitch)
        start = min(n.start for n in chord)
        end = max(n.end for n in chord)
        bass = pitches[0]
        melody = pitches[-1]
        inner = pitches[1:-1]

        if "trumpet" in tracks:
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
                tracks["clarinet"].notes.append(pretty_midi.Note(
                    velocity=70, pitch=max(melody.pitch - 12, 0), start=start, end=end
                ))
        if "organ" in tracks:
            tracks["organ"].notes.append(pretty_midi.Note(
                velocity=65, pitch=max(bass.pitch - 12, 0), start=start, end=end
            ))

    return tracks


def build_bass_track(chords, tempo_bpm: float) -> pretty_midi.Instrument:
    """Basse qui pulse le tempo en jouant la fondamentale de chaque accord."""
    bass = pretty_midi.Instrument(program=PROGRAM_BASS, name="Bass")
    beat = 60.0 / max(tempo_bpm, 40)
    for chord in chords:
        pitches = sorted(chord, key=lambda n: n.pitch)
        root = max(pitches[0].pitch - 12, 0)
        start = min(n.start for n in chord)
        end = max(n.end for n in chord)
        t = start
        while t < end:
            note_end = min(t + beat * 0.9, end)
            bass.notes.append(pretty_midi.Note(velocity=90, pitch=root, start=t, end=note_end))
            t += beat
    return bass


def build_drum_track(total_duration: float, tempo_bpm: float) -> pretty_midi.Instrument:
    """Groove basique en 4/4 : grosse caisse 1-3, caisse claire 2-4, charleston en croches."""
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    beat = 60.0 / max(tempo_bpm, 40)
    t = 0.0
    i = 0
    while t < total_duration:
        beat_in_bar = i % 4
        if beat_in_bar in (0, 2):
            drums.notes.append(pretty_midi.Note(velocity=105, pitch=DRUM_KICK, start=t, end=t + 0.1))
        if beat_in_bar in (1, 3):
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=DRUM_SNARE, start=t, end=t + 0.1))
        drums.notes.append(pretty_midi.Note(velocity=70, pitch=DRUM_HIHAT_CLOSED, start=t, end=t + beat * 0.4))
        drums.notes.append(pretty_midi.Note(velocity=55, pitch=DRUM_HIHAT_CLOSED, start=t + beat / 2, end=t + beat * 0.9))
        t += beat
        i += 1
    return drums


def build_canon_track(melody_notes: List[pretty_midi.Note], tempo_bpm: float, delay_beats: float = 2.0) -> pretty_midi.Instrument:
    """Reprend la ligne mélodique en écho, décalée dans le temps, sur une flûte."""
    canon = pretty_midi.Instrument(program=PROGRAM_FLUTE, name="Canon")
    delay = delay_beats * (60.0 / max(tempo_bpm, 40))
    for n in melody_notes:
        canon.notes.append(pretty_midi.Note(
            velocity=60, pitch=n.pitch, start=n.start + delay, end=n.end + delay
        ))
    return canon


def build_ornament_track(melody_notes: List[pretty_midi.Note]) -> pretty_midi.Instrument:
    """
    Ajoute des notes de passage / d'échappée entre les notes mélodiques
    quand l'intervalle est un saut (>= une tierce mineure), pour densifier
    la ligne. Notes courtes, jouées juste avant la note d'arrivée.
    """
    ornaments = pretty_midi.Instrument(program=PROGRAM_OBOE, name="Ornaments")
    for i in range(len(melody_notes) - 1):
        n1 = melody_notes[i]
        n2 = melody_notes[i + 1]
        interval = n2.pitch - n1.pitch
        if abs(interval) >= 3:
            direction = 1 if interval > 0 else -1
            passing_pitch = n1.pitch + direction * 2
            gap = max(n2.start - n1.end, 0)
            dur = min(0.15, gap / 2) if gap > 0 else 0.1
            start = max(n1.end, n2.start - 0.15)
            ornaments.notes.append(pretty_midi.Note(
                velocity=55, pitch=passing_pitch, start=start, end=start + dur
            ))
    return ornaments


def orchestrate(
    pm: pretty_midi.PrettyMIDI,
    instruments: List[str],
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

    all_tracks = build_solo_tracks(chords, instruments)

    if add_rhythm:
        all_tracks["bass"] = build_bass_track(chords, tempo)
        all_tracks["drums"] = build_drum_track(total_duration, tempo)

    melody_notes = [sorted(c, key=lambda n: n.pitch)[-1] for c in chords]

    if add_canon:
        all_tracks["canon"] = build_canon_track(melody_notes, tempo)

    if add_ornaments:
        all_tracks["ornaments"] = build_ornament_track(melody_notes)

    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    if keep_piano:
        piano.program = PROGRAM_PIANO
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

        # Étape 1 : synthèse MIDI -> WAV
        result_wav = subprocess.run(
            ["fluidsynth", "-ni", SOUNDFONT_PATH, midi_path, "-F", wav_path, "-r", "44100"],
            capture_output=True, timeout=60,
        )
        if result_wav.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(
                f"Échec du rendu WAV (fluidsynth). stderr: {result_wav.stderr.decode(errors='ignore')[:500]}"
            )

        # Étape 2 : WAV -> MP3 via lame (encodeur MP3 dédié, plus robuste qu'un ffmpeg
        # dont le support libmp3lame dépend du build de l'image).
        result_mp3 = subprocess.run(
            ["lame", "-b", "192", "-q", "2", wav_path, mp3_path],
            capture_output=True, timeout=60,
        )
        if result_mp3.returncode != 0 or not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1000:
            raise RuntimeError(
                f"Échec de l'encodage MP3 (lame). stderr: {result_mp3.stderr.decode(errors='ignore')[:500]}"
            )

        with open(mp3_path, "rb") as f:
            return f.read()


@app.post("/orchestrate")
async def orchestrate_endpoint(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
    instrument: str = "trumpet",       # ex: "trumpet" | "clarinet,organ" | "band"
    add_rhythm: bool = False,          # basse + batterie
    add_canon: bool = False,           # écho mélodique décalé (flûte)
    add_ornaments: bool = False,       # notes de passage / d'échappée (hautbois)
    keep_piano: bool = True,
    format: str = "mp3",               # "mp3" ou "midi"
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    instruments = parse_instruments(instrument)

    # "band"/"orchestra" implique toujours le rythme, même si la case n'est pas cochée
    if instrument.strip().lower() in ("band", "orchestra", "all"):
        add_rhythm = True

    raw = await file.read()
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fichier MIDI invalide: {e}")

    try:
        result = orchestrate(
            pm,
            instruments=instruments,
            add_rhythm=add_rhythm,
            add_canon=add_canon,
            add_ornaments=add_ornaments,
            keep_piano=keep_piano,
        )
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
    import shutil
    soundfont_ok = os.path.exists(SOUNDFONT_PATH)
    return {
        "status": "ok",
        "soundfont_found": soundfont_ok,
        "soundfont_path": SOUNDFONT_PATH,
        "fluidsynth_found": shutil.which("fluidsynth") is not None,
        "lame_found": shutil.which("lame") is not None,
    }
