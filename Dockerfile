FROM python:3.11-slim

# fluidsynth       : synthétise le MIDI en audio
# fluid-soundfont-gm : banque de sons General MIDI
# lame             : encodeur MP3 dédié, sert aussi à décoder (MP3 -> WAV)
#                    pour la transcription MP3->MIDI
# libsndfile1      : dépendance native de soundfile/librosa (lecture audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fluidsynth \
    fluid-soundfont-gm \
    lame \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
