# MIDI Orchestrator

Micro-service qui prend un MIDI piano (accords, max ~4 notes) et répartit
les voix sur orgue, trompette et clarinette pour obtenir un instrumental
plus riche, prêt pour y poser une voix.

## Déploiement sur EasyPanel

1. Pousse ce dossier sur un repo GitHub (public ou privé).
2. Dans EasyPanel : **Create Service > App > From GitHub**.
   - Connecte ton compte GitHub, sélectionne le repo.
   - Build method : EasyPanel détecte le `Dockerfile` automatiquement.
   - Port interne : `8000`.
3. Dans **Environment Variables**, ajoute :
   - `ORCHESTRATOR_API_KEY` = une clé secrète longue et aléatoire.
4. Déploie. EasyPanel te donne :
   - une URL publique (ex: `https://midi-orchestrator.tondomaine.com`)
   - et/ou un nom de service interne sur le réseau Docker si n8n tourne
     sur le même host EasyPanel (ex: `http://midi-orchestrator:8000`) —
     à privilégier si possible, plus rapide et pas exposé publiquement.

## Tester en ligne de commande

```bash
curl -X POST "https://midi-orchestrator.tondomaine.com/orchestrate" \
  -H "x-api-key: TA_CLE_SECRETE" \
  -F "file=@mon_piano.mid" \
  -o orchestrated.mid
```

## Notes d'implémentation

- Le tri des notes d'un accord (grave -> aigu) détermine qui joue quoi :
  la note la plus grave va à l'orgue, la plus aiguë à la trompette,
  les notes intermédiaires à la clarinette.
- `keep_piano=true` (par défaut) conserve la piste piano d'origine en plus
  des nouvelles pistes ; passe `keep_piano=false` en query param si tu
  veux seulement les nouveaux instruments.
- Les numéros de programme General MIDI utilisés :
  - Orgue (Church Organ) = 19
  - Trompette = 56
  - Clarinette = 71
  Tu peux les changer directement dans `main.py` (constantes en haut du
  fichier) si tu préfères un Drawbar Organ (16) ou un Reed Organ (20), etc.
- Le résultat est un `.mid` multi-pistes, pas un audio. Pour l'écouter,
  ouvre-le dans ton DAW (FL Studio, Ableton, Logic...) avec de vrais VST,
  ou ajoute une étape de rendu FluidSynth côté serveur si tu veux un
  aperçu audio direct depuis n8n (voir section "Aller plus loin").

## Aller plus loin (optionnel)

- **Rendu audio serveur** : ajoute FluidSynth + un soundfont (`.sf2`) dans
  le Dockerfile, et un appel `fluidsynth -ni soundfont.sf2 orchestrated.mid
  -F output.wav` après l'orchestration, pour renvoyer directement un `.wav`.
- **File d'attente Redis** : utile seulement si tu veux traiter beaucoup
  de fichiers en parallèle ou éviter un timeout HTTP sur des fichiers
  longs. Pour du piano de 20-30s, l'appel HTTP synchrone est largement
  suffisant.
