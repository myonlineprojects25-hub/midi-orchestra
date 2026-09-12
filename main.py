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

Dans les deux cas, le résultat est enrichi avec 
