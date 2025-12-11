# Roadmap de modularisation

Cette feuille de route vise à extraire progressivement le code du fichier monolithique `orchestrateur.py` sans perte de fonctionnalités ni de prompts. Elle suit une priorisation par impact et risque, avec des critères d'acceptation pour sécuriser chaque étape.

## 1) Tranche rapide et à faible risque
- **Créer `config.py`** : regrouper les constantes globales (modèle, limites de taille, timeouts, autonomie) en une structure de configuration unique (dataclass ou dictionnaire). Prévoir surcharge via variables d'environnement ou arguments CLI.
- **Créer `utils/`** : déplacer `log_verbose`, `read_file` et helpers transverses (gestion d'encodage, affichage riche) dans `utils/logging.py` et `utils/io.py`. Ajouter tests unitaires sur la troncature et la verbosité.
- **Critères de validation** : le CLI existant importe les nouveaux modules sans modification de comportement observé sur les logs et la collecte de fichiers.

## 2) Registre d'outils et frontière de sécurité
- **Créer `tools/registry.py`** : exposer une API unique pour déclarer et récupérer les outils (read_file, list_files, write_file, execute_code, create_venv, git_*). Permettre d'activer/désactiver des outils selon le modèle (compat natif vs fallback JSON) via `MODEL_COMPAT`.
- **Déplacer les implémentations** dans `tools/filesystem.py` et `tools/git.py` pour clarifier la surface d'exposition. Encapsuler les règles de sécurité (exclusions, messages d'erreur) dans chaque module.
- **Critères de validation** : le prompt généré liste les mêmes outils avec les mêmes paramètres obligatoires ; les appels existants restent compatibles.

## 3) Construction et parsing de prompts
- **Créer `prompt/builder.py`** : extraire `build_prompt`, structurer les sections (outils, règles, contenu des fichiers, workflow). Prévoir deux modes : `native_tools` et `json_fallback` pour la compatibilité modèles.
- **Créer `prompt/parser.py`** : extraire `parse_json_tool_calls` et les validations associées (champs obligatoires, types). Garder une API idempotente et testable.
- **Critères de validation** : prompts identiques à l'existant pour un même mode ; parsing JSON robuste aux erreurs actuelles et aux futurs ajouts de champs.

## 4) Collecte et sûreté
- **Créer `ingestion/collector.py`** : déplacer `collect_files`, les filtres d'exclusion et la logique de troncature. Rendre les seuils et patterns injectables depuis `config.py`.
- **Créer `safety/detector.py`** : isoler `detect_dangerous_patterns` avec une liste extensible de patterns critiques et des hooks pour d'autres vérifications.
- **Critères de validation** : même nombre de fichiers collectés dans un run de référence ; les avertissements de troncature et de danger restent affichés.

## 5) Orchestration CLI
- **Créer `cli/main.py`** : orchestrer l'argument parsing, le chargement de configuration, la collecte, la construction du prompt, l'appel modèle (Ollama) et la boucle interactive. `orchestrateur.py` devient une façade mince qui appelle `cli/main.py`.
- **Critères de validation** : le binaire CLI conserve les mêmes options ; les interactions utilisateur (rich/markdown) restent inchangées.

## 6) Sécurité et qualité continue
- **Tests unitaires** : ajouter des tests ciblés pour `collector`, `builder`, `parser`, `tools`. Couvrir les cas de troncature, de validation JSON et de registres d'outils.
- **Documentation** : ajouter un README court par module (objectif, API publique, exemples). Documenter le flux global (collecte → build prompt → parse tools → boucle CLI).
- **Migrations progressives** : conserver temporairement des wrappers dans `orchestrateur.py` qui appellent les nouveaux modules. Supprimer la duplication uniquement après passage des tests.

## 7) Priorisation et jalons
1. **Semaine 1** : config + utils (faible risque) → mesure : le script tourne sans modification des réponses.
2. **Semaine 2** : tools/registry + déplacement des outils existants → mesure : prompt identique, tool-calls inchangés.
3. **Semaine 3** : prompt builder/parser + ingestion → mesure : même prompt et même parsing sur un diff de référence.
4. **Semaine 4** : CLI orchestrateur fin → mesure : refactor complet, suppression de l'ancien monolithe après validation des tests.

## 8) Gardes-fous anti-régression
- Comparer les prompts générés avant/après à l'octet près pour un dossier de test.
- Capturer un jeu de conversations de référence (questions/réponses) et vérifier la stabilité après chaque tranche.
- Bloquer toute régression sur les outils exposés : mêmes noms, mêmes paramètres obligatoires, mêmes messages d'erreur.

Cette roadmap permet une migration incrémentale en maintenant la compatibilité des prompts et de la surface d'API visible par le modèle.
