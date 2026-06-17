# Projet Lyra : Modèles de Mondes 3D Génératifs Explorables 🚀

Bienvenue dans ce fork personnalisé de **Project Lyra** (Lyra 1.0 & Lyra 2.0), développé à l'origine par le *NVIDIA Spatial Intelligence Lab*. 

Cette version a été enrichie et configurée spécifiquement pour simplifier l'installation, optimiser le rendu 3D temps réel sur les architectures matérielles récentes (telles que NVIDIA Blackwell & Grace) et fournir une interface utilisateur moderne et interactive.

---

## 🎨 Vue d'ensemble et Différences avec le Dépôt Original

Par rapport au dépôt NVIDIA Lyra original, les composants majeurs suivants ont été ajoutés et intégrés pour transformer le projet de recherche en une **application clé en main** :

### 1. 🖥️ Tableau de bord Web interactif (Lyra 2.0 Web Dashboard)
Une application Web moderne développée avec **FastAPI** et du **Vanilla JS/CSS** premium, accessible directement sur le port `7860`. Elle permet de :
*   **Téléverser une image de départ** via une interface drag-and-drop.
*   **Saisir un prompt textuel** pour guider la génération de l'environnement 3D.
*   **Suivre la progression en temps réel** des deux étapes (génération de vidéo + reconstruction 3D Gaussian Splatting) avec affichage en direct des logs.
*   **Visualiser et interagir en direct** avec le rendu vidéo de la caméra et télécharger le fichier de nuage de points 3D (`.ply`).
*   **Surveiller les performances système** en temps réel (pourcentage d'utilisation du CPU, RAM, consommation de VRAM GPU et charge de calcul GPU) via l'API NVML.

### 2. 🔌 Wrapper CUDA d'émulation (`cuda_fake.c` / `libcuda_fake.so`)
Un wrapper CUDA d'émulation sur mesure a été développé pour contourner certaines contraintes d'initialisation des pilotes CUDA hôtes au sein du conteneur Docker :
*   Il intercepte certains appels de l'API CUDA et simule la présence de pilotes spécifiques.
*   Injecté dynamiquement via `LD_PRELOAD`, il garantit que le pipeline PyTorch et gsplat s'exécutent de manière stable sans plantages liés aux pilotes systèmes.

### 3. ⚡ Optimisation Blackwell / Grace et Compilation Automatisée (`compile_install.sh`)
Un script d'installation système et de compilation sur mesure a été conçu :
*   Cible spécifiquement les dernières architectures GPU comme **Blackwell** en forçant la liste d'architectures CUDA (`TORCH_CUDA_ARCH_LIST="12.0"`, `FLASH_ATTN_CUDA_ARCHS="120"`).
*   Gère la compilation complexe et les conflits de dépendance de `flash-attn` (v2.6.3), `transformer_engine`, `MoGe`, `vipe_ext` et le module de reconstruction `depth_anything_3`.

### 4. 🧮 Intégration de `fused-ssim`
Intégration d'un module personnalisé hautement optimisé en CUDA pour le calcul de l'indice de similarité structurelle (SSIM) en 2D et 3D. Ce module permet d'accélérer considérablement le rendu et la reconstruction géométrique lors de la phase de Gaussian Splatting (étape 2).

### 5. 🛠️ Script de Contrôle Unifié (`manage.sh`)
Un script bash d'administration tout-en-un à la racine du projet pour contrôler l'intégralité du cycle de vie de l'application :
*   **Gestion des dossiers hôtes** : Résolution automatique des chemins et création de liens symboliques de compatibilité pour Docker.
*   **Gestion du conteneur** : Lancement automatique du conteneur Docker NGC officiel (`nvcr.io/nvidia/pytorch:25.01-py3`) avec accès GPU complet.
*   **Gestion de l'application** : Démarrage, arrêt, et redémarrage du serveur FastAPI en arrière-plan.
*   **Diagnostics** : Affichage de l'état de santé du matériel, du conteneur, de l'application et de la présence des fichiers de modèles (checkpoints).

---

## 🏗️ Structure du Projet

```text
├── Lyra-1/               # Implémentation officielle de Lyra 1.0 (anonymisée)
├── Lyra-2/               # Code source de Lyra 2.0 (génération & reconstruction)
│   ├── web_app/          # Application FastAPI du Tableau de Bord (Backend + Frontend)
│   ├── checkpoints/      # Dossier contenant les modèles pré-entraînés (VAE, UNet, etc.)
│   └── lyra_2/_src/      # Moteur d'inférence, VIPE, Depth Anything 3
├── fused-ssim/           # Module CUDA optimisé pour le calcul du SSIM 2D/3D
├── manage.sh             # Script de contrôle global du projet (Démarrer/Arrêter/Status/Logs)
├── compile_install.sh    # Script de compilation des extensions CUDA et Python
├── docker_setup.sh       # Script de configuration initiale du conteneur Docker
├── cuda_fake.c           # Code source du wrapper CUDA d'émulation
└── libcuda_fake.so       # Version compilée du wrapper CUDA d'émulation
```

---

## 🚦 Démarrage Rapide

Le script `manage.sh` automatise toutes les étapes. Voici comment l'utiliser :

### 1. Vérifier l'état du système
Vérifiez si vos GPU NVIDIA sont détectés et si les modèles pré-entraînés sont correctement placés dans `Lyra-2/checkpoints/` :
```bash
./manage.sh status
```

### 2. Démarrer le projet
Démarrez le conteneur Docker et lancez le Tableau de Bord Web en arrière-plan :
```bash
./manage.sh start
```
Une fois démarré, ouvrez votre navigateur et accédez à l'adresse suivante :
👉 **[http://localhost:7860](http://localhost:7860)**

### 3. Consulter les logs en direct
Pour suivre l'exécution de l'application ou diagnostiquer un problème :
```bash
./manage.sh logs
```

### 4. Arrêter le projet
Pour arrêter proprement le serveur web et le conteneur Docker :
```bash
./manage.sh stop
```

---

## 📥 Téléchargement des Modèles (Checkpoints)

Si certains modèles sont signalés manquants par `./manage.sh status`, vous devez les télécharger depuis HuggingFace :

```bash
# Installer huggingface_hub si nécessaire
pip install huggingface_hub

# Télécharger les checkpoints de Lyra 2.0 dans le sous-dossier Lyra-2
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir Lyra-2
```

---

## 🛠️ Compilation Manuelle (Si nécessaire)
Si vous souhaitez forcer la recompilation des extensions CUDA à l'intérieur du conteneur Docker :
```bash
./manage.sh setup
```
Ce script exécutera `compile_install.sh` directement dans le conteneur actif, configurera les variables d'environnement appropriées et compilera les extensions nécessaires.

---

## 📝 Licence et Crédits

*   **Code source original** : Propriété de NVIDIA Corporation sous licence Apache 2.0.
*   **Modèles Lyra 2.0** : Publiés sous la licence de recherche scientifique interne de NVIDIA (*NVIDIA Internal Scientific Research and Development Model License*).
*   **Modifications & Dashboard** : Développés par **StepMan91** (Bastien) pour simplifier le déploiement et l'interaction avec le modèle.
