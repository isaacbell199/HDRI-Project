# Guide d'Installation Windows - NexaStory Tauri

## 📦 Contenu du projet

Ce projet est une application de création d'histoires avec IA locale utilisant :
- **Tauri v2** - Framework desktop (Rust + Web)
- **llama.cpp** - Inférence de modèles GGUF locaux
- **React/Next.js** - Interface utilisateur
- **SQLite** - Base de données locale

**Windows ONLY** - 100% Desktop, 100% Local

---

## 🚀 Installation Étape par Étape

### Étape 1 : Installer Visual Studio Build Tools

**OBLIGATOIRE pour compiler Rust sur Windows**

1. Téléchargez **Visual Studio Build Tools** :
   - https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Lancez l'installateur

3. Sélectionnez **"Desktop development with C++"**

4. Cliquez sur **Installer**

5. Redémarrez votre ordinateur après l'installation

---

### Étape 2 : Installer Rust

1. Allez sur https://rustup.rs/

2. Téléchargez `rustup-init.exe`

3. Exécutez l'installateur :
   - Choisissez l'option **1** (installation par défaut)
   - Acceptez les termes

4. Ouvrez un **NOUVEAU** terminal PowerShell et vérifiez :
   ```powershell
   rustc --version
   cargo --version
   ```

---

### Étape 3 : Installer Node.js

1. Allez sur https://nodejs.org/

2. Téléchargez la version **LTS** (Long Term Support)

3. Installez avec les options par défaut

4. Vérifiez dans un nouveau terminal :
   ```powershell
   node --version
   npm --version
   ```

---

### Étape 4 : Préparer le projet

1. **Extrayez l'archive** dans un dossier de votre choix
   ```
   Par exemple : C:\Projects\nexastory-desktop
   ```

2. **Ouvrez le dossier dans VS Code** :
   - Fichier > Ouvrir le dossier
   - Sélectionnez `nexastory-desktop`

---

### Étape 5 : Installer les dépendances

Ouvrez le terminal dans VS Code (`Ctrl + ù`) et exécutez :

```powershell
bun install
```

ou

```powershell
npm install
```

---

### Étape 6 : Compiler le backend Rust

**Option A - Compilation CPU (fonctionne partout)** :
```powershell
bun run build:cpu
```

**Option B - Compilation avec CUDA (NVIDIA GPU)** :
```powershell
bun run build:rust:cuda
```
*Note: Nécessite CUDA Toolkit installé*

⏳ **La première compilation peut prendre 10-15 minutes**

---

### Étape 7 : Lancer l'application

```powershell
bun run tauri:dev
```

L'application devrait s'ouvrir dans une nouvelle fenêtre !

---

## 🤖 Étape 8 : Télécharger un modèle GGUF

Pour utiliser l'IA locale, vous avez besoin d'un modèle GGUF :

### Modèles recommandés (gratuits) :

1. **Llama 3.2 3B** (recommandé pour débuter)
   - https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
   - Téléchargez `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~2GB)

2. **Mistral 7B** (plus puissant)
   - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
   - Téléchargez `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4GB)

3. **Phi-3 Mini** (très léger)
   - https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
   - Téléchargez `Phi-3-mini-4k-instruct-q4.gguf` (~2GB)

### Placer le modèle :

Créez le dossier et placez-y le fichier `.gguf` :
```
C:\Users\[VOTRE_NOM]\.nexastory\models\
```

---

## 🛠️ Dépannage

### Erreur : "linker 'link.exe' not found"
→ Installez Visual Studio Build Tools (Étape 1)

### Erreur : "cargo not found"
→ Redémarrez votre terminal après avoir installé Rust

### Erreur : "npm not found"
→ Redémarrez votre terminal après avoir installé Node.js

### L'application ne se lance pas
→ Vérifiez les logs :
```powershell
cd src-tauri
cargo tauri dev
```

### Performance lente
→ Utilisez un modèle plus petit (Q4_K_M au lieu de Q8) ou activez GPU

---

## 📁 Structure des fichiers

```
nexastory-desktop/
├── src-tauri/              # Backend Rust
│   ├── src/
│   │   ├── main.rs         # Point d'entrée
│   │   ├── commands.rs     # API commands
│   │   ├── database.rs     # SQLite
│   │   ├── llm.rs          # llama.cpp
│   │   └── models.rs       # Types
│   └── Cargo.toml          # Dépendances Rust
├── src/                    # Frontend React
│   ├── lib/
│   │   └── tauri-api.ts    # API TypeScript
│   └── components/         # UI
└── package.json
```

---

## ✅ Checklist d'installation

- [ ] Visual Studio Build Tools installé
- [ ] Rust installé (`rustc --version` fonctionne)
- [ ] Node.js installé (`node --version` fonctionne)
- [ ] Dépendances npm installées (`bun install`)
- [ ] Backend Rust compilé (`bun run build:cpu`)
- [ ] Application lancée (`bun run tauri:dev`)
- [ ] Modèle GGUF téléchargé et placé dans le dossier models

---

## 🎮 Commandes utiles

| Commande | Description |
|----------|-------------|
| `bun run tauri:dev` | Lancer en développement |
| `bun run tauri:build` | Créer l'exécutable |
| `bun run tauri:build:cuda` | Créer l'exécutable avec CUDA |
| `bun run lint` | Vérifier le code |

---

## 💡 Conseils

1. **Premier lancement** : Soyez patient, la compilation Rust est longue
2. **Modèle GGUF** : Commencez avec un petit modèle (3B paramètres)
3. **RAM** : Prévoyez au moins 8GB de RAM libre pour les modèles
4. **GPU** : Si vous avez une carte NVIDIA, utilisez CUDA pour 10x plus rapide

Bonne création d'histoires ! 📚✨
