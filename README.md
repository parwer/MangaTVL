# MangaTVL

MangaTVL translates manga and comic pages on any website, right where you read them. It reads 14 languages (English, Japanese, Korean, Chinese (Simplified and Traditional), Thai, Vietnamese, Indonesian, French, German, Spanish, Italian, Portuguese, Russian) and translates into any of them.

It has two parts:

- **Browser extension** (Chrome, Edge, Brave and other Chromium browsers): finds the manga pages on the site you are reading and swaps each one for its translated version.
- **Companion app** (Windows): runs on your own PC and does the image work. It finds the speech balloons and text, reads the text, erases the original lettering, and writes the translation back into the balloons.

The translation is done by an AI of your choice. It sees the page and the text, and it keeps the story context from page to page, so names and the way characters speak stay consistent across a chapter.

## Features

- Translate one page or every page of a chapter, with pages swapped in as they finish
- Context-aware translation: speaker, tone, character names and story carried across pages
- Clean results: only the letters are erased, and the translated text is fitted to each balloon's shape
- Choice of AI:
  - **OpenRouter**: one-click sign in, no API key to copy
  - **Gemini**: with your own API key
  - **Local AI**: Ollama, LM Studio or any OpenAI-compatible server on your PC; free and offline
  - **Chrome built-in translator**: offline, but it translates text only (it cannot see the page)
- Text-only mode for AI models that cannot read images
- Runs on any graphics card (NVIDIA, AMD, Intel) through DirectML, and on PCs without one (slower)

## Install

Download both files from the [latest release](https://github.com/parwer/MangaTVL/releases/latest):

- `MangaTVL-Companion-Setup-<version>.exe`: the Companion app
- `MangaTVL-Extension-<version>.zip`: the browser extension

### 1. Companion app

1. Run `MangaTVL-Companion-Setup-<version>.exe`. It installs for your user only, so it needs no admin rights and no Python.
2. When it finishes, the app starts and shows an icon in the system tray. It loads its models on first start, which takes a few seconds.
3. Optional: turn on the start-with-Windows option in the tray icon's menu.

### 2. Browser extension

1. Unzip `MangaTVL-Extension-<version>.zip` into a folder you will keep.
2. Open `chrome://extensions` (or `edge://extensions` / `brave://extensions`).
3. Turn on **Developer mode**.
4. Click **Load unpacked** and select the unzipped folder.
5. Pin the MangaTVL icon to the toolbar.

### 3. Choose an AI

Click the MangaTVL icon to open the popup. It should say the Companion app is ready. Then do one of these:

- Click **Sign in with OpenRouter** and approve. The extension receives a key by itself.
- Or open the advanced section and enter a Gemini API key.
- Or run a local model (for example `ollama pull qwen3-vl:4b-instruct`) and choose **Local AI**.

## Use

1. Open a chapter on a manga site.
2. In the popup, pick the language of the pages (**from**) and the language you want (**to**), then click the button that turns MangaTVL on for that site. This shows the translate button on its pages.
3. Click the floating button to translate the pages. You can also right-click an image and choose **Translate this image (MangaTVL)**.

## Update

Download the new installer from the [latest release](https://github.com/parwer/MangaTVL/releases/latest) and run it over the old version. For the extension, replace the unzipped folder with the new one and click the reload button on `chrome://extensions`.

## Uninstall

- **Companion app:** Windows Settings → Apps → *MangaTVL Companion* → Uninstall.
- **Extension:** remove it on `chrome://extensions`.
