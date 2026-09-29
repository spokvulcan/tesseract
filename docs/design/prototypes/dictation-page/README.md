# Dictation page, round 2 prototypes (throwaway, never merge)

Five HTML prototypes of the Dictation page, the overlay and the fix flow, made after the five in-app
variants on this branch were turned down. Each one runs on a fake Mac: Terminal and Notes on the left,
Tesseract's Dictation window on the right, and the overlay floating over both. Dictation mishears the
owner's real terms the way Whisper did in the 2026-09-28 replay (`docs/research/2026-09-28-dictation-errors-and-learning.md`):
Claude comes out as "cloud", Tesseract as "SRACT" or "SRAX", DFlash2 as "D flash two".

## Run it

Open `index.html` in a browser, or serve the folder:

    python3 -m http.server 8765 --directory docs/design/prototypes/dictation-page

Hold <kbd>`</kbd> to dictate (it stands in for ⌥Space, which the running Tesseract keeps) and press
<kbd>Tab</kbd> for ⌃⌥Space. The **Hold to talk** button works with a mouse. <kbd>←</kbd> <kbd>→</kbd>
or the bar at the bottom switch prototypes; each switch starts fresh. The top bar lists what to try
and the line you'll say next. Mistakes are marked in yellow so you can see them; the real app wouldn't.

## The five

| | Idea | You fix a word by | The page is |
|---|---|---|---|
| 1 Roll Call | Teach it your words before it gets them wrong | saying each word once on a card; a missed one: ⌃⌥Space, type it, say it | a prompter: one word from your projects and memory at a time |
| 2 Catch | See it before it lands | tapping ⇧ while talking; the take waits in the lens for ← → pick, type, ↩ | a catch record: per word, mistakes before you taught it and catches since |
| 3 Ask Me | It notices; you answer | pressing ⌃⌥Space when the overlay asks "SRACT → Tesseract?" | a conversation with your dictation: its questions, your answers |
| 4 Just Type It | Type the word you meant | typing `;;claude` then space right where you are | one field that fixes a word everywhere: takes, the agent's memory, from now on |
| 5 Places | Your words belong to where you use them | ⌃⌥Space, the letter over the word, the number of the right one | a board of places: each project and app with its own words |

## What's faked

The recognizer, the audio and the sound-alike matching. Mishearings come from a script, and "sounds
like" uses the script's answer key; the real thing would use a phonetic key. Nothing is saved.
