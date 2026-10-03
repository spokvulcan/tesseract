# Dictation page prototypes (throwaway, never merge)

HTML prototypes of the Dictation page, the overlay and the fix flow. Each one runs on a fake Mac:
Terminal and Notes on the left, Tesseract's Dictation window on the right, and the overlay floating
over both. Dictation mishears the owner's real terms the way Whisper did in the 2026-09-28 replay
(`docs/research/2026-09-28-dictation-errors-and-learning.md`): Claude comes out as "cloud",
Tesseract as "SRACT" or "SRAX", DFlash2 as "D flash two".

Round 2 made five concepts. The owner picked **Catch**, and round 3 refines it; the page opens on it.
The other four stay in the switcher for comparison.

## Run it

Open `index.html` in a browser, or serve the folder:

    python3 -m http.server 8765 --directory docs/design/prototypes/dictation-page

Hold <kbd>`</kbd> to dictate (it stands in for ⌥Space, which the running Tesseract keeps) and press
<kbd>Tab</kbd> for ⌃⌥Space. The **Hold to talk** button works with a mouse. <kbd>←</kbd> <kbd>→</kbd>
or the bar at the bottom switch prototypes; each switch starts fresh. The top bar lists what to try
and the line you'll say next. Mistakes are marked in yellow so you can see them; the real app wouldn't.

## Catch, refined

See it before it lands. The lens at the bottom of the screen streams your words while you talk.

- **Streaming the way a recognizer does.** The newest words are grey while they can still change,
  learned words flip to your spelling as they arrive, and when you let go the final pass lands: a
  word the live preview got wrong ("this" for "the") settles into place. What pastes is the final
  pass, so streaming costs no accuracy.
- **Type the word you meant; it finds the one it misheard.** No cursor walk: in the lens, type
  `claude` and the word that sounds like it (Cloud) is picked, with completion from your words. ↩
  fixes, learns and pastes. ⇥ fixes and stays for another word. ← → pick a word by hand; a click
  works too.
- **Before it pastes, or after.** Tap ⇧ while talking and the take waits in the lens. Missed one
  after it pasted? ⌃⌥Space (Tab here) reopens the last take and the fix lands in the app if the
  text is still there. Esc on a waiting take keeps it unpasted; ⌃⌥Space brings it back.
- **Every fix teaches, and a wrong lesson can be taken back.** A fix learns "heard → meant". If a
  learned word flips in where you meant the plain word (Claude in a Notes line about the weather),
  fix it back and it is left alone in that app from then on. Ordinary words ("the", "this") are
  fixed in that take only, never learned.
- **The page is the catch record.** How many mistakes were caught before they reached an app this
  week, one tile per word you taught (every way it was heard, before and after, where it is left
  alone, Forget with Undo), and today's takes. Click a take to fix a word in it.

The steps in the top bar walk through all of it in order.

## The five from round 2

| | Idea | You fix a word by | The page is |
|---|---|---|---|
| 1 Roll Call | Teach it your words before it gets them wrong | saying each word once on a card; a missed one: ⌃⌥Space, type it, say it | a prompter: one word from your projects and memory at a time |
| 2 Catch | See it before it lands | tapping ⇧ while talking, then typing the word you meant in the lens | a catch record: per word, mistakes before you taught it and catches since |
| 3 Ask Me | It notices; you answer | pressing ⌃⌥Space when the overlay asks "SRACT → Tesseract?" | a conversation with your dictation: its questions, your answers |
| 4 Just Type It | Type the word you meant | typing `;;claude` then space right where you are | one field that fixes a word everywhere: takes, the agent's memory, from now on |
| 5 Places | Your words belong to where you use them | ⌃⌥Space, the letter over the word, the number of the right one | a board of places: each project and app with its own words |

## What's faked

The recognizer, the audio and the timing. Mishearings come from a script, and the live preview's
misses are scripted too. Catch's "sounds like" is a small consonant key in JavaScript; the real
thing would use a proper phonetic key. Nothing is saved.
