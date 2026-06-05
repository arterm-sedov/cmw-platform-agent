# Gradio 6.x: chatbot image and share button behavior

## What we see in the UI today

For a single assistant turn that produces an image, our app appends three
assistant bubbles in this order:

1. **Image component** — renders the picture plus its intrinsic label
   `<filename> — <size>`.
2. **Caption bubble** — short text description of the image.
3. **Stats bubble** — the styled "📊 Статистика" card with token counts, API
   tokens, provider/model, cost, and execution time.

In Gradio 6.x, the chatbot groups consecutive same-role messages by default
(`group_consecutive_messages=True`) and renders one **ButtonPanel** (copy /
like / download / share) at the end of each group. The panel picks the first
media file in the group via `get_file()` in
`js/chatbot/shared/Message.svelte:93`, so the **download** is logically
attached to the image — but visually the panel renders after the last bubble
in the group, which is the stats bubble. That is why the download button lands
below the stats card rather than next to the picture.

The same panel also shows a **share** button for any assistant group that
contains a file. The share button is non-functional outside Hugging Face
Spaces: it calls `uploadToHuggingFace` in `ButtonPanel.svelte:88`, which throws
`ShareError("Must be on Spaces to share.")` when
`window.__gradio_space__ == null` (`js/utils/src/utils.svelte.ts:92-94`). On
click outside Spaces, the user gets a `console.error` and a dispatched
`"error"` event with that message. So our app currently shows a button that
doesn't work and prints a console error. This is the real UX wart, bigger
than the download displacement.

## Gradio 5.x vs 6.x recap

- In Gradio 5.x, the image preview component itself carried the download
  button when `allow_file_downloads` was on, and the share button was
  per-component too.
- 6.x moved the affordances out to the group-level ButtonPanel and
  hardcoded the per-message `show_share_button` prop to `true` in
  `Message.svelte:151` — it is no longer driven by the user's `buttons=`
  list, which now only controls the top-level toolbar.

## What we already control

- The **top-level** toolbar share button is already off in our app:
  `agent_ng/tabs/chat_tab.py` passes `buttons=["copy", "copy_all"]`, and
  `Index.svelte:74` derives `show_share_button` from that list. Verified:
  no top-level share renders.
- The **per-message** share button cannot be turned off from the app side
  today. It is hardcoded in Gradio 6.x.

## Acceptance decision (current)

Accepted as-is. The download button still works, the filename + size label
under the image is the primary download cue, and users can right-click the
image as a fallback. The per-message share button being non-functional is a
known Gradio 6.x wart, not something we can fix from the app without
patching Gradio. None of this affects our backend logic; it is a pure
visual-model question.

## If it needs to be fixed, paths in priority order

1. **Tiny Gradio patch: thread `show_share_button` from `ChatBot` to
   `Message`**, defaulting from the same `gradio.props.buttons` source as the
   top-level one. This is the smallest possible Gradio change and fixes the
   real UX wart (a non-functional share button), without re-introducing the
   5.x-style per-image download button. It does not fork any behavior, it
   just exposes an existing prop. **Recommended.** Cost: carry a small
   local patch on top of our pinned Gradio, and rebase occasionally.
2. **`group_consecutive_messages=False` on our `gr.Chatbot`** — one line in
   `agent_ng/tabs/chat_tab.py`. Each bubble becomes its own group; the image
   group gets its download button right under the image; caption and stats
   stand alone. **App-only, no Gradio patch needed.** Side effects:
   - Visual: one turn becomes three bubbles instead of one merged block,
     taking more vertical space and showing empty ButtonPanel rows under the
     caption and stats bubbles.
   - Any future turn that streams multiple short assistant messages
     (thinking → answer) will also stop merging, which can feel choppier.
   - The non-functional share button still appears on the image group, so
     this path alone doesn't fix the share issue.
3. **Reorder the appended assistant bubbles so the image is the last in the
   group**: emit stats → caption → image. The ButtonPanel lands right under
   the image. **App-only.** Side effect: metadata reads before the
   artifact, which is backwards from what users expect.
4. **Fold caption and conversation-stats into the image component's
   `caption` field**, so the image is the only assistant bubble of the
   turn. **App-only.** "Stats" here means the conversation-stats bubble
   (`token_metadata_message` in `agent_ng/app_ng_modular.py`) — token
   counts, API tokens, provider/model, cost, execution time. These are the
   per-turn metrics, not image metadata (filename and size, which the image
   component already shows). Side effect: the visual stats card with its
   rounded box and 📊 icon becomes plain text inside the image caption, so
   we lose the styled presentation.

If we revisit this, path 1 alone fixes the share wart with a one-line
Gradio change. Pairing path 1 with path 2 fixes the download displacement
without touching how the chat groups assistant messages globally. Paths 3
and 4 are app-only and trade different things.

## What `group_consecutive_messages=False` does NOT break

Verified in the current code:

- *Backend code*: nothing. `agent_ng/` builds `working_history` as a flat
  list of role/content pairs; no reference to `group_consecutive_messages` or
  `display_consecutive_in_same_bubble` anywhere in our code.
- *Frontend code*: nothing. No custom JS traverses Chatbot bubbles.
- *Tests*: no test asserts on grouped vs ungrouped rendering.

What changes is purely visual layout: more whitespace, the image's download
button moves to right under the image, empty toolbar rows appear under the
caption and stats bubbles. The per-message non-functional share button is
unaffected (it still appears on the image group), so this path does not fix
the share issue by itself.

## Files in our app involved in the message flow

- `agent_ng/tabs/chat_tab.py` — `gr.Chatbot(...)` construction (where
  `group_consecutive_messages` would be set for path 2).
- `agent_ng/app_ng_modular.py` — where the image component, the caption, and
  the `token_metadata_message` are appended to the working history during a
  turn (where reordering in path 3 would happen).
- `agent_ng/_file_attachment.py` — `build_file_bubbles()` produces the image
  bubble; `caption` field is what path 4 would consume.

## References

- Gradio Chatbot docs: <https://www.gradio.app/main/docs/gradio/chatbot>
  (`group_consecutive_messages`, `allow_file_downloads`, `buttons`).
- Gradio source: `js/chatbot/shared/Message.svelte`,
  `js/chatbot/shared/ButtonPanel.svelte`, `js/chatbot/shared/ChatBot.svelte`,
  `js/chatbot/Index.svelte`, `js/utils/src/utils.svelte.ts`.
