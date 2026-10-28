# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project is released under GPL-3.0.

## [Unreleased]

- Add `docs/INSTALL_CN.md`: exact model file names and target folders, required vs optional plugins, the `nvidia-vfx` dependency boundary for `nvidia_rtx_vsr`, and a checklist for the "validation failed / 校验失败" error on the first-pass `Final Video` link.

## [1.0.0] — 2026-10-18

Independent pack: MiniMax H3 Segment Studio.

- Windowed Refine **Window Frames** is a dropdown (`5.9s` / `6.6s` / `7.3s` / `8s` / `10s` / **whole 15s**), not a typed number. Default remains 5.9s (141) at 2.0 on 16GB. Lower **Output Megapixels** before picking a longer window.
- Finite Segment Sampler and Windowed Refine host a KJ-style tiny-VAE live preview on the node itself. No extra Model Preview Override node. `models/vae_approx/MiniMax-H3-TAE.safetensors` is used when present.
- Windowed Refine `nvidia_rtx_vsr` now follows the original dual-pass order (CLIP/refs → VSR pixels → 2.0 encode → light T8) and unloads between phases so 16GB can use the sharp VSR path without holding the whole clip.

- Ship Timeline Director (GPL-3.0) plus the 16GB 0.4→2.0 path as one plugin.
- Flagship graphs are named by seconds-per-segment, not segment count: `8秒-多段` and `15秒-多段`, each with T8 and 3D-lift variants.
- Flagship graph `example_workflows/H3_SegmentStudio_8秒-多段_T8二采_2.0.json`: first pass at 0.4, windowed refine at 2.0. The old whole-clip VSR+T8 subgraph is removed.
- Windowed Refine with `h3_latent` and Refine Steps 0 now 3D-lifts then decodes (no T8). Short lift graph: `example_workflows/H3_SegmentStudio_8秒-多段_仅3D放大_2.0.json`.
- Windowed 2.0 refine reuses first-pass prompt+refs, caps T8 at 141 frames, and resets AIMDO CUDA graphs between windows.
- 15s-per-segment T8 graph `example_workflows/H3_SegmentStudio_15秒-多段_T8二采_2.0.json`: Windowed Refine reads the first-pass Final Video one window at a time when Merged Frames is the 8x8 omit stub.
- Lift-only ~15s boxes: `split_long_windows` on Windowed Refine slices 3D lift to 141-frame hops. Graph: `example_workflows/H3_SegmentStudio_15秒-多段_仅3D放大_2.0.json`.

## [Unreleased] — upstream Timeline Director history below

### Added

- Windowed Refine and sequential Finite Segment Sampler now use Director-style 2.0 continuity on Timeline's T8 path: encode the 0.4 window, `h3_latent` 3D lift, re-pin the previous 2.0 AV tail with Drift-Control, sample 2 T8 steps at the output canvas, then trim the pin. 16 GB cards still never encode a whole-clip 2.0 tensor.
- Cap 2.0 T8 refine windows at 192 frames (8s). First-pass 243-frame (~10.1s) GEN windows are split into 8s hops with a 39-frame re-pin; 16GB cards OOM from 9s at 2.0.
- Windowed 2.0 refine now lifts the 3D latent on a clean GPU *before* T8 CLIP (~15GB). Loading CLIP then LatentResizer3D together OOMed 16GB even on 8s windows. Temporal lift chunks at 16 tokens.
- Clamp 2.0 T8 sample windows to 141 frames (~5.9s). After CLIP was moved out of the way, 192-frame 2.0 SageAttention still OOMed the MiniMax H3 UNET on 16GB.
- Windowed refine no longer maps 2.0 hop indices onto the first-pass planner. A 3-segment source split into 6×141-frame T8 windows was dying on spill with `Prompt index 4 exceeds the planner's 3 material segments` after the first 2.0 window had already sampled.
- Evict dynamic UNET/CLIP before each 3D lift. Window 1 2.0 T8 succeeded; window 2 LatentResizer3D conv3d OOMed because Comfy will not unload one dynamic model for another (`0 models unloaded`) and the 15.2GB CUDA cache stayed reserved. Lift chunks are 8 tokens.
- Windowed / sequential 2.0 refine now reuses the first-pass prompt and reference images (same idea as MiniMaxH3-双采超清-最初版 Get_positive). Capped hops pick the overlapping GEN box. Falls back to the generic T8 refine prompt only when the planner has no materials.
- Reset Comfy AIMDO CUDA graphs between 2.0 refine windows (the same cleanup Comfy only runs when a node returns). Window 1 T8 left 4.2GB free for window 2 LatentResizer3D even after `unload_all_models`. Cached first-pass cond is CPU-offloaded; lift chunks shrink if free VRAM is still tight.
- Add `h3_latent` upscale method and an optional latent-upscaler combo on Windowed Refine (`auto` picks a compatible H3 checkpoint). Pixel VSR/lanczos remain as heavier fallbacks.
- Add `MiniMax H3 Windowed Refine` for overlapping 0.4-MP windows so whole-clip 2.0 T8 is never required on a 16 GB card.

### Added

- Port Director's long-job loop onto Finite Segment Sampler: `execution_mode=auto` switches from graph expansion to a one-segment Python loop at 8+ windows, with optional VRAM unload between segments and disk resume under `output/minimax_seg_cache`. 20–40 segment 0.4 MP first passes no longer keep every AV latent linked in one DAG.

### Added

- Stitch FaceRefine at planner resolution, fade the paste at segment seams, then VSR the whole clip so the face is not a hard sticker on 2.0 frames. Pin each exported mp4's audio length to its frame count to stop lip-sync drift after the second H3 pass.
- Add optional `face_refine` on Finite Segment Sampler: connect MiniMax H3 Director FaceRefine to YOLO-crop and H3-restitch faces after 2.0 VSR, without running the slow Director generation loop. Keep Material Planner `secondPass` off on this path.
- Add `output_megapixels` (default 2.0 in the UI) on Finite Segment Sampler: after overlap trim, each segment is RTX-VSR/lanczos upscaled and optionally light-refined (beta / 2 steps / denoise 0.2) before the crash-safe mp4 write, so long videos finish at 2MP without merging RGB.
- Add `export_mode` (`segments` / `all`) on Finite Segment Sampler and the low-memory VIDEO sampler. `segments` writes crash-safe `output/minimax_seg_export/<timestamp>/seg_XXXX.mp4` as soon as each clip finishes, plus `full.mp4` after join — the Timeline equivalent of Director segment export, without the slower Director Refine loop.
- Add a low-memory finite segment VIDEO sampler that encodes each decoded segment to ComfyUI's temporary directory and joins encoded sources without retaining the full RGB timeline. It saves no latent checkpoints and cleans temporary spill files.
- Add a matching all-in-one low-memory example workflow that connects the sampler's `VIDEO` output directly to `SaveVideo`.
- Add local configuration-only and complete-media preset folders, a preset browser/loader node, an export node accepting either Material Plan or Segment Plan, and an editable preset-import input on Material Planner.
- Persist an executed preset back into the Material Planner UI so imported materials, segment timing, prompts, and assignments remain editable after disconnecting the loader.
- Add a native ComfyUI `Start Loop` / `End Loop` workflow path with public nodes for loop-state initialization, per-iteration segment selection, continuation preparation, overlap-aware accumulation, and final output resolution.
- Expose per-segment two-stage settings so native Loop workflows can freely choose ordinary sampling, the bundled H3 two-stage sampler, previews, saves, or custom processing inside the loop body.

### Changed

- Make the H3 Plan Encoder and bundled H3 Two-Stage Sampler visible for explicit native Loop workflow construction.
- Keep the original Finite Segment Sampler as a backward-compatible one-node workflow path.

### Fixed

- Keep uploaded image/audio removal controls above thumbnails and drag surfaces so their delete buttons remain clickable.
- Reset the active segment before applying imported preset widget values, allowing repeated imports when the next preset has fewer segments than the previous one.
- Prefer and auto-repair selection to a structurally compatible MiniMax H3 latent upscaler when the shared model directory also contains LTX upscalers; reject incompatible checkpoints before low-resolution sampling and report the selected filename clearly.
- [spill-seg-offload] Evict per-segment decoded frames and the superseded accumulated tensor after merge in FiniteSegmentFinalize; sampled latent, merged output and audio untouched. Prevents RAM OOM on long multi-segment sampler jobs. Set H3_MERGED_FP16=1 to store merged frames in float16 (halves retained bytes; blind-tested, no perceivable difference).

## [0.8.0] - 2026-09-17

### Added

- Bundle the MiniMax H3 progressive-resolution runtime so two-stage segmented generation works without a separate `comfyui-SelfLift` installation.
- Support locked-audio long-video digital humans through the native AV mask/sigma path while preserving low- and high-resolution continuation state between segments.
- Add Material Planner controls for enabling two-stage sampling, selecting any checkpoint from `models/latent_upscale_models`, and choosing the number of high-resolution steps.

### Changed

- In both one-stage and two-stage reference-video generation, enabling Video Original Audio now promotes the edited soundtrack to the locked native AV path: every segment receives the exact timeline slice with a zero audio denoise mask, and final output uses the continuous original waveform. Disabling it fixes the AV audio stream and final master to silence instead of asking H3 to generate replacement sound; an explicitly uploaded locked audio asset still takes priority.
- Reduce routine two-stage timing, memory, upscaler, and tiling messages to debug logging.
- Give the retained side of a learned 3D latent lift real low-resolution predecessor context, eliminating the fixed dark/bright pulse at a trimmed segment boundary without restoring a high-resolution latent paste.
- Preserve both native low-resolution tails and final high-resolution Drift-Control anchors across every segmented generation mode, including text, image, audio, image-plus-audio, reference video, and digital-human workflows.

### Fixed

- Eliminate progressive blur, white flashes, and visible joins at two-stage long-video boundaries by applying continuation consistently at both resolution stages.
- Keep locked digital-human audio on the native AV mask/sigma path through both SelfLift stages and restore one continuous original waveform during final assembly.
- Preserve the PR #9 DOM-widget host width pin while integrating the two-stage Material Planner UI.

## [0.7.0] - 2026-09-15

### Added

- Add a compact all-in-one workflow covering material-free T2V, image/audio reference generation, editable-video reference, character replacement, motion transfer, digital humans, and manually segmented long-form generation.
- Support user-authored GEN windows, global or per-segment prompts, per-segment media assignments, seam overlap calculation, and final tail trimming directly in Material Planner plans.
- Add locked-original-audio generation for long-form digital humans and singing avatars. Source audio is sliced per segment, injected through the native AV-latent mask/sigma path for lip-sync guidance, and restored unchanged during final assembly.
- Remove global upload-count limits from the standalone image and audio libraries. Each generated segment continues to enforce MiniMax H3's native maximum of 9 reference images and 3 reference audio clips.

### Changed

- Simplify the primary long-video architecture to Material Planner → Finite Segment Sampling. Remove Long Reference Auto Segmentation, keep all segment boundaries user-authored, and hide Plan Encoder from the new-node menu while retaining it for internal expanded graphs and legacy workflows.
- Consolidate the shipped examples to the all-in-one workflow and the Omni prompt-generation workflow, and rewrite both READMEs around these two entry points.
- Make adaptive Drift-Control the only finite-segment video continuation path and remove the public continuation-mode selector. The mask derives its temporal prefix from the H3-aligned overlap selected by the user, preserves up to four clean-boundary taper steps, and follows the connected external sigma schedule, including accelerated 4-step and 8-step configurations. Continued audio uses Soft AV: its final eight latent ticks follow a half-cosine release, and assembly gives the incoming segment ownership of the same aligned overlap so the transition reaches the final soundtrack.
- Use English as the base language for node definitions, tooltips, status messages, and the custom timeline UI. Add official ComfyUI `locales/en` and `locales/zh` resources; the dynamic timeline follows the active ComfyUI locale and safely falls back to English.
- Simplify finite-segment sampling by removing the per-segment seed-increment and Guide-mask switches. Every segment now reuses the same seed.

### Fixed

- Merge community Pull Request #9 to keep the timeline DOM-widget host width pinned to the current node width after ComfyUI frontend layout passes.

### Security

- Replace the plugin-owned network-reachable chunk writer with ComfyUI's official `/upload/image` endpoint and keep all timeline uploads in a fixed input subfolder.
- Restrict media inspection and preview generation to direct Timeline Director uploads, change both helpers to POST, and add file-size, duration, dimension, concurrency, timeout, and proxy-cache limits.
- Keep detailed decoder/FFmpeg failures in server logs while returning only bounded public errors to browser clients.

## [0.6.0] - 2026-09-01

### Added

- Add optional `提示词序号` input to the material planner for per-segment image/audio selection.
- Add an explicit segment-count setup flow and drag-and-drop per-segment material boards; Picture/Audio ordinals restart from 1 inside every segment.
- Add a Chinese Agent specification for writing parseable finite-segment prompts.

### Compatibility

- Existing workflows with no prompt-index connection or a segment count of 0 continue to use the complete reference library.

### Fixed

- Restore full timeline editing under Nodes 2.0: timeline presses are captured before canvas gestures, active pointers remain captured outside the node, video clips explicitly select their host node, and the cyan generation range now has a full-width move rail plus in-bounds resize handles.
- Allow an empty material plan to run as native MiniMax H3 text-to-video instead of raising “no reference materials”. Empty plans now tokenize only the prompt, create the normal empty AV latent, and skip the Guide dependency entirely.
- Keep timeline dragging and preview playback responsive in Nodes 2.0 by updating only the active clip, generation range, playhead, snap guide, and paired audio geometry during animation frames; the expensive timeline/media DOM rebuild now happens once after drag release.
- Stop the cyan generation-range overlay from stealing pointer input from overlapping video clips. Video bodies and trim handles remain directly draggable, while the `GEN` grip, cyan edge handles, and empty cyan area move or resize the generation range.
- Remove stale invisible DOM hit areas after node resizing or segment-panel contraction: the director now measures natural content height, shrinks as well as grows, and leaves unused DOM-widget space transparent to ComfyUI canvas pointer events.
- Keep the hidden `timeline_data` multiline widget hidden after ComfyUI recreates or resets its DOM wrapper during resize/configure. Its real `.dom-widget` host is now collapsed to zero and cannot intercept canvas drags.
- Allow independent image/audio cards to be copied into per-prompt segment bins while retaining move-based reordering inside the source library. This fixes drag-over highlighting followed by a cancelled drop.
- Lay out multiple image/audio cards horizontally inside each per-prompt segment bin.
- Keep the custom director DOM host at its measured content height and retain a stable node-chrome allowance, preventing Nodes 2.0 from shrinking the node to half-height after selection or resize.

### Changed

- Split the finite long-video path into a pure `MiniMax H3 有限分段展开` planning node and a separate `MiniMax H3 有限分段采样` graph-expansion node. Sampler, scheduler, model, CLIP, and VAEs are no longer inputs of the planning node.

### Removed

- Remove the old generic-loop helper nodes, Loop-dependent workflows/tests/docs, and the ComfyUI PR #15923 runtime dependency.

### Experimental

- Add plugin-owned finite planning and sampling for known segment counts. It performs per-segment selection/encoding/sampling, direct AV-latent continuation, linear temporal masking, overlap removal, and ordered image/audio assembly.
- Add `MiniMax时间线插件内置有限分段工作流.json`, using standard sampler/scheduler/video nodes around the two finite nodes.
- Verify the finite path end to end at 512x288: two 56-frame segments with a 22-frame overlap produced exactly 90 frames at 24 fps with audio.
- Add a dynamic 0→1 linear temporal noise mask to latent-loop continuation. The mask length follows the aligned overlap frames, directly places the previous sampled AV-latent tail at the next segment's opening, and progressively releases video/audio denoising toward the end of the overlap.
- Add an opt-in `MiniMax H3 Direct Latent Guide` node for controlled continuation tests that reuse a sampled H3 video-latent tail without an RGB decode/encode round trip.
- Add an objective video-difference node and a reproducible API workflow comparing native RGB Guide, direct latent Guide, and a single VAE round trip. Production planner, encoder, and compatibility-director behavior is unchanged.

## [0.4.1] - 2026-08-27

### Fixed

- Keep the director DOM fully inside the node after manual width/height resizing by measuring wrapped content, accounting for ComfyUI DOM-widget chrome, and automatically enforcing the resulting minimum node height.
- Allow narrow layouts to wrap settings and inspector controls as complete fields instead of squeezing Chinese labels into vertical characters.

## [0.4.0] - 2026-08-27

### Added

- Split workflow architecture with `MiniMax H3 Material Planner` and `MiniMax H3 Plan Encoder`, removing the dependency cycle when timeline media is inspected by `MiniMax-H3 Prompt Rewriter Omni`.
- Compact `MINIMAX_H3_OMNI_MEDIA_BUNDLE` connection that carries exact Picture, Video, standalone Audio, and paired video-soundtrack order through one socket.
- `MiniMax H3 Omni Media-Bundle Prompt Bridge`, which calls the installed Prompt Rewriter Omni backend and returns only `rewritten_prompt`.

### Changed

- Material Planner outputs were reduced from 23 sockets to two: the H3 plan and the ordered Omni media bundle.
- Every reference video is now strictly cropped to its intersection with the cyan generation range. Outside source frames no longer enter prompt rewriting, H3 video-reference encoding, or per-clip Guides.
- Intersecting videos are numbered left-to-right on the timeline, and the same interval plan is shared by the Omni bridge and final H3 encoder.
- The original monolithic Timeline Director remains available as a backward-compatible node.

### Documentation

- Added a Chinese operational guide for video-generation agents covering long-video segmentation, fixed-Guide overlap, H3 prompt alignment, audio continuity, reference ordering, and deduplicated assembly.
- Added three lightweight example workflows for the compatibility timeline, split planner/encoder, and Prompt Rewriter Omni pipeline.
- Reworked the Chinese and English READMEs into a shorter quick-start guide with compressed workflow screenshots, explicit Prompt Rewriter dependency information, and project credits.

## [0.3.2] - 2026-08-27

### Added

- Video, image, and audio picker buttons now support selecting and importing multiple files in one operation.

### Fixed

- Mouse-wheel events over the director DOM are forwarded to the ComfyUI canvas so canvas zoom continues to work under the cursor.
- Reduced the director DOM widget height from 750 to its 674-pixel content height and removed the forced oversized node height.

## [0.3.1] - 2026-08-26

### Added

- Drop image, audio, and video files directly anywhere inside the director widget to upload them to the correct media collection.
- A full-widget drop overlay provides clear visual feedback during external file dragging.

### Fixed

- External file drops are captured inside the director and prevented from reaching ComfyUI's canvas handlers, avoiding unwanted Load Image nodes or embedded-workflow loading.

## [0.3.0] - 2026-08-26

### Added

- Per-clip video-purpose selector with `Fixed Guide`, `Editable Reference`, and `Boundary Only` modes.
- Editable-reference mode for character, clothing, and style replacement: the selected overlap becomes `<Video N>` without any hard guide.
- Boundary-only mode keeps the overlap as `<Video N>` while anchoring only its first and last frames.
- Drag-and-drop reordering for standalone image and audio bins; prompt ordinals and backend `ref_image_N` / `ref_audio_N` inputs update immediately.

### Compatibility

- Existing workflows and newly uploaded clips default to `Fixed Guide`, preserving the 0.2 behavior until the user explicitly changes a clip mode.

## [0.2.0] - 2026-08-26

### Added

- Native `MiniMaxH3AddGuide` integration for fixed image, video-clip, and optional soundtrack guides at generated-frame positions.
- Timeline visualization of legal fixed-guide frame counts.

### Changed

- Video overlap inside the cyan range is now hard-anchored as a guide; only source context outside crossed edges becomes an ordinary `<Video N>` reference.
- Empty gaps now anchor the nearest left/right stills at the generated first/last frame without consuming `<Picture>` slots.
- Multi-frame guides follow H3's official `5 + 17n` constraint; legal batches plus single-frame guides preserve every source frame (24 frames become `22 + 1 + 1`).
- Long fixed overlaps are decoded in bounded 15-second windows at the node canvas resolution to avoid materializing an oversized source-frame tensor.

## [0.1.1] - 2026-08-24

### Added

- Global video-soundtrack reference toggle on the original-audio track. Disabling it keeps video references active while omitting `ref_video_audio_N` inputs.

### Fixed

- Do not create automatic boundary frames when the cyan selection fully contains a reference video or exactly matches its edges.
- Keep the front-end reference-label preview consistent with the backend boundary-frame rules.

## [0.1.0] - 2026-08-23

### Added

- Editable multi-video reference timeline for MiniMax H3 Ref2VA.
- Bound original-audio waveform track for uploaded videos.
- Clip move, trim, split, delete, and precision editing controls.
- Draggable red playhead with low-resolution proxy monitoring.
- Cached `480×270 / 12fps` silent preview proxies.
- Cyan generation range with move, resize, snapping, and gap matching.
- Bidirectional synchronization between the cyan range and `generation_seconds`.
- Partial-overlap reference extraction with paired soundtrack timing.
- Automatic selection-edge boundary frames and empty-gap bridging.
- Standalone reference image and audio bins with independent numbering from 1.
- H3 presentation ordering that keeps independent media labels stable.
- Reference-media resizing to the node output canvas during decoding.
- Chunked media upload endpoints.
- `视频原声合并` timeline-mixed audio output.
- `独立音频合并` concatenated standalone-audio output.
- Workflow JSON persistence, example workflow, and media smoke tests.
- English and Simplified Chinese documentation.
