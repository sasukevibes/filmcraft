# ADR 0002: Hardware decoding on Linux through Vulkan Video

- **Status:** proposed (2026-10-06). Implemented for H.264; **not yet run on a GPU** (see
  [Verification](#verification)).
- **Issue:** #30 (hardware acceleration)
- **Builds on:** [ADR 0001](0001-platform-ffi.md) (`unsafe` OS media FFI in `crates/platform` only)

## Context

Linux has no hardware decoding path. Decoding 4K H.264 / HEVC in software costs 85–120 ms of CPU
per frame ([performance.md](../performance.md)), so 4K playback drops frames. The roadmap named
VA-API as the Linux API. The choices are:

| API | Vendors | Notes |
|---|---|---|
| VA-API (libva) | Intel and AMD natively; NVIDIA only through the community `nvidia-vaapi-driver` shim | the long-standing Linux standard |
| Vulkan Video (`VK_KHR_video_decode_*`) | NVIDIA's own driver; Mesa RADV (AMD) and ANV (Intel) | also on Windows; decoded pictures are Vulkan images, the API wgpu renders with on Linux |
| NVDEC / CUVID | NVIDIA only | proprietary SDK |

VA-API and Vulkan Video are both *stateless*: the application parses the headers, computes picture
order counts, manages the decoded picture buffer (DPB) and output order, and the GPU only
reconstructs pixels. VideoToolbox, by contrast, takes compressed access units and returns pictures.

The first machine this will run on (the project owner's) has an NVIDIA GPU.

## Decision

Use **Vulkan Video** for Linux hardware decoding, in `crates/platform` under ADR 0001's rules
(FFI module with `#[allow(unsafe_code)]`, a `// SAFETY:` comment on every `unsafe` block, a safe
`Result` API, the software decoder as the tested fallback). VA-API can be added later behind the same
front-end if Intel / AMD machines without a working Vulkan Video driver need it.

Why Vulkan Video:

- It is native on NVIDIA, which VA-API reaches only through a third-party shim, and Mesa ships it
  for AMD and Intel.
- The same API exists on Windows, so one backend can serve both systems later.
- Decoded pictures are Vulkan images: the route to zero-copy display through wgpu later.
- No new dependency: `ash` (MIT / Apache-2.0) is already in the tree through wgpu. `libvulkan.so.1`
  is loaded at runtime (`ash::Entry::load`), so a machine without it starts normally and reports
  hardware decoding as unavailable.

## Design

1. **The front-end is our own decoder.** `filmcraft_h264::accel` runs the software decoder's own
   NAL handling, slice-header parsing, POC computation, reference marking (sliding window, MMCOs)
   and DPB output process, but skips pixel reconstruction. Per access unit it emits events:
   *decode picture P* (active SPS / PPS, `frame_num`, `idr_pic_id`, POCs, reference flags, the
   slice NAL units, and the DPB pictures P may reference) and *output picture Q* (in the exact
   order the software decoder outputs, with its pts, crop and colour). Output order, cropping and
   colour therefore match the software decoder by construction.
2. **The backend** (`crates/platform/src/vulkan/`) owns a Vulkan device with a video decode queue,
   one video session per stream, session parameters built from the stream's SPS / PPS, a layered
   DPB image (one array layer per DPB slot), and a host-visible bitstream buffer. For each *decode*
   event it records the reference slots and decodes into the picture's slot; for each *output*
   event it copies the slot's NV12 planes into a host buffer and then into planar `Yuv8`, cropped
   to the conformance window.
3. **Failure handling.** Every Vulkan call returns a `Result`; a decode error (result-status
   query), a device loss or a fence that does not signal within 2 s is an error, which
   `HybridDecoder` answers by continuing in software (ADR 0001's fallback guarantee).
4. **First-use verification.** Untested hardware paths must never change a picture. For the first
   hardware stream of each kind in a process, `HybridDecoder` runs our reference decoder
   (single-threaded, so each call returns exactly what the front-end outputs in it) in lockstep,
   compares the first 8 pictures bit for bit and returns the reference's pictures meanwhile. On a
   mismatch it continues with the reference decoder (already in step) and turns that hardware path
   off for the rest of the run (logged, `perf.stats` `decode.hardware.mismatches`).
5. **Declined streams** (software is used): no `libvulkan`, no device with a video decode queue and
   H.264 decode support, unsupported profile / level / size, field coding, anything other than
   8-bit 4:2:0 (the software H.264 decoder's own limits), and streams with `frame_num` gaps
   (non-existing reference frames).

## Verification

| What | Where | Status (2026-10-06) |
|---|---|---|
| Front-end against the software decoder on every libx264 fixture: same outputs (order, pts, POC, crop, colour, aspect), the same outputs per call as a single-threaded decoder, decode-before-output, DPB slot simulation with `max_dpb_frames + 1` slots (`crates/h264/tests/accel.rs`) | every machine | passes |
| Std SPS / PPS / picture structures field by field against `ffmpeg -bsf:v trace_headers` on three libx264 streams (`crates/platform/src/vulkan/h264.rs`) | Linux | passes |
| No Vulkan Video device (Mesa lavapipe): the probe declines cleanly, H.264 goes to software, counted (`tests/vulkan_video.rs`) | Linux | passes |
| First-use verification with stand-in decoders (`tests/verify.rs`) | every machine | passes |
| Bit-exact parity with the software decoder on five H.264 fixtures, seeks, flush, forced failures, factory verification, damaged input (`tests/vulkan_video.rs`) | Linux with a Vulkan Video driver | **not run yet** |
| Benchmark (`cargo xtask bench --hw off` / `--hw auto`) | same | **not run yet** |

Until the GPU rows pass, first-use verification is what keeps a wrong picture from reaching
the user: the first stream's first pictures are compared with our decoder, and on any difference
hardware decoding stays off for that run.

## Consequences

- Linux gets the same fallback guarantee as macOS: a stream is either decoded bit-exactly on the
  GPU or decoded by our own decoder.
- The front-end refactor is reusable: VA-API, Windows D3D11 video decoding and Vulkan Video for
  HEVC / AV1 take the same per-picture information.
- First version copies pictures back to system memory (12 MB per 4K frame). Zero-copy into wgpu is
  a follow-up, shared with VideoToolbox.

## On hold: agent-first hardware decoding

Ideas for agents (Claude Code and others, as Omarchy pre-wires them) editing video through the MCP
server, the CLI and the control channel. Parked by the project owner on 2026-10-06; none is
implemented. Each would be a normal engine command so MCP, CLI and the control channel get it.

1. **Headless first.** Vulkan Video needs no window or display, so hardware decoding works in
   `filmcraft-cli mcp`, over SSH and in background jobs. Keep the decoder independent of the UI's
   GPU device (copy-back stays first-class; zero-copy is a desktop extra).
2. **Sampling, not just playback.** Agents jump around (scene detection, `render_frame`, checking
   an edit). Keep the video session across seeks (only the front-end resets), and add a
   "preview quality" request that scales on the GPU (≈1.4 MB per 720p frame instead of 12 MB at 4K),
   flagged like draft frames and never used for edits or export.
3. **Sharing the GPU.** The desktop app and several agent sessions may decode at once. Cap hardware
   sessions per process, fall back to software when the cap is reached, report it in `perf.stats`.
4. **Tell the agent what's happening.** `hw.info` (also an MCP resource such as
   `filmcraft://hardware`): device, driver, codecs, limits, and why hardware decoding is off with
   the fix ("libvulkan not installed", "GPU not visible in this sandbox"). Per-clip decode path and
   speed so an agent can choose `media.createProxies` first; fallbacks as structured events.
5. **Same result on every machine.** Keep hardware output bit-exact; add `hw.verify {item}` so an
   agent can check a clip both ways before a long unattended batch.
6. **Recoverable crashes.** A driver crash kills the process outside Rust's panic handling.
   Breadcrumbs: note the codec and driver before each session; if the next start finds the note,
   turn that path off and write a machine-readable crash report (Omarchy's crash watcher or any
   agent can read it). Stronger, if needed: decode in a helper process.
7. **Agents developing it.** GPU tests need a real GPU: a local agent runs `tests/vulkan_video.rs`
   and the bench. On a mismatch it can log every structure we pass to the driver with the Vulkan
   API dump layer (`VK_LAYER_LUNARG_api_dump`) and compare it with what an external decoder
   passes for the same file (ffmpeg as an outside oracle only, AGENTS.md §2).
