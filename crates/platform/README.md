# filmcraft-platform

OS media integration for FilmCraft (layer L5): hardware video decoding through the operating
system's codecs, behind `filmcraft_codecs::VideoDecoder`. It holds OS media FFI and nothing else.
It is the one crate of the workspace allowed to contain `unsafe`, under the rules of
[ADR 0001](../../docs/adr/0001-platform-ffi.md) and [AGENTS.md](../../AGENTS.md) §0.3; the Linux
backend follows [ADR 0002](../../docs/adr/0002-linux-vulkan-video.md).

```rust
// at startup (the desktop app, filmcraft-cli, the bench)
let availability = filmcraft_platform::register(); // Available("VideoToolbox") on macOS,
                                                   // Available("Vulkan Video") on Linux
```

## What it does

- **macOS: VideoToolbox H.264 (`avcC`) and HEVC (`hvcC`)**, 8- and 10-bit, 4:2:0 and 4:2:2
  (`videotoolbox.rs`). The session is created from the sample entry's parameter sets with a
  hardware decoder *required*; samples go in as `CMSampleBuffer`s with asynchronous decompression
  (two access units in flight); the output callback copies each NV12 / P010-style biplanar
  `CVPixelBuffer` into planar `Yuv8` / `Yuv16` (chroma deinterleaved, 10-bit samples shifted down
  from the high bits, cropped to the conformance window when the buffer is the coded size). A
  reorder buffer of the stream's own depth (`max_num_reorder_frames` /
  `sps_max_num_reorder_pics`) restores presentation order; a run starting at an HEVC CRA leaves
  out its RASL pictures, as our decoder does. Each seek (`reset`) starts a fresh session.
- **Linux: Vulkan Video H.264 (`avcC`)**, 8-bit 4:2:0 progressive (`vulkan/`), **not yet run on
  a GPU** (see Tests). `register()` only checks that the system Vulkan loader (`libvulkan.so.1`,
  loaded at runtime through `ash`) opens; the GPU is opened when the first stream needs it: the
  best GPU (discrete first) with Vulkan 1.3, a video decode queue that takes H.264, timeline
  semaphores and synchronization2. Decoding is stateless: FilmCraft's own H.264 front-end
  (`filmcraft_h264::accel`) parses the headers, computes picture order counts, marks references
  and decides output order, and the GPU reconstructs pixels. Each stream gets a video session,
  session parameters built from the `avcC` SPS / PPS (`vulkan/h264.rs`: every list given
  explicitly, no fall-back left to the driver), and one layered image whose array layers are the
  DPB slots and also the decode output (`DPB_AND_OUTPUT_COINCIDE`). A picture keeps its slot while
  the front-end lists it in the DPB (`vulkan/slots.rs`); each output copies the slot's NV12 planes
  into a host buffer and then into planar `Yuv8`, cropped. Every submission (decode on the video
  queue, copy on it or on a transfer queue) waits for the previous one through a timeline
  semaphore and is waited for on the host with a 2 s limit; a decode-status query reports decode
  errors where the driver supports it. A seek (`reset`) keeps the session and resets its DPB state
  on the next decode.
- **Other systems:** `register()` does nothing and returns `Availability::Unavailable`.
- **`HybridDecoder`** (`hybrid.rs`, safe code): the hardware decoder plus the means to build our
  software decoder for the same `SampleEntry` (`filmcraft_codecs::software_video_decoder`). On a
  mid-stream failure (decode error, invalidated session, changed in-band parameter sets) it replays
  the samples since the last restart point (IDR / IRAP; for an HEVC CRA the one before, so its RASL
  pictures decode) through the software decoder, drops pictures already returned, keeps the ones
  the hardware had decoded but not returned, and stays in software for that instance. The replay
  log is bounded (600 samples / 256 MB); beyond it one error is returned and the next seek restarts
  in software. Streams our decoders cannot decode (HEVC 4:2:2) have no fallback: the error stands.
  **First-use verification** (`HybridDecoder::verifying`, used by the Vulkan Video factory): the
  first decoder of a hardware path in a process runs our reference decoder
  (`filmcraft_codecs::reference_video_decoder`, single-threaded so each call returns what the
  front-end outputs in it) in lockstep and compares every picture of the first calls (8 pictures)
  bit for bit, returning the reference's pictures. A match marks the path verified for the run; a
  mismatch continues with the reference decoder and turns the path off for the run (logged,
  `perf.stats` `decode.hardware.mismatches`); a hardware error or a drop before the end gives
  the check to the next decoder. Other decoders use software while a check runs.

## Guarantees

- **Never undecodable:** the factory declines (returns `None`, so the software decoder is used)
  when Settings ▸ Playback ▸ Hardware decoding is Off, for formats it does not take (field-coded
  H.264, bit depths other than 8 / 10, 4:4:4 or monochrome, luma / chroma depth mismatch, larger
  than 8192×8192) and when VideoToolbox cannot create a hardware session. The Vulkan Video factory
  declines for everything but 8-bit 4:2:0 progressive H.264 that the software decoder also takes,
  for profiles / levels / sizes / DPB sizes beyond the GPU's capabilities, when no GPU has a
  Vulkan Video H.264 decoder, and when the decoded format cannot be copied out; mid-stream it
  fails over to software on a decode error, a `frame_num` gap (non-existing reference frames), a
  lost device or a 2 s timeout.
- **Interchangeable:** colour, pixel aspect, pts, presentation order, `is_random_access` and
  `is_disposable` come from the software decoders' own helpers (`filmcraft_codecs::hw`,
  `video::vui_color`, `sar_par`).
- **Never crash:** no `unwrap` / `expect` / `panic!` outside tests; the output callback runs under
  `catch_unwind`; every `unsafe` block has a `// SAFETY:` comment; the public API is safe.
- **Counted:** `perf.stats` `decode.hardware` (frames, software frames, sessions, declined,
  fallbacks, mismatches; `filmcraft_codecs::hw::hw_stats`).

## Tests

| test | what |
|---|---|
| `tests/videotoolbox.rs` (macOS) | H.264 High, HEVC Main (open GOP: CRA + RASL) and HEVC Main 10, 640×360 (coded 368: cropping) with B-frames: every picture **bit-exact** with our software decoder, same pts order, count, colour and aspect, also after `reset` + reseek to every later sync sample, mid-stream `flush`, and a full pass after resets; forced mid-stream failures (`VtDecoder::fail_after`) at five points continue with the software decoder's exact output; seeded mutation of samples and parameter sets (bit flips, truncation, corrupt length prefixes) never panics or hangs; HEVC 4:2:2 10-bit is bit-exact with ffmpeg's decode |
| `tests/fallback.rs` (every OS) | `HybridDecoder` with a stand-in hardware decoder failing after N samples (every sync sample ± a few, first / last sample, after a seek): output identical to the software decoder; in-band parameter sets identical to the sample entry's stay in hardware, different ones switch to software |
| `tests/setting.rs` | Hardware decoding Off gives the software decoder through `make_video_decoder` and the media stack (no hardware frames); Auto gives VideoToolbox where available |
| `tests/verify.rs` (every OS) | first-use verification with stand-in hardware decoders: matching pictures verify the path; a damaged picture (first or fourth output) never reaches the caller, turns the path off and is counted; a hardware error, an early drop or a damaged stream give the check back; the check continues across seeks |
| `src/vulkan/h264.rs` (Linux) | every field of the std SPS / PPS structures and of each picture's decode info (`frame_num`, `idr_pic_id`, IDR and reference flags) against `ffmpeg -bsf:v trace_headers` on three libx264 streams (custom scaling matrices, weighted prediction, cropping, CAVLC / CABAC, POC types 0 and 2, four references); levels, profiles, bitstream layout |
| `src/vulkan/slots.rs`, `decoder.rs` (Linux) | slot reuse rules; NV12 → planar cropping and deinterleaving, hostile crops and short buffers |
| `tests/vulkan_video.rs` (Linux) | **without a Vulkan Video device** (CI, cloud, Mesa lavapipe): H.264 goes to the software decoder, declined and counted, never left claimed. **With one** (not run yet): five libx264 fixtures (High B-pyramid 640×360 cropped, Main ref=4 weightp, Baseline 4 slices, High cqm=jvt, 1080p) bit-exact with the software decoder, also after reset + reseek to every later sync sample and after resets; forced failures at five points continue with the software output; the factory verifies on first use and Off gives software; damaged samples and parameter sets never crash or hang |

Fixtures are made with ffmpeg into `target/fixtures/platform/` (generator only, never linked);
tests skip without ffmpeg or without a hardware decoder.

### Running the GPU tests (Linux)

```sh
vulkaninfo --summary | grep -iE "deviceName|driverName"     # the GPU and driver Vulkan sees
vulkaninfo | grep -E "VK_KHR_video_decode_h264|VIDEO_DECODE" # its video decode support
cargo test --release -p filmcraft-platform --test vulkan_video -- --nocapture --test-threads 1
```

The first line of output names the device (`Vulkan Video device: …`); `SKIPPED` means none was
found and says why. In the app, `perf.stats` (control channel / MCP) shows
`decode.hardware.frames` against `softwareFrames`, and `declined` / `fallbacks` / `mismatches`.

## Performance

M4 Pro, load 150–190 (`cargo xtask bench --hw off|auto`): CPU per decoded frame H.264 2160p
119 → 4.2 ms, HEVC 2160p 86 → 3.7 ms; decode 35 → 107 fps and 49 → 217 fps; 4K H.264 and HEVC
playback with no dropped frames at Full, 1/2 and 1/4. Details in
[docs/performance.md](../../docs/performance.md).

## Not yet

Zero-copy upload of `CVPixelBuffer`s / Vulkan images into wgpu textures; hardware encoding; Media
Foundation / D3D11 (Windows) decoders; field-coded H.264. Linux: a run on real GPUs; HEVC and AV1
through Vulkan Video; drivers that only offer a decode output separate from the DPB
(`DPB_AND_OUTPUT_DISTINCT`, some AMD GPUs) or no copy out of DPB images; pipelining (each picture
is decoded and copied synchronously); VA-API for machines without a Vulkan Video driver.
