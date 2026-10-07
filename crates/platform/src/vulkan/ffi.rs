//! Vulkan Video FFI (Linux): the decode device, one video session per stream, decode and
//! copy-back submissions.
//!
//! The only module of the Linux backend with `unsafe` code (docs/adr/0001-platform-ffi.md,
//! docs/adr/0002-linux-vulkan-video.md). Rules: every `unsafe` block has a `// SAFETY:` comment;
//! Vulkan handles stay inside this module's types; every failure is a `Result` that the decoder
//! answers by switching to the software decoder; no extension or core-1.2/1.3 function is called
//! unless the driver returned a pointer for it (ash's stand-ins for missing functions abort).
//!
//! Synchronisation is deliberately simple: every submission of a session (decode or copy, on the
//! decode queue or a transfer-capable one) waits for the previous one through the session's
//! timeline semaphore and is waited for on the host (2 s limit) before the next is recorded. So
//! command buffers, the bitstream buffer and the read-back buffer are never in use by the GPU when
//! the host touches them, and consecutive submissions are fully ordered (semaphore signal / wait
//! carry the memory dependencies; barriers chain with them through `ALL_COMMANDS`).

use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use ash::vk;
use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH265PictureInfo, StdVideoDecodeH265ReferenceInfo, StdVideoH264ProfileIdc,
    StdVideoH265ProfileIdc,
};

use super::{h264, h265};

pub(crate) type R<T> = Result<T, String>;

/// How long the host waits for one submission before treating the GPU as hung.
const TIMEOUT_NS: u64 = 2_000_000_000;
/// Largest bitstream buffer a picture may need (a corrupt or hostile stream beyond it is an error).
const MAX_BITSTREAM: u64 = 128 << 20;

fn vk_err(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |e| format!("{what}: {e}")
}

/// Whether the system Vulkan loader can be opened (no driver is touched).
pub(crate) fn loader_available() -> R<()> {
    // SAFETY: opening libvulkan.so.1 runs only the loader's own initialisation; the `Entry` is
    // dropped right away and no handle created from it outlives it.
    unsafe { ash::Entry::load() }.map(drop).map_err(|e| format!("no Vulkan loader: {e}"))
}

/// One queue family the device offers.
#[derive(Clone, Copy)]
struct Family {
    flags: vk::QueueFlags,
    codecs: vk::VideoCodecOperationFlagsKHR,
    status_queries: bool,
}

/// The decode device: a Vulkan instance and logical device with a video decode queue that takes
/// H.264 and / or HEVC, and a transfer-capable queue for read-back (the same queue when it can
/// copy).
pub(crate) struct Device {
    _entry: ash::Entry,
    instance: ash::Instance,
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    video_instance: ash::khr::video_queue::Instance,
    video: ash::khr::video_queue::Device,
    decode: ash::khr::video_decode_queue::Device,
    decode_family: u32,
    copy_family: u32,
    decode_queue: Mutex<vk::Queue>,
    /// `None`: copies go to the decode queue.
    copy_queue: Option<Mutex<vk::Queue>>,
    mem: vk::PhysicalDeviceMemoryProperties,
    status_queries: bool,
    /// The codecs the decode queue takes (with their extensions enabled).
    codecs: vk::VideoCodecOperationFlagsKHR,
    pub(crate) name: String,
    lost: AtomicBool,
}

/// Device extensions the backend needs.
const EXTENSIONS: [&CStr; 2] = [ash::khr::video_queue::NAME, ash::khr::video_decode_queue::NAME];

/// The codecs the backend decodes and their extensions (at least one is needed).
const CODECS: [(vk::VideoCodecOperationFlagsKHR, &CStr); 2] = [
    (vk::VideoCodecOperationFlagsKHR::DECODE_H264, ash::khr::video_decode_h264::NAME),
    (vk::VideoCodecOperationFlagsKHR::DECODE_H265, ash::khr::video_decode_h265::NAME),
];

fn any_codec() -> vk::VideoCodecOperationFlagsKHR {
    CODECS.iter().fold(vk::VideoCodecOperationFlagsKHR::empty(), |a, (c, _)| a | *c)
}

/// Device-level functions called through pointers (checked before use).
const DEVICE_FNS: [&CStr; 13] = [
    c"vkCreateVideoSessionKHR",
    c"vkDestroyVideoSessionKHR",
    c"vkGetVideoSessionMemoryRequirementsKHR",
    c"vkBindVideoSessionMemoryKHR",
    c"vkCreateVideoSessionParametersKHR",
    c"vkDestroyVideoSessionParametersKHR",
    c"vkCmdBeginVideoCodingKHR",
    c"vkCmdEndVideoCodingKHR",
    c"vkCmdControlVideoCodingKHR",
    c"vkCmdDecodeVideoKHR",
    c"vkQueueSubmit2",
    c"vkCmdPipelineBarrier2",
    c"vkWaitSemaphores",
];

/// Instance-level functions called through pointers (checked before use).
const INSTANCE_FNS: [&CStr; 2] = [c"vkGetPhysicalDeviceVideoCapabilitiesKHR", c"vkGetPhysicalDeviceVideoFormatPropertiesKHR"];

fn version(v: u32) -> String {
    format!("{}.{}", vk::api_version_major(v), vk::api_version_minor(v))
}

/// The queue families of a physical device with their video codecs and status-query support.
fn queue_families(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Vec<(vk::QueueFamilyProperties, Family)> {
    // SAFETY: `pdev` was enumerated from this live instance.
    let n = unsafe { instance.get_physical_device_queue_family_properties2_len(pdev) }.min(64);
    let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); n];
    let mut status = vec![vk::QueueFamilyQueryResultStatusPropertiesKHR::default(); n];
    let basic: Vec<vk::QueueFamilyProperties> = {
        let mut props: Vec<vk::QueueFamilyProperties2> =
            video.iter_mut().zip(status.iter_mut()).map(|(v, s)| vk::QueueFamilyProperties2::default().push_next(v).push_next(s)).collect();
        // SAFETY: `props` holds `n` initialised structures whose pNext chains point at elements of
        // `video` / `status`, which outlive the call.
        unsafe { instance.get_physical_device_queue_family_properties2(pdev, &mut props) };
        props.iter().map(|p| p.queue_family_properties).collect()
    };
    basic
        .into_iter()
        .zip(video.iter().zip(&status))
        .map(|(b, (v, s))| (b, Family { flags: b.queue_flags, codecs: v.video_codec_operations, status_queries: s.query_result_status_support == vk::TRUE }))
        .collect()
}

/// What makes a physical device usable.
struct Suitable {
    score: u32,
    decode_family: u32,
    copy_family: u32,
    status_queries: bool,
    codecs: vk::VideoCodecOperationFlagsKHR,
}

/// A physical device that can decode H.264 or HEVC, or why not.
fn suitable(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> R<Suitable> {
    // SAFETY: `pdev` was enumerated from this live instance.
    let props = unsafe { instance.get_physical_device_properties(pdev) };
    if props.api_version < vk::API_VERSION_1_3 {
        return Err(format!("Vulkan {} device (1.3 needed)", version(props.api_version)));
    }
    let score = match props.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
        _ => return Err("not a GPU".into()),
    };
    // SAFETY: `pdev` was enumerated from this live instance.
    let exts = unsafe { instance.enumerate_device_extension_properties(pdev) }.map_err(vk_err("vkEnumerateDeviceExtensionProperties"))?;
    let has = |want: &CStr| exts.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == want));
    for want in EXTENSIONS {
        if !has(want) {
            return Err(format!("no {}", want.to_string_lossy()));
        }
    }
    let usable = CODECS.iter().filter(|(_, ext)| has(ext)).fold(vk::VideoCodecOperationFlagsKHR::empty(), |a, (c, _)| a | *c);
    if usable.is_empty() {
        return Err("no H.264 or HEVC decode extension".into());
    }
    // the video queue structures are only queried once the device has the extension
    let fams = queue_families(instance, pdev);
    // the decode family taking the most of our codecs
    let decode = fams
        .iter()
        .enumerate()
        .filter(|(_, (_, f))| f.flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) && f.codecs.intersects(usable))
        .max_by_key(|(_, (_, f))| (f.codecs & usable).as_raw().count_ones())
        .map(|(i, _)| i)
        .ok_or("no video decode queue for H.264 or HEVC")?;
    let codecs = fams.get(decode).map(|(_, f)| f.codecs & usable & any_codec()).unwrap_or_default();
    let can_copy = |f: &Family| f.flags.intersects(vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
    let copy =
        if fams.get(decode).is_some_and(|(_, f)| can_copy(f)) { decode } else { fams.iter().position(|(_, f)| can_copy(f)).ok_or("no transfer queue")? };
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    {
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut f13);
        // SAFETY: the pNext chain points at `f12` / `f13`, which outlive the call.
        unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    }
    if f12.timeline_semaphore != vk::TRUE || f13.synchronization2 != vk::TRUE {
        return Err("no timeline semaphores / synchronization2".into());
    }
    let status_queries = fams.get(decode).is_some_and(|(_, f)| f.status_queries);
    let as_u32 = |i: usize| u32::try_from(i).map_err(|_| "queue family index overflows".to_string());
    Ok(Suitable { score: score * 4 + codecs.as_raw().count_ones(), decode_family: as_u32(decode)?, copy_family: as_u32(copy)?, status_queries, codecs })
}

impl Device {
    /// Open the best GPU that decodes H.264 or HEVC through Vulkan Video, or say why there is none.
    pub(crate) fn open() -> R<Device> {
        // SAFETY: see `loader_available`; the `Entry` is kept in the returned `Device`, so the
        // library stays loaded while any handle created from it exists.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader: {e}"))?;
        // SAFETY: a plain query through the loaded entry.
        let loader = unsafe { entry.try_enumerate_instance_version() }.map_err(vk_err("vkEnumerateInstanceVersion"))?.unwrap_or(vk::API_VERSION_1_0);
        if loader < vk::API_VERSION_1_3 {
            return Err(format!("Vulkan {} loader (1.3 needed)", version(loader)));
        }
        let app = vk::ApplicationInfo::default().application_name(c"FilmCraft").engine_name(c"FilmCraft").api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: valid create info whose pointers outlive the call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(vk_err("vkCreateInstance"))?;
        match Self::open_device(&entry, &instance) {
            Ok(parts) => Ok(Self::assemble(entry, instance, parts)),
            Err(e) => {
                // SAFETY: nothing created from the instance is alive (open_device cleans up).
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    fn open_device(entry: &ash::Entry, instance: &ash::Instance) -> R<OpenedDevice> {
        for f in INSTANCE_FNS {
            // SAFETY: a valid instance handle and a NUL-terminated name.
            if unsafe { entry.get_instance_proc_addr(instance.handle(), f.as_ptr()) }.is_none() {
                return Err(format!("the Vulkan loader lacks {}", f.to_string_lossy()));
            }
        }
        // SAFETY: the instance is live.
        let pdevs = unsafe { instance.enumerate_physical_devices() }.map_err(vk_err("vkEnumeratePhysicalDevices"))?;
        let mut best: Option<(Suitable, vk::PhysicalDevice)> = None;
        let mut why = Vec::new();
        for pdev in pdevs {
            match suitable(instance, pdev) {
                Ok(s) if best.as_ref().is_none_or(|b| s.score > b.0.score) => best = Some((s, pdev)),
                Ok(_) => {}
                Err(e) => why.push(e),
            }
        }
        let Some((Suitable { decode_family, copy_family, status_queries, codecs, .. }, pdev)) = best else {
            return Err(if why.is_empty() { "no GPU".to_string() } else { format!("no GPU decodes H.264 or HEVC through Vulkan Video ({})", why.join("; ")) });
        };
        let prio = [1.0f32];
        let mut queues = vec![vk::DeviceQueueCreateInfo::default().queue_family_index(decode_family).queue_priorities(&prio)];
        if copy_family != decode_family {
            queues.push(vk::DeviceQueueCreateInfo::default().queue_family_index(copy_family).queue_priorities(&prio));
        }
        let ext_names: Vec<*const c_char> =
            EXTENSIONS.iter().map(|n| n.as_ptr()).chain(CODECS.iter().filter(|(c, _)| codecs.contains(*c)).map(|(_, n)| n.as_ptr())).collect();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&ext_names).push_next(&mut f12).push_next(&mut f13);
        // SAFETY: `pdev` belongs to this instance; the create info and everything it points at
        // outlive the call; the requested extensions and features were checked to be supported.
        let device = unsafe { instance.create_device(pdev, &info, None) }.map_err(vk_err("vkCreateDevice"))?;
        for f in DEVICE_FNS {
            // SAFETY: a valid device handle and a NUL-terminated name.
            if unsafe { instance.get_device_proc_addr(device.handle(), f.as_ptr()) }.is_none() {
                // SAFETY: nothing was created from the device yet.
                unsafe { device.destroy_device(None) };
                return Err(format!("the driver lacks {}", f.to_string_lossy()));
            }
        }
        // SAFETY: both families were requested with one queue each at device creation.
        let decode_queue = unsafe { device.get_device_queue(decode_family, 0) };
        // SAFETY: as above.
        let copy_queue = (copy_family != decode_family).then(|| unsafe { device.get_device_queue(copy_family, 0) });
        // SAFETY: `pdev` belongs to this instance.
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        // SAFETY: as above.
        let mem = unsafe { instance.get_physical_device_memory_properties(pdev) };
        let name = props.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|_| "GPU".into());
        Ok(OpenedDevice { pdev, device, decode_family, copy_family, decode_queue, copy_queue, mem, status_queries, codecs, name })
    }

    fn assemble(entry: ash::Entry, instance: ash::Instance, d: OpenedDevice) -> Device {
        let video_instance = ash::khr::video_queue::Instance::new(&entry, &instance);
        let video = ash::khr::video_queue::Device::new(&instance, &d.device);
        let decode = ash::khr::video_decode_queue::Device::new(&instance, &d.device);
        Device {
            _entry: entry,
            instance,
            pdev: d.pdev,
            device: d.device,
            video_instance,
            video,
            decode,
            decode_family: d.decode_family,
            copy_family: d.copy_family,
            decode_queue: Mutex::new(d.decode_queue),
            copy_queue: d.copy_queue.map(Mutex::new),
            mem: d.mem,
            status_queries: d.status_queries,
            codecs: d.codecs,
            name: d.name,
            lost: AtomicBool::new(false),
        }
    }

    /// Whether the device was lost (every session fails; new ones are declined).
    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }

    /// Which of H.264 / HEVC the device decodes.
    pub(crate) fn decodes(&self, codec: &Codec) -> bool {
        self.codecs.contains(codec.operation())
    }

    /// The codecs the device decodes, for logs and `probe` ("H.264 and HEVC").
    pub(crate) fn codec_names(&self) -> String {
        let names: Vec<&str> = [(vk::VideoCodecOperationFlagsKHR::DECODE_H264, "H.264"), (vk::VideoCodecOperationFlagsKHR::DECODE_H265, "HEVC")]
            .into_iter()
            .filter(|(c, _)| self.codecs.contains(*c))
            .map(|(_, n)| n)
            .collect();
        names.join(" and ")
    }

    /// The first memory type in `bits` with all of `want`.
    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<(u32, vk::MemoryPropertyFlags)> {
        let count = self.mem.memory_type_count.min(vk::MAX_MEMORY_TYPES as u32);
        (0..count).find_map(|i| {
            let t = self.mem.memory_types.get(i as usize)?;
            (bits & (1 << i) != 0 && t.property_flags.contains(want)).then_some((i, t.property_flags))
        })
    }

    /// Allocate memory for `req` from the first preference that has a type and succeeds.
    fn allocate(&self, req: vk::MemoryRequirements, prefs: &[vk::MemoryPropertyFlags]) -> R<(vk::DeviceMemory, vk::MemoryPropertyFlags)> {
        let mut last = "no suitable memory type".to_string();
        for want in prefs {
            let Some((index, flags)) = self.memory_type(req.memory_type_bits, *want) else { continue };
            let info = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index);
            // SAFETY: a valid allocate info for this device's memory type.
            match unsafe { self.device.allocate_memory(&info, None) } {
                Ok(m) => return Ok((m, flags)),
                Err(e) => last = format!("vkAllocateMemory: {e}"),
            }
        }
        Err(last)
    }

    /// Submit `cmd` to `queue`, after timeline value `wait`, signalling `signal`.
    fn submit(&self, queue: &Mutex<vk::Queue>, cmd: vk::CommandBuffer, timeline: vk::Semaphore, wait: u64, signal: u64) -> R<()> {
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let waits = [vk::SemaphoreSubmitInfo::default().semaphore(timeline).value(wait).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let signals = [vk::SemaphoreSubmitInfo::default().semaphore(timeline).value(signal).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit =
            vk::SubmitInfo2::default().command_buffer_infos(&cmds).wait_semaphore_infos(if wait > 0 { &waits } else { &[] }).signal_semaphore_infos(&signals);
        let q = queue.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: the queue is externally synchronised by its mutex; `cmd` was recorded and ended
        // and is not pending (every earlier submission was waited for); all handles are live.
        let r = unsafe { self.device.queue_submit2(*q, &[submit], vk::Fence::null()) };
        r.map_err(|e| self.failed("vkQueueSubmit2", e))
    }

    /// Wait on the host until `timeline` reaches `value`.
    fn wait(&self, timeline: vk::Semaphore, value: u64) -> Result<(), Wait> {
        let sems = [timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values);
        // SAFETY: a live timeline semaphore of this device.
        match unsafe { self.device.wait_semaphores(&info, TIMEOUT_NS) } {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => Err(Wait::Hung("the GPU did not finish decoding within 2 s".into())),
            Err(e) => Err(Wait::Hung(self.failed("vkWaitSemaphores", e))),
        }
    }

    /// Note a failed call (device loss marks the device lost).
    fn failed(&self, what: &str, e: vk::Result) -> String {
        if e == vk::Result::ERROR_DEVICE_LOST {
            self.lost.store(true, Ordering::Relaxed);
        }
        format!("{what}: {e}")
    }
}

/// What `open_device` made, before the function tables are loaded.
struct OpenedDevice {
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    decode_family: u32,
    copy_family: u32,
    decode_queue: vk::Queue,
    copy_queue: Option<vk::Queue>,
    mem: vk::PhysicalDeviceMemoryProperties,
    status_queries: bool,
    codecs: vk::VideoCodecOperationFlagsKHR,
    name: String,
}

/// A host wait that did not complete: the session's GPU work is in an unknown state.
enum Wait {
    Hung(String),
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: every session holds an `Arc<Device>`, so none is alive; after an idle wait no work
        // is pending. A lost device needs no wait (it never completes) and is destroyed as is.
        unsafe {
            if !self.is_lost() {
                let _ = self.device.device_wait_idle();
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The codec and profile of a session (progressive 4:2:0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    /// 8-bit H.264.
    H264(StdVideoH264ProfileIdc),
    /// HEVC at 8 or 10 bits.
    H265(StdVideoH265ProfileIdc, u32),
}

impl Codec {
    fn operation(&self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            Codec::H264(_) => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Codec::H265(..) => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
        }
    }
    fn bit_depth(&self) -> u32 {
        match self {
            Codec::H264(_) => 8,
            Codec::H265(_, d) => *d,
        }
    }
    /// The decoded picture format: NV12, or P010 (10 bits in the high bits of 16).
    fn format(&self) -> vk::Format {
        if self.bit_depth() > 8 { vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 } else { vk::Format::G8_B8R8_2PLANE_420_UNORM }
    }
    fn std_header(&self) -> (&'static CStr, u32) {
        match self {
            Codec::H264(_) => (h264::STD_HEADER_NAME, h264::STD_HEADER_VERSION),
            Codec::H265(..) => (h265::STD_HEADER_NAME, h265::STD_HEADER_VERSION),
        }
    }
}

/// What a stream's session needs.
pub(crate) struct SessionSpec {
    pub(crate) codec: Codec,
    /// The std level (`StdVideoH264LevelIdc` / `StdVideoH265LevelIdc`).
    pub(crate) level: u32,
    /// Coded picture size in luma samples (multiples of the minimum block size).
    pub(crate) coded: (u32, u32),
    /// DPB slots needed (the DPB size + 1).
    pub(crate) slots: u32,
    /// Reference pictures a picture may use.
    pub(crate) max_refs: u32,
}

/// A stream's std parameter sets.
#[derive(Clone, Copy)]
pub(crate) enum Params<'a> {
    H264(&'a h264::ParameterSets),
    H265(&'a h265::ParameterSets),
}

/// A picture's std decode info.
#[derive(Clone, Copy)]
pub(crate) enum PictureInfo {
    H264(StdVideoDecodeH264PictureInfo),
    H265(StdVideoDecodeH265PictureInfo),
}

/// A DPB slot's std reference info.
#[derive(Clone, Copy)]
pub(crate) enum RefInfo {
    H264(StdVideoDecodeH264ReferenceInfo),
    H265(StdVideoDecodeH265ReferenceInfo),
}

/// One picture to decode, in the session's terms.
pub(crate) struct DecodeJob<'a> {
    /// Slice (segment) NAL units, each after a start code.
    pub(crate) bitstream: &'a [u8],
    pub(crate) slice_offsets: &'a [u32],
    pub(crate) picture: PictureInfo,
    /// DPB slot the picture is decoded into, and how it is stored for reference.
    pub(crate) setup: (u32, RefInfo),
    /// Reference pictures: their slots and reference info.
    pub(crate) refs: &'a [(u32, RefInfo)],
    /// First decode after creation or a seek: reset the session's DPB state.
    pub(crate) reset: bool,
}

/// A host-visible buffer, persistently mapped.
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    map: *mut u8,
    size: u64,
    coherent: bool,
}

impl HostBuffer {
    const NULL: HostBuffer = HostBuffer { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), map: std::ptr::null_mut(), size: 0, coherent: true };
}

/// A video session for one stream: session and parameters, the layered DPB image (one array
/// layer per slot, also the decode output), the bitstream and read-back buffers, command buffers,
/// the timeline semaphore and an optional decode-status query.
pub(crate) struct Session {
    dev: Arc<Device>,
    codec: Codec,
    session: vk::VideoSessionKHR,
    session_memory: Vec<vk::DeviceMemory>,
    params: vk::VideoSessionParametersKHR,
    format: vk::Format,
    /// Image extent (coded size aligned to the picture access granularity) and coded extent.
    extent: vk::Extent2D,
    coded: vk::Extent2D,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    view: vk::ImageView,
    layouts: Vec<vk::ImageLayout>,
    bitstream: HostBuffer,
    bitstream_align: (u64, u64),
    readback: HostBuffer,
    decode_pool: vk::CommandPool,
    decode_cmd: vk::CommandBuffer,
    copy_pool: vk::CommandPool,
    copy_cmd: vk::CommandBuffer,
    timeline: vk::Semaphore,
    value: u64,
    query: vk::QueryPool,
    /// A wait timed out or the device was lost: GPU work may still use the session's objects, so
    /// they are leaked instead of destroyed.
    hung: bool,
}

// SAFETY: the Vulkan handles are plain identifiers usable from any thread; the session's objects
// are externally synchronised through `&mut self` (queues have their own mutexes); the mapped
// pointers are only dereferenced through `&mut self` while the GPU does not use the buffers.
unsafe impl Send for Session {}

/// Run `f` with the stream's video profile (progressive 4:2:0).
fn with_profile<T>(codec: Codec, f: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> T) -> T {
    let depth = if codec.bit_depth() > 8 { vk::VideoComponentBitDepthFlagsKHR::TYPE_10 } else { vk::VideoComponentBitDepthFlagsKHR::TYPE_8 };
    let base = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(codec.operation())
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(depth)
        .chroma_bit_depth(depth);
    match codec {
        Codec::H264(idc) => {
            let mut h264 =
                vk::VideoDecodeH264ProfileInfoKHR::default().std_profile_idc(idc).picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
            f(&base.push_next(&mut h264))
        }
        Codec::H265(idc, _) => {
            let mut h265 = vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(idc);
            f(&base.push_next(&mut h265))
        }
    }
}

/// What the driver reports for a profile.
struct Caps {
    decode_flags: vk::VideoDecodeCapabilityFlagsKHR,
    offset_align: u64,
    size_align: u64,
    granularity: vk::Extent2D,
    min_extent: vk::Extent2D,
    max_extent: vk::Extent2D,
    max_slots: u32,
    max_refs: u32,
    max_level: u32,
    header: vk::ExtensionProperties,
}

fn capabilities(dev: &Device, profile: &vk::VideoProfileInfoKHR<'_>, codec: Codec) -> R<Caps> {
    let mut h264 = vk::VideoDecodeH264CapabilitiesKHR::default();
    let mut h265 = vk::VideoDecodeH265CapabilitiesKHR::default();
    let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default().push_next(&mut decode);
    caps = match codec {
        Codec::H264(_) => caps.push_next(&mut h264),
        Codec::H265(..) => caps.push_next(&mut h265),
    };
    // SAFETY: the function pointer was checked at device creation; the profile chain and the
    // capability chain point at live locals for the duration of the call.
    let r = unsafe { (dev.video_instance.fp().get_physical_device_video_capabilities_khr)(dev.pdev, profile, &mut caps) };
    if r != vk::Result::SUCCESS {
        return Err(format!("profile not supported for decoding ({r})"));
    }
    let c = Caps {
        decode_flags: vk::VideoDecodeCapabilityFlagsKHR::empty(),
        offset_align: caps.min_bitstream_buffer_offset_alignment.max(1),
        size_align: caps.min_bitstream_buffer_size_alignment.max(1),
        granularity: caps.picture_access_granularity,
        min_extent: caps.min_coded_extent,
        max_extent: caps.max_coded_extent,
        max_slots: caps.max_dpb_slots,
        max_refs: caps.max_active_reference_pictures,
        max_level: 0,
        header: caps.std_header_version,
    };
    let max_level = match codec {
        Codec::H264(_) => h264.max_level_idc,
        Codec::H265(..) => h265.max_level_idc,
    };
    Ok(Caps { decode_flags: decode.flags, max_level, ..c })
}

/// The image parameters the driver gives `format` for `usage` with this profile, if it offers it.
fn video_format(
    dev: &Device,
    profile: &vk::VideoProfileInfoKHR<'_>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> R<Option<(vk::ImageCreateFlags, vk::ImageTiling)>> {
    let profiles = [*profile];
    let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
    let info = vk::PhysicalDeviceVideoFormatInfoKHR::default().image_usage(usage).push_next(&mut list);
    let fp = dev.video_instance.fp().get_physical_device_video_format_properties_khr;
    let mut n = 0u32;
    // SAFETY: function pointer checked at device creation; a count query with a null array.
    let r = unsafe { fp(dev.pdev, &info, &mut n, std::ptr::null_mut()) };
    if r == vk::Result::ERROR_FORMAT_NOT_SUPPORTED || (r == vk::Result::SUCCESS && n == 0) {
        return Ok(None);
    }
    if r != vk::Result::SUCCESS {
        return Err(format!("vkGetPhysicalDeviceVideoFormatPropertiesKHR: {r}"));
    }
    let mut props = vec![vk::VideoFormatPropertiesKHR::default(); n.min(64) as usize];
    let mut n = props.len() as u32;
    // SAFETY: `props` holds `n` initialised structures.
    let r = unsafe { fp(dev.pdev, &info, &mut n, props.as_mut_ptr()) };
    if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
        return Err(format!("vkGetPhysicalDeviceVideoFormatPropertiesKHR: {r}"));
    }
    props.truncate(n as usize);
    Ok(props.iter().find(|p| p.format == format && p.image_usage_flags.contains(usage)).map(|p| (p.image_create_flags, p.image_tiling)))
}

fn align_up(v: u64, a: u64) -> Option<u64> {
    let a = a.max(1);
    v.checked_add(a - 1).map(|x| x / a * a)
}

impl Session {
    /// A session for a stream, or why the GPU cannot decode it.
    pub(crate) fn new(dev: Arc<Device>, spec: &SessionSpec, params: Params<'_>) -> R<Session> {
        if dev.is_lost() {
            return Err("the GPU was lost".into());
        }
        if !dev.decodes(&spec.codec) {
            return Err(format!("{} does not decode this codec through Vulkan Video", dev.name));
        }
        let mut s = Session {
            dev: dev.clone(),
            codec: spec.codec,
            session: vk::VideoSessionKHR::null(),
            session_memory: Vec::new(),
            params: vk::VideoSessionParametersKHR::null(),
            format: spec.codec.format(),
            extent: vk::Extent2D::default(),
            coded: vk::Extent2D { width: spec.coded.0, height: spec.coded.1 },
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            layouts: Vec::new(),
            bitstream: HostBuffer::NULL,
            bitstream_align: (1, 1),
            readback: HostBuffer::NULL,
            decode_pool: vk::CommandPool::null(),
            decode_cmd: vk::CommandBuffer::null(),
            copy_pool: vk::CommandPool::null(),
            copy_cmd: vk::CommandBuffer::null(),
            timeline: vk::Semaphore::null(),
            value: 0,
            query: vk::QueryPool::null(),
            hung: false,
        };
        // On an error `s` is dropped, which destroys whatever was created so far.
        with_profile(spec.codec, |profile| s.create(profile, spec, params))?;
        Ok(s)
    }

    fn create(&mut self, profile: &vk::VideoProfileInfoKHR<'_>, spec: &SessionSpec, params: Params<'_>) -> R<()> {
        let dev = self.dev.clone();
        let caps = capabilities(&dev, profile, spec.codec)?;
        let (w, h) = spec.coded;
        if w < caps.min_extent.width || h < caps.min_extent.height || w > caps.max_extent.width || h > caps.max_extent.height {
            return Err(format!(
                "{w}x{h} is outside the decoder's {}x{} to {}x{}",
                caps.min_extent.width, caps.min_extent.height, caps.max_extent.width, caps.max_extent.height
            ));
        }
        if spec.level > caps.max_level {
            return Err("level above the decoder's maximum".into());
        }
        if spec.slots > caps.max_slots || spec.max_refs > caps.max_refs || spec.slots < 2 {
            return Err(format!("needs {} DPB slots / {} references, the decoder has {} / {}", spec.slots, spec.max_refs, caps.max_slots, caps.max_refs));
        }
        let (header, version) = spec.codec.std_header();
        if caps.header.extension_name_as_c_str().ok() != Some(header) || caps.header.spec_version < version {
            return Err("unexpected std header from the driver".into());
        }
        if !caps.decode_flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE) {
            return Err("decode output separate from the DPB is not supported yet".into());
        }
        let usage = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR | vk::ImageUsageFlags::TRANSFER_SRC;
        let (create_flags, tiling) = video_format(&dev, profile, self.format, usage)?.ok_or("no NV12 / P010 decode format that can be copied out")?;
        let gw = caps.granularity.width.max(1);
        let gh = caps.granularity.height.max(1);
        self.extent = vk::Extent2D { width: w.div_ceil(gw).saturating_mul(gw), height: h.div_ceil(gh).saturating_mul(gh) };
        if self.extent.width > caps.max_extent.width
            || self.extent.height > caps.max_extent.height
            || !self.extent.width.is_multiple_of(2)
            || !self.extent.height.is_multiple_of(2)
        {
            return Err("aligned picture size outside the decoder's limits".into());
        }
        self.bitstream_align = (caps.offset_align, caps.size_align);
        self.create_session(profile, spec, &caps)?;
        self.create_parameters(params)?;
        self.create_image(profile, spec.slots, create_flags, tiling, usage)?;
        let bits = align_up(u64::from(w) * u64::from(h), caps.size_align).ok_or("bitstream size overflows")?.max(1 << 20);
        self.bitstream = self.host_buffer(
            bits,
            vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR,
            Some(profile),
            &[vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, vk::MemoryPropertyFlags::HOST_VISIBLE],
        )?;
        let frame = u64::from(self.extent.width) * u64::from(self.extent.height) * 3 / 2 * self.bytes_per_sample();
        self.readback = self.host_buffer(
            frame,
            vk::BufferUsageFlags::TRANSFER_DST,
            None,
            &[
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ],
        )?;
        (self.decode_pool, self.decode_cmd) = self.command_buffer(dev.decode_family)?;
        (self.copy_pool, self.copy_cmd) = self.command_buffer(dev.copy_family)?;
        let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: a valid create info; the semaphore is destroyed in `Drop`.
        self.timeline = unsafe { dev.device.create_semaphore(&info, None) }.map_err(vk_err("vkCreateSemaphore"))?;
        if dev.status_queries {
            let mut p = *profile;
            let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::RESULT_STATUS_ONLY_KHR).query_count(1).push_next(&mut p);
            // SAFETY: a valid create info whose profile chain points at live locals.
            self.query = unsafe { dev.device.create_query_pool(&info, None) }.map_err(vk_err("vkCreateQueryPool"))?;
        }
        Ok(())
    }

    fn create_session(&mut self, profile: &vk::VideoProfileInfoKHR<'_>, spec: &SessionSpec, caps: &Caps) -> R<()> {
        let dev = self.dev.clone();
        let (name, version) = spec.codec.std_header();
        let header = vk::ExtensionProperties::default().extension_name(name).map_err(|_| "std header name too long".to_string())?.spec_version(version);
        let max_refs = spec.max_refs.max(1).min(caps.max_refs);
        let info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(dev.decode_family)
            .video_profile(profile)
            .picture_format(self.format)
            .max_coded_extent(self.extent)
            .reference_picture_format(self.format)
            .max_dpb_slots(spec.slots)
            .max_active_reference_pictures(max_refs)
            .std_header_version(&header);
        let fp = dev.video.fp();
        // SAFETY: function pointer checked at device creation; the create info and the structures
        // it points at outlive the call; the session is destroyed in `Drop`.
        let r = unsafe { (fp.create_video_session_khr)(dev.device.handle(), &info, std::ptr::null(), &mut self.session) };
        if r != vk::Result::SUCCESS {
            self.session = vk::VideoSessionKHR::null();
            return Err(format!("vkCreateVideoSessionKHR: {r}"));
        }
        let mut n = 0u32;
        // SAFETY: a count query on the live session.
        let r = unsafe { (fp.get_video_session_memory_requirements_khr)(dev.device.handle(), self.session, &mut n, std::ptr::null_mut()) };
        if r != vk::Result::SUCCESS {
            return Err(format!("vkGetVideoSessionMemoryRequirementsKHR: {r}"));
        }
        let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); n.min(64) as usize];
        let mut n = reqs.len() as u32;
        // SAFETY: `reqs` holds `n` initialised structures.
        let r = unsafe { (fp.get_video_session_memory_requirements_khr)(dev.device.handle(), self.session, &mut n, reqs.as_mut_ptr()) };
        if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
            return Err(format!("vkGetVideoSessionMemoryRequirementsKHR: {r}"));
        }
        reqs.truncate(n as usize);
        let mut binds = Vec::with_capacity(reqs.len());
        for r in &reqs {
            let (memory, _) = dev.allocate(r.memory_requirements, &[vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty()])?;
            self.session_memory.push(memory);
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(r.memory_bind_index)
                    .memory(memory)
                    .memory_offset(0)
                    .memory_size(r.memory_requirements.size),
            );
        }
        if !binds.is_empty() {
            // SAFETY: each bind names memory allocated for that requirement, bound once.
            let r = unsafe { (fp.bind_video_session_memory_khr)(dev.device.handle(), self.session, binds.len() as u32, binds.as_ptr()) };
            if r != vk::Result::SUCCESS {
                return Err(format!("vkBindVideoSessionMemoryKHR: {r}"));
            }
        }
        Ok(())
    }

    fn create_parameters(&mut self, params: Params<'_>) -> R<()> {
        let dev = self.dev.clone();
        let count = |n: usize| u32::try_from(n).map_err(|_| "too many parameter sets".to_string());
        let h264_add;
        let h265_add;
        let mut h264;
        let mut h265;
        let mut info = vk::VideoSessionParametersCreateInfoKHR::default().video_session(self.session);
        match params {
            Params::H264(sets) => {
                h264_add = vk::VideoDecodeH264SessionParametersAddInfoKHR::default().std_sp_ss(&sets.sps).std_pp_ss(&sets.pps);
                h264 = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
                    .max_std_sps_count(count(sets.sps.len())?)
                    .max_std_pps_count(count(sets.pps.len())?)
                    .parameters_add_info(&h264_add);
                info = info.push_next(&mut h264);
            }
            Params::H265(sets) => {
                h265_add = vk::VideoDecodeH265SessionParametersAddInfoKHR::default().std_vp_ss(&sets.vps).std_sp_ss(&sets.sps).std_pp_ss(&sets.pps);
                h265 = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
                    .max_std_vps_count(count(sets.vps.len())?)
                    .max_std_sps_count(count(sets.sps.len())?)
                    .max_std_pps_count(count(sets.pps.len())?)
                    .parameters_add_info(&h265_add);
                info = info.push_next(&mut h265);
            }
        }
        // SAFETY: function pointer checked at device creation; the std structures and the storage
        // their pointers point into (`params`) outlive the call; the driver copies what it keeps.
        let r = unsafe { (dev.video.fp().create_video_session_parameters_khr)(dev.device.handle(), &info, std::ptr::null(), &mut self.params) };
        if r != vk::Result::SUCCESS {
            self.params = vk::VideoSessionParametersKHR::null();
            return Err(format!("vkCreateVideoSessionParametersKHR: {r}"));
        }
        Ok(())
    }

    fn create_image(
        &mut self,
        profile: &vk::VideoProfileInfoKHR<'_>,
        layers: u32,
        flags: vk::ImageCreateFlags,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> R<()> {
        let dev = self.dev.clone();
        let profiles = [*profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let families = [dev.decode_family, dev.copy_family];
        let mut info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(self.format)
            .extent(vk::Extent3D { width: self.extent.width, height: self.extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut list);
        info = if dev.copy_family != dev.decode_family {
            info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        // SAFETY: a valid create info whose pointers outlive the call; destroyed in `Drop`.
        self.image = unsafe { dev.device.create_image(&info, None) }.map_err(vk_err("vkCreateImage"))?;
        // SAFETY: the image is live.
        let req = unsafe { dev.device.get_image_memory_requirements(self.image) };
        let (memory, _) = dev.allocate(req, &[vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty()])?;
        self.image_memory = memory;
        // SAFETY: memory allocated for this image's requirements, bound once at offset 0.
        unsafe { dev.device.bind_image_memory(self.image, memory, 0) }.map_err(vk_err("vkBindImageMemory"))?;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(layers);
        let view =
            vk::ImageViewCreateInfo::default().image(self.image).view_type(vk::ImageViewType::TYPE_2D_ARRAY).format(self.format).subresource_range(range);
        // SAFETY: a view of the live image covering its layers; destroyed in `Drop`.
        self.view = unsafe { dev.device.create_image_view(&view, None) }.map_err(vk_err("vkCreateImageView"))?;
        self.layouts = vec![vk::ImageLayout::UNDEFINED; layers as usize];
        Ok(())
    }

    /// A persistently mapped host-visible buffer (the video profile in its chain for bitstream
    /// buffers). The caller stores it in `self`, whose `Drop` frees it.
    fn host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        profile: Option<&vk::VideoProfileInfoKHR<'_>>,
        prefs: &[vk::MemoryPropertyFlags],
    ) -> R<HostBuffer> {
        let dev = &self.dev;
        let profiles: Vec<vk::VideoProfileInfoKHR<'_>> = profile.into_iter().copied().collect();
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let mut info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        if !profiles.is_empty() {
            info = info.push_next(&mut list);
        }
        // SAFETY: a valid create info whose pointers outlive the call.
        let buffer = unsafe { dev.device.create_buffer(&info, None) }.map_err(vk_err("vkCreateBuffer"))?;
        let mut out = HostBuffer { buffer, ..HostBuffer::NULL };
        let made = (|| {
            // SAFETY: the buffer is live.
            let req = unsafe { dev.device.get_buffer_memory_requirements(buffer) };
            let (memory, flags) = dev.allocate(req, prefs)?;
            out.memory = memory;
            out.coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            // SAFETY: memory allocated for this buffer's requirements, bound once at offset 0.
            unsafe { dev.device.bind_buffer_memory(buffer, memory, 0) }.map_err(vk_err("vkBindBufferMemory"))?;
            // SAFETY: host-visible memory, mapped once for the buffer's lifetime (unmapped when freed).
            let map = unsafe { dev.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.map_err(vk_err("vkMapMemory"))?;
            out.map = map.cast::<u8>();
            out.size = size;
            R::Ok(())
        })();
        match made {
            Ok(()) => Ok(out),
            Err(e) => {
                destroy_buffer(&dev.device, &out);
                Err(e)
            }
        }
    }

    fn command_buffer(&self, family: u32) -> R<(vk::CommandPool, vk::CommandBuffer)> {
        let dev = &self.dev;
        let info = vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER).queue_family_index(family);
        // SAFETY: a valid create info; the pool is destroyed in `Drop` (or below on error).
        let pool = unsafe { dev.device.create_command_pool(&info, None) }.map_err(vk_err("vkCreateCommandPool"))?;
        let alloc = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        // SAFETY: allocating from the live pool.
        match unsafe { dev.device.allocate_command_buffers(&alloc) }.map(|v| v.first().copied()) {
            Ok(Some(cmd)) => Ok((pool, cmd)),
            other => {
                // SAFETY: nothing from the pool is in use.
                unsafe { dev.device.destroy_command_pool(pool, None) };
                Err(format!("vkAllocateCommandBuffers: {:?}", other.err()))
            }
        }
    }

    /// The decoded picture size (copied-out planes are this size).
    pub(crate) fn image_size(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    /// Bytes per copied-out sample: 1 (NV12) or 2 (P010).
    pub(crate) fn bytes_per_sample(&self) -> u64 {
        if self.codec.bit_depth() > 8 { 2 } else { 1 }
    }

    fn check(&self) -> R<()> {
        if self.hung || self.dev.is_lost() {
            return Err("the hardware decoder stopped responding".into());
        }
        Ok(())
    }

    /// Submit `cmd` after every earlier submission of this session and wait for it.
    fn run(&mut self, cmd: vk::CommandBuffer, copy: bool) -> R<()> {
        let dev = self.dev.clone();
        let queue = match (&dev.copy_queue, copy) {
            (Some(q), true) => q,
            _ => &dev.decode_queue,
        };
        let next = self.value.checked_add(1).ok_or("timeline overflow")?;
        dev.submit(queue, cmd, self.timeline, self.value, next)?;
        self.value = next;
        match dev.wait(self.timeline, next) {
            Ok(()) => Ok(()),
            Err(Wait::Hung(e)) => {
                self.hung = true;
                Err(e)
            }
        }
    }

    /// Make room in the bitstream buffer for `size` bytes.
    fn reserve_bitstream(&mut self, size: u64) -> R<()> {
        if size <= self.bitstream.size {
            return Ok(());
        }
        if size > MAX_BITSTREAM {
            return Err(format!("picture bitstream of {size} bytes"));
        }
        let want = align_up(size.saturating_mul(2).min(MAX_BITSTREAM), self.bitstream_align.1).ok_or("bitstream size overflows")?;
        let buffer = with_profile(self.codec, |profile| {
            self.host_buffer(
                want,
                vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR,
                Some(profile),
                &[vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, vk::MemoryPropertyFlags::HOST_VISIBLE],
            )
        })?;
        // The old buffer is idle: every submission was waited for.
        destroy_buffer(&self.dev.device, &self.bitstream);
        self.bitstream = buffer;
        Ok(())
    }

    /// Decode one picture into its DPB slot.
    pub(crate) fn decode(&mut self, job: &DecodeJob<'_>) -> R<()> {
        self.check()?;
        let dev = self.dev.clone();
        let len = job.bitstream.len() as u64;
        if len == 0 || job.slice_offsets.is_empty() {
            return Err("picture without slices".into());
        }
        let range = align_up(len, self.bitstream_align.1).ok_or("bitstream size overflows")?;
        self.reserve_bitstream(range)?;
        let layers = self.layouts.len() as u32;
        let setup_layer = job.setup.0;
        if setup_layer >= layers || job.refs.iter().any(|(l, _)| *l >= layers || *l == setup_layer) {
            return Err("DPB slot out of range".into());
        }
        // SAFETY: the mapped buffer holds at least `range` bytes (reserved above) and the GPU is
        // not reading it (the previous submission was waited for).
        unsafe {
            std::ptr::copy_nonoverlapping(job.bitstream.as_ptr(), self.bitstream.map, job.bitstream.len());
            std::ptr::write_bytes(self.bitstream.map.add(job.bitstream.len()), 0, (range - len) as usize);
        }
        if !self.bitstream.coherent {
            let r = vk::MappedMemoryRange::default().memory(self.bitstream.memory).offset(0).size(vk::WHOLE_SIZE);
            // SAFETY: the range lies in the mapped allocation.
            unsafe { dev.device.flush_mapped_memory_ranges(&[r]) }.map_err(vk_err("vkFlushMappedMemoryRanges"))?;
        }

        match (job.picture, job.setup.1) {
            (PictureInfo::H264(pic), RefInfo::H264(setup)) => {
                let refs = job
                    .refs
                    .iter()
                    .map(|(_, r)| match r {
                        RefInfo::H264(i) => Ok(*i),
                        RefInfo::H265(_) => Err("HEVC reference for an H.264 picture".to_string()),
                    })
                    .collect::<R<Vec<_>>>()?;
                let mut ref_dpb: Vec<vk::VideoDecodeH264DpbSlotInfoKHR<'_>> =
                    refs.iter().map(|i| vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(i)).collect();
                let mut setup_dpb = vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup);
                let mut info = vk::VideoDecodeH264PictureInfoKHR::default().std_picture_info(&pic).slice_offsets(job.slice_offsets);
                self.record_decode(job, range, &mut setup_dpb, &mut ref_dpb, &mut info)
            }
            (PictureInfo::H265(pic), RefInfo::H265(setup)) => {
                let refs = job
                    .refs
                    .iter()
                    .map(|(_, r)| match r {
                        RefInfo::H265(i) => Ok(*i),
                        RefInfo::H264(_) => Err("H.264 reference for an HEVC picture".to_string()),
                    })
                    .collect::<R<Vec<_>>>()?;
                let mut ref_dpb: Vec<vk::VideoDecodeH265DpbSlotInfoKHR<'_>> =
                    refs.iter().map(|i| vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(i)).collect();
                let mut setup_dpb = vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&setup);
                let mut info = vk::VideoDecodeH265PictureInfoKHR::default().std_picture_info(&pic).slice_segment_offsets(job.slice_offsets);
                self.record_decode(job, range, &mut setup_dpb, &mut ref_dpb, &mut info)
            }
            _ => Err("picture and reference infos of different codecs".into()),
        }
    }

    /// Record, submit and wait for one decode; `setup_dpb` / `ref_dpb` / `info` are the
    /// codec-specific structures of the setup slot, the reference slots and the picture.
    fn record_decode<D: vk::ExtendsVideoReferenceSlotInfoKHR, P: vk::ExtendsVideoDecodeInfoKHR>(
        &mut self,
        job: &DecodeJob<'_>,
        range: u64,
        setup_dpb: &mut D,
        ref_dpb: &mut [D],
        info: &mut P,
    ) -> R<()> {
        let dev = self.dev.clone();
        let setup_layer = job.setup.0;
        let resource = |layer: u32| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(self.coded)
                .base_array_layer(layer)
                .image_view_binding(self.view)
        };
        let setup_res = resource(setup_layer);
        let ref_res: Vec<vk::VideoPictureResourceInfoKHR<'_>> = job.refs.iter().map(|(l, _)| resource(*l)).collect();
        let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = ref_dpb
            .iter_mut()
            .zip(&ref_res)
            .zip(job.refs)
            .map(|((d, r), (layer, _))| vk::VideoReferenceSlotInfoKHR::default().slot_index(*layer as i32).picture_resource(r).push_next(d))
            .collect();
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default().slot_index(setup_layer as i32).picture_resource(&setup_res).push_next(setup_dpb);
        // The coding scope binds the references and, inactive (-1), the slot being set up.
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = ref_res
            .iter()
            .zip(job.refs)
            .map(|(r, (layer, _))| vk::VideoReferenceSlotInfoKHR::default().slot_index(*layer as i32).picture_resource(r))
            .collect();
        begin_slots.push(vk::VideoReferenceSlotInfoKHR::default().slot_index(-1).picture_resource(&setup_res));
        let decode = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream.buffer)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(setup_res)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&ref_slots)
            .push_next(info);

        // The slot being set up is overwritten (old contents discarded); references go back to the
        // DPB layout if they were last copied out.
        let mut barriers = vec![self.barrier(
            setup_layer,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
        )];
        for (layer, _) in job.refs {
            let old = self.layouts.get(*layer as usize).copied().unwrap_or(vk::ImageLayout::UNDEFINED);
            if old != vk::ImageLayout::VIDEO_DECODE_DPB_KHR {
                barriers.push(self.barrier(
                    *layer,
                    old,
                    vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                ));
            }
        }
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        let begin = vk::VideoBeginCodingInfoKHR::default().video_session(self.session).video_session_parameters(self.params).reference_slots(&begin_slots);
        let control = vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end = vk::VideoEndCodingInfoKHR::default();
        let cmd = self.decode_cmd;
        let video = dev.video.fp();
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is not pending (the previous submission was waited for) and its
        // pool allows resets; every structure recorded points at locals that outlive recording;
        // function pointers were checked at device creation; slots and layers were range-checked.
        unsafe {
            dev.device.begin_command_buffer(cmd, &begin_info).map_err(vk_err("vkBeginCommandBuffer"))?;
            if self.query != vk::QueryPool::null() {
                dev.device.cmd_reset_query_pool(cmd, self.query, 0, 1);
            }
            dev.device.cmd_pipeline_barrier2(cmd, &dependency);
            (video.cmd_begin_video_coding_khr)(cmd, &begin);
            if job.reset {
                (video.cmd_control_video_coding_khr)(cmd, &control);
            }
            if self.query != vk::QueryPool::null() {
                dev.device.cmd_begin_query(cmd, self.query, 0, vk::QueryControlFlags::empty());
            }
            (dev.decode.fp().cmd_decode_video_khr)(cmd, &decode);
            if self.query != vk::QueryPool::null() {
                dev.device.cmd_end_query(cmd, self.query, 0);
            }
            (video.cmd_end_video_coding_khr)(cmd, &end);
            dev.device.end_command_buffer(cmd).map_err(vk_err("vkEndCommandBuffer"))?;
        }
        self.run(cmd, false)?;
        for l in std::iter::once(setup_layer).chain(job.refs.iter().map(|(l, _)| *l)) {
            if let Some(slot) = self.layouts.get_mut(l as usize) {
                *slot = vk::ImageLayout::VIDEO_DECODE_DPB_KHR;
            }
        }
        if self.query != vk::QueryPool::null() {
            let mut status = [0i32];
            // SAFETY: one 32-bit status result of the live query, which has completed (waited).
            unsafe { dev.device.get_query_pool_results(self.query, 0, &mut status, vk::QueryResultFlags::WITH_STATUS_KHR) }
                .map_err(vk_err("vkGetQueryPoolResults"))?;
            if status[0] < 0 {
                return Err(format!("the GPU reported a decode error (status {})", status[0]));
            }
        }
        Ok(())
    }

    fn barrier(
        &self,
        layer: u32,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(layer)
            .layer_count(1);
        vk::ImageMemoryBarrier2::default()
            // chains with the timeline wait of the submission (ALL_COMMANDS)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(range)
    }

    /// Copy the picture in DPB slot `layer` to the host: the luma plane (`image_size`) followed
    /// by the interleaved chroma plane at half size, both tightly packed.
    pub(crate) fn read_picture(&mut self, layer: u32) -> R<&[u8]> {
        self.check()?;
        let dev = self.dev.clone();
        let old = *self.layouts.get(layer as usize).ok_or("DPB slot out of range")?;
        if old == vk::ImageLayout::UNDEFINED {
            return Err("DPB slot holds no picture".into());
        }
        let (w, h) = (self.extent.width, self.extent.height);
        let luma = u64::from(w) * u64::from(h) * self.bytes_per_sample();
        let size = luma * 3 / 2;
        if size > self.readback.size {
            return Err("read-back buffer too small".into());
        }
        let barrier = [self.barrier(layer, old, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ)];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barrier);
        let plane = |aspect, offset, width, height| {
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).mip_level(0).base_array_layer(layer).layer_count(1))
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width, height, depth: 1 })
        };
        let regions = [plane(vk::ImageAspectFlags::PLANE_0, 0, w, h), plane(vk::ImageAspectFlags::PLANE_1, luma, w / 2, h / 2)];
        let cmd = self.copy_cmd;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is not pending; the regions lie inside the image's layer and
        // inside the read-back buffer (size checked above); the barrier names a live image.
        unsafe {
            dev.device.begin_command_buffer(cmd, &begin_info).map_err(vk_err("vkBeginCommandBuffer"))?;
            dev.device.cmd_pipeline_barrier2(cmd, &dependency);
            dev.device.cmd_copy_image_to_buffer(cmd, self.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.readback.buffer, &regions);
            dev.device.end_command_buffer(cmd).map_err(vk_err("vkEndCommandBuffer"))?;
        }
        self.run(cmd, true)?;
        if let Some(l) = self.layouts.get_mut(layer as usize) {
            *l = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
        if !self.readback.coherent {
            let r = vk::MappedMemoryRange::default().memory(self.readback.memory).offset(0).size(vk::WHOLE_SIZE);
            // SAFETY: the range lies in the mapped allocation; the copy has completed.
            unsafe { dev.device.invalidate_mapped_memory_ranges(&[r]) }.map_err(vk_err("vkInvalidateMappedMemoryRanges"))?;
        }
        // SAFETY: the mapping holds at least `size` bytes; the copy that wrote them completed and
        // was made visible to the host; the slice borrows `self`, and only `&mut self` methods
        // submit work that could write the buffer again.
        Ok(unsafe { std::slice::from_raw_parts(self.readback.map, size as usize) })
    }
}

fn destroy_buffer(device: &ash::Device, b: &HostBuffer) {
    // SAFETY: the buffer and memory belong to `device`, are idle, and are destroyed once (callers
    // replace or drop the `HostBuffer` afterwards); null handles are skipped.
    unsafe {
        if b.memory != vk::DeviceMemory::null() {
            if !b.map.is_null() {
                device.unmap_memory(b.memory);
            }
            device.free_memory(b.memory, None);
        }
        if b.buffer != vk::Buffer::null() {
            device.destroy_buffer(b.buffer, None);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let dev = self.dev.clone();
        if !self.hung && self.value > 0 && dev.wait(self.timeline, self.value).is_err() {
            self.hung = true;
        }
        if self.hung && !dev.is_lost() {
            log::warn!("leaking a Vulkan video session that may still be in use by the GPU");
            return;
        }
        let d = &dev.device;
        destroy_buffer(d, &self.bitstream);
        destroy_buffer(d, &self.readback);
        // SAFETY: no submission of this session is pending (the last one was waited for, or the
        // device is lost); each handle is destroyed once; null handles are skipped.
        unsafe {
            if self.query != vk::QueryPool::null() {
                d.destroy_query_pool(self.query, None);
            }
            if self.timeline != vk::Semaphore::null() {
                d.destroy_semaphore(self.timeline, None);
            }
            for pool in [self.decode_pool, self.copy_pool] {
                if pool != vk::CommandPool::null() {
                    d.destroy_command_pool(pool, None);
                }
            }
            if self.view != vk::ImageView::null() {
                d.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                d.destroy_image(self.image, None);
            }
            if self.image_memory != vk::DeviceMemory::null() {
                d.free_memory(self.image_memory, None);
            }
            let video = dev.video.fp();
            if self.params != vk::VideoSessionParametersKHR::null() {
                (video.destroy_video_session_parameters_khr)(d.handle(), self.params, std::ptr::null());
            }
            if self.session != vk::VideoSessionKHR::null() {
                (video.destroy_video_session_khr)(d.handle(), self.session, std::ptr::null());
            }
            for m in &self.session_memory {
                d.free_memory(*m, None);
            }
        }
    }
}
