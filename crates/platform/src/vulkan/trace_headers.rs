//! ffmpeg's `trace_headers` bitstream filter output, parsed for the std-structure tests (an
//! external oracle: ffmpeg runs as a separate program, never linked or shipped). Each parameter set
//! and slice header becomes a block of syntax elements with their bit positions and values.

use std::path::Path;

/// One syntax element: its bit position in the NAL unit's RBSP, its name (with any `[i]` indices)
/// and its value.
#[derive(Clone, Debug)]
pub(crate) struct Field {
    pub(crate) pos: u64,
    pub(crate) name: String,
    pub(crate) value: i64,
}

/// A block of the trace: its title ("Sequence Parameter Set", "Slice Segment Header", …) and its
/// syntax elements in bitstream order.
#[derive(Debug)]
pub(crate) struct Block {
    pub(crate) title: String,
    pub(crate) fields: Vec<Field>,
}

/// The header trace of the elementary stream at `path`, in bitstream order.
pub(crate) fn trace(ff: &Path, path: &Path) -> Vec<Block> {
    let out = std::process::Command::new(ff)
        .args(["-hide_banner", "-loglevel", "info", "-i"])
        .arg(path)
        .args(["-c:v", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stderr);
    let mut blocks: Vec<Block> = Vec::new();
    for line in text.lines() {
        let Some((_, body)) = line.strip_prefix("[trace_headers @ ").and_then(|r| r.split_once("] ")) else { continue };
        let body = body.trim();
        let field = body.split_once(" = ").and_then(|(lhs, v)| {
            let mut t = lhs.split_whitespace();
            let pos = t.next()?.parse::<u64>().ok()?;
            Some((pos, t.next().unwrap_or_default().to_string(), v.trim().parse::<i64>().ok()))
        });
        match field {
            Some((pos, name, Some(value))) => {
                if let Some(b) = blocks.last_mut() {
                    b.fields.push(Field { pos, name, value });
                }
            }
            Some((_, _, None)) => {}
            None => blocks.push(Block { title: body.to_string(), fields: Vec::new() }),
        }
    }
    blocks
}

/// The value of syntax element `name`, or `absent` when it is not coded.
pub(crate) fn get(fields: &[Field], name: &str, absent: i64) -> i64 {
    fields.iter().find(|f| f.name == name).map_or(absent, |f| f.value)
}
