//! DPB slots of a hardware decoding session, assigned from the front-end's events
//! (`filmcraft_h264::accel`): a picture keeps its slot while the front-end lists it in the decoded
//! picture buffer of a later picture; every other slot can be reused. Outputs are copied out as
//! soon as their event arrives, so a picture waiting for output is always still in its slot.

/// Which picture each slot holds.
#[derive(Clone, Debug)]
pub(crate) struct Slots {
    held: Vec<Option<u32>>,
}

impl Slots {
    pub(crate) fn new(n: usize) -> Self {
        Self { held: vec![None; n] }
    }

    /// Forget every picture (a new decoding run).
    pub(crate) fn clear(&mut self) {
        self.held.iter_mut().for_each(|s| *s = None);
    }

    /// Free the slots of pictures no longer in `dpb`, then give picture `id` a free slot.
    pub(crate) fn assign(&mut self, dpb: &[u32], id: u32) -> Result<u32, String> {
        for s in self.held.iter_mut() {
            if s.is_some_and(|held| !dpb.contains(&held)) {
                *s = None;
            }
        }
        if self.slot_of(id).is_some() {
            return Err(format!("picture {id} already has a slot"));
        }
        let free = self.held.iter().position(Option::is_none).ok_or_else(|| format!("no free DPB slot for picture {id} ({} slots)", self.held.len()))?;
        let slot = u32::try_from(free).map_err(|_| "DPB slot index overflows".to_string())?;
        if let Some(s) = self.held.get_mut(free) {
            *s = Some(id);
        }
        Ok(slot)
    }

    /// The slot holding picture `id`.
    pub(crate) fn slot_of(&self, id: u32) -> Option<u32> {
        self.held.iter().position(|s| *s == Some(id)).and_then(|i| u32::try_from(i).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_reused_once_pictures_leave_the_dpb() {
        let mut s = Slots::new(3);
        assert_eq!(s.assign(&[], 1), Ok(0));
        assert_eq!(s.assign(&[1], 2), Ok(1));
        assert_eq!(s.assign(&[1, 2], 3), Ok(2));
        assert!(s.assign(&[1, 2, 3], 4).is_err(), "all three still held");
        // picture 2 left the DPB: its slot is reused
        assert_eq!(s.assign(&[1, 3], 4), Ok(1));
        assert_eq!(s.slot_of(4), Some(1));
        assert_eq!(s.slot_of(2), None);
        assert!(s.assign(&[1, 3, 4], 4).is_err(), "a picture is decoded once");
        s.clear();
        assert_eq!(s.slot_of(1), None);
        assert_eq!(s.assign(&[], 9), Ok(0));
        assert!(Slots::new(0).assign(&[], 1).is_err());
    }
}
