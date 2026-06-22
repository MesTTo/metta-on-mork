use std::convert::TryInto;

#[derive(Clone, Copy)]
enum Tag {
    NewVar,
    VarRef,
    SymbolSize(u8),
    Arity,
}

const GROUNDED_MARK: u8 = 0x00;
const GROUNDED_REF_MARK: u8 = 0x01;

fn byte_item(b: u8) -> Tag {
    if b == 0b1100_0000 {
        Tag::NewVar
    } else if (b & 0b1100_0000) == 0b1100_0000 {
        Tag::SymbolSize(b & 0b0011_1111)
    } else if (b & 0b1100_0000) == 0b1000_0000 {
        Tag::VarRef
    } else {
        Tag::Arity
    }
}

fn decode_atom_head_shape(bytes: &[u8], pos: &mut usize) -> Option<()> {
    let tag = byte_item(*bytes.get(*pos)?);
    *pos += 1;
    match tag {
        Tag::SymbolSize(s) => {
            let end = pos.checked_add(s as usize)?;
            let raw = bytes.get(*pos..end)?;
            *pos = end;
            if let Some((&GROUNDED_REF_MARK, id_bytes)) = raw.split_first() {
                let id_bytes: [u8; 8] = id_bytes.try_into().ok()?;
                let _id = u64::from_le_bytes(id_bytes);
            } else if let Some((&GROUNDED_MARK, _display)) = raw.split_first() {
            }
            Some(())
        }
        Tag::Arity | Tag::NewVar | Tag::VarRef => Some(()),
    }
}

#[kani::proof]
#[kani::unwind(12)]
fn decode_atom_head_does_not_panic_on_short_grounded_refs() {
    let bytes: [u8; 10] = kani::any();
    let len: usize = kani::any();
    kani::assume(len <= bytes.len());
    let slice = &bytes[..len];
    let mut pos = 0usize;

    let _ = decode_atom_head_shape(slice, &mut pos);
    assert!(pos <= slice.len());
}
