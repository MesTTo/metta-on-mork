use vstd::prelude::*;

verus! {

// A reduced model of Hyperon's `Bindings` store.
//
// `live` is the set of binding ids present in the backing store. `var_to_binding`
// maps every variable to the binding id for its equivalence class. The real bug was
// an equal-value class merge that removed the right binding id while retargeting only
// one representative variable. The invariant is simple: every variable must point at
// a live binding id.

pub open spec fn binding_live(live: Map<nat, ()>, id: nat) -> bool {
    live.dom().contains(id)
}

pub open spec fn rewrite_binding_id(id: nat, removed: nat, kept: nat) -> nat {
    if id == removed {
        kept
    } else {
        id
    }
}

pub proof fn move_binding_to_binding_preserves_variable_pointer(
    live: Map<nat, ()>,
    id: nat,
    removed: nat,
    kept: nat,
)
    requires
        binding_live(live, id),
        binding_live(live, removed),
        binding_live(live, kept),
        removed != kept,
    ensures
        binding_live(live.remove(removed), rewrite_binding_id(id, removed, kept)),
{
    if id == removed {
        assert(rewrite_binding_id(id, removed, kept) == kept);
        assert(live.remove(removed).dom().contains(kept));
    } else {
        assert(rewrite_binding_id(id, removed, kept) == id);
        assert(live.remove(removed).dom().contains(id));
    }
}

pub open spec fn live_value_filter(captured: int, query: int) -> bool {
    captured == query
}

pub proof fn live_value_filter_is_exact(captured: int, query: int)
    ensures
        live_value_filter(captured, query) <==> captured == query,
{
}

} // verus!

fn main() {}
