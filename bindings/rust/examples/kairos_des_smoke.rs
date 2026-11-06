//! Bounded native DES qualification against the retained Kairos pin.

use kairo_ecs_core::Scheduler;
use kairo_ecs_des::Resource;
use kairo_ecs_rng::DeterministicStream;
use kairo_ecs_types::{EntityId, EventKind, ScheduleRequest, SimTime, StepOutcome};

fn replay(seed: u64) -> Vec<(u128, u32)> {
    let mut scheduler = Scheduler::new();
    let mut rng = DeterministicStream::from_run_seed(seed);
    for kind in 0..8 {
        scheduler.schedule(ScheduleRequest {
            at: SimTime::from_ticks(u128::from(rng.next_u32() % 16)),
            priority: 0,
            entity: None,
            kind: EventKind::Custom(kind),
        });
    }
    let mut trace = Vec::new();
    while let StepOutcome::Dispatched(event) = scheduler.step() {
        let EventKind::Custom(kind) = event.kind;
        trace.push((event.at.ticks(), kind));
    }
    assert_eq!(scheduler.pending_events(), 0);
    assert_eq!(trace.len(), 8);
    assert!(trace.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    trace
}

fn main() {
    let trace = replay(42);
    assert_eq!(trace, replay(42));
    assert_ne!(trace, replay(43));
    let mut resource = Resource::new("adoption_channel", 1);
    let first = EntityId::new(0, 0);
    let second = EntityId::new(1, 0);
    assert!(resource.request(first));
    assert!(!resource.request(second));
    assert_eq!(resource.release(), Some(second));
    assert_eq!(resource.release(), None);
    assert_eq!(resource.available_count(), 1);
    println!("Kairos native DES: deterministic replay and FIFO assertions passed");
}
