//! Bounded native agent-state qualification against the retained Kairos pin.

use kairo_ecs_abm::ABMContext;
use kairo_ecs_types::{SimTime, StepOutcome};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Adoption(bool);

fn replay() -> Vec<(u128, bool)> {
    // ABMContext owns Kairos' kairo_ecs_core scheduler and kairo_ecs_state
    // component registry; Innovate does not implement another scheduler.
    let mut context = ABMContext::new(42);
    let first = context.spawn_agent();
    let second = context.spawn_agent();
    context.attach(first, Adoption(false));
    context.attach(second, Adoption(false));
    context.schedule_behaviour(second, 1, SimTime::from_ticks(2));
    context.schedule_behaviour(first, 1, SimTime::from_ticks(1));
    let mut trace = Vec::new();
    while let StepOutcome::Dispatched(event) = context.step() {
        let agent = event.entity.expect("agent event must retain identity");
        assert_eq!(context.get::<Adoption>(agent), Some(&Adoption(false)));
        context.attach(agent, Adoption(true));
        trace.push((event.at.ticks(), context.get::<Adoption>(agent).unwrap().0));
    }
    assert_eq!(context.get::<Adoption>(first), Some(&Adoption(true)));
    assert_eq!(context.get::<Adoption>(second), Some(&Adoption(true)));
    assert_eq!(context.step(), StepOutcome::Empty);
    trace
}

fn main() {
    let trace = replay();
    assert_eq!(trace, vec![(1, true), (2, true)]);
    assert_eq!(trace, replay());
    println!("Kairos native ABM: agent identity, updates and replay assertions passed");
}
