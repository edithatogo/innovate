//! Kairos entity state and dispatch for the simulation paths without spatial indexing.
use kairo_ecs_core::Scheduler;
use kairo_ecs_state::{ComponentRegistry, World};
use kairo_ecs_types::{EntityId, EventKind, ScheduleRequest, SimTime, StepOutcome};

pub(crate) struct AgentState {
    pub(crate) scheduler: Scheduler,
    world: World,
    components: ComponentRegistry,
}

impl AgentState {
    pub(crate) fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            world: World::new(),
            components: ComponentRegistry::new(),
        }
    }

    pub(crate) fn spawn_agent(&mut self) -> EntityId {
        self.world.spawn()
    }

    pub(crate) fn attach<T: 'static>(&mut self, agent: EntityId, component: T) -> bool {
        self.components.insert(agent, component)
    }

    pub(crate) fn get<T: 'static>(&self, agent: EntityId) -> Option<&T> {
        self.components.get(agent)
    }

    pub(crate) fn schedule_behaviour(&mut self, agent: EntityId, kind: u32, at: SimTime) {
        self.scheduler.schedule(ScheduleRequest {
            at,
            priority: 0,
            entity: Some(agent),
            kind: EventKind::Custom(kind),
        });
    }

    pub(crate) fn step(&mut self) -> StepOutcome {
        self.scheduler.step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};
    use serde_json::json;

    #[test]
    fn pinned_abm_replays_entity_state_and_rng() {
        let replay = || {
            let mut context = AgentState::new();
            let entity = context.spawn_agent();
            assert!(context.get::<u64>(entity).is_none());
            assert!(context.attach(entity, 0_u64));
            let mut rng = DeterministicStream::from_entity_purpose(42, entity, "innovate.test");
            let value = rng.sample(Uniform::new(1_u64, 100).unwrap());
            let tick = SimTime::from_ticks(7);
            context.schedule_behaviour(entity, 1, tick);
            let StepOutcome::Dispatched(event) = context.step() else {
                panic!("expected a Kairos behavior dispatch");
            };
            assert_eq!(event.entity, Some(entity));
            assert_eq!(event.kind, EventKind::Custom(1));
            assert_eq!(event.at, tick);
            assert!(context.attach(entity, value));
            assert_eq!(context.get::<u64>(entity), Some(&value));
            (entity, value, json!(rng.checkpoint()))
        };
        assert_eq!(replay(), replay());
    }
}
