# RSG architecture

The package owns exactly three ROS 2 nodes:

1. `rsg_preprocessor` prepares synchronized RGB-D, CameraInfo, and pose data.
2. `rsg_phase1_semantic_coordinator` runs NanoSAM, geometry estimation,
   persistent tracking, RAP retrieval, VLM fallback, and semantic-image output.
3. `rsg_scene_graph_fuser` combines Hydra DSG updates with final Phase 1
   slot-to-label events and publishes the fused RViz graph.

Hydra, Chroma, and the Qwen server are external processes. They are started by
`rsg_full_stack.launch.py` or `rsg_all.launch.py` but are not RSG-owned nodes.

## Semantic data flow

```text
preprocessor -> Phase 1 -> Hydra semantic image
                     \-> final slot-to-label event -> fuser
Hydra DSG ------------------------------------------> fuser -> RViz markers
```

The fuser uses slot ID as the primary join key. A cached final label and its
validated mobility metadata are applied to the actual Hydra object nodes that
carry the matching slot. The source DSG remains unchanged; temporal confidence
only affects the derived fused metadata and RViz marker alpha.

## Best-crop semantic scheduling

A new persistent track publishes its temporary slot to Hydra immediately. Phase
1 then collects candidate crops for at least the configured semantic settling
interval. Once eligible, RAP and VLM queues carry only the persistent track ID.
The best crop remains mutable while that ID waits in either queue; RAP or VLM
creates an immutable crop snapshot only when it dequeues the ID. A VLM result is
stored to RAP memory with the same crop that was actually supplied to the VLM.
The RAP reference also stores label confidence, mobility class, and mobility
confidence so later visual retrieval restores the complete semantic contract.

## Mobility-aware persistence

The first dynamic-object extension does not estimate motion. Humans, animals,
and self-propelled mobile robots are tagged `dynamic`; other objects are tagged
`static`, with `unknown` retained for uncertain results. Active observations
reset confidence to one. Unobserved static/unknown slots decay with a ten-minute
half-life, while dynamic slots decay with a two-minute half-life. No DSG object
is deleted. The fuser retains a minimum marker alpha and uses cubes for dynamic
objects and spheres for all others. See `MOBILITY_AWARE_DECAY.md`.


## Fuser execution model

The fuser consumes final Phase 1 labels in a dedicated callback group and
renders the complete fused graph at a capped rate. Semantic messages only update
a slot-label cache; they never trigger a full RViz rebuild. This preserves late
semantic labels while the Hydra graph grows.
