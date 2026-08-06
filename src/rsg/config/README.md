# Runtime configuration

- `rsg_pipeline.yaml`: topics, preprocessing, SAM, tracking, RAP, VLM, and
  semantic-label publisher QoS.
- `rsg_scene_graph_fuser.yaml`: Hydra DSG input, final semantic-label fuser,
  RViz projection, and graph display policy.
- `hydra/`: Hydra sensor-input and static slot-label dictionary.
- `rviz/`: visualisation layout.

Do not put learned RAP/VLM class labels into the Hydra slot-label YAML. During a
mapping session Hydra receives stable `unknown_slot_*` IDs; the fuser overlays
the current RAP/VLM class label by slot ID.

Mobility-aware persistence is configured in both files. `rsg_pipeline.yaml`
controls the structured VLM contract, validation thresholds, label hints, and
optional VLM/RAP result logging. `rsg_scene_graph_fuser.yaml` controls the
600-second static/unknown and 120-second dynamic half-lives, minimum marker
alpha, cube rendering for dynamic-capable objects, compact multiline labels,
and optional fuser metadata logging.
