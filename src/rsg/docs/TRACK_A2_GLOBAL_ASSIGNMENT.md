# Track-aware mask redundancy and global frame assignment

This revision implements three controlled tracking changes.

## A2: track-aware union redundancy

Before tracks are updated, all SAM observations are evaluated against the existing track set. An enclosing mask is suppressed only when:

- at least two smaller masks are substantially contained by it;
- the union of those smaller masks explains at least 90% of the enclosing mask;
- the smaller masks prefer at least two distinct established tracks; and
- the summed child-track assignment utility is stronger than the enclosing mask's best single-track utility.

This avoids the unsafe generic rule of always preferring the largest mask.

## B: 2D IoU threshold

`persistent_tracking.min_2d_iou` is raised from 0.20 to 0.30.

## E: global frame-level assignment

All retained observations and all existing tracks are evaluated before any track is changed. A rectangular Hungarian solver then selects the maximum-utility one-to-one assignment. Private dummy columns allow any observation to create a new track, so weak matches are never forced.

Route utilities preserve the current association hierarchy:

1. accumulated 3D footprint;
2. recent 3D continuation;
3. centroid 3D;
4. recent 2D IoU fallback.

The selected assignment is then committed in the original observation order using forced frame decisions, eliminating first-mask advantage.

## A3: Same-track nested duplicate suppression

A3 handles the complementary case where SAM emits both a stable whole-object mask and a nested partial mask, and both observations prefer the same established physical track. Before global assignment, nested pairs are compared using a continuity ordering:

1. association-route priority,
2. previous-track 2D IoU,
3. 3D centroid distance,
4. volume-ratio closeness to 1,
5. route residual,
6. mask area as the final tie-breaker.

The weaker nested duplicate is suppressed with reason `same_track_nested_duplicate`. This prevents the nested subsection from consuming the historical track and forcing the full continuation into a new ID. The rule does not blindly retain the larger mask.

Configuration:

```yaml
same_track_nested_suppression_enabled: true
same_track_nested_containment_threshold: 0.90
```

## A3 broader-mask takeover for one established track

When a smaller and a broader nested mask both independently prefer the same established physical track, the broader mask may inherit that track and the smaller duplicate is suppressed. This permits a physical track to expand when SAM reveals previously unseen coherent surface area.

The broader mask is promoted only when:

- the smaller mask satisfies the configured containment threshold;
- A2 has not identified the broader mask as a union of multiple established tracks;
- the parent/child area ratio is bounded;
- the broader mask adds a minimum amount of new visible area;
- the broader mask has accumulated, continuation, or centroid 3D support rather than only 2D IoU;
- no hard 3D contradiction is present; and
- its volume ratio remains within the broader-mask expansion limit.

Configuration:

```yaml
same_track_nested_suppression_enabled: true
same_track_nested_containment_threshold: 0.90
same_track_max_parent_child_area_ratio: 3.0
same_track_min_added_area_fraction: 0.05
same_track_broader_max_route_priority: 2
same_track_broader_max_volume_ratio: 6.0
```

A suppressed narrow mask records `same_track_broader_mask_takeover`. If the broader mask is not coherent enough, the implementation falls back to the conservative temporal-continuity comparison used previously.
