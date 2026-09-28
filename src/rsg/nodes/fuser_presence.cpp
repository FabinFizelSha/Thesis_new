/**
 * @file fuser_presence.cpp
 * @brief Presence-confidence resolution (mirrors phase1's dynamic-track-expiry decay formula).
 */
#include "fuser.hpp"

namespace rsg {

  double SemanticSceneGraphFuser::currentReferenceTimeSec() const {
    const auto clock = const_cast<SemanticSceneGraphFuser*>(this)->get_clock();
    const double clock_sec = static_cast<double>(clock->now().nanoseconds()) * 1.0e-9;
    if (clock_sec > 0.0) {
      return clock_sec;
    }
    const double stamp_sec = static_cast<double>(latest_header_.stamp.sec) +
                             static_cast<double>(latest_header_.stamp.nanosec) * 1.0e-9;
    if (stamp_sec > 0.0) {
      return stamp_sec;
    }
    return 0.0;
  }

  const SemanticOverlay* SemanticSceneGraphFuser::overlayForNode(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved) const {
    const auto found = resolved.find(node.id);
    return found == resolved.end() ? nullptr : &found->second.overlay;
  }

  double SemanticSceneGraphFuser::presenceHalfLifeSec(const std::string& mobility_class) const {
    return normaliseMobilityClass(mobility_class) == "dynamic"
               ? dynamic_presence_half_life_sec_
               : static_presence_half_life_sec_;
  }

  ResolvedPresence SemanticSceneGraphFuser::resolvePresenceForSlot(
      uint32_t slot_id,
      const PresenceCache& presence,
      const std::string& mobility_class) const {
    ResolvedPresence resolved;
    if (slot_id == 0U) {
      return resolved;
    }
    const auto it = presence.find(slot_id);
    if (it == presence.end()) {
      return resolved;
    }
    resolved.observation = it->second;
    const double now_sec = currentReferenceTimeSec();
    resolved.age_sec = std::max(0.0, now_sec - it->second.last_observed_timestamp_sec);

    // This is the TRUE, honest recency signal: for a restored-but-not-yet
    // -reobserved slot, age_sec reflects phase1's deliberate day-plus save-time
    // shift (see memory/README.md), so confidence correctly comes out
    // vanishingly small here. That is exactly what the halo in
    // appendObjectMarkers needs -- a restored object hasn't actually been seen
    // this session, so it must not glow just because it's known map content.
    // Rendering it as fully opaque anyway (not invisible) is handled
    // separately in objectDisplayColor, which is the only place this value
    // gets used for alpha rather than recency.
    const bool observed = resolved.age_sec <= presence_observed_epsilon_sec_;
    resolved.state = it->second.is_restored ? "RESTORED" : (observed ? "OBSERVED" : "DECAYING");
    const double decay_age_sec = std::max(0.0, resolved.age_sec - presence_observed_epsilon_sec_);
    resolved.confidence = observed
                              ? 1.0
                              : std::pow(0.5, decay_age_sec / presenceHalfLifeSec(mobility_class));
    resolved.confidence = clampValue(resolved.confidence, 0.0, 1.0);
    return resolved;
  }

}  // namespace rsg
