uint8 candidate_id
int8 gear
float64 speed_anchor
float64 steering_anchor
float64 duration
float64 learned_score
# V4.2 complete hybrid-sequence metadata.  Legacy V1/V2 candidates leave
# these arrays empty and continue to use the scalar gear/duration fields.
int8[] action_gears
bool[] action_mask
float64[] action_durations
bool[] shift_required
float64[] transition_duration
int8[] motion_gears
float64[] time
float64[] x
float64[] y
float64[] yaw
float64[] speed
float64[] steering
