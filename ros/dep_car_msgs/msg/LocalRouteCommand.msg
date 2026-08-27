int8 REVERSE=-1
int8 NEUTRAL=0
int8 FORWARD=1

uint8 NAVIGATION_CONNECTIVITY=0
uint8 NAVIGATION_MEMORY_GOAL=1
uint8 NAVIGATION_MEMORY_BACKTRACK=2
uint8 NAVIGATION_MEMORY_RESUME=3
uint8 NAVIGATION_FAR_DEAD_END_EGRESS=4

uint8 ROUTE_CONFIDENCE_LOCAL=0
uint8 ROUTE_CONFIDENCE_PARTIAL=1
uint8 ROUTE_CONFIDENCE_CONNECTED=2

std_msgs/Header header
geometry_msgs/Pose target
int8 requested_gear
uint32 segment_index
bool segment_end
uint8 navigation_mode
# Navigation authority is part of the same stamped transaction as the route
# and target.  A separate status callback may arrive before or after either
# half and therefore cannot safely identify a LOCAL->FAR handoff by itself.
string route_id
string route_source
uint32 route_revision
uint64 authority_epoch
bool rolling_target_latched
# P6 V4.3.2 speed authority.  These fields describe only the current online
# route transaction; they are never persisted or indexed by map/scenario ID.
uint8 route_confidence
bool goal_connected
bool route_stable
float64 verified_prefix_m
