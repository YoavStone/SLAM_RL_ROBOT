# ---------------- # ---------------- #
# a file to store all the constants of this pkg
# ---------------- # ---------------- #

# ----------------
# map params
# ----------------
RESOLUTION = 0.15  # map res: 15 cm
CROP_SIZE_METERS = 6.0  # 6m x 6m area
MAP_SIZE = int((CROP_SIZE_METERS / RESOLUTION) * (CROP_SIZE_METERS / RESOLUTION))  # map size in cells count

# ---------------- # ---------------- #

# ----------------
# robot params
# ----------------
LINEAR_SPEED = 0.3  # m/s
ANGULAR_SPEED = 1.2  # rad/s
RAD_OF_ROBOT = 0.34  # 34 cm
ROBOT_RAD_SAFE_FACTOR = 1.3  # a safe factor to multiply the rad of the robot so that it doesn't come to close to the wall

# ---------------- # ---------------- #

# ----------------
# learning params
# ----------------

# Agent Hyperparameters
GAMMA = 0.99
LEARNING_RATE_START = 2.5e-4
LEARNING_RATE_END = 0.8e-4  # 0.5e-4
LEARNING_RATE_DECAY = 550000
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000  # Minimum experiences in buffer before learning starts
EPSILON_START = 1.0
EPSILON_END = 0.04
EPSILON_DECAY = 300000  # Steps over which epsilon decays
TARGET_UPDATE_FREQ = 1000 # 2500  # Steps between updating the target network

SAVE_NETWORK_STEP_COUNT_THRESHOLD = 100

# Reward  params
CONTINUES_PUNISHMENT = -0.9  # amount of punishment for every sec
INCOMPLETE_MAP_PUNISHMENT = 5
HIT_WALL_PUNISHMENT = -500.0
CLOSE_TO_WALL_PUNISHMENT = 0.35  # calc dis to wall pun = calced punishment by dis to wall*CLOSE_TO_WALL_PUNISHMENT
WALL_POWER = 6.5
EXPLORATION_REWARD = 3.5  # reward for every newly discovered cell
MOVEMENT_REWARD = 0.7  # reward for moving beyond a threshold (so it wont stay in place)
REVISIT_PENALTY = -0.17  # punishment for revisiting a cell in the map
REMEMBER_VISIT_TIME = 1.5  # how long to keep the visit time of a spot so it counts as visited in seconds
GO_BACK_PUNISH = -0.4  # punishment for going backwards

# ---------------- # ---------------- #