from dolphin import event, gui, savestate, memory, controller
import numpy as np
class MKWMemory:
    class Addresses:
        def __init__(self):
            # RaceManagerPlayer
            self.RaceCompletion = self.resolve_address(0x809BD730, [0xC, 0x0, 0xC])
            # LapCompletion was on a per-checkpoint basis, RaceCompletion is interpolated

            self.currentLap = self.resolve_address(0x809BD730, [0xC, 0x0, 0x24])
            self.countdownTimer = self.resolve_address(0x809BD730, [0x22])
            self.stage = self.resolve_address(0x809BD730, [0x28])

            # KartDynamics - Iterate 3 times with 4 bytes offset to get X, Y and Z.
            self.position = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x18])
            self.acceleration_KartDynamics = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0x80])
            self.mainRotation = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0xF0])
            self.internalVelocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0x14C])
            self.externalVelocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0x74])
            self.angularVelocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0xA4])
            self.velocity = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x4, 0xD4])

            # KartMove
            self.speed = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x20])
            self.acceleration_KartMove = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x30])
            self.miniturboCharge = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x44, 0xFE])

            # Can be used as a mushroom timer as well
            self.offroadInvincibility = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x148])

            self.wheelieFrames = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x2A8])
            self.wheelieCooldown = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x2B6])
            self.leanRot = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x294])

            # KartState
            self.bitfield2 = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x4, 0xC])

            # KartCollide
            self.surfaceFlags = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x18, 0x2C])

            # Misc
            self.mushroomCount = self.resolve_address(0x809C3618, [0x14, 0x90])
            self.hopPos = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x44, 0x22C])

            # I added
            self.mt_boost_timer = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x102])
            self.airtime = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x4, 0x1C])
            self.allmt = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x10C])
            self.mush_and_boost = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x110])
            self.floor_collision_count = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x40])
            self.race_position = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x3C])
            self.respawn_timer = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x18, 0x48])

            # this is called m_types in KartPhysics->CollisionGroup->CollisionData
            self.wall_collide = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x8, 0x8])

            self.soft_speed_limit = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x18])

            self.trickableTimer = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x4, 0xA6])

            self.trick_cooldown = self.resolve_address(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x258, 0x38])

        @staticmethod
        def resolve_address(base_address, offsets):
            """
            This is a helper function to allow multiple ptr dereferences in
            quick succession. base_address is dereferenced first, and then
            offsets are applied.
            """
            current_address = memory.read_u32(base_address)
            for offset in offsets:
                value_address = current_address + offset
                current_address = memory.read_u32(current_address + offset)

            return value_address

    def __init__(self):
        self.addresses = self.Addresses()

        # RaceManagerPlayer
        self.RaceCompletion: float = 0.0
        self.currentLap: int = 0
        self.countdownTimer: int = 0
        self.stage: int = 0

        # KartDynamics = list[float]
        self.position = np.array([0.0, 0.0, 0.0])
        self.acceleration_KartDynamics = np.array([0.0, 0.0, 0.0])
        self.mainRotation = np.array([0.0, 0.0, 0.0, 0.0])
        self.mainRotationEuler = np.array([0.0, 0.0, 0.0])
        self.internalVelocity = np.array([0.0, 0.0, 0.0])
        self.externalVelocity = np.array([0.0, 0.0, 0.0])
        self.angularVelocity = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

        # KartMove
        self.speed: float = 0.0
        self.acceleration_KartMove: float = 0.0
        self.miniturboCharge: int = 0
        self.offroadInvincibility = False
        self.wheelieFrames: int = 0
        self.wheelieCooldown: int = 0
        self.leanRot: float = 0.0

        # KartState
        self.bitfield2: int = 0
        self.isWheelie = False

        # KartCollide
        self.surfaceFlags: int = 0
        self.isAboveOffroad = False
        self.isTouchingOffroad = False

        # Misc
        self.mushroomCount = 0
        self.hopPos = 0

        # Oil Spill Radians and Distance
        self.oilSpillRadians = 0.0
        self.oilSpillDistance = 0.0

        # Boost Panel Radians and Distance
        self.boostPanelRadians = 0.0
        self.boostPanelDistance = 0.0

        # I added
        self.mt_boost_timer = 0
        self.airtime = 0
        self.allmt = 0
        self.mush_and_boost = 0
        self.floor_collision_count = 2
        self.race_position = 12
        self.respawn_timer = 0
        self.wall_collide = 0
        self.speed_limit = 100

        self.trickableTimer = 0
        self.trick_cooldown = 0

    def update(self):

        # my ones
        self.mt_boost_timer = memory.read_u16(self.addresses.mt_boost_timer)
        self.airtime = memory.read_u16(self.addresses.airtime)

        self.allmt = memory.read_u16(self.addresses.allmt)
        self.mush_and_boost = memory.read_u16(self.addresses.mush_and_boost)
        self.floor_collision_count = memory.read_u16(self.addresses.floor_collision_count)

        self.race_position = memory.read_u8(self.addresses.race_position)
        self.respawn_timer = memory.read_u16(self.addresses.respawn_timer)
        self.wall_collide = memory.read_u32(self.addresses.wall_collide)

        self.speed_limit = memory.read_f32(self.addresses.soft_speed_limit)

        # RaceManagerPlayer
        self.RaceCompletion = memory.read_f32(self.addresses.RaceCompletion)

        self.currentLap = memory.read_u16(self.addresses.currentLap)
        self.countdownTimer = memory.read_u16(self.addresses.countdownTimer)
        self.stage = memory.read_u32(self.addresses.stage)

        # KartMove
        self.speed = memory.read_f32(self.addresses.speed)

        self.offroadInvincibility = memory.read_u16(self.addresses.offroadInvincibility)

        # KartCollide
        self.isTouchingOffroad = self.surfaceFlags & (1 << (7 - 1)) != 0

        # Misc
        self.mushroomCount = memory.read_u32(self.addresses.mushroomCount)
        self.hopPos = memory.read_f32(self.addresses.hopPos)

        self.trickableTimer = memory.read_u16(self.addresses.trickableTimer)
        self.trick_cooldown = memory.read_u16(self.addresses.trick_cooldown)

    @staticmethod
    def Quat2Euler(quaternion):

        x, y, w, z = quaternion

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])