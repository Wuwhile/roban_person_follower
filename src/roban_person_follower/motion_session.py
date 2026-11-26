from dataclasses import replace
from time import monotonic

from .bodyhub import OWNER
from .motion_policy import Corridor, FollowMotion


class MotionSession:
    """Serial motion worker; the ROS input thread can revoke permission at any time."""

    def __init__(self, device, settings, near_m, far_m, inputs, notify, command_lock, clock=monotonic):
        self.device = device
        self.cfg = settings
        self.near_m, self.far_m = near_m, far_m
        self.policy = FollowMotion(settings, near_m, far_m)
        self.inputs = inputs
        self.notify = notify
        self.clock = clock
        self.command_lock = command_lock
        self.active = False
        self.fault = False
        self.head = 0.0
        self.walking = False
        self.steps = 0
        self.step_since = None
        self.pose_since = 0
        self.pitch_checked_at = 0
        self.preparing_pitch = False
        self.wait_since = None
        self.last_audit = 0

    def report(self, phase, action="none", reason=""):
        self.notify(phase, action, reason, self.steps)

    def arm(self, once):
        if self.active or self.fault:
            return
        now = self.clock()
        sample, revoked = self.inputs()
        needs_pitch = bool(self.cfg.walk_head_pitch_deg)
        intent = self.initial_intent(now, sample)
        if revoked or intent.action == "stop" or (
                not needs_pitch and (not sample.corridor.visible or sample.corridor.blocked)):
            self.report("standby", reason=intent.reason or "initial_corridor_not_clear")
            return
        self.policy = FollowMotion(self.cfg, self.near_m, self.far_m)
        self.steps = 0
        self.report("arming")
        try:
            self.device.claim(self.permit)
        except Exception as error:
            self.stop(f"claim: {error}")
            return
        self.active = True
        self.once = once
        self.until = now + self.cfg.trial_timeout_s
        self.step_since = self.wait_since = None
        self.head = 0.0
        self.walking = False
        self.pose_since = self.last_audit = self.clock()
        self.pitch_checked_at = 0
        if needs_pitch:
            try:
                self.prepare_pitch()
            except Exception as error:
                self.stop(f"head_pitch: {error}")
        else:
            self.report("armed")

    def initial_intent(self, now, sample):
        # Only the head can be commanded until the camera reaches its calibrated pose.
        if self.cfg.walk_head_pitch_deg and sample is not None:
            sample = replace(sample, corridor=Corridor(visible=True, blocked=False))
        return FollowMotion(self.cfg, self.near_m, self.far_m).decide(now, sample, 0.0)

    def prepare_pitch(self):
        if not self.device.head(0.0, self.permit, self.command_lock):
            self.stop("head_pitch_revoked")
            return
        self.pose_since = self.clock()
        self.preparing_pitch = True
        self.report("armed", "head", "setting_walk_pitch")

    def permit(self):
        sample, revoked = self.inputs()
        return not revoked and sample is not None and sample.enabled and (
            0 <= self.clock()-sample.received <= self.cfg.perception_timeout_s
        )

    def permit_neutral(self):
        sample, revoked = self.inputs()
        if revoked or sample is None or not sample.corridor.visible or sample.corridor.blocked:
            return False
        if not self.frame_after_settle(sample):
            return False
        return self.policy.decide(self.clock(), sample, self.head).action in ("forward", "turn")

    def frame_after_settle(self, sample):
        return sample.received >= max(self.pose_since + self.cfg.head_settle_s, self.pitch_checked_at)

    def stop(self, reason):
        self.active = False
        self.preparing_pitch = False
        self.report("stopping", reason=reason)
        try:
            self.device.close()
        except Exception as error:
            self.fault = True
            self.report("stop_unconfirmed", reason=f"{reason}: {error}")
            return
        self.fault = False
        self.walking = False
        self.step_since = self.wait_since = None
        self.report("estop" if reason == "estop" else "standby", reason=reason)

    def tick(self):
        if not self.active:
            return
        try:
            self.update()
        except Exception as error:
            self.stop(f"bodyhub_error: {error}")

    def update(self):
        now = self.clock()
        sample, revoked = self.inputs()
        if revoked:
            self.stop(revoked)
            return
        if now >= self.until:
            self.stop("trial_timeout")
            return

        if self.preparing_pitch:
            intent = self.initial_intent(now, sample)
            if intent.action == "stop":
                self.stop(intent.reason)
            elif now >= self.pose_since + self.cfg.head_settle_s:
                self.device.confirm_head(0.0, self.cfg.neutral_tolerance_deg)
                self.pitch_checked_at = self.clock()
                self.preparing_pitch = False
                self.report("armed", "head", "walk_pitch_confirmed")
            return

        if self.pitch_checked_at and sample is not None and sample.received <= self.pitch_checked_at:
            intent = self.initial_intent(now, sample)
            if intent.action == "stop":
                self.stop(intent.reason)
            return

        intent = self.policy.decide(now, sample, self.head)
        if intent.action == "stop":
            self.stop(intent.reason)
            return
        if now-self.last_audit >= .7:
            self.device.check_exclusive()
            if self.device.owner() != OWNER:
                raise RuntimeError("BodyHub owner changed")
            self.last_audit = now

        if self.step_since is not None:
            if not sample.corridor.visible:
                self.stop("forward_coverage_lost")
                return
            if self.device.gait_finished(self.step_since):
                self.step_since = None
                self.wait_since = None
                self.report("armed", "step_complete")
                if self.once:
                    self.stop("single_step_complete")
                return
            if now-self.step_since > self.cfg.step_timeout_s:
                self.stop("walking_feedback_timeout")
            return

        if not self.frame_after_settle(sample):
            return
        if intent.action in ("head", "hold"):
            self.wait_since = None
            if self.walking:
                self.device.stand()
                self.walking = False
                self.head = 0.0
                self.pose_since = self.clock()
                if self.cfg.walk_head_pitch_deg:
                    self.prepare_pitch()
                else:
                    self.report("armed", "hold", "standing")
                return
            if intent.action == "head":
                if not self.device.head(intent.value * self.cfg.head_yaw_sign, self.permit, self.command_lock):
                    return
                self.head = intent.value
                self.pose_since = self.clock()
            self.report("armed", intent.action, intent.reason)
            return

        if not self.walking:
            self.pose_since = self.device.start_walking(self.cfg.neutral_tolerance_deg, self.permit_neutral)
            self.walking = True
            self.wait_since = None
            self.report("armed", "walking_ready")
            return

        if self.wait_since is None:
            self.wait_since = now
        elif now-self.wait_since > self.cfg.request_timeout_s:
            self.stop("gait_request_timeout")
            return
        forward = intent.value if intent.action == "forward" else 0.0
        turn = intent.value * self.cfg.gait_yaw_sign if intent.action == "turn" else 0.0

        def authorize():
            latest, revoked_now = self.inputs()
            if revoked_now or latest is None or not self.frame_after_settle(latest):
                return False
            if not latest.corridor.visible or latest.corridor.blocked:
                return False
            return self.policy.decide(self.clock(), latest, self.head) == intent

        sent = self.device.gait(forward, turn, self.cfg.request_max_age_s, authorize, self.command_lock)
        if sent is not None:
            self.step_since = sent
            self.steps += 1
            self.report("armed", intent.action, str(intent.value))
