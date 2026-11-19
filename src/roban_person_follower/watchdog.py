class FrameWatchdog:
    def __init__(self, timeout_s, started_at):
        self.timeout_s = timeout_s
        self.reset(started_at)

    def reset(self, received_at):
        self.received_at = received_at
        self.reported = False

    def expired(self, now):
        if self.reported or now - self.received_at <= self.timeout_s:
            return False
        self.reported = True
        return True
