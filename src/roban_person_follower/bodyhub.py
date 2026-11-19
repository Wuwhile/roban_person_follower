"""BodyHub transport for a bounded, supervised ROS 1 trial."""

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic, sleep
import xmlrpc.client


GAIT = "/gaitCommand"
HUB = "/BodyHubNode"
PREFIX = "/MediumSize/BodyHub/"
OWNER = 47
EXPECTED_PUBLISHERS = frozenset({
    "/head_toward_sound", "/joystick_handle_node", "/ros_aiui_node",
    "/ros_socket_node", "/slam_mapping_node",
})


@dataclass(frozen=True)
class Process:
    name: str
    pid: int
    start: str

    @classmethod
    def inspect(cls, name, pid):
        path = Path("/proc") / str(pid)
        if path.stat().st_uid != os.getuid():
            raise RuntimeError(f"Unexpected owner for {name}")
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return cls(name, pid, fields[19]), fields[0]

    def state(self):
        identity, state = self.inspect(self.name, self.pid)
        if identity != self:
            raise RuntimeError(f"Process identity changed for {self.name}")
        return state


class GaitSources:
    def __init__(self, master, journal):
        self.master = master
        self.journal = journal
        self.parked = []
        self.recorded = False

    def publishers(self):
        graph, _, _ = self.master.getSystemState()
        return frozenset(dict(graph).get(GAIT, ()))

    def pause(self):
        if self.journal.exists():
            raise RuntimeError("A prior gait isolation needs idle recovery")
        if self.publishers() != EXPECTED_PUBLISHERS:
            raise RuntimeError("Unexpected /gaitCommand publishers")
        sources = []
        for name in sorted(EXPECTED_PUBLISHERS):
            with xmlrpc.client.ServerProxy(self.master.lookupNode(name)) as node:
                code, _, pid = node.getPid("/roban_follow_motion")
            if code != 1:
                raise RuntimeError(f"Cannot identify {name}")
            process, state = Process.inspect(name, pid)
            if state in ("T", "t", "Z", "X"):
                raise RuntimeError(f"Cannot park {name} in state {state}")
            sources.append(process)
        if len({source.pid for source in sources}) != len(sources):
            raise RuntimeError("Gait publishers are not independent processes")
        pending = self.journal.with_suffix(".tmp")
        pending.write_text(json.dumps([asdict(source) for source in sources]), encoding="utf-8")
        pending.replace(self.journal)
        self.recorded = True
        for source in sources:
            self.parked.append(source)
            os.kill(source.pid, signal.SIGSTOP)
        deadline = monotonic() + 1.2
        while monotonic() < deadline:
            if self.all_parked():
                return
            sleep(.02)
        raise RuntimeError("Could not pause gait publishers")

    def all_parked(self):
        return len(self.parked) == len(EXPECTED_PUBLISHERS) and all(
            source.state() == "T" for source in self.parked
        )

    def resume(self):
        failures = []
        for source in reversed(self.parked):
            try:
                if source.state() == "T":
                    os.kill(source.pid, signal.SIGCONT)
            except (OSError, RuntimeError) as error:
                failures.append((source, str(error)))
        self.parked = [source for source, _ in failures]
        if failures:
            raise RuntimeError(f"Gait publisher restoration incomplete: {failures}")
        if self.recorded:
            self.journal.unlink(missing_ok=True)
            self.recorded = False

    def recover_idle(self):
        if not self.journal.exists():
            return
        sources = [Process(**entry) for entry in json.loads(self.journal.read_text(encoding="utf-8"))]
        if len(sources) != len(EXPECTED_PUBLISHERS) or {p.name for p in sources} != EXPECTED_PUBLISHERS:
            raise RuntimeError("Invalid publisher isolation journal")
        for source in sources:
            try:
                if source.state() == "T":
                    os.kill(source.pid, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                continue
        self.journal.unlink()


class BodyHub:
    def __init__(self, rospy, journal, timeout=4):
        import rosgraph
        from bodyhub.msg import JointControlPoint
        from std_msgs.msg import Bool, Float64, Float64MultiArray, UInt16

        self.ros = rospy
        self.head_type = JointControlPoint
        self.gait_type = Float64MultiArray
        self.master = rosgraph.Master(rospy.get_name())
        self.sources = GaitSources(self.master, journal)
        self.service_timeout = timeout
        self.requests = deque(maxlen=16)
        self.walking = deque(maxlen=64)
        self.unexpected = deque(maxlen=5)
        self.hub_state = None
        self.gait_pub = None
        self.head_pub = None
        self.touched = False
        self.active = False
        self.last_request = 0.0
        self.subs = (
            rospy.Subscriber("/requestGaitCommand", Bool, self.receive_request, queue_size=5),
            rospy.Subscriber(PREFIX+"WalkingStatus", Float64, self.receive_walking, queue_size=8),
            rospy.Subscriber(PREFIX+"Status", UInt16, self.receive_state, queue_size=8),
            rospy.Subscriber(GAIT, Float64MultiArray, self.receive_gait, queue_size=6),
        )

    def from_hub(self, msg):
        return getattr(msg, "_connection_header", {}).get("callerid") == HUB

    def receive_request(self, msg):
        if self.from_hub(msg) and msg.data:
            self.requests.append(monotonic())

    def receive_walking(self, msg):
        if self.from_hub(msg) and msg.data in (0.0, 1.0):
            self.walking.append((monotonic(), bool(msg.data)))

    def receive_state(self, msg):
        if self.from_hub(msg):
            self.hub_state = msg.data

    def receive_gait(self, msg):
        if self.active and getattr(msg, "_connection_header", {}).get("callerid") != self.ros.get_name():
            self.unexpected.append(monotonic())

    def service(self, name, argument):
        import yaml
        result = subprocess.run(["rosservice", "call", PREFIX+name, argument],
                                capture_output=True, text=True, timeout=self.service_timeout)
        if result.returncode:
            raise RuntimeError(f"BodyHub {name} failed: {result.stderr.strip()}")
        reply = yaml.safe_load(result.stdout)
        if not isinstance(reply, dict):
            raise RuntimeError(f"Invalid reply for BodyHub {name}")
        return reply

    def state(self):
        return self.service("GetStatus", "{str: status}")["data"]

    def owner(self):
        return int(self.service("GetMasterID", "{str: status}")["data"])

    def jump(self, motion):
        self.service("StateJump", "{masterID: %s, stateReq: %s}" % (OWNER, motion))

    def native_links_ready(self):
        own = self.ros.get_node_uri()
        with xmlrpc.client.ServerProxy(self.master.lookupNode(HUB)) as node:
            code, _, rows = node.getBusInfo(self.ros.get_name())
        incoming = {row[4] for row in rows if len(row)>5 and row[1] in (own, self.ros.get_name())
                    and row[2]=="i" and row[5]}
        return code == 1 and {GAIT, PREFIX+"HeadPosition"} <= incoming

    def claim(self, permit):
        if (self.owner(), self.state()) != (0, "preReady"):
            raise RuntimeError("BodyHub must be idle and unclaimed")
        self.sources.pause()
        self.active = True
        self.unexpected.clear()
        # Synchronous one-shot gait publishing prevents old gait messages staying queued.
        self.gait_pub = self.ros.Publisher(GAIT, self.gait_type, queue_size=None)
        self.head_pub = self.ros.Publisher(PREFIX+"HeadPosition", self.head_type, queue_size=1)
        deadline = monotonic()+2
        while monotonic()<deadline and not self.native_links_ready():
            sleep(.025)
        if not self.native_links_ready():
            raise RuntimeError("BodyHub transport connections are unavailable")
        self.check_exclusive(check_state=False)
        if not permit() or (self.owner(), self.state()) != (0, "preReady"):
            raise RuntimeError("BodyHub changed during acquisition")
        if not permit():
            raise RuntimeError("Motion permission revoked during acquisition")
        self.touched = True
        self.jump("setStatus")
        if (self.owner(), self.state()) != (OWNER, "ready"):
            raise RuntimeError("BodyHub control acquisition not confirmed")
        self.hub_state = 22

    def check_exclusive(self, check_state=True):
        own = self.ros.get_name()
        if (not self.sources.all_parked()
                or self.sources.publishers() != EXPECTED_PUBLISHERS | {own}
                or self.unexpected):
            raise RuntimeError("Gait publisher authority lost")
        if check_state and self.hub_state in (20, 21, 26, 27, 29):
            raise RuntimeError(f"BodyHub reports unsafe state {self.hub_state}")

    def start_walking(self, tolerance, permit):
        self.check_exclusive()
        current = self.state()
        if current not in ("ready", "pause") or self.owner() != OWNER:
            raise RuntimeError(f"Cannot enter walking from {current}")
        if not permit():
            raise RuntimeError("Motion permission revoked before entering walking")
        since = monotonic()
        self.jump("walking")
        if (self.owner(), self.state()) != (OWNER, "walking"):
            raise RuntimeError("BodyHub walking state was not confirmed")
        self.hub_state = 28
        # The vendor walking transition samples Present_Position on all servos.
        measured = self.service("GetJointAngle", "{idArray: [21, 22], idCnt: 2}")["getData"]
        measured = measured[20:22] if len(measured) >= 22 else measured
        if len(measured) != 2 or not all(math.isfinite(angle) and abs(angle)<=tolerance
                                         for angle in measured):
            raise RuntimeError("BodyHub head is not neutral after entering walking")
        self.last_request = since
        return monotonic()

    def head(self, angle, authorize, lock):
        self.check_exclusive()
        if self.hub_state == 28:
            raise RuntimeError("Head gesture cannot overlap gait")
        with lock:
            if not authorize():
                return False
            self.head_pub.publish(self.head_type(positions=[angle, 0.0], mainControlID=OWNER))
        return True

    def gait(self, forward, turn, max_age, authorize, lock):
        self.check_exclusive()
        if self.hub_state != 28 or not self.requests:
            return None
        stamp = self.requests[-1]
        if stamp <= self.last_request or not 0 <= monotonic()-stamp <= max_age:
            return None
        with lock:
            if not authorize() or monotonic()-stamp > max_age or self.unexpected:
                return None
            self.last_request = stamp
            sent = monotonic()
            self.gait_pub.publish(self.gait_type(data=[forward, 0.0, turn]))
            return sent

    def gait_finished(self, since):
        moving = False
        for timestamp, value in list(self.walking):
            if timestamp <= since:
                continue
            if value:
                moving = True
            elif moving:
                return True
        return False

    def stand(self):
        if self.owner() != OWNER:
            raise RuntimeError("Cannot request stop from a different BodyHub owner")
        if self.state() != "ready":
            self.jump("stop")
        if (self.owner(), self.state()) != (OWNER, "ready"):
            raise RuntimeError("BodyHub stop feedback was not confirmed")
        self.hub_state = 22

    def recover_idle(self):
        if (self.owner(), self.state()) != (0, "preReady"):
            raise RuntimeError("Confirm BodyHub preReady and unclaimed before restoring sources")
        self.sources.recover_idle()

    def close(self):
        if self.touched:
            if self.owner() == OWNER:
                self.stand()
                self.jump("reset")
            if (self.owner(), self.state()) != (0, "preReady"):
                # Keep competing publishers paused while the stop is unconfirmed.
                raise RuntimeError("BodyHub stop/release unconfirmed")
            self.touched = False
        self.active = False
        try:
            for publisher in (self.gait_pub, self.head_pub):
                if publisher is not None:
                    publisher.unregister()
        finally:
            self.gait_pub = self.head_pub = None
            self.sources.resume()
            self.hub_state = None
