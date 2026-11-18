import argparse
import socket
import struct

"""
Keep listening to the T block pose published over LCM and print it.

The message is camera_driver_lcm.object_pose_t (see
trio_infra/src/pose_estimation/lcmtypes/object_pose_t.lcm), decoded here by
hand so this script has no dependency on the generated Python types.

The LCM UDP multicast packets are received with a plain socket instead of the
lcm library, because lcm always joins the multicast group on the default-route
interface (Wi-Fi) and cannot be told to use the wired port to the NUC.
"""

parser = argparse.ArgumentParser(description="Print the T block pose from LCM.")
parser.add_argument("--channel", default="T_SHAPE_POSE", help="LCM channel name")
parser.add_argument("--group", default="239.255.76.67", help="LCM multicast group")
parser.add_argument("--port", type=int, default=7667, help="LCM multicast port")
parser.add_argument(
    "--iface",
    default="192.168.50.1",
    help="Local IP of the interface connected to the publisher (default: wired port)",
)
args = parser.parse_args()


def decode_object_pose(data: bytes) -> dict:
    """Decode an object_pose_t message (big-endian, after an 8-byte fingerprint)."""
    offset = 8
    timestamp_us, frame_number = struct.unpack_from(">qq", data, offset)
    offset += 16

    def read_string() -> str:
        nonlocal offset
        (n,) = struct.unpack_from(">i", data, offset)  # length incl. null byte
        offset += 4
        s = data[offset : offset + n - 1].decode()
        offset += n
        return s

    object_name = read_string()
    frame_name = read_string()
    values = struct.unpack_from(">7d", data, offset)  # position(3) + quat xyzw(4)
    offset += 56
    (reproj_err,) = struct.unpack_from(">d", data, offset)
    offset += 8
    (num_tags,) = struct.unpack_from(">i", data, offset)
    offset += 4
    tag_ids = struct.unpack_from(f">{num_tags}i", data, offset)
    return {
        "timestamp_us": timestamp_us,
        "frame_number": frame_number,
        "object_name": object_name,
        "frame_name": frame_name,
        "position_m": values[:3],
        "quaternion_xyzw": values[3:],
        "reprojection_error_px": reproj_err,
        "tag_ids": tag_ids,
    }


def on_pose(channel: str, data: bytes) -> None:
    pose = decode_object_pose(data)
    px, py, pz = pose["position_m"]
    qx, qy, qz, qw = pose["quaternion_xyzw"]
    print(
        f"[{pose['timestamp_us'] / 1e6:.3f}] frame={pose['frame_number']} "
        f"{pose['object_name']} in {pose['frame_name']} | "
        f"pos=({px:+.4f}, {py:+.4f}, {pz:+.4f}) m | "
        f"quat_xyzw=({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f}) | "
        f"reproj={pose['reprojection_error_px']:.2f}px tags={list(pose['tag_ids'])}"
    )


LCM_MAGIC_SHORT = 0x4C433032  # "LC02": a message that fits in one UDP packet

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", args.port))
sock.setsockopt(
    socket.IPPROTO_IP,
    socket.IP_ADD_MEMBERSHIP,
    socket.inet_aton(args.group) + socket.inet_aton(args.iface),
)
print(
    f"Listening on '{args.channel}' ({args.group}:{args.port} via {args.iface}). "
    "Ctrl+C to stop."
)
try:
    while True:
        packet, _ = sock.recvfrom(65535)
        (magic,) = struct.unpack_from(">I", packet)
        if magic != LCM_MAGIC_SHORT:
            continue  # fragmented messages are not needed for this small pose
        end = packet.index(b"\0", 8)  # channel name follows magic + seqno
        channel = packet[8:end].decode()
        if channel == args.channel:
            on_pose(channel, packet[end + 1 :])
except KeyboardInterrupt:
    print("\nStopped.")
