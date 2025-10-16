#!/usr/bin/env python3
# bridge37_producer.py  (Python 3.7)
import os, mmap, struct, json, time
import socket
import carla  # 0.9.14

PATH = "/home/you/project/ipc/carla_ring.dat"
CAPACITY = 256                 # number of slots in ring
SLOT_DATA_BYTES = 2048         # max payload per slot (bytes)
HEADER_FMT = "!4sIIIIQ"        # magic(4s), version(u32), capacity(u32), slot_data_bytes(u32), write_idx(u32), reserved(u64)
SLOT_HDR_FMT = "!QI"           # seq(u64), size(u32)
HEADER_SIZE = struct.calcsize(HEADER_FMT) + 4  # +4 reserved to align to 32
SLOT_HDR_SIZE = struct.calcsize(SLOT_HDR_FMT) + 4
SLOT_SIZE = SLOT_HDR_SIZE + SLOT_DATA_BYTES
MAGIC = b"CRLA"
VERSION = 1

def create_or_open(size):
    fd = os.open(PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        current = os.fstat(fd).st_size
        if current != size:
            os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        return fd, mm
    except Exception:
        os.close(fd)
        raise

def init_header(mm):
    mm.seek(0)
    header = mm.read(HEADER_SIZE)
    if len(header) == HEADER_SIZE and header[:4] == MAGIC:
        return  # already initialized
    mm.seek(0)
    mm.write(struct.pack(HEADER_FMT, MAGIC, VERSION, CAPACITY, SLOT_DATA_BYTES, 0, 0))
    mm.write(b"\x00"*4)  # reserved pad

def write_slot(mm, idx, seq, payload_bytes):
    # slot offset
    off = HEADER_SIZE + idx * SLOT_SIZE
    size = len(payload_bytes)
    if size > SLOT_DATA_BYTES:
        raise ValueError("payload too large")

    # Write header except seq first to avoid torn reads
    mm.seek(off)
    mm.write(struct.pack("!Q", 0))              # temp seq=0 (not published)
    mm.write(struct.pack("!I", size))
    mm.write(b"\x00"*4)                         # reserved
    # Write data
    mm.write(payload_bytes)
    # Pad rest
    remaining = SLOT_DATA_BYTES - size
    if remaining:
        mm.write(b"\x00"*remaining)
    mm.flush()
    # Publish by writing seq last
    mm.seek(off)
    mm.write(struct.pack("!Q", seq))
    mm.flush()

def update_write_idx(mm, widx):
    # write_idx lives at offset: magic(4)+version(4)+capacity(4)+slot_bytes(4) = 16 bytes into header
    mm.seek(4+4+4+4)
    mm.write(struct.pack("!I", widx))
    mm.flush()

def main():
    total_size = HEADER_SIZE + CAPACITY * SLOT_SIZE
    fd, mm = create_or_open(total_size)
    try:
        init_header(mm)

        # Connect to CARLA
        client = carla.Client("localhost", 2000)   # adjust host/port
        client.set_timeout(5.0)
        world = client.get_world()

        seq = 1
        write_idx = 0

        print("Producer ready. Writing to", PATH)
        while True:
            ego = None
            for a in world.get_actors().filter('vehicle.*'):
                if a.attributes.get('role_name') == 'hero':
                    ego = a
                    break

            if ego:
                t = ego.get_transform()
                v = ego.get_velocity()
                payload = {
                    "ts": time.time(),
                    "pose": {
                        "x": t.location.x, "y": t.location.y, "z": t.location.z,
                        "roll": t.rotation.roll, "pitch": t.rotation.pitch, "yaw": t.rotation.yaw,
                    },
                    "vel": {"x": v.x, "y": v.y, "z": v.z},
                }
            else:
                payload = {"ts": time.time(), "pose": None, "vel": None}

            data = json.dumps(payload).encode("utf-8")
            write_slot(mm, write_idx, seq, data)

            write_idx = (write_idx + 1) % CAPACITY
            update_write_idx(mm, write_idx)
            seq += 1

            world.wait_for_tick()
            # small sleep to reduce CPU; tune as needed
            time.sleep(0.01)
    finally:
        mm.close()
        os.close(fd)

if __name__ == "__main__":
    main()
