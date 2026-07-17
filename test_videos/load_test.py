import sys, os, time, json, asyncio, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import websockets

WS_URL = "ws://localhost:8765"
VIDEO_PATH = "test_videos/tawuran_grogol.mp4"
FRAMES_PER_CLIENT = 50
JPEG_QUALITY = 50


async def simulate_client(cid: int, video_path: str, max_frames: int, results: list):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        results.append({"cid": cid, "error": f"Cannot open {video_path}"})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    frames_sent = 0
    frames_ok = 0
    latencies = []
    t0 = time.time()

    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            for fidx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ret, frame = cap.read()
                if not ret:
                    continue

                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ret:
                    continue

                t_send = time.time()
                await ws.send(jpeg.tobytes())
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=30)
                    t_recv = time.time()
                    latencies.append(t_recv - t_send)
                    frames_ok += 1
                except asyncio.TimeoutError:
                    pass

                frames_sent += 1
    except Exception as e:
        results.append({"cid": cid, "error": str(e)})
    finally:
        cap.release()

    elapsed = time.time() - t0
    fps = frames_sent / elapsed if elapsed > 0 else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0
    results.append({
        "cid": cid,
        "frames_sent": frames_sent,
        "frames_ok": frames_ok,
        "elapsed": round(elapsed, 2),
        "fps": round(fps, 2),
        "avg_latency_ms": round(avg_lat * 1000, 1),
        "max_latency_ms": round(max_lat * 1000, 1),
    })


async def run_load_test(n_clients: int, video_path: str, max_frames: int):
    print(f"\n{'='*60}")
    print(f"  LOAD TEST: {n_clients} concurrent client(s)")
    print(f"  Video   : {video_path}")
    print(f"  Frames  : {max_frames} per client")
    print(f"{'='*60}\n")

    results = []
    tasks = [simulate_client(i + 1, video_path, max_frames, results) for i in range(n_clients)]
    t_start = time.time()
    await asyncio.gather(*tasks)
    total_elapsed = time.time() - t_start

    errors = [r for r in results if "error" in r]
    successes = [r for r in results if "fps" in r]

    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    Client#{e['cid']}: {e['error']}")

    if successes:
        total_frames = sum(r["frames_ok"] for r in successes)
        total_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
        all_latencies = []
        for r in successes:
            if "avg_latency_ms" in r:
                all_latencies.append(r["avg_latency_ms"])
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        max_latency = max(r.get("max_latency_ms", 0) for r in successes)

        print(f"\n  {'='*60}")
        print(f"  RESULTS")
        print(f"  {'='*60}")
        print(f"  {'Client':>8} | {'Sent':>6} | {'OK':>6} | {'FPS':>6} | {'AvgLat(ms)':>10} | {'MaxLat(ms)':>10}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
        for r in successes:
            print(f"  {r['cid']:>8} | {r['frames_sent']:>6} | {r['frames_ok']:>6} | {r['fps']:>6.1f} | {r.get('avg_latency_ms', 0):>10.1f} | {r.get('max_latency_ms', 0):>10.0f}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
        print(f"  {'TOTAL':>8} | {total_frames:>6} | {'':>6} | {total_fps:>6.1f} | {avg_latency:>10.1f} | {max_latency:>10.0f}")
        print(f"  {'='*60}")
        print(f"  Wall time: {total_elapsed:.1f}s")
        print(f"  {'='*60}\n")
    else:
        print("  No successful clients.")

    return successes, errors


def main():
    parser = argparse.ArgumentParser(description="CCTV Load Test - Concurrent WS Clients")
    parser.add_argument("--clients", type=int, default=1, help="Number of concurrent WS clients (default: 1)")
    parser.add_argument("--frames", type=int, default=FRAMES_PER_CLIENT, help=f"Frames per client (default: {FRAMES_PER_CLIENT})")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help=f"Video path (default: {VIDEO_PATH})")
    parser.add_argument("--all", nargs="?", const="1,2,4,6,8", help="Run sequence of client counts (e.g. 1,2,4)")
    args = parser.parse_args()

    server_proc = start_ws_server_if_needed()

    try:
        if args.all:
            counts = [int(x.strip()) for x in args.all.split(",")]
            print(f"Running sequential load test: {counts}")
            summary = []
            for n in counts:
                s, e = asyncio.run(run_load_test(n, args.video, args.frames))
                total_fps = sum(r["fps"] for r in s) / len(s) if s else 0
                summary.append({"clients": n, "avg_fps": round(total_fps, 2), "ok": sum(r["frames_ok"] for r in s)})
                print(f"\n  --- Waiting 5s between tests ---")
                time.sleep(5)

            print(f"\n{'='*60}")
            print(f"  LOAD TEST SUMMARY")
            print(f"{'='*60}")
            print(f"  {'Clients':>8} | {'Avg FPS/client':>16} | {'Total Frames':>12}")
            print(f"  {'-'*8}-+-{'-'*16}-+-{'-'*12}")
            for row in summary:
                print(f"  {row['clients']:>8} | {row['avg_fps']:>16.1f} | {row['ok']:>12}")
            print(f"{'='*60}\n")
        else:
            asyncio.run(run_load_test(args.clients, args.video, args.frames))
    finally:
        if server_proc:
            print("[LoadTest] Shutting down WS server...")
            server_proc.kill()
            server_proc.wait()


def start_ws_server_if_needed():
    import socket, subprocess
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 8765))
        s.close()
        print("[LoadTest] WS server already running on port 8765")
        return None
    except ConnectionRefusedError:
        print("[LoadTest] Starting WS server...", flush=True)
        proc = subprocess.Popen(
            ["C:\\Python314\\python.exe", "ws_detect_server.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Wait for ready
        for i in range(60):
            try:
                s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s2.settimeout(1)
                s2.connect(("127.0.0.1", 8765))
                s2.close()
                print(f"[LoadTest] WS server ready after {i+1}s", flush=True)
                return proc
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        print("[LoadTest] WS server failed to start within 60s")
        proc.kill()
        sys.exit(1)


if __name__ == "__main__":
    main()
