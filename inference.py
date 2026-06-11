import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import deque

from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.evidence import EvidenceManager
from preprocessing.extract_frames import FrameExtractor
from preprocessing.feature_extractor import TemporalFeatureExtractor
from core.mil_ranking import MILRankingModel, MILBagProcessor
from core.yolo_detector import YOLODetector
from alert.telegram_bot import TelegramAlert
from alert.whatsapp_sender import WhatsAppAlert


class AnomalyDetector:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config["general"]["log_level"])
        self.device = self.config["general"]["device"]
        self.evidence = EvidenceManager(self.config["evidence"]["output_dir"])

        self._init_models()
        self._init_alerts()

        self.score_buffer = deque(maxlen=self.config["thresholds"]["confirmation_frames"])
        self.alert_count = 0
        self.cooldown_until = datetime.min

    def _init_models(self):
        self.logger.info("Loading models...")
        self.yolo = YOLODetector(
            model_path=self.config["model"]["yolo"]["model_path"],
            confidence_threshold=self.config["model"]["yolo"]["confidence_threshold"],
            device=self.device,
        )
        temporal_cfg = self.config["model"]["temporal"]
        self.extractor = TemporalFeatureExtractor(
            architecture=temporal_cfg["architecture"],
            device=self.device,
        )
        mil_model = MILRankingModel(
            input_dim=temporal_cfg["feature_dim"],
        )
        self.mil = MILBagProcessor(mil_model, device=self.device)
        self.logger.info("Models loaded successfully")

    def _init_alerts(self):
        alert_cfg = self.config["alert"]
        self.telegram = None
        self.whatsapp = None
        if alert_cfg["provider"]["telegram"]:
            tg = alert_cfg["telegram"]
            self.telegram = TelegramAlert(tg["bot_token"], tg["chat_id"])
        if alert_cfg["provider"]["whatsapp"]:
            wa = alert_cfg["whatsapp"]
            self.whatsapp = WhatsAppAlert(wa["phone_number"], wa["api_key"])

    def _generate_report_id(self) -> str:
        return f"ALRT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def _compute_fused_score(self, mil_score: float, yolo_objects: Dict) -> float:
        w_mil = self.config["model"]["fusion"]["mil_weight"]
        w_obj = self.config["model"]["fusion"]["object_weight"]
        has_weapons = len(yolo_objects.get("weapons", [])) > 0
        has_fight = len(yolo_objects.get("persons", [])) >= 2
        obj_score = 0.5 + (0.5 if has_weapons else 0) + (0.3 if has_fight else 0)
        obj_score = min(obj_score, 1.0)
        return w_mil * mil_score + w_obj * obj_score

    def _check_anti_spam(self) -> bool:
        now = datetime.now()
        if now < self.cooldown_until:
            self.logger.info(f"Cooldown active until {self.cooldown_until}")
            return True
        if self.alert_count >= self.config["alert"]["max_alerts_per_hour"]:
            self.logger.warning("Max alerts per hour reached")
            return True
        return False

    def _build_alert_message(self, report_id: str, score: float, objects: Dict, plate_info: Dict) -> str:
        loc = self.config["location"]
        now = datetime.now().strftime("%d %B %Y, %H:%M:%S WIB")

        weapon_str = ", ".join(
            [f"{w.get('confidence', 0):.0%}" for w in objects.get("weapons", [])]
        ) or "Tidak terdeteksi"
        person_count = len(objects.get("persons", []))

        message = (
            f"LAPORAN INDIKASI ANOMALI\n"
            f"{'='*35}\n"
            f"ID: {report_id}\n"
            f"Waktu: {now}\n"
            f"Lokasi: {loc['address']}\n"
            f"Map: {loc['maps_link']}\n\n"
            f"Skor Anomali: {score:.2f} / 1.00\n\n"
            f"Objek Bahaya: {weapon_str}\n"
            f"Jumlah Orang: {person_count}\n"
        )

        if plate_info and plate_info.get("plate"):
            message += f"Plat Nomor: {plate_info['plate']}\n"

        message += (
            f"\nLaporan otomatis dari sistem monitoring AI.\n"
            f"Ini adalah INDIKASI awal - untuk verifikasi dan\n"
            f"tindak lanjut oleh pihak kepolisian."
        )
        return message

    def process_video(self, video_path: str):
        self.logger.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (448, 448))
            frames_buffer.append(resized)

            if len(frames_buffer) >= 16:
                mil_features = self.extractor.extract(frames_buffer[-16:])
                mil_score = self.mil.predict_bag(mil_features)

                yolo_objects = self.yolo.detect(resized)

                faces = self.yolo.detect_faces(resized)
                if faces:
                    yolo_objects["faces"] = faces

                plate = self.yolo.detect_plate(resized)
                yolo_objects["plate"] = plate

                fused_score = self._compute_fused_score(mil_score, yolo_objects)
                self.score_buffer.append(fused_score)

                self._check_alert(fused_score, yolo_objects, frames_buffer)

                cv2.imshow("Detection", self._draw_detections(resized, yolo_objects, fused_score))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frames_buffer = frames_buffer[-8:]

        cap.release()
        cv2.destroyAllWindows()

    def _check_alert(self, score: float, objects: Dict, frames_buffer: List):
        threshold_alert = self.config["thresholds"]["alert"]
        threshold_warning = self.config["thresholds"]["warning"]

        if score >= threshold_alert and len(self.score_buffer) >= self.score_buffer.maxlen:
            if all(s >= threshold_alert for s in self.score_buffer):
                if self._check_anti_spam():
                    return

                report_id = self._generate_report_id()
                self.logger.info(f"ALERT: {report_id} | Score: {score:.2f}")

                self.evidence.save_screenshot(frames_buffer[-1], report_id)
                self.evidence.save_video_clip(
                    list(self.score_buffer), report_id
                )

                plate_info = objects.get("plate", {})
                message = self._build_alert_message(report_id, score, objects, plate_info)

                if self.telegram:
                    self.telegram.send_alert_sync(
                        message=message,
                        photo_path=self.evidence.output_dir / f"{report_id}_screenshot.jpg",
                    )
                if self.whatsapp:
                    self.whatsapp.send_alert(message=message)

                self.alert_count += 1

    def _draw_detections(self, frame: np.ndarray, objects: Dict, score: float) -> np.ndarray:
        for obj_type, items in objects.items():
            if obj_type == "plate":
                continue
            for item in items:
                x1, y1, x2, y2 = map(int, item["bbox"])
                conf = item["confidence"]
                label = f"{obj_type} {conf:.2f}"
                color = (0, 255, 0) if obj_type == "persons" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Score: {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", help="Override video path")
    args = parser.parse_args()

    detector = AnomalyDetector(args.config)
    video_path = args.video or detector.config["input"]["file"]
    detector.process_video(video_path)
