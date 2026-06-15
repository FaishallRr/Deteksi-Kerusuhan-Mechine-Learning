from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    print(f"Training YOLO11n on 7 weapon classes (Celurit, Golok, Kapak, Pedang, Pisau, Pistol, Senapan)")
    print(f"Dataset: Weapon-Detection-3 (27,687 images)")

    results = model.train(
        data="Weapon-Detection-3/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",
        workers=2,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        patience=10,
        seed=42,
        project="runs/train",
        name="yolo_indo_weapons",
        exist_ok=True,
        verbose=True,
    )

    print(f"Training complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
