from datetime import datetime


def build_professional_alert(
    report_id: str,
    score: float,
    weapons: list,
    persons_count: int,
    running_count: int,
    scene: str,
    status: str,
    camera: str = "",
    camera_owner: str = "",
    address: str = "",
    maps_link: str = "",
    coords: str = "",
    plate_numbers: list = None,
    escalation: str = "stable",
    alert_level: str = "SIAGA",
) -> str:
    weapon_str = ", ".join(
        sorted(set(
            f"{w.get('class', 'senjata')} ({int(w.get('confidence', 0) * 100)}%)"
            for w in weapons
        ))
    ) or "Tidak terdeteksi"

    now = datetime.now()
    time_str = now.strftime("%d %B %Y, %H:%M:%S WIB")

    level_icon = {"SIAGA": "🟡", "KRITIS": "🔴", "WASPADA": "🟠"}.get(alert_level, "⚠️")
    scene_label = {"demo_rusuh": "RUSUH / AKSI ANARKIS", "demo_damai": "DEMO DAMAI", "normal": "NORMAL"}.get(scene, scene.upper())
    running_pct = int((running_count / max(persons_count, 1)) * 100) if persons_count > 0 else 0

    msg = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{level_icon} PERINGATAN DINI KERUSUHAN\n"
        "SISTEM MONITORING CCTV AI KOTA SEMARANG\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"TINGKAT ANCAMAN: {alert_level} (Skor: {score:.2f}/1.00)\n"
        f"Klasifikasi Adegan: {scene_label}\n"
        f"Tren: {escalation.upper()}\n\n"
        "━━━ LOKASI & WAKTU ━━━\n"
    )

    if camera:
        owner_info = f" ({camera_owner})" if camera_owner else ""
        msg += f"📹 Kamera: {camera}{owner_info}\n"
    if address:
        msg += f"📍 Alamat: {address}\n"
    if coords:
        msg += f"🗺️ Koordinat: {coords}\n"
    if maps_link:
        msg += f"📍 Peta: {maps_link}\n"
    msg += f"🕐 Waktu: {time_str}\n"
    msg += f"🆔 Laporan: {report_id}\n\n"

    msg += (
        "━━━ DATA DETEKSI ━━━\n"
        f"👥 Jumlah Orang: {persons_count}\n"
        f"🏃 Bergerak Cepat: {running_count} orang ({running_pct}%)\n"
        f"🔪 Senjata Terdeteksi: {len(weapons)}\n"
        f"   Jenis: {weapon_str}\n"
    )

    if plate_numbers:
        plates = ", ".join(plate_numbers)
        msg += f"🔢 Plat Nomor: {plates}\n"

    msg += "\n━━━ EVIDENCE ━━━\n"
    msg += "📸 Foto situasi: [TERLAMPIR]\n"
    msg += f"🎥 Video: evidence/{report_id}_clip.mp4\n\n"

    msg += "━━━ TINDAKAN ───\n"
    if alert_level == "KRITIS":
        msg += "🔴 Laporan otomatis ke pihak kepolisian.\n"
        msg += "🚔 Harap segera tindak lanjuti.\n"
    elif alert_level == "SIAGA":
        msg += "🟡 Laporan ke pusat monitoring.\n"
        msg += "⏳ Menunggu verifikasi petugas (60 detik).\n"
    else:
        msg += "🟠 Terpantau, belum perlu tindakan.\n"

    msg += (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Sistem AI Monitoring Kamtibmas\n"
        "DINAS PERHUBUNGAN KOTA SEMARANG\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return msg
