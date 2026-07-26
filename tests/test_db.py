from datetime import date

import config
import db
from ocr_pipeline import Medicine, Prescription


def test_save_list_get_active_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    prescription = Prescription(
        medicines=[
            Medicine(
                brand="Napa",
                generic="Paracetamol",
                strength="500 mg",
                dose_pattern="1+0+1",
                duration="5 days",
                confidence=0.94,
            )
        ],
        overall_confidence=0.92,
        model_source="cloud 31B",
        model_id="gemma-test",
    )

    entry_id = db.save_prescription(prescription, label="Synthetic test")
    entries = db.list_prescriptions()
    restored = db.get_prescription(entry_id)
    active = db.get_active_medicines(as_of=date.today())

    assert entries[0].label == "Synthetic test"
    assert entries[0].medicine_count == 1
    assert restored is not None
    assert restored.medicines[0].brand == "Napa"
    assert active[0].generic == "Paracetamol"
    assert db.delete_prescription(entry_id)
    assert db.get_prescription(entry_id) is None
