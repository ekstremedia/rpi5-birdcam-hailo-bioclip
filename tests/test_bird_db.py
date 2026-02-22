"""Tests for BirdDB — SQLite visit logging and labeling."""

from datetime import date, datetime


class TestSchema:
    def test_creates_tables(self, in_memory_db):
        conn = in_memory_db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "visits" in names
        assert "detections" in names

    def test_visits_has_label_columns(self, in_memory_db):
        conn = in_memory_db._get_conn()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(visits)").fetchall()]
        assert "user_species" in cols
        assert "labeled_at" in cols


class TestVisitLifecycle:
    def test_log_visit_start(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.85, "/tmp/crop.jpg")
        visits = in_memory_db.get_recent_visits(10)
        assert len(visits) == 1
        assert visits[0]["track_id"] == 1
        assert visits[0]["max_confidence"] == 0.85
        assert visits[0]["end_time"] is None

    def test_log_visit_end(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8)
        in_memory_db.log_visit_end(1, "2026-02-22T10:01:00")
        visits = in_memory_db.get_recent_visits(10)
        assert visits[0]["end_time"] == "2026-02-22T10:01:00"

    def test_update_visit_species(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8)
        in_memory_db.update_visit_species(1, "Kjøttmeis")
        visits = in_memory_db.get_recent_visits(10)
        assert visits[0]["species"] == "Kjøttmeis"

    def test_species_on_start(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.9, species="Blåmeis")
        visits = in_memory_db.get_recent_visits(10)
        assert visits[0]["species"] == "Blåmeis"


class TestTodayCount:
    def test_counts_today_only(self, in_memory_db):
        today = date.today().isoformat()
        in_memory_db.log_visit_start(1, f"{today}T08:00:00", 0.8)
        in_memory_db.log_visit_start(2, f"{today}T09:00:00", 0.7)
        # Old visit from last year
        in_memory_db.log_visit_start(3, "2025-01-01T10:00:00", 0.6)
        assert in_memory_db.get_today_count() == 2

    def test_empty_db_returns_zero(self, in_memory_db):
        assert in_memory_db.get_today_count() == 0


class TestRecentVisits:
    def test_returns_most_recent_first(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T08:00:00", 0.8)
        in_memory_db.log_visit_start(2, "2026-02-22T09:00:00", 0.9)
        visits = in_memory_db.get_recent_visits(10)
        assert visits[0]["track_id"] == 2
        assert visits[1]["track_id"] == 1

    def test_respects_limit(self, in_memory_db):
        for i in range(5):
            in_memory_db.log_visit_start(i, f"2026-02-22T{10+i}:00:00", 0.8)
        assert len(in_memory_db.get_recent_visits(3)) == 3


class TestLabeling:
    def test_set_label(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8, "/tmp/c.jpg")
        visits = in_memory_db.get_recent_visits(1)
        vid = visits[0]["id"]
        in_memory_db.set_label(vid, "Kjøttmeis")
        visits = in_memory_db.get_recent_visits(1)
        assert visits[0]["user_species"] == "Kjøttmeis"
        assert visits[0]["labeled_at"] is not None

    def test_skip_visit(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8, "/tmp/c.jpg")
        vid = in_memory_db.get_recent_visits(1)[0]["id"]
        in_memory_db.skip_visit(vid)
        assert in_memory_db.get_recent_visits(1)[0]["user_species"] == "skip"

    def test_undo_label(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8, "/tmp/c.jpg")
        vid = in_memory_db.get_recent_visits(1)[0]["id"]
        in_memory_db.set_label(vid, "Blåmeis")
        in_memory_db.undo_label(vid)
        visits = in_memory_db.get_recent_visits(1)
        assert visits[0]["user_species"] is None
        assert visits[0]["labeled_at"] is None

    def test_unlabeled_requires_crop(self, in_memory_db):
        """Visits without crop_path should not appear in the unlabeled queue."""
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8, crop_path=None)
        assert in_memory_db.get_unlabeled_visits(10) == []

    def test_get_label_stats(self, in_memory_db):
        in_memory_db.log_visit_start(1, "2026-02-22T10:00:00", 0.8, "/tmp/a.jpg")
        in_memory_db.log_visit_start(2, "2026-02-22T10:01:00", 0.7, "/tmp/b.jpg")
        in_memory_db.log_visit_start(3, "2026-02-22T10:02:00", 0.6, "/tmp/c.jpg")
        in_memory_db.log_visit_start(4, "2026-02-22T10:03:00", 0.5)  # no crop

        vid1 = in_memory_db.get_recent_visits(10)[-3]["id"]  # visit 1
        vid2 = in_memory_db.get_recent_visits(10)[-2]["id"]  # visit 2

        in_memory_db.set_label(vid1, "Kjøttmeis")
        in_memory_db.skip_visit(vid2)

        stats = in_memory_db.get_label_stats()
        assert stats["total"] == 3      # only visits with crops
        assert stats["labeled"] == 1
        assert stats["skipped"] == 1
        assert stats["unlabeled"] == 1
