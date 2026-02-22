"""Tests for species file loading (species_classifier._load_species_file)."""

from species_classifier import _load_species_file


class TestLoadSpeciesFile:
    def test_parses_pipe_format(self, tmp_path):
        f = tmp_path / "species.txt"
        f.write_text("Great Tit | Kjøttmeis\nBlue Tit | Blåmeis\n")
        en, no_map = _load_species_file(f)
        assert en == ["Great Tit", "Blue Tit"]
        assert no_map["Great Tit"] == "Kjøttmeis"
        assert no_map["Blue Tit"] == "Blåmeis"

    def test_skips_comments_and_blanks(self, tmp_path):
        f = tmp_path / "species.txt"
        f.write_text("# header\n\nGreat Tit | Kjøttmeis\n# another comment\n")
        en, no_map = _load_species_file(f)
        assert en == ["Great Tit"]

    def test_no_pipe_uses_line_as_both(self, tmp_path):
        f = tmp_path / "species.txt"
        f.write_text("Robin\n")
        en, no_map = _load_species_file(f)
        assert en == ["Robin"]
        assert no_map["Robin"] == "Robin"

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "species.txt"
        f.write_text("  Great Tit  |  Kjøttmeis  \n")
        en, no_map = _load_species_file(f)
        assert en == ["Great Tit"]
        assert no_map["Great Tit"] == "Kjøttmeis"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "species.txt"
        f.write_text("")
        en, no_map = _load_species_file(f)
        assert en == []
        assert no_map == {}
