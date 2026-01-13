import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

from add_publisher_leaning import add_publisher_leaning, publisher_leaning
from separate_by_category import separate_data_by_category
from update_categories import update_categories

class TestAddPublisherLeaning:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'title': ['Article 1', 'Article 2', 'Article 3'],
            'source': ['CNBC', 'Fox News', 'Unknown Source'],
            'content': ['Content 1', 'Content 2', 'Content 3']
        })

    def test_add_publisher_leaning_known_sources(self, sample_data, tmp_path):
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        sample_data.to_csv(input_file, index=False)
        
        result_df = add_publisher_leaning(str(input_file), str(output_file))
        
        assert result_df is not None
        assert 'publisher_leaning' in result_df.columns
        assert result_df.loc[0, 'publisher_leaning'] == 'Center-Left'  # CNBC
        assert result_df.loc[1, 'publisher_leaning'] == 'Right'  # Fox News
        assert result_df.loc[2, 'publisher_leaning'] == 'Unknown'  # Unknown Source

    def test_add_publisher_leaning_file_not_found(self, tmp_path):
        input_file = tmp_path / "nonexistent.csv"
        output_file = tmp_path / "output.csv"
        
        result_df = add_publisher_leaning(str(input_file), str(output_file))
        
        assert result_df is None

    def test_publisher_leaning_dict_not_empty(self):
        assert len(publisher_leaning) > 0
        assert 'CNBC' in publisher_leaning
        assert 'Fox News' in publisher_leaning

    def test_output_file_created(self, sample_data, tmp_path):
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        sample_data.to_csv(input_file, index=False)
        
        add_publisher_leaning(str(input_file), str(output_file))
        
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert len(output_df) == len(sample_data)
        assert 'publisher_leaning' in output_df.columns


class TestSeparateByCategory:
    @pytest.fixture
    def sample_category_data(self):
        return pd.DataFrame({
            'title': ['Article 1', 'Article 2', 'Article 3', 'Article 4'],
            'Categories': ['Economy', 'Social Issues', 'Economy', 'Immigration & Security'],
            'content': ['Content 1', 'Content 2', 'Content 3', 'Content 4']
        })

    def test_separate_by_category(self, sample_category_data, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
        sample_category_data.to_csv('data_annotated.csv', index=False)
        
        separate_data_by_category()
        
        data_dir = Path('data')
        assert data_dir.exists()
        
        economy_file = data_dir / 'economy.csv'
        social_file = data_dir / 'social_issues.csv'
        immigration_file = data_dir / 'immigration_and_security.csv'
        
        assert economy_file.exists()
        assert social_file.exists()
        assert immigration_file.exists()
        
        economy_df = pd.read_csv(economy_file)
        assert len(economy_df) == 2
        assert all(economy_df['Categories'] == 'Economy')

    def test_safe_filename_creation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
        test_data = pd.DataFrame({
            'title': ['Test'],
            'Categories': ['Immigration & Security'],
            'content': ['Test content']
        })
        test_data.to_csv('data_annotated.csv', index=False)
        
        separate_data_by_category()
        
        expected_file = Path('data') / 'immigration_and_security.csv'
        assert expected_file.exists()


class TestUpdateCategories:
    @pytest.fixture
    def sample_category_data(self):
        return pd.DataFrame({
            'title': ['Article 1', 'Article 2', 'Article 3'],
            'Categories': ['Social', 'Corruption/Scandal', 'Economy'],
            'content': ['Content 1', 'Content 2', 'Content 3']
        })

    def test_update_categories_mapping(self, sample_category_data, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
        sample_category_data.to_csv('data_annotated.csv', index=False)
        
        update_categories()
        
        updated_df = pd.read_csv('data_annotated.csv')
        
        assert 'Social Issues' in updated_df['Categories'].values
        assert 'Corruption & Scandal' in updated_df['Categories'].values
        assert 'Social' not in updated_df['Categories'].values
        assert 'Corruption/Scandal' not in updated_df['Categories'].values
        assert 'Economy' in updated_df['Categories'].values  # unchanged

    def test_update_categories_no_target_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
        try:
            update_categories()
        except Exception:
            pytest.fail("update_categories() should handle missing files gracefully")

    def test_update_categories_both_files(self, sample_category_data, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
        sample_category_data.to_csv('data_annotated.csv', index=False)
        sample_category_data.to_csv('data_annotated_with_leaning.csv', index=False)
        
        update_categories()
        
        for filename in ['data_annotated.csv', 'data_annotated_with_leaning.csv']:
            updated_df = pd.read_csv(filename)
            assert 'Social Issues' in updated_df['Categories'].values
            assert 'Social' not in updated_df['Categories'].values

    def test_update_categories_preserves_data(self, sample_category_data, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
        sample_category_data.to_csv('data_annotated.csv', index=False)
        
        update_categories()
        
        updated_df = pd.read_csv('data_annotated.csv')
        
        assert len(updated_df) == len(sample_category_data)
        assert list(updated_df.columns) == list(sample_category_data.columns)
        assert all(updated_df['title'] == sample_category_data['title'])