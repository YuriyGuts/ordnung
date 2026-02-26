"""
Validate that an agent-organized directory meets the system prompt requirements.

Usage:
    pytest tests/manual/test_organized_directory.py --dir /path/to/organized/directory
"""


def test_no_top_level_files(organized_dir):
    # GIVEN an agent-organized directory
    # WHEN listing top-level entries
    top_level_files = [p for p in organized_dir.iterdir() if p.is_file()]

    # THEN there must be no uncategorized files at the top level
    assert top_level_files == [], (
        f"Found uncategorized files at top level: {[f.name for f in top_level_files]}"
    )


def test_only_directories_at_top_level(organized_dir):
    # GIVEN an agent-organized directory
    # WHEN listing top-level entries
    entries = list(organized_dir.iterdir())

    # THEN every entry must be a directory
    non_dirs = [p.name for p in entries if not p.is_dir()]
    assert non_dirs == [], f"Non-directory entries at top level: {non_dirs}"


def test_at_least_one_category(organized_dir):
    # GIVEN an agent-organized directory
    # WHEN listing top-level directories
    categories = [p for p in organized_dir.iterdir() if p.is_dir()]

    # THEN there must be at least one category
    assert len(categories) >= 1, "No category directories found"


def test_single_level_categories(organized_dir):
    # GIVEN an agent-organized directory
    # WHEN checking each category subdirectory
    categories = [p for p in organized_dir.iterdir() if p.is_dir()]

    # THEN no category should contain nested subdirectories (only one level allowed)
    for category in categories:
        nested_dirs = [p for p in category.iterdir() if p.is_dir()]
        assert nested_dirs == [], (
            f"Category '{category.name}' contains subdirectories "
            f"(no subcategories allowed): {[d.name for d in nested_dirs]}"
        )


def test_categories_are_not_empty(organized_dir):
    # GIVEN an agent-organized directory
    # WHEN checking each category subdirectory
    categories = [p for p in organized_dir.iterdir() if p.is_dir()]

    # THEN every category must contain at least one file
    for category in categories:
        files = [p for p in category.iterdir() if p.is_file()]
        assert len(files) >= 1, f"Category '{category.name}' is empty"
