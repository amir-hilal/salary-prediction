"""Shared label maps and category-ordering lists for visualization modules."""

_EXP_LABELS: dict[int, str] = {
    0: "Entry-level", 1: "Mid-level", 2: "Senior", 3: "Executive"
}
_EXP_ORDER: list[str] = ["Entry-level", "Mid-level", "Senior", "Executive"]

_REGION_LABELS: dict[int, str] = {
    0: "Rest of World", 1: "Asia Pacific", 2: "Europe", 3: "North America"
}
_REGION_ORDER: list[str] = ["Rest of World", "Asia Pacific", "Europe", "North America"]

_FAMILY_LABELS: dict[int, str] = {
    0: "Other", 1: "Analytics", 2: "Data Science",
    3: "Data Engineering", 4: "ML/AI", 5: "Leadership",
}

_REMOTE_LABELS: dict[int, str] = {0: "On-site", 50: "Hybrid", 100: "Remote"}
_REMOTE_ORDER: list[str] = ["On-site", "Hybrid", "Remote"]

_SIZE_LABELS: dict[int, str] = {0: "Small", 1: "Medium", 2: "Large"}
_SIZE_ORDER: list[str] = ["Small", "Medium", "Large"]
