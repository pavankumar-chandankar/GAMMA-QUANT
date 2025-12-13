def get_activity_groups(label_dict):
    low_freq_names = [
        "stretching (triceps)",
        "stretching (lunging)",
        "stretching (shoulders)",
        "stretching (hamstrings)",
        "stretching (lumbar rotation)",
    ]

    high_freq_names = [
        "jogging",
        "jogging (rotating arms)",
        "jogging (skipping)",
        "jogging (sidesteps)",
        "jogging (butt-kicks)",
        "push-ups",
        "push-ups (complex)",
        "sit-ups",
        "sit-ups (complex)",
        "burpees",
        "lunges",
        "lunges (complex)",
        "bench-dips",
    ]

    null_name = "null"
    null_id = label_dict[null_name]

    def to_ids(names):
        return [label_dict[n] for n in names]

    low_ids = to_ids(low_freq_names) + [null_id]
    high_ids = to_ids(high_freq_names) + [null_id]

    return {
        "low_freq": sorted(list(set(low_ids))),
        "high_freq": sorted(list(set(high_ids))),
    }
