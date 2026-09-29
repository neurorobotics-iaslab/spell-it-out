"""Numeric software-event codes recorded by ROS-Neuro during acquisition."""

from enum import IntEnum


class EventCode(IntEnum):
    """ROS-Neuro event codes embedded in the original EEG recordings."""

    OFFSET = 0x8000
    TRIAL_START = 1
    FIXATION = 786
    CUE_1 = 3138
    CUE_2 = 3139
    CUE_3 = 3140
    CUE_4 = 3141
    CUE_5 = 3142
    CUE_6 = 3143
    CUE_7 = 3144
    CUE_8 = 3145
    CUE_9 = 3146
    CUE_10 = 3147
    CUE_11 = 3148
    CUE_12 = 3149
    CUE_13 = 3150
    CUE_14 = 3151
    CUE_15 = 3152
    CUE_16 = 3153
    CUE_17 = 3154
    CUE_18 = 3155
    CUE_19 = 3156
    CUE_20 = 3157
    REPETITION = 782
    DOTS = 780
    SYLLABATION = 781
    TRIAL_END = 897
