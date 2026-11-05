"""Tests for plumescout.firms (pure parts)."""

from datetime import date

from plumescout import firms


def test_date_windows_merge_and_split() -> None:
    d = [date(2023, 1, 10), date(2023, 1, 11), date(2023, 3, 1)]
    w = list(firms.date_windows(d, pad_days=1, max_days=10))
    assert w == [(date(2023, 1, 9), 4), (date(2023, 2, 28), 3)]


def test_date_windows_respects_max_days() -> None:
    d = [date(2023, 1, k) for k in range(1, 25)]
    w = list(firms.date_windows(d, pad_days=0, max_days=10))
    assert [n for _, n in w] == [10, 10, 4]
    assert w[1][0] == date(2023, 1, 11)


def test_parse_csv_utc_times() -> None:
    text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
        "confidence,version,bright_ti5,frp,daynight\n"
        "29.01,47.95,330.1,0.4,0.4,2023-04-05,1012,N,VIIRS,n,2,290.0,5.2,D\n"
        "29.02,47.96,331.0,0.4,0.4,2023-04-05,0047,N,VIIRS,h,2,291.0,7.0,N\n"
    )
    df = firms.parse_csv(text, "VIIRS_SNPP_SP")
    assert len(df) == 2
    assert str(df.datetime_utc.iloc[1]) == "2023-04-05 00:47:00+00:00"
    assert df.lon.iloc[0] == 47.95


def test_parse_csv_handles_empty_and_errors() -> None:
    assert firms.parse_csv("", "x").empty
    assert firms.parse_csv("Invalid MAP_KEY.", "x").empty


def test_concat_with_empty_frames_keeps_datetime() -> None:
    import pandas as pd

    full = firms.parse_csv(
        "latitude,longitude,acq_date,acq_time,confidence,frp\n29.0,47.9,2023-04-05,1012,n,5\n", "a"
    )
    out = pd.concat([firms.parse_csv("", "b"), full], ignore_index=True)
    out["datetime_utc"] = pd.to_datetime(out.datetime_utc, utc=True)
    assert out.datetime_utc.dt.floor("min").iloc[0] == pd.Timestamp("2023-04-05 10:12", tz="UTC")
