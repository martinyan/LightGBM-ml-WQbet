-- HKJC SQLite schema (v1)

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meetings (
  meeting_id INTEGER PRIMARY KEY,
  racedate TEXT NOT NULL,        -- YYYY/MM/DD
  venue TEXT NOT NULL,           -- ST|HV
  going TEXT,
  rail TEXT,
  surface_hint TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(racedate, venue)
);

CREATE TABLE IF NOT EXISTS races (
  race_id INTEGER PRIMARY KEY,
  meeting_id INTEGER NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  race_no INTEGER NOT NULL,
  distance_m INTEGER,
  class_num INTEGER,
  surface TEXT,                  -- turf|awt
  course TEXT,
  scheduled_time TEXT,           -- HH:MM
  race_name TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(meeting_id, race_no)
);

CREATE TABLE IF NOT EXISTS horses (
  horse_code TEXT PRIMARY KEY,   -- e.g. K165
  horse_name_zh TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runners (
  runner_id INTEGER PRIMARY KEY,
  race_id INTEGER NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
  horse_code TEXT REFERENCES horses(horse_code),
  horse_no INTEGER,              -- saddle cloth number in that race
  horse_name_zh TEXT,
  draw INTEGER,
  weight INTEGER,
  jockey TEXT,
  trainer TEXT,
  win_odds REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(race_id, horse_no)
);

CREATE TABLE IF NOT EXISTS results (
  runner_id INTEGER PRIMARY KEY REFERENCES runners(runner_id) ON DELETE CASCADE,
  finish_pos INTEGER,
  finish_time_sec REAL,
  margin_text TEXT,
  margin_len REAL,
  time_delta_sec REAL
);

-- Legacy/compat: fixed 3-segment representation. Deprecated for modeling; use `sectional_splits`.
CREATE TABLE IF NOT EXISTS sectionals (
  runner_id INTEGER PRIMARY KEY REFERENCES runners(runner_id) ON DELETE CASCADE,
  pos1 INTEGER,
  pos3 INTEGER,
  seg2_time REAL,
  seg3_time REAL,
  kick_time REAL
);

CREATE INDEX IF NOT EXISTS idx_runners_horse_code ON runners(horse_code);
CREATE INDEX IF NOT EXISTS idx_races_meeting ON races(meeting_id);
CREATE INDEX IF NOT EXISTS idx_meetings_date ON meetings(racedate);
