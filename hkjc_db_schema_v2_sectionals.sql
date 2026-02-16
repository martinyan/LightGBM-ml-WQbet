-- HKJC SQLite schema upgrade v2: dynamic sectional splits

CREATE TABLE IF NOT EXISTS sectional_splits (
  runner_id INTEGER NOT NULL REFERENCES runners(runner_id) ON DELETE CASCADE,
  split_idx INTEGER NOT NULL,          -- 1..K in order of columns
  split_label TEXT,                    -- optional header label (e.g. 400M/800M/...)
  pos INTEGER,
  split_time REAL,                     -- seconds
  PRIMARY KEY (runner_id, split_idx)
);

CREATE INDEX IF NOT EXISTS idx_sectional_splits_runner ON sectional_splits(runner_id);
