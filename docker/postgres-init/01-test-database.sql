-- Runs once, the first time the dgb_pgdata volume is created.
-- Separate database for the automated tests (the test suite wipes it; its name must end in _test).
CREATE DATABASE dgb_test OWNER dgb;
