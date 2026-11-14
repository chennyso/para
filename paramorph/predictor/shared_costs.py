import json
from pathlib import Path
import sqlite3

from predictor.profile_database import ProfileDatabase, profile_execution_request, signature


TABLES = ('profiles', 'profile_sources', 'communication_profiles', 'overlap_profiles',
    'operator_costs', 'operator_families', 'operator_models')


def import_cost_databases(target, sources):
    database = ProfileDatabase(target)
    connection = database.connection
    connection.execute('CREATE TABLE IF NOT EXISTS imported_cost_sources '
        '(path TEXT PRIMARY KEY, identity TEXT NOT NULL)')
    result = dict(sources=0, inserted={table: 0 for table in TABLES})
    for source in sources:
        source = Path(source).resolve()
        if source == Path(target).resolve():
            continue
        wal = source.with_name(source.name+'-wal')
        identity = json.dumps([source.stat().st_mtime_ns, source.stat().st_size,
            [wal.stat().st_mtime_ns, wal.stat().st_size] if wal.is_file() else None])
        previous = connection.execute('SELECT identity FROM imported_cost_sources WHERE path=?',
            (str(source),)).fetchone()
        if previous is not None and previous[0] == identity:
            continue
        with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as origin:
            origin.execute('BEGIN')
            tables = {row[0] for row in origin.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            with connection:
                for table in TABLES:
                    if table not in tables:
                        continue
                    columns = [row[1] for row in connection.execute(f'PRAGMA table_info({table})')]
                    assert columns == [row[1] for row in origin.execute(f'PRAGMA table_info({table})')]
                    for values in origin.execute(f'SELECT * FROM {table}'):
                        if table == 'profiles':
                            document = json.loads(values[2])
                            values = (values[0], signature(profile_execution_request(document)), values[2])
                        existing = connection.execute(f'SELECT * FROM {table} WHERE {columns[0]}=?',
                            (values[0],)).fetchone()
                        if existing is not None:
                            assert existing == values, (table, values[0], str(source))
                            continue
                        connection.execute(f'INSERT INTO {table} VALUES ({",".join("?" for _ in values)})', values)
                        result['inserted'][table] += 1
                connection.execute('INSERT INTO imported_cost_sources VALUES (?, ?) '
                    'ON CONFLICT(path) DO UPDATE SET identity=excluded.identity', (str(source), identity))
        result['sources'] += 1
    connection.close()
    return result
