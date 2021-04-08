import psycopg2


def connect():
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(dbname='gleiph', user='heleno', password='heleno', port=32146, host='localhost')
        cur = conn.cursor()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        return conn, cur

def close(conn, cur):
    try:
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        conn.close()


def get_conflict(chunk_id):
    conn, cur = connect()
    conflict = []
    try:
        query = f"select content from conflictingcontent where conflictingchunk_id = {chunk_id} order by id"
        cur.execute(query)
        for row in cur.fetchall():
            conflict.append(row[0])
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        close(conn, cur)
        return conflict

def get_conflict_position(chunk_id):
    conn, cur = connect()
    beginline = 0
    endline = 0
    try:
        query = f"select beginline, endline from conflictingchunk where id = {chunk_id}"
        cur.execute(query)
        for row in cur.fetchall():
            beginline, endline = row
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        close(conn, cur)
        return beginline, endline

def get_solution(chunk_id):
    conn, cur = connect()
    solution = []
    try:
        query = f"select content from solutioncontent where conflictingchunk_id = {chunk_id} order by id"
        cur.execute(query)
        for row in cur.fetchall():
            solution.append(row[0])
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        close(conn, cur)
        return solution