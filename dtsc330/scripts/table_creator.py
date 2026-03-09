import sqlite3

def minimal_create_database():
    """The bare minimum to create a new sqlite database.
    """
    # Connections are just that-- connecting to a database
    # For sqlite, if the database doesn't exist, it will be created
    connection = sqlite3.connect('data/article_grant_db.sqlite')

    # We can run a query directly, but usually we use pandas.
    # In this case, we just wanted to check that it worked.
    print(connection.execute("SELECT sqlite_version();").fetchall())

    # There exist cursors, which you can think of as python generators
    # Know this exists, but ignore for now
    cursor = connection.cursor()
    cursor.execute('SELECT something FROM somewhere;')

minimal_create_database()