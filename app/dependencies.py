from run import app

def get_db_pool():
    return app.state.db_pool