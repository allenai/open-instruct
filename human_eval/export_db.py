import sqlite3
import pandas as pd


if __name__ == "__main__":
    # database connection
    DATABASE = "data/evaluation.db"
    DB_CONN = sqlite3.connect(DATABASE, check_same_thread=False)
    DB_CURSOR = DB_CONN.cursor()

    # export the evaluation results as excel
    evaluation_results = pd.read_sql_query("SELECT * from evaluation_record", DB_CONN)
    evaluation_results.to_excel("data/eval_annotations.xlsx", index=False)

