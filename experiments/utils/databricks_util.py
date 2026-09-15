CATALOG = "workspace"
SCHEMA = "lilypond_experiments"

get_catalog_path = lambda ent_name: f"{CATALOG}.{SCHEMA}.{ent_name}"

def get_spark():
    from dotenv import load_dotenv
    from databricks.connect import DatabricksSession
    load_dotenv()
    spark = DatabricksSession.builder.getOrCreate()
    return spark
