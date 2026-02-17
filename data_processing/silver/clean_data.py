from loguru import logger
import pandas as pd
import typer

from data_processing.check_s3 import wait_for_s3_object
from iris_project.config import S3_BRONZE_PREFIX, S3_BUCKET, S3_SILVER_PREFIX

app = typer.Typer()


@app.command()
def main(
    input_path: str = f"s3://{S3_BUCKET}/{S3_BRONZE_PREFIX}/iris.parquet",
    output_path: str = f"s3://{S3_BUCKET}/{S3_SILVER_PREFIX}/iris.parquet",
):
    """Silver Layer Cleaning - Load Bronze dataset, apply cleaning, and save to Silver."""
    logger.info(f"Loading Bronze dataset from: {input_path}")
    wait_for_s3_object(S3_BUCKET, f"{S3_BRONZE_PREFIX}/iris.parquet", timeout=60)

    df = pd.read_parquet(input_path, storage_options={"anon": False})

    # Rename columns to snake_case
    logger.info("Renaming columns to snake_case...")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Drop id column if present
    if "id" in df.columns:
        logger.info("Dropping 'id' column...")
        df = df.drop(columns=["id"])

    # Normalize species names (strip 'Iris-' prefix)
    if "species" in df.columns:
        logger.info("Normalizing species names...")
        df["species"] = df["species"].str.replace("Iris-", "", regex=False).str.lower().str.strip()

    # Ensure float64 types for measurement columns
    logger.info("Ensuring correct data types...")
    measurement_cols = [c for c in df.columns if c != "species" and c != "ingestion_time"]
    for col in measurement_cols:
        df[col] = df[col].astype("float64")

    # Save Silver
    logger.info(f"Saving Silver dataset to: {output_path}")
    df.to_parquet(output_path, index=False, storage_options={"anon": False})
    logger.success("Silver dataset created successfully!")
    print(df.head())


if __name__ == "__main__":
    app()
